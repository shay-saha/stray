"""Project only observation IDs/times/taxa to census actual unique fix intervals."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(identity, output):
    started = time.time()
    identity = Path(identity)
    evidence = json.loads(identity.read_text())
    physical = evidence["fold_groups"]
    parent = json.loads(Path(evidence["parent_manifest"]["path"]).read_text())
    groups = {"new_clock_roots": {r["animal"] for r in evidence["new_animal_parent_bindings"]},
              "parent_v3": set(parent["identity_mapping"])}
    groups["parent_plus_new"] = groups["new_clock_roots"] | groups["parent_v3"]
    times, taxa = collections.defaultdict(list), collections.defaultdict(set)
    inputs = []
    for item in evidence["observations"]:
        path = Path(item["path"])
        file = pq.ParquetFile(path)
        # Source hashes were already verified by the frozen identity audit.
        # Verify again before deriving a new report, never read target caches.
        if sha(path) != item["sha256"]:
            raise ValueError(f"Observation hash mismatch: {path}")
        count = 0
        for batch in file.iter_batches(batch_size=250_000, columns=["animal", "timestamp", "species"], use_threads=False):
            frame = batch.to_pandas()
            if frame["timestamp"].isna().any():
                raise ValueError("Missing clock-compatible observation time")
            mapped = frame["animal"].map(physical)
            if mapped.isna().any():
                raise ValueError("Observation identity missing from frozen audit")
            frame["ns"] = pd.to_datetime(frame["timestamp"], utc=True).astype("int64")
            for group, block in frame.groupby("animal", observed=True, sort=False):
                times[group].append(block["ns"].to_numpy(copy=True))
                taxa[group].update(block["species"].astype(str).unique())
            count += len(frame)
        inputs.append({"path": str(path), "sha256": item["sha256"], "rows": count})
        print(json.dumps({"completed": str(path), "rows": count, "elapsed_seconds": time.time()-started}), flush=True)
    rows = []
    limits = [300, 900, 3600]
    for group, pieces in sorted(times.items()):
        stamps = np.unique(np.concatenate(pieces))
        seconds = np.diff(stamps) / 1e9
        row = {"source_animal_id": group, "frozen_physical_id": physical[group], "raw_species": sorted(taxa[group]), "unique_fix_times": len(stamps),
               "positive_intervals": len(seconds), "span_days": float((stamps[-1]-stamps[0]) / 86400e9),
               "median_seconds": float(np.median(seconds)) if len(seconds) else None,
               "intervals_at_most_seconds": {str(limit): int((seconds <= limit).sum()) for limit in limits}}
        rows.append(row)
    summaries = {}
    for scope, members in groups.items():
        selected = [row for row in rows if row["source_animal_id"] in members]
        denominator = sum(row["positive_intervals"] for row in selected)
        summaries[scope] = {"source_animal_ids": len(selected), "unique_fix_times": sum(r["unique_fix_times"] for r in selected),
                            "positive_intervals": denominator, "thresholds": {}}
        for limit in limits:
            count = sum(row["intervals_at_most_seconds"][str(limit)] for row in selected)
            summaries[scope]["thresholds"][str(limit)] = {
                "intervals": count, "pooled_interval_fraction": count / denominator if denominator else None,
                "source_ids_with_median_at_most": sum(r["median_seconds"] is not None and r["median_seconds"] <= limit for r in selected),
                "source_ids_with_eight_intervals_and_one_hour_span": sum(r["intervals_at_most_seconds"][str(limit)] >= 8 and r["span_days"] >= 1/24 for r in selected),
                "equal_source_id_mean_interval_fraction": float(np.mean([r["intervals_at_most_seconds"][str(limit)] / r["positive_intervals"] for r in selected if r["positive_intervals"]]))}
    result = {"schema": "wildcast-actual-fix-cadence-audit-v1", "code_sha256": sha(__file__),
              "identity_audit": {"path": str(identity), "sha256": sha(identity)}, "inputs": inputs,
              "scope": "Raw/QC observation times only; no model cases, target arrays or outcomes read",
              "units": "Consecutive distinct observation-time differences in seconds per original source animal ID; these are not independent biological animals",
              "summaries": summaries, "source_animal_ids": rows, "elapsed_seconds": time.time()-started,
              "limitations": ["Cadence is irregular: short bursts do not establish sustained coverage or independent origins",
                              "Raw taxon strings are preserved, not new named-species or model eligibility claims",
                              "Times are unique within original source ID across roots; aliases can duplicate releases and hence pooled weights",
                              "Frozen physical aliases are retained only as provenance, never used to blend separate source trajectories for cadence",
                              "Parent nominal clock uncertainty remains as documented; new roots use frozen clock compatibility policies",
                              "Observed interval distribution is not a velocity/turning feature and supplies no future input to models"]}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "elapsed_seconds": time.time()-started,
                      "summaries": summaries}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.identity, args.output)
