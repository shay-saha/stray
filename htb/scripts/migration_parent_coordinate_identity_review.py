"""Recheck frozen coordinate-derived parent identities without editing them."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(parent, metadata, output):
    parent, metadata = Path(parent), Path(metadata)
    manifest, fields = json.loads(parent.read_text()), json.loads(metadata.read_text())
    links = [r for r in manifest["identity_links"] if r["evidence"] == "exact_species_time_GPS_overlap"]
    names = {r[k] for r in links for k in ["a", "b"]}
    frames, inputs = [], []
    for source in manifest["sources"]:
        path = Path(source["path"])
        if sha(path) != source["sha256"]:
            raise ValueError("Parent source hash mismatch")
        frame = pq.read_table(path, columns=["animal", "timestamp", "lat", "lon", "species"],
                              filters=[("animal", "in", sorted(names))]).to_pandas()
        frames.append(frame)
        inputs.append({"path": str(path), "sha256": source["sha256"], "selected_rows": len(frame)})
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(["animal", "timestamp", "lat", "lon"])
    by_animal = {name: block for name, block in frame.groupby("animal", observed=True, sort=False)}
    rows = []
    for link in links:
        a, b = [by_animal[link[k]] for k in ["a", "b"]]
        both = a.merge(b, on="timestamp", suffixes=("_a", "_b"))
        exact = (both.lat_a == both.lat_b) & (both.lon_a == both.lon_b)
        phi1, lam1, phi2, lam2 = [np.radians(both[column].to_numpy()) for column in ["lat_a", "lon_a", "lat_b", "lon_b"]]
        distance = 12742 * np.arcsin(np.sqrt(np.clip(np.sin((phi1-phi2)/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin((lam1-lam2)/2)**2, 0, 1)))
        fraction = float(exact.mean()) if len(both) else None
        alias = bool(len(both) >= 5 and exact.all())
        concern = bool(len(both) >= 20 and fraction < .1 and (distance > 1).sum() >= 3)
        groups = {manifest["identity_mapping"][link[k]] for k in ["a", "b"]}
        rows.append({"a": link["a"], "b": link["b"], "parent_link": link,
                     "parent_physical_ids": sorted(groups), "a_fixes": len(a), "b_fixes": len(b),
                     "a_start": a.timestamp.min().isoformat(), "a_end": a.timestamp.max().isoformat(),
                     "b_start": b.timestamp.min().isoformat(), "b_end": b.timestamp.max().isoformat(),
                     "simultaneous_fix_pairs": len(both), "exact_same_time_position_pairs": int(exact.sum()),
                     "exact_fraction_of_simultaneous_pairs": fraction,
                     "simultaneous_separation_km_quantiles_0_50_90_95_99_100": np.quantile(distance, [0,.5,.9,.95,.99,1]).tolist() if len(both) else None,
                     "simultaneous_pairs_above_1km": int((distance > 1).sum()),
                     "all_simultaneous_positions_identical": alias,
                     "rounded_coincidence_identity_concern": concern,
                     "conflicting_reference_sex_in_component": any(g in fields["sex_conflicting_physical_ids"] for g in groups)})
    concerns = [r for r in rows if r["rounded_coincidence_identity_concern"]]
    affected_groups = sorted({g for r in concerns for g in r["parent_physical_ids"]})
    affected_names = sorted(n for n, g in manifest["identity_mapping"].items() if g in affected_groups)
    result = {"schema": "wildcast-frozen-parent-coordinate-identity-review-v1", "code_sha256": sha(__file__),
              "parent_manifest": {"path": str(parent), "sha256": sha(parent)},
              "metadata_audit": {"path": str(metadata), "sha256": sha(metadata)}, "inputs": inputs,
              "summary": {"coordinate_links": len(rows),
                          "all_simultaneous_positions_identical_links": sum(r["all_simultaneous_positions_identical"] for r in rows),
                          "rounded_coincidence_identity_concern_links": len(concerns),
                          "concern_physical_components": len(affected_groups), "concern_source_animal_ids": len(affected_names)},
              "concern_physical_components": affected_groups, "concern_source_animal_ids": affected_names,
              "pair_reviews": rows,
              "scope": "Raw observation identity evidence only. No target cache labels, model predictions, errors or outcomes read; no identity/cache edits.",
              "interpretation": ["Full identical tracks support duplicate observations despite contradictory reference sex; biological identity still has source-metadata uncertainty",
                                 "Only a small coincident subset plus many simultaneous positions kilometres apart does not establish a common physical animal",
                                 "Concern rule is a conservative review flag: >=20 simultaneous fixes, <10% exact matches, >=3 simultaneous separations >1km",
                                 "Parent coordinate links were accepted by >=5 exact positions across >=1h; that threshold can match distinct nearby herd animals when coordinates are rounded",
                                 "Never split existing exclusion groups across folds to repair coordinates; any future physical separation must retain inherited conservative exclusion roles"]}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "summary": result["summary"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    review(args.parent, args.metadata, args.output)
