"""Audit a conflicting printed-ring alias against untouched source records.

This source-QC audit constructs no forecast cases or target labels. Animal and
record IDs establish lineage; coordinates only check measurement concordance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wildcast.data import distance


SOURCE_ID = "8318572a-a9d8-4d9e-b685-7db8a73bdf86"
LOCAL_IDS = ("GG_2017_D84_Leinewijk", "GG_2017_D85_Leinewijk")
REFERENCE_FIELDS = ("tag-id", "animal-id", "animal-taxon", "deploy-on-date",
                    "deploy-off-date", "animal-life-stage", "animal-ring-id",
                    "animal-sex", "deployment-id", "manipulation-type",
                    "tag-manufacturer-name", "tag-mass")


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def compare(frame):
    """Compare exact shared nominal instants, without nearest-time matching."""
    groups = []
    for name in LOCAL_IDS:
        part = frame.loc[frame.local_id.eq(name)].copy()
        if part.duplicated("timestamp").any():
            raise ValueError("This pair review requires unambiguous within-raw-ID timestamps")
        groups.append(part)
    paired = groups[0].merge(groups[1], on="timestamp", suffixes=("_a", "_b"), validate="one_to_one")
    separation = distance(paired.lat_a.to_numpy(), paired.lon_a.to_numpy(),
                          paired.lat_b.to_numpy(), paired.lon_b.to_numpy())
    exact = (paired.lat_a == paired.lat_b) & (paired.lon_a == paired.lon_b)
    proof = paired[["record_a", "record_b"]].to_numpy(dtype="<i8")
    return {"raw_counts": {name: len(part) for name, part in zip(LOCAL_IDS, groups)},
            "shared_exact_nominal_timestamps": len(paired),
            "exact_coordinate_matches": int(exact.sum()),
            "nonidentical_coordinate_pairs": int((~exact).sum()),
            "maximum_separation_metres": float(separation.max()),
            "separation_quantiles_metres": np.quantile(separation, [.5, .9, .99]).tolist(),
            "pairs_above_1m": int((separation > 1).sum()),
            "pairs_above_100m": int((separation > 100).sum()),
            "pairs_above_1km": int((separation > 1000).sum()),
            "ordered_original_record_pairs_sha256": hashlib.sha256(proof.tobytes()).hexdigest()}


def review(source_root, observations, repair, output):
    source_root, observations, repair, output = map(Path, (source_root, observations, repair, output))
    if output.exists():
        raise FileExistsError(output)
    source_path = source_root / "source.json"
    manifest = json.loads(source_path.read_text())
    verified = []
    for record in manifest["files"]:
        if record.get("role") not in {"events", "identity_reference", "source_documentation"}:
            continue
        path = source_root / record["name"]
        if not path.exists():
            path = source_root / record["stored_as"]
        checksum = record["published_checksum"]
        algorithm = checksum["checkSumAlgorithm"].lower().replace("-", "")
        if (record.get("status") != "complete" or digest(path) != record["sha256"]
                or digest(path, algorithm) != checksum["value"].lower()):
            raise ValueError("Original/public checksum differs: " + str(path))
        verified.append({"path": str(path), "sha256": record["sha256"],
                         "role": record["role"], "published_checksum": checksum,
                         "bytes": path.stat().st_size})
    events = next(row for row in verified if row["role"] == "events")
    reference = next(row for row in verified if row["role"] == "identity_reference")
    refs = []
    with Path(reference["path"]).open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            if row["animal-id"] in LOCAL_IDS:
                refs.append({"line": line, **{key: row.get(key, "") for key in REFERENCE_FIELDS}})
    if len(refs) != 2 or {row["animal-id"] for row in refs} != set(LOCAL_IDS):
        raise ValueError("Expected exactly two independent reference records")
    raw = []
    with Path(events["path"]).open(newline="", encoding="utf-8-sig") as stream:
        for record, row in enumerate(csv.DictReader(stream), start=1):
            if row["individual-local-identifier"] not in LOCAL_IDS:
                continue
            raw.append({"local_id": row["individual-local-identifier"], "record": record,
                        "timestamp_reported": row["timestamp"], "latitude_reported": row["location-lat"],
                        "longitude_reported": row["location-long"], "tag": row["tag-local-identifier"],
                        "event": row["event-id"], "sensor": row["sensor-type"], "visible": row["visible"]})
    raw = pd.DataFrame(raw)
    if not raw.sensor.str.upper().eq("GPS").all() or not raw.visible.str.lower().eq("true").all():
        raise ValueError("Selected original rows require separately documented source QC")
    raw["timestamp"] = pd.to_datetime(raw.timestamp_reported, utc=True).dt.as_unit("ns")
    raw["lat"] = pd.to_numeric(raw.latitude_reported)
    raw["lon"] = pd.to_numeric(raw.longitude_reported)
    contract = json.loads(repair.read_text())
    names = sorted(name for name in contract["physical_mapping"] if name.startswith(SOURCE_ID + ":")
                   and name.rsplit(":", 1)[-1] in LOCAL_IDS)
    if len(names) != 2:
        raise ValueError("Repair must contain the exact two original namespaces")
    columns = ["animal", "timestamp", "lat", "lon", "original_source_record", "original_source_sha256",
               "individual_id_reported", "tag_id_reported", "event_id_reported", "timestamp_reported",
               "latitude_reported", "longitude_reported"]
    qc = pd.read_parquet(observations, columns=columns, filters=[("animal", "in", names)])
    qc["record"] = pd.to_numeric(qc.original_source_record).astype("int64")
    qc["local_id"] = qc.individual_id_reported
    qc["timestamp"] = pd.to_datetime(qc.timestamp, utc=True).dt.as_unit("ns")
    if not qc.original_source_sha256.eq(events["sha256"]).all() or qc.record.duplicated().any():
        raise ValueError("QC rows lack unique original-file/record lineage")
    joined = qc.merge(raw, on="record", suffixes=("_qc", "_raw"), how="left", validate="one_to_one")
    if len(joined) != len(qc) or joined.local_id_raw.isna().any():
        raise ValueError("QC source records are missing from the original")
    for key in ["local_id", "timestamp", "lat", "lon", "timestamp_reported", "latitude_reported", "longitude_reported"]:
        if not np.array_equal(joined[key + "_qc"].to_numpy(), joined[key + "_raw"].to_numpy()):
            raise ValueError("QC/original lineage differs: " + key)
    for key, rawkey in [("tag_id_reported", "tag"), ("event_id_reported", "event")]:
        if not np.array_equal(joined[key].to_numpy(), joined[rawkey].to_numpy()):
            raise ValueError("QC/original identifiers differ: " + key)
    raw_comparison, qc_comparison = compare(raw), compare(qc)
    result = {"schema": "wildcast-conflicting-ring-reference-review-v1", "complete": True,
              "scope": "Original measurement/reference identity QC only, with no forecast cases, targets, predictions or model-error selection",
              "code": {"path": str(Path(__file__)), "sha256": digest(__file__)},
              "source_manifest": {"path": str(source_path), "sha256": digest(source_path)},
              "originals": verified, "repair": {"path": str(repair), "sha256": digest(repair)},
              "observations": {"path": str(observations), "sha256": digest(observations)},
              "reference_records": refs, "original_GPS": raw_comparison, "retained_QC_GPS": qc_comparison,
              "original_row_lineage_verified": {"rows": len(qc), "key": ["original_source_sha256", "original_source_record"],
                  "exactly_equal": ["raw_animal_ID", "timestamp", "lat", "lon", "reported_decimal_strings", "event_ID", "tag_ID"]},
              "retained_edges": [], "retracted_edges": [{"a": names[0], "b": names[1],
                  "type": "unsupported_physical_alias_despite_duplicate_printed_ring",
                  "reason": "Distinct animal/deployment IDs and tags3884/3885 have conflicting simultaneous original GPS. Shared printed ring8064094, sex/vendor and coarse common deployment date do not independently certify one biological animal. Conservatively retract both duplicate physical primitives and preserve their whole-animal exclusion union.",
                  "status": "risk_only_do_not_merge_coordinates"}],
              "limitations": ["Cannot determine which printed ring/reference field is mistaken or certify biological identity from GPS separation alone.",
                  "GPS sensor/source clock/printed coordinate precision is not measurement accuracy.",
                  "No coordinates are rounded or reallocated; no per-timestamp first-row preference or inferred replacement animal is selected.",
                  "This retracts one unsupported physical join only. All inherited role, exclusion and nonfreshness protections remain."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "sha256": digest(output), "qc_rows_verified": len(qc),
                      "original_shared": raw_comparison, "qc_shared": qc_comparison}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="data/migration-breadth-stage1/raw/" + SOURCE_ID)
    parser.add_argument("--observations", default="data/migration-breadth-clock-v1/processed/observations.parquet")
    parser.add_argument("--repair", default="reports/migration-parent-physical-retraction-v3.json")
    parser.add_argument("--output", required=True)
    review(**vars(parser.parse_args()))
