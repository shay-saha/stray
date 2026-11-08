"""Certify eight source-native record preferences without coordinate tolerance."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SOURCE_ID = "a1d2d3ab-adcd-4101-8775-e0559445b612"
LOCAL_IDS = {"Osen 2678", "Smena 2675", "Vesna 2676"}
PREFIX = SOURCE_ID + ":LifeTrack Rough-legged buzzards:Buteo lagopus:"


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def certify(source_root, observations, parent_manifest, repair, output):
    source_root, observations, parent_manifest, repair, output = map(
        Path, (source_root, observations, parent_manifest, repair, output))
    if output.exists():
        raise FileExistsError(output)
    ledger_path = source_root / "source.json"
    ledger = json.loads(ledger_path.read_text())
    originals = []
    for record in ledger["files"]:
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
        originals.append({"path": str(path), "sha256": record["sha256"], "role": record["role"],
                          "published_checksum": checksum, "bytes": path.stat().st_size})
    event = next(row for row in originals if row["role"] == "events")
    raw_rows = []
    with Path(event["path"]).open(newline="", encoding="utf-8-sig") as stream:
        for record, row in enumerate(csv.DictReader(stream), start=1):
            if row["individual-local-identifier"] in LOCAL_IDS:
                raw_rows.append({"record": record, **row})
    raw = pd.DataFrame(raw_rows).set_index("record", drop=False)
    contract = json.loads(repair.read_text())
    names = sorted(PREFIX + name for name in LOCAL_IDS)
    bindings = [row for row in contract["raw_coordinate_ownership"]
                if row["input_animal"] in names and row["input_path"] == str(observations)]
    hashes = {row["input_sha256"] for row in bindings}
    if len(bindings) != 3 or len(hashes) != 1 or digest(observations) not in hashes:
        raise ValueError("Unexpected processed observation source binding")
    input_hash = next(iter(hashes))
    columns = ["animal", "timestamp", "lat", "lon", "original_source_record", "original_source_sha256",
               "individual_id_reported", "tag_id_reported", "event_id_reported", "timestamp_reported",
               "latitude_reported", "longitude_reported"]
    qc = pd.read_parquet(observations, columns=columns, filters=[("animal", "in", names)])
    qc["record"] = pd.to_numeric(qc.original_source_record).astype("int64")
    qc["timestamp"] = pd.to_datetime(qc.timestamp, utc=True).dt.as_unit("ns")
    if (qc.record.duplicated().any() or not qc.original_source_sha256.eq(event["sha256"]).all()
            or not set(qc.record) <= set(raw.index)):
        raise ValueError("Original-file/record ownership is not unique")
    selected = raw.loc[qc.record]
    pairs = [("individual_id_reported", "individual-local-identifier"),
             ("tag_id_reported", "tag-local-identifier"), ("event_id_reported", "event-id"),
             ("timestamp_reported", "timestamp"), ("latitude_reported", "location-lat"),
             ("longitude_reported", "location-long")]
    for current, original in pairs:
        if not np.array_equal(qc[current].to_numpy(), selected[original].to_numpy()):
            raise ValueError("Reported original-row fields differ: " + current)
    for current, original in [("lat", "location-lat"), ("lon", "location-long")]:
        if not np.array_equal(qc[current].to_numpy(), pd.to_numeric(selected[original]).to_numpy()):
            raise ValueError("Processed numeric parser differs: " + current)
    if not np.array_equal(qc.timestamp.to_numpy(), pd.to_datetime(selected.timestamp, utc=True).dt.as_unit("ns").to_numpy()):
        raise ValueError("Processed timestamp differs from original reported time")
    positions = qc.drop_duplicates(["animal", "timestamp", "lat", "lon"])
    conflicts = positions.loc[positions.duplicated(["animal", "timestamp"], keep=False)]
    preferences = []
    native_fields = ["eobs:key-bin-checksum", "eobs:status", "eobs:type-of-fix",
                     "eobs:start-timestamp", "eobs:used-time-to-get-fix"]
    for (animal, time), group in conflicts.groupby(["animal", "timestamp"], sort=True):
        if len(group) != 2:
            raise ValueError("Expected two published variants at each reviewed instant")
        copies = raw.loc[group.record]
        native = copies.loc[copies["eobs:key-bin-checksum"].ne("")]
        sparse = copies.loc[copies["eobs:key-bin-checksum"].eq("")]
        if len(native) != 1 or len(sparse) != 1:
            raise ValueError("Native-fix record preference is not uniquely supported")
        native, sparse = native.iloc[0], sparse.iloc[0]
        if (native["eobs:status"] != "A" or native["eobs:type-of-fix"] != "3"
                or any(sparse[field] for field in native_fields)):
            raise ValueError("Unexpected source-native/sparse field semantics")
        start = pd.Timestamp(native["eobs:start-timestamp"], tz="UTC")
        if start + pd.Timedelta(float(native["eobs:used-time-to-get-fix"]), unit="s") != time:
            raise ValueError("Native-fix acquisition metadata does not reproduce its issue time")
        for field in ["individual-local-identifier", "tag-local-identifier", "timestamp", "sensor-type", "visible"]:
            if native[field] != sparse[field]:
                raise ValueError("Published variants have different source identities/times/QC")
        if native["sensor-type"].lower() != "gps" or native["visible"].lower() != "true":
            raise ValueError("Preferred original is not an observed visible GPS record")
        arithmetic = []
        for axis in ["location-lat", "location-long"]:
            integer = Decimal(native[axis]) * 10**7
            if integer != integer.to_integral_value():
                raise ValueError("Native value is not exactly the stated integer representation")
            integer = int(integer)
            if float(native[axis]) != integer / 10**7 or float(sparse[axis]) != integer * 10**-7:
                raise ValueError("The two explicit arithmetic representations do not reproduce publisher values exactly")
            arithmetic.append({"axis": axis, "common_integer": integer,
                "native_publisher_decimal": native[axis], "sparse_publisher_decimal": sparse[axis],
                "native_float64_hex": float(native[axis]).hex(), "sparse_float64_hex": float(sparse[axis]).hex(),
                "native_equals_integer_divide_1e7": True, "sparse_equals_integer_multiply_1e_minus7": True})
        preferences.append({"input_animal": animal, "nominal_timestamp": time.isoformat(),
            "original_source_sha256": event["sha256"], "preferred_original_record": int(native.record),
            "preferred_event_id": native["event-id"], "suppressed_original_record": int(sparse.record),
            "suppressed_event_id": sparse["event-id"], "tag_id": native["tag-local-identifier"],
            "native_fix_evidence": {field: native[field] for field in native_fields},
            "exact_float64_arithmetic": arithmetic})
    if len(preferences) != 8 or len(conflicts) != 16:
        raise ValueError("Reviewed source has a different conflict census")
    # Recover FULL processed-input row positions without assuming the filtered
    # table preserves Parquet's RangeIndex. No coordinates/future labels read.
    needed = {value for row in preferences for value in
              [row["preferred_original_record"], row["suppressed_original_record"]]}
    found = {}
    parquet = pq.ParquetFile(observations)
    count = 0
    for batch in parquet.iter_batches(batch_size=65536, columns=["original_source_sha256", "original_source_record", "event_id_reported", "animal"]):
        matching = pc.and_(pc.equal(batch.column(0), event["sha256"]),
                           pc.is_in(batch.column(1), value_set=pa.array(sorted(needed), type=batch.column(1).type)))
        offsets = np.flatnonzero(matching.to_numpy(zero_copy_only=False))
        for index in offsets:
            original_record = int(batch.column(1)[int(index)].as_py())
            if original_record in found:
                raise ValueError("An original source record appears twice in the processed input")
            found[original_record] = {"position": count + int(index),
                "event_id": batch.column(2)[int(index)].as_py(), "animal": batch.column(3)[int(index)].as_py()}
        count += len(batch)
    if set(found) != needed or count != parquet.metadata.num_rows:
        raise ValueError("Full-input positional ownership is incomplete")
    for row in preferences:
        for label, record_key, event_key in [("preferred", "preferred_original_record", "preferred_event_id"),
                                             ("suppressed", "suppressed_original_record", "suppressed_event_id")]:
            entry = found[row[record_key]]
            if entry["event_id"] != row[event_key] or entry["animal"] != row["input_animal"]:
                raise ValueError("Full-input position does not identify the audited source record")
            row[label + "_input_row_position"] = entry["position"]
    parent = json.loads(parent_manifest.read_text())
    old_names = sorted(name for name in parent["animals"] if name.startswith(SOURCE_ID + ":"))
    if old_names:
        raise ValueError("This new-source-only preference now requires explicit frozen-parent origin accounting")
    result = {"schema": "wildcast-source-native-record-review-v1", "complete": True,
        "scope": "Published-source representation/measurement lineage QC only. No forecast cases, future target matching, model error, training or test scoring.",
        "code": {"path": str(Path(__file__)), "sha256": digest(__file__)},
        "source_manifest": {"path": str(ledger_path), "sha256": digest(ledger_path)}, "originals": originals,
        "observations": {"path": str(observations), "sha256": input_hash, "rows": count},
        "repair": {"path": str(repair), "sha256": digest(repair)},
        "frozen_parent": {"path": str(parent_manifest), "sha256": digest(parent_manifest), "affected_parent_animals": old_names,
            "old_parent_origin_coordinate_changes": 0},
        "summary": {"raw_IDs": 3, "QC_rows_exactly_bound_to_original_record": len(qc),
            "conflicting_nominal_timestamps": 8, "preferred_native_original_rows": 8,
            "suppressed_sparse_original_rows": 8, "coordinate_values_modified": 0,
            "original_event_IDs_are_distinct": True, "across_root_parser_discrepancy": False,
            "all_16_published_axes_reproduced_by_exact_integer_arithmetic": True},
        "record_preferences": preferences,
        "proposed_source_record_exclusions": [{"input_sha256": input_hash,
            "row_positions": sorted(row["suppressed_input_row_position"] for row in preferences),
            "row_number_basis": "zero_based_full_input_order", "input_rows": count,
            "reason": "Eight individually certified sparse import/representation variants; retain the documented checksum/type3 native GPS original event unchanged. No numeric/geographic tolerance or rounding."}],
        "limitations": ["Distinct event IDs and battery metadata are not byte-identical source events. Same source animal/tag/time plus exact decoder arithmetic strongly support coordinate representation variants, not publisher certification that event IDs are aliases.",
            "This is an explicit source-native record preference for these eight content-bound pairs, not a generic distance tolerance, inferred GPS accuracy, or blanket decimal normalization.",
            "No primary coordinate is rewritten: the preferred complete native event is retained as published; eight sparse copies are excluded by exact processed-input row position with original-record evidence.",
            "1e-7 here is an exact arithmetic explanation of publisher strings, not a statement of observed resolution or instrument accuracy.",
            "Another input SHA/order, source copy or new parent ancestry requires a new review. Preserve all original bytes, lineage and global physical/exclusion/role/freshness protections."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "sha256": digest(output), "summary": result["summary"],
                      "proposed_source_record_exclusions": result["proposed_source_record_exclusions"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="data/migration-breadth-stage1/raw/" + SOURCE_ID)
    parser.add_argument("--observations", default="data/migration-breadth-clock-v1/processed/observations.parquet")
    parser.add_argument("--parent-manifest", default="data/migration-expanded-dev-v3/manifest.json")
    parser.add_argument("--repair", default="reports/migration-parent-physical-retraction-v3.json")
    parser.add_argument("--output", required=True)
    certify(**vars(parser.parse_args()))
