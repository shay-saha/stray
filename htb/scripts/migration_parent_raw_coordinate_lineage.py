"""Verify retained parent times/positions against checksum-pinned publisher originals."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(parent, review, output):
    parent, review = Path(parent), Path(review)
    manifest, checked = json.loads(parent.read_text()), json.loads(review.read_text())
    edges = checked["pair_reviews"]
    names = {r[k] for r in edges for k in ["a", "b"]}
    source_ids = sorted({n.split(":", 1)[0] for n in names})
    result = {"schema": "wildcast-parent-original-coordinate-lineage-v1", "code_sha256": digest(__file__),
              "parent_manifest": {"path": str(parent), "sha256": digest(parent)},
              "coordinate_review": {"path": str(review), "sha256": digest(review)}, "sources": [],
              "interpretation": ["Every compared QC time/latitude/longitude tuple must occur unchanged in a publisher original for the same raw animal namespace",
                                 "The cleaner thins by an hour bucket but retains the selected ORIGINAL timestamp; it never replaces timestamps with bucket starts",
                                 "Direct original-tuple membership excludes cleaner-introduced timestamp flooring and coordinate rounding as explanations for the reviewed coincidences",
                                 "Source-native rounding and source-native repeated timestamps remain source properties, not independently measured biological identity",
                                 "No case/target cache, model errors, fates or outcomes are used"]}
    for source_id in source_ids:
        source_dir = Path("data/migration-expanded/raw") / source_id
        source_path = source_dir / "source.json"
        source = json.loads(source_path.read_text())
        source_names = sorted(n for n in names if n.startswith(source_id + ":"))
        frames = [pq.read_table(s["path"], columns=["animal", "timestamp", "lat", "lon"],
                               filters=[("animal", "in", source_names)]).to_pandas() for s in manifest["sources"]]
        qc = pd.concat(frames, ignore_index=True).drop_duplicates(["animal", "timestamp", "lat", "lon"])
        qc["ns"] = qc.timestamp.astype("int64")
        qc_index = pd.MultiIndex.from_frame(qc[["animal", "ns", "lat", "lon"]])
        matched = np.zeros(len(qc), dtype=bool)
        files, raw_count, selected_count, off_hour = [], 0, 0, 0
        samples = []
        selected_fields = ["event-id", "timestamp", "location-long", "location-lat", "sensor-type",
                           "individual-taxon-canonical-name", "tag-local-identifier", "individual-local-identifier", "study-name"]
        for item in source["files"]:
            if not item.get("derived"):
                continue
            original = source_dir / item["stored_as"]
            actual = digest(original)
            if actual != item["sha256"]:
                raise ValueError("Original SHA mismatch")
            published = item["published_checksum"]
            algorithm = published["checkSumAlgorithm"].lower().replace("-", "")
            if digest(original, algorithm) != published["value"].lower():
                raise ValueError("Published original checksum mismatch")
            for derived in item["derived"]:
                if not derived.get("chunks"):
                    continue
                members = [derived.get("archive_member")]
                member = members[0]
                archive = zipfile.ZipFile(original) if member else None
                stream = archive.open(member) if member else original
                member_info = archive.getinfo(member) if archive else None
                stream_hash = hashlib.sha256()
                # Record parsed source field strings in bounded witnesses. Full
                # source bytes retain their original checksum and row lineage.
                records = 0
                for block in pd.read_csv(stream, usecols=selected_fields, dtype=str, keep_default_na=False,
                                         encoding=derived.get("source_encoding", "utf-8-sig"), chunksize=150_000):
                    block["source_row"] = np.arange(records+2, records+len(block)+2)
                    records += len(block)
                    animal = (source_id + ":" + block["study-name"] + ":" + block["individual-taxon-canonical-name"]
                              + ":" + block["individual-local-identifier"])
                    selected = block.loc[animal.isin(source_names)].copy()
                    selected["animal"] = animal.loc[selected.index]
                    raw_count += len(block)
                    selected_count += len(selected)
                    if not len(selected):
                        continue
                    stamps = pd.to_datetime(selected.timestamp, utc=True, format="mixed", errors="raise")
                    selected["ns"] = stamps.astype("int64")
                    selected["lat"] = pd.to_numeric(selected["location-lat"], errors="raise")
                    selected["lon"] = pd.to_numeric(selected["location-long"], errors="raise")
                    raw_index = pd.MultiIndex.from_frame(selected[["animal", "ns", "lat", "lon"]])
                    matched |= qc_index.isin(raw_index)
                    off_hour += int(((selected["ns"] % int(3600e9)) != 0).sum())
                    # Show both a native non-hour timestamp and the decisive
                    # MD18F0082/MD18M0104 separated/coincident observations.
                    witness = ((selected["individual-local-identifier"].isin(["MD18F0082", "MD18M0104"]))
                               & selected.timestamp.str.startswith(("2019-01-04 07:00:00", "2019-03-07 19:00:00")))
                    for record in selected.loc[witness, [*selected_fields, "source_row"]].to_dict("records"):
                        record["publisher_original"] = str(original)
                        record["archive_member"] = member
                        samples.append(record)
                    if not any(r.get("witness_kind") == "native_non_hour_timestamp" for r in samples):
                        subset = selected.loc[(selected["ns"] % int(3600e9)) != 0]
                        if len(subset):
                            record = subset[[*selected_fields, "source_row"]].iloc[0].to_dict()
                            record.update({"publisher_original": str(original), "archive_member": member,
                                           "witness_kind": "native_non_hour_timestamp"})
                            samples.append(record)
                if archive:
                    stream.close(); archive.close()
                chunk_proofs = []
                for chunk in derived["chunks"]:
                    path = source_dir / chunk["name"]
                    if digest(path) != chunk["sha256"]:
                        raise ValueError("Derived chunk SHA mismatch")
                    chunk_proofs.append({"path": str(path), "sha256": chunk["sha256"], "rows": chunk["rows"]})
                files.append({"path": str(original), "sha256": actual, "published_checksum": published,
                              "archive_member": member, "archive_member_CRC32": member_info.CRC if member_info else None,
                              "raw_parsed_rows": records, "derived_chunks": chunk_proofs})
        unmatched = qc.loc[~matched, ["animal", "timestamp", "lat", "lon"]]
        if len(unmatched):
            raise ValueError(f"{len(unmatched)} retained observations do not occur exactly in publisher originals")
        result["sources"].append({"source_id": source_id, "source_manifest": str(source_path),
                                  "source_manifest_sha256": digest(source_path), "files": files,
                                  "selected_source_ids": source_names, "publisher_rows_scanned": raw_count,
                                  "publisher_rows_for_selected_ids": selected_count,
                                  "QC_rows_verified_exact_original_tuple": len(qc), "QC_unmatched_rows": 0,
                                  "native_selected_rows_not_on_hour_boundary": off_hour,
                                  "raw_witnesses": samples})
    result["normalizer_lineage"] = {
        "data_reader": {"path": "wildcast/data.py", "current_sha256": digest("wildcast/data.py"),
                        "frozen_parent_sha256": manifest["preparation_code_sha256"]["data.py"]},
        "cleaner": {"path": "scripts/migration_clean.py", "current_sha256": digest("scripts/migration_clean.py"),
                    "historical_cleaner_code_hash_recorded": False,
                    "limitation": "Direct original-row verification supplies the evidence; current cleaner hash is not represented as a historical snapshot"},
        "quality_report": {"path": "data/migration-expanded-clean-v2/processed/quality-report.json",
                           "sha256": digest("data/migration-expanded-clean-v2/processed/quality-report.json")},
        "acquisition_chunk_writer": {"path": "scripts/migration_acquire.py", "current_sha256": digest("scripts/migration_acquire.py"),
                                     "limitation": "Current source only; exact ORIGINAL and derived-chunk hashes are authoritative"}}
    if result["normalizer_lineage"]["data_reader"]["current_sha256"] != manifest["preparation_code_sha256"]["data.py"]:
        raise ValueError("Current data reader differs from frozen parent source")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": digest(output),
                      "verified_QC_rows": sum(r["QC_rows_verified_exact_original_tuple"] for r in result["sources"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.parent, args.review, args.output)
