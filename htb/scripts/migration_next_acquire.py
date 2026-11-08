#!/usr/bin/env python3
"""Normalize the finite CC0 harrier release, preserving GPS and author labels.

The companion goose files are acquired by migration_acquire.py. This adapter
never modifies a prepared cache. High-resolution correction data remain only
as originals, because they are not independent annual tracks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wildcast.sources import download, dump, get_json, session

SOURCE = "dryad-84jk0"
STUDY = "Montagu annual cycles"
TAXON = "Circus pygargus"
DOI = "10.5061/dryad.84jk0"
LIMIT = 1_000_000_000
FLOOR = 30_000_000_000
CHUNK = 32_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def size(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink())


class Quota:
    def __init__(self, root):
        self.root, self.used = Path(root), size(root)
        self.check(0)

    def check(self, amount):
        if self.used + amount > LIMIT or shutil.disk_usage(self.root).free - amount < FLOOR:
            raise ValueError("Finite intake exceeds its 1GB quota or 30GB free-space floor")
        self.used += amount


def acquire_harrier(root):
    """Public Dryad metadata plus its byte-identical, publicly licensed mirror."""
    root = Path(root); directory = root / "raw" / SOURCE
    existing = directory / "source.json"
    if existing.exists() and json.loads(existing.read_text()).get("status") == "complete":
        return verify_complete(root)
    s = session()
    dataset = get_json(s, "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.84jk0")
    version = get_json(s, "https://datadryad.org" + dataset["_links"]["stash:version"]["href"])
    dryad_files = get_json(s, "https://datadryad.org" + version["_links"]["stash:files"]["href"])
    rights = get_json(s, "https://api.datacite.org/dois/" + DOI)
    mirror = get_json(s, "https://zenodo.org/api/records/4971089")
    if (not any(row.get("rightsIdentifier") == "cc0-1.0" for row in rights["data"]["attributes"]["rightsList"]) or
            mirror["metadata"]["license"]["id"] != "cc-zero"):
        raise ValueError("Harrier source/mirror requires explicit matching CC0 rights")
    metadata = {"dryad-dataset": dataset, "dryad-version": version, "dryad-files": dryad_files,
                "datacite": rights, "zenodo-record": mirror}
    for name, payload in metadata.items():
        dump(directory / "metadata" / (name + ".json"), payload)
    originals = {f["path"]: f for f in dryad_files["_embedded"]["stash:files"]}
    if len(originals) != 2:
        raise ValueError("Harrier finite intake expected exactly two declared CSV files")
    records = []
    for f in mirror["files"]:
        if f["key"] not in originals:
            continue
        original = originals[f["key"]]
        if (Path(f["key"]).name != f["key"] or f["checksum"] != "md5:" + original["digest"] or
                f["size"] != original["size"] or f["size"] > 150_000_000):
            raise ValueError("Public mirror metadata differs from original published bytes/checksum")
        path = directory / "originals" / f["key"]
        if not path.exists():
            Quota(root).check(f["size"])
        download(s, f["links"]["self"], path, f["size"], ("md5", original["digest"]))
        records.append({"name": f["key"], "stored_as": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                        "sha256": sha(path), "url": f["links"]["self"], "published_checksum": {"algorithm": "md5", "value": original["digest"]},
                        "mirror_checksum": f["checksum"], "description": original["description"],
                        "retrieved_utc": datetime.now(timezone.utc).isoformat()})
    if len(records) != 2:
        raise ValueError("The mirror did not provide both verified files")
    result = {"id": SOURCE, "title": dataset["title"], "doi": DOI, "url": "https://datadryad.org/dataset/doi:" + DOI,
              "mirror": "https://zenodo.org/records/4971089", "version": dataset["versionNumber"], "files": records,
              "metadata": {"dc.identifier.doi": ["doi:" + DOI], "dc.rights": ["CC0 1.0 Universal"],
                           "dc.rights.uri": ["https://creativecommons.org/publicdomain/zero/1.0/"], "sensor.name": ["GPS"]},
              "status": "originals_complete_pending_schema_audit",
              "retrieval_note": "Anonymous Dryad bytes returned HTTP401/403 locally. The public CC0 Zenodo mirror agrees exactly with original names, sizes and MD5. No private access used."}
    dump(directory / "source.json", result)
    return result


class CsvChunks:
    def __init__(self, directory, header, quota):
        self.directory, self.header, self.quota = directory, header, quota
        self.records, self.stream, self.n, self.rows = [], None, 0, 0
        self.buffer = io.StringIO(newline=""); self.writer = csv.writer(self.buffer)

    def encoded(self, row):
        self.buffer.seek(0); self.buffer.truncate(0); self.writer.writerow(row)
        return self.buffer.getvalue().encode("utf-8")

    def close(self):
        if self.stream is not None:
            self.stream.close(); self.stream = None
            record = self.records[-1]
            part = self.directory / (record["name"] + ".part")
            record.update({"bytes": self.n, "rows": self.rows, "sha256": sha(part)})

    def add(self, row):
        data = self.encoded(row)
        if self.stream is None or self.n + len(data) > CHUNK:
            self.close()
            name = f"gps_observed__part{len(self.records) + 1:04d}.csv"
            path = self.directory / (name + ".part")
            if (self.directory / name).exists() or path.exists():
                raise FileExistsError("Harrier derived tables are immutable; existing output needs explicit audit")
            self.stream = path.open("wb"); header = self.encoded(self.header)
            self.quota.check(len(header)); self.stream.write(header)
            self.records.append({"name": name}); self.n, self.rows = len(header), 0
        self.quota.check(len(data)); self.stream.write(data); self.n += len(data); self.rows += 1

    def publish(self):
        self.close()
        for record in self.records:
            (self.directory / (record["name"] + ".part")).replace(self.directory / record["name"])


def normalize(root):
    root = Path(root); directory = root / "raw" / SOURCE
    meta = json.loads((directory / "source.json").read_text())
    if meta.get("status") == "complete":
        raise FileExistsError("Harrier source is already complete and immutable")
    main = next(row for row in meta["files"] if "_gps_tracking_data_" in row["name"])
    path = directory / main["stored_as"]
    if sha(path) != main["sha256"]:
        raise ValueError("Original harrier bytes changed before normalization")
    quota = Quota(root)
    event_header = ["event-id", "timestamp", "location-lat", "location-long", "individual-local-identifier",
                    "individual-taxon-canonical-name", "study-name", "sensor-type"]
    writer = CsvChunks(directory, event_header, quota)
    annotations = directory / "annotations"; annotations.mkdir(exist_ok=True)
    annotation_path = annotations / "author-retrospective-phase.parquet"
    if annotation_path.exists() or annotation_path.with_suffix(".part").exists():
        raise FileExistsError("Retrospective annotations are immutable")
    annotation_part = annotation_path.with_suffix(".part")
    schema = pa.schema([("source_row", pa.int64()), ("animal", pa.string()), ("season", pa.string()),
                        ("phase", pa.string()), ("population", pa.string()), ("sex", pa.string())])
    phase_writer = pq.ParquetWriter(annotation_part, schema, compression="zstd")
    season_records, animal_records, phase_counts, total = {}, {}, Counter(), 0
    seen_annotation_bytes = 0
    try:
        for frame in pd.read_csv(path, dtype=str, chunksize=40_000, keep_default_na=False):
            needed = ["season", "population", "sex", "date_time", "ann.rough", "latitude", "longitude", "speed_2d"]
            if list(frame) != needed:
                raise ValueError("Unsupported original harrier table columns")
            utc = pd.to_datetime(frame.date_time, utc=True, errors="raise")
            if not (pd.to_numeric(frame.latitude).between(-90, 90).all() and pd.to_numeric(frame.longitude).between(-180, 180).all()):
                raise ValueError("Invalid original GPS coordinates require source review")
            labels = {key: [] for key in schema.names}
            for offset, row in enumerate(frame.itertuples(index=False, name=None)):
                season, population, sex, timestamp, phase, lat, lon, speed = row
                match = re.fullmatch(r"(.+)_(20\d{2})_(20\d{2})", season)
                if match is None or int(match[3]) != int(match[2]) + 1:
                    raise ValueError("Annual-cycle name cannot be mapped explicitly to original bird identity")
                animal, year = match[1], int(match[2])
                if utc.iloc[offset].year not in {year, year + 1}:
                    raise ValueError("Original annual-cycle name and actual timestamp disagree")
                static = {"animal": animal, "sex": sex, "population": population}
                if animal in animal_records and animal_records[animal] != static:
                    raise ValueError("Original bird name has conflicting sex/population metadata")
                animal_records[animal] = static
                record = season_records.setdefault(season, {**static, "season": season, "annual_cycle_start_year": year,
                                                             "rows": 0, "first_timestamp": timestamp, "last_timestamp": timestamp})
                record["rows"] += 1
                record["first_timestamp"] = min(record["first_timestamp"], timestamp)
                record["last_timestamp"] = max(record["last_timestamp"], timestamp)
                source_row = total + offset + 1  # Original data-record ordinal, excluding the header.
                writer.add([source_row, timestamp, lat, lon, animal, TAXON, STUDY, "gps"])
                for key, value in zip(schema.names, (source_row, animal, season, phase, population, sex)):
                    labels[key].append(value)
                phase_counts[phase] += 1
            phase_writer.write_table(pa.Table.from_pydict(labels, schema=schema))
            current = annotation_part.stat().st_size
            quota.check(current - seen_annotation_bytes); seen_annotation_bytes = current
            total += len(frame)
        phase_writer.close()
        quota.check(annotation_part.stat().st_size - seen_annotation_bytes)
        if len(animal_records) != 29 or len(season_records) != 40 or total != 1_134_664:
            raise ValueError("Observed harrier animal/season/row counts differ from the declared primary paper")
        writer.publish(); annotation_part.replace(annotation_path)
    except Exception:
        writer.close(); phase_writer.close()
        annotation_part.unlink(missing_ok=True)
        for record in writer.records:
            (directory / (record["name"] + ".part")).unlink(missing_ok=True)
        raise
    with (directory / "animal-reference-data.csv").open("w", newline="") as stream:
        ref = csv.writer(stream); ref.writerow(["animal-id", "animal-taxon", "animal-sex", "study-site"])
        for animal in sorted(animal_records):
            static = animal_records[animal]; ref.writerow([animal, TAXON, static["sex"], static["population"]])
    dump(annotations / "annual-cycle-reference.json", {"rows": sorted(season_records.values(), key=lambda r: r["season"]),
          "identity_rule": "Original season is explicitly defined as bird-name plus annual-cycle years. Strip only the exact _YYYY_YYYY suffix; verify sex/population consistency and source-paper counts. Names are source-local; no external identity asserted."})
    meta.update({"status": "complete", "derived_events": writer.records, "event_rows": total,
                 "animals": len(animal_records), "author_complete_annual_cycles": len(season_records),
                 "phase_counts": dict(phase_counts), "normalizer_sha256": sha(__file__),
                 "normalization": "Field renaming and explicit original bird-name/year parsing only. Original timestamp/latitude/longitude strings copied unchanged; UTC is stated by the source; no interpolation, resampling or coordinates manufactured.",
                 "phase_annotations": {"path": str(annotation_path.relative_to(directory)), "sha256": sha(annotation_path),
                     "rows": total, "join_key": "source_row equals event-id, 1-based original main-file data-record ordinal",
                     "kind": "author_retrospective_visual_annual_phase_delineation", "causal_predictor": False,
                     "evidence": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5493734/", "note": "Authors assigned daily tracks to annual phases using maps. These are retrospective analytical annotations, not contemporaneous biological measurements."},
                 "ancillary_exclusion": "High-resolution file is preserved only in originals; it supplies sampling-interval correction factors and is not an independent annual-track release.",
                 "completed_utc": datetime.now(timezone.utc).isoformat()})
    dump(directory / "source.json", meta)
    if size(root) > LIMIT or shutil.disk_usage(root).free < FLOOR:
        raise ValueError("Intake quota failed final verification")
    return meta


def verify_complete(root):
    root = Path(root); directory = root / "raw" / SOURCE
    meta = json.loads((directory / "source.json").read_text())
    if meta.get("status") != "complete":
        raise ValueError("The harrier intake has not completed")
    for record in meta["files"]:
        if sha(directory / record["stored_as"]) != record["sha256"]:
            raise ValueError("An immutable original changed")
    for record in meta["derived_events"]:
        if sha(directory / record["name"]) != record["sha256"]:
            raise ValueError("An immutable derived event table changed")
    if sha(directory / meta["phase_annotations"]["path"]) != meta["phase_annotations"]["sha256"]:
        raise ValueError("Immutable phase annotations changed")
    Quota(root)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/migration-next")
    parser.add_argument("--acquire-harrier", action="store_true")
    args = parser.parse_args()
    if args.acquire_harrier:
        acquire_harrier(args.root)
    source = Path(args.root) / "raw" / SOURCE / "source.json"
    result = verify_complete(args.root) if source.exists() and json.loads(source.read_text()).get("status") == "complete" else normalize(args.root)
    print(json.dumps({k: result[k] for k in ("status", "event_rows", "animals", "author_complete_annual_cycles")}, indent=2))


if __name__ == "__main__":
    main()
