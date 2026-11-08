"""Clean bounded acquired chunks without retaining dense raw GPS in memory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from wildcast.data import read_events
from wildcast.sources import dump


def clean(root, output):
    root, output = Path(root), Path(output)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest = output / "processed"
    if (dest / "observations.parquet").exists():
        raise FileExistsError("Preserve snapshots: choose a new clean output directory")
    dest.mkdir(parents=True, exist_ok=True)
    frames, files = [], []
    for directory in sorted((root / "raw").iterdir()):
        source_path = directory / "source.json"
        if not source_path.exists(): continue
        source = json.loads(source_path.read_text())
        rights = source.get("metadata", {}).get("dc.rights.uri", [])
        allowed = ["/publicdomain/zero/1.0", "/licenses/by/4.0", "/licenses/by/3.0"]
        if not rights or not all(any(x in u.lower() for x in allowed) for u in rights):
            files.append({"path": str(directory), "status": "quarantined: unsupported or contradictory rights"}); continue
        for path in sorted(directory.glob("*.csv")):
            if "reference" in path.name.lower(): continue
            try:
                frame = read_events(path, directory.name, preserve_identifiers=True)
            except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as e:
                files.append({"path": str(path), "status": "read_error", "error": str(e)}); continue
            if frame is None:
                files.append({"path": str(path), "status": "unsupported_schema"}); continue
            raw_rows = len(frame)
            valid = frame.timestamp.notna() & frame.lat.between(-90, 90) & frame.lon.between(-180, 180)
            valid &= frame.species.notna() & ~frame.animal.str.endswith(":nan")
            frame = frame.loc[valid].sort_values("timestamp", kind="stable")
            frame["hour"] = frame.timestamp.dt.floor("h")
            frame = frame.drop_duplicates(["animal", "hour"]).drop(columns="hour")
            files.append({"path": str(path), "status": "accepted", "sensor_filtered_rows": raw_rows,
                          "chunk_hourly_rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            frames.append(frame)
            print(json.dumps({"file": path.name, "sensor_rows": raw_rows, "hourly_rows": len(frame)}), flush=True)
    if not frames: raise ValueError("No supported licensed GPS chunks found")
    f = pd.concat(frames, ignore_index=True).sort_values(["animal", "timestamp"], kind="stable")
    del frames
    f["hour"] = f.timestamp.dt.floor("h")
    f = f.drop_duplicates(["animal", "hour"]).drop(columns="hour")
    # Cross-source physical identity is resolved jointly with the existing corpus
    # by migration_data.prepare. No 48-hour segmentation truncates personal memory.
    f = f.loc[f.groupby("animal").animal.transform("size") >= 50].reset_index(drop=True)
    f["segment"] = 0
    f.to_parquet(dest / "observations.parquet", index=False)
    report = {"source_root": str(root), "files": files, "rows": len(f), "animals": f.animal.nunique(),
              "cleaning_code_sha256": code_hash,
              "identifier_rule": "Read local animal, species and study identifiers as strings; preserve leading zeros.",
              "species": sorted(f.species.unique()), "min_observations_per_animal": 50,
              "observations_sha256": hashlib.sha256((dest / "observations.parquet").read_bytes()).hexdigest(),
              "rule": "Explicit GPS/visible/outlier filters; valid coordinates/timestamps; earliest actual observed fix per animal UTC hour, across chunk boundaries; no interpolation or gap reset. Physical identity and speed-case audit occur in migration preparation."}
    dump(dest / "quality-report.json", report)
    print(json.dumps({k: report[k] for k in ["rows", "animals", "species"]}, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    clean(a.root, a.output)
