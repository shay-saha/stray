#!/usr/bin/env python3
"""Prepare the copied, completed server model locally. Never reads target labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models/server-repaired-v2"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8*1024*1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    bundle = BUNDLE
    manifest_path = bundle / "data/reference/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((bundle/"model/summary.json").read_text())
    assert sha256(bundle/"model/selected.pt") == "76fc72a65f14b2304fc07d9e8286e164ac25642ca413f68f12862db2c6b0d959"
    assert sha256(manifest_path) == summary["manifest_sha256"]
    assert sha256(bundle/"data/reference/cases.npz") == summary["cases_sha256"]
    assert sha256(bundle/"data/reference/origins.parquet") == manifest["origins_sha256"]
    # These are the learned architecture and the exact feature builder used by training.
    for name in ("migration_model.py","migration_data.py","data.py","multiscale_data.py","migration_repaired_features.py"):
        assert sha256(bundle/"wildcast"/name) == summary["code_sha256"][name], name
    output = bundle/"data/observations.parquet"
    report = {"source": "Actual masked history GPS fixes from the repaired model cache",
              "future_targets_read": False, "interpolated_observations": False,
              "cases_sha256": summary["cases_sha256"], "origins_sha256":manifest["origins_sha256"]}
    if not output.exists():
        origins = pd.read_parquet(bundle/"data/reference/origins.parquet",columns=["canonical_animal","species","study"])
        with np.load(bundle/"data/reference/cases.npz",allow_pickle=False) as archive:
            # Explicit whitelist: y/target_latlon/target_time are never opened.
            mask, times, coordinates = (archive[name] for name in ("history_mask","history_time","history_latlon"))
        assert len(origins) == len(mask)
        groups = origins.groupby(["canonical_animal","species","study"],sort=True).indices
        pending, written, conflicts, writer = [], 0, 0, None
        temp = output.with_suffix(".partial.parquet")
        try:
            for i, ((animal,taxon,study),ids) in enumerate(groups.items()):
                valid = mask[ids]
                ts = times[ids][valid]
                ll = coordinates[ids][valid]
                frame = pd.DataFrame({"seconds":ts,"lat":ll[:,0],"lon":ll[:,1]})
                frame = frame.drop_duplicates().sort_values("seconds",kind="stable")
                duplicate = frame.seconds.duplicated(keep=False)
                conflicts += int(duplicate.sum())
                frame = frame.loc[~duplicate].copy()
                if frame.empty:
                    continue
                frame["timestamp"] = pd.to_datetime(frame.pop("seconds"),unit="s",utc=True)
                frame["animal"],frame["species"],frame["study"] = animal,taxon,study
                pending.append(frame)
                if sum(len(x) for x in pending)>=100000 or i+1==len(groups):
                    data = pd.concat(pending,ignore_index=True)
                    table = pa.Table.from_pandas(data,preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temp,table.schema,compression="zstd")
                    writer.write_table(table);written += len(data);pending.clear()
                if i%500==0:
                    print(f"Recovered {i}/{len(groups)} observed histories; {written} GPS fixes",flush=True)
            if pending:
                data=pd.concat(pending,ignore_index=True);table=pa.Table.from_pandas(data,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression="zstd")
                writer.write_table(table);written+=len(data)
        finally:
            if writer is not None:writer.close()
        temp.replace(output)
        report.update(rows=written,source_histories=len(groups),conflicting_timestamp_rows_removed=conflicts,
                      observed_taxon_labels=int(origins.species.nunique()))
        (bundle/"data/observation-lineage.json").write_text(json.dumps(report,indent=2)+"\n")
    files = {str(p.relative_to(bundle)):sha256(p) for p in bundle.rglob("*")
             if p.is_file() and p.name not in {"package.json","verify_package.py"} and "__pycache__" not in str(p)}
    package = {"model_kind":"wildcast_migration_joint","protocol":manifest["protocol"],
               "trained_taxa":len(manifest["species"]),"source_host":"89.22.197.55:62892",
               "source_checkpoint":"/workspace/wildcast/runs/migration-speed-v2-gpu0/scaled-history_direct/best.pt",
               "source_checkpoint_sha256":files["model/selected.pt"],"files":files}
    (bundle/"package.json").write_text(json.dumps(package,indent=2)+"\n")
    print("Prepared verified local model and observed GPS snapshot",flush=True)


if __name__=="__main__":
    main()
