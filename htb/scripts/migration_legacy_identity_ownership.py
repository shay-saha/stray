"""Certify historical chosen-row lineage by replay, never by nearest coordinates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def certify(review,recovery,ownership,output):
    review,recovery,ownership,output=map(Path,(review,recovery,ownership,output))
    if output.exists():raise FileExistsError(output)
    r,m=json.loads(review.read_text()),json.loads(recovery.read_text())
    qpath=Path(r["quality_report"]["path"])
    if sha(qpath)!=r["quality_report"]["sha256"]:raise ValueError("Historical quality report changed")
    quality=json.loads(qpath.read_text())
    rec=m["outputs"][0];rawpath=Path(rec["path"])
    if not m["complete"] or sha(rawpath)!=rec["sha256"]:raise ValueError("Recovered source rows changed")
    raw=pd.read_parquet(rawpath)
    membership={name:c["collapsed_canonical"] for c in r["original_recovery"] for name in c["raw_ids"]}
    raw["old_component"]=raw.animal.map(membership)
    rank={row["path"]:i for i,row in enumerate(quality["files"]) if row.get("accepted_sensor_rows",0)>0}
    raw["original_file_order"]=raw.input_path.map(rank)
    if raw.old_component.isna().any() or raw.original_file_order.isna().any():raise ValueError("Unknown historical source membership/order")
    selected=raw.sort_values(["old_component","timestamp","original_file_order","row_number"],kind="stable").drop_duplicates(["old_component","timestamp"])
    selected["original_hour"]=selected.timestamp.dt.floor("h")
    selected=selected.drop_duplicates(["old_component","original_hour"])
    minimum=quality["config"]["min_points"]
    selected=selected.loc[selected.groupby("old_component").old_component.transform("size")>=minimum]
    selected=selected.sort_values(["old_component","timestamp"]).reset_index(drop=True)
    oldinfo=r["original_recovery"][0];oldpath=Path(oldinfo["input_path"])
    if sha(oldpath)!=oldinfo["input_sha256"]:raise ValueError("Old observation bytes changed")
    old=pq.read_table(oldpath,columns=["animal","timestamp","lat","lon"],filters=[("animal","in",sorted(set(membership.values())))]).to_pandas().rename(columns={"animal":"old_component"})
    old["timestamp"]=old.timestamp.dt.as_unit("ns");old=old.sort_values(["old_component","timestamp"]).reset_index(drop=True)
    keys=["old_component","timestamp","lat","lon"]
    if len(selected)!=len(old) or not all(np.array_equal(selected[k].to_numpy(),old[k].to_numpy()) for k in keys):raise ValueError("Historical selector replay differs from old observations")
    table=pd.read_parquet(ownership)
    pd.testing.assert_frame_equal(selected[list(table.columns)],table,check_exact=True)
    if not table.original_row_id.is_unique or table.duplicated(["old_component","timestamp"]).any():raise ValueError("Selected-row ownership is ambiguous")
    report={"schema":"wildcast-certified-historical-selected-row-replay-v1","complete":True,"code_sha256":sha(__file__),
        "review":{"path":str(review),"sha256":sha(review)},"recovery":{"path":str(recovery),"sha256":sha(recovery)},
        "old_observations":{"path":str(oldpath),"sha256":sha(oldpath)},"quality_report":{"path":str(qpath),"sha256":sha(qpath)},
        "historical_selected_rows":{"path":str(ownership),"sha256":sha(ownership),"row_id_column":"original_row_id","rows":len(table),"columns":list(table.columns)},
        "original_ordering":{"source_file_rank":rank,"within_file":"Original logical CSV record number, header=1","sort":"Old component + timestamp; source-file/record order breaks exact-time ties stably", "selectors":["First original record per component/timestamp","First actual fix per component/UTC-container hour","Old per-component minimum observed fix count"],"minimum_points":minimum},
        "checks":{"all_old_tuples_replayed_bit_exact":True,"published_ownership_table_recomputed_exactly":True,"unique_original_row_ID_per_old_row":True,"no_nearest_time_or_coordinate_assignment":True},
        "lookup_key":["old_input_sha256","old_component","timestamp"],
        "scope":"Documents which original record the historical algorithm selected. Coordinates only verify equality. Does not certify biological identity or make old mixed histories/model labels valid. Preserve inherited global exclusions and rebuild physical histories from complete original source rows."}
    output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"sha256":sha(output),"rows":len(table),"table_sha256":sha(ownership)},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["review","recovery","ownership","output"]:p.add_argument("--"+name,required=True)
    certify(**vars(p.parse_args()))
