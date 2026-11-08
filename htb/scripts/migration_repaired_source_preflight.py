"""Audit simultaneous source measurements without constructing forecast targets."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from wildcast import migration_data as legacy
from wildcast import migration_physical_repair as repair
from wildcast import multiscale_data as previous
from wildcast.multiscale_repaired import select_source_rows
from wildcast.data import distance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch",required=True);parser.add_argument("--repair",required=True)
    parser.add_argument("--report",required=True)
    args=parser.parse_args(); output=Path(args.report)
    if output.exists():raise FileExistsError(output)
    started=time.time();launch=json.loads(Path(args.launch).read_text())
    parent=previous._load_parent(launch["parent_cache"],2020,2021)
    contract,provenance=repair.load_contract(args.repair,parent)
    paths,evidence=previous._input_paths(launch["roots"],parent,2020,2021)
    by_hash=defaultdict(list)
    for row in contract["raw_coordinate_ownership"]:by_hash[row["input_sha256"]].append(row)
    records=[];raw_rows=0;retained_rows=0
    for path,source in zip(paths,evidence):
        owned=by_hash[source["sha256"]]
        excluded=[r["input_animal"] for r in owned if r["status"]!="unmixed_original_source_id"]
        rules={"sha256":source["sha256"],"allowed_raw_animals":sorted({r["input_animal"] for r in owned}-set(excluded)),
               "excluded_raw_animals":excluded,"excluded_intervals":[r for r in contract.get("source_interval_exclusions",[]) if r["input_sha256"]==source["sha256"]]}
        frame=pd.read_parquet(path,columns=["timestamp","lat","lon","animal"])
        raw_rows+=len(frame)
        frame=select_source_rows(frame,source["sha256"],{"observation_source_rules":[rules]})
        retained_rows+=len(frame);records.append(frame)
        print("SOURCE_QC_READ",str(path),len(frame),flush=True)
    for source in repair.recovery_outputs(contract):
        frame=pd.read_parquet(source["path"],columns=["timestamp","lat","lon","animal"])
        raw_rows+=len(frame);retained_rows+=len(frame);records.append(frame)
        evidence.append({"path":source["path"],"sha256":source["sha256"]})
    frame=pd.concat(records,ignore_index=True);del records
    frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True).dt.as_unit("ns")
    frame["raw_animal"]=frame.animal
    frame["animal"]=frame.animal.map(contract["physical_mapping"])
    if frame.animal.isna().any():raise ValueError("Unreviewed source identity")
    points=frame.drop_duplicates(["animal","timestamp","lat","lon"])
    conflict=points.loc[points.duplicated(["animal","timestamp"],keep=False)].copy()
    del points
    report=[]
    for animal,group in conflict.groupby("animal",sort=True):
        first=group.groupby("timestamp")[['lat','lon']].transform("first")
        gap=distance(group.lat.to_numpy(),group.lon.to_numpy(),first.lat.to_numpy(),first.lon.to_numpy())
        raw=sorted(set(frame.loc[frame.animal.eq(animal),"raw_animal"]))
        report.append({"canonical_animal":animal,"raw_animals":raw,
                       "simultaneous_times_with_different_reported_coordinates":group.timestamp.nunique(),
                       "nonidentical_positions":len(group),"max_separation_from_first_metres":float(gap.max()),
                       "separation_quantiles_metres":np.quantile(gap,[.5,.9,.99]).tolist(),
                       "points_above_1mm":int((gap>.001).sum()),"points_above_1m":int((gap>1).sum()),
                       "points_above_100m":int((gap>100).sum()),"points_above_1km":int((gap>1000).sum()),
                       "max_abs_latitude_difference_degrees":float(abs(group.lat-first.lat).max()),
                       "max_abs_longitude_difference_degrees":float(abs(group.lon-first.lon).max())})
    result={"schema":"wildcast-repaired-simultaneous-source-QC-v1","repair":provenance,"sources":evidence,
            "code_sha256":legacy.sha256(__file__),"raw_rows":raw_rows,"retained_after_source_selectors":retained_rows,
            "canonical_motion_groups":frame.animal.nunique(),"affected_motion_groups":len(report),"groups":report,
            "elapsed_seconds":time.time()-started,
            "scope":"Licensed-source simultaneous measurement identity/QC audit only; no forecast origins/windows, target matching, final cohort outcomes, model predictions or scoring. Coordinates are not emitted."}
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("SOURCE_QC_COMPLETE",json.dumps({k:v for k,v in result.items() if k not in ["sources","repair","groups"]}),flush=True)


if __name__=="__main__":main()
