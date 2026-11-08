"""Finite supervised multiscale preparation with frozen input/code checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from wildcast.migration_data import sha256
from wildcast.multiscale_data import prepare


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temporary.replace(path)


def verify(launch):
    for record in launch["frozen_files"]:
        if not Path(record["path"]).is_file() or sha256(record["path"]) != record["sha256"]:
            raise ValueError("Frozen launch asset changed: "+record["path"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch",required=True);parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args();path=Path(args.launch);launch_hash=sha256(path);launch=json.loads(path.read_text())
    verify(launch)
    if args.verify_only:
        print(json.dumps({"scope":"Frozen metadata/code/source byte checks only; no cases or labels derived",
                          "launch_sha256":launch_hash,"verified_files":len(launch["frozen_files"])}));return
    job=Path(launch["job_root"]);job.mkdir(parents=True,exist_ok=True)
    state={"scope":"Finite sealed multiscale case preparation; no model fit or final truth release",
           "pid":os.getpid(),"launch":str(path),"launch_sha256":launch_hash,
           "started_unix":time.time(),"status":"preparing","result":launch["result"]}
    save(job/"prepare-status.json",state)
    print("FROZEN_PREPARE_START",json.dumps(state),flush=True)
    try:
        manifest=prepare(launch["roots"],launch["result"],parent_cache=launch["parent_cache"],
                         identity_audit=launch["identity_audit"],reference_audit=launch.get("reference_audit"),
                         existing_manifest=launch.get("existing_manifest"),parent_clock_audit=launch["parent_clock_audit"],
                         train_end_year=launch["train_end_year"],validation_year=launch["validation_year"],seed=launch["seed"],
                         horizons=launch["horizons_days"])
        verify(launch)
        if sha256(path) != launch_hash:raise ValueError("Frozen launch declaration changed during preparation")
        root=Path(launch["result"])
        with np.load(root/"cases.npz",allow_pickle=False) as arrays:
            sealed=arrays["split"] >= 3
            if arrays["target_mask"][sealed].any() or any(arrays[k][sealed].any() for k in ["y","target_latlon","target_time"]):
                raise ValueError("Reserved label arrays are not sealed")
        support=pd.read_csv(root/"development-support.csv")
        train=support[(support["split"] == "train")&(support.cases > 0)]
        taxa={str(float(h)):int(g.species.nunique()) for h,g in train.groupby("horizon_days")}
        state.update(status="complete",ended_unix=time.time(),manifest_sha256=sha256(root/"manifest.json"),
                     cases=manifest["cases"],training_taxa=manifest["training_taxa_with_any_target"],
                     training_taxa_by_horizon=taxa,reserved_labels_sealed=True,
                     parent_origins_preserved=manifest["parent_cache"]["parent_cases_preserved"],
                     parent_day_targets_lost=sum(manifest["parent_cache"]["shared_horizon_support"]["lost_development_targets"]))
        save(job/"prepare-status.json",state)
        print("FROZEN_PREPARE_COMPLETE",json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status="failed",ended_unix=time.time(),error_type=type(error).__name__,error=str(error))
        save(job/"prepare-status.json",state)
        raise


if __name__ == "__main__":main()
