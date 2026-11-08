"""Finite supervised corrected-physical-history preparation, with frozen inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from wildcast.migration_data import sha256
from wildcast.multiscale_repaired import prepare


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temporary.replace(path)


def verify(launch):
    for record in launch["frozen_files"]:
        if not Path(record["path"]).is_file() or sha256(record["path"]) != record["sha256"]:
            raise ValueError("Frozen repaired launch asset changed: "+record["path"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch",required=True);parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args();path=Path(args.launch);launch_hash=sha256(path);launch=json.loads(path.read_text())
    verify(launch)
    if args.verify_only:
        print(json.dumps({"scope":"Frozen code/metadata/source byte checks only; no cases or labels derived",
                          "launch_sha256":launch_hash,"verified_files":len(launch["frozen_files"])}));return
    job=Path(launch["job_root"]);job.mkdir(parents=True,exist_ok=True)
    state={"scope":"Finite corrected-physical-history preparation; no model fit or final truth release",
           "pid":os.getpid(),"launch":str(path),"launch_sha256":launch_hash,
           "started_unix":time.time(),"status":"preparing","result":launch["result"]}
    def progress(stage,counts):
        state.update(stage=stage,updated_unix=time.time(),progress=counts)
        save(job/"prepare-status.json",state)
        print("REPAIRED_PREPARE_PROGRESS",json.dumps({"stage":stage,**counts}),flush=True)
    save(job/"prepare-status.json",state)
    print("REPAIRED_PREPARE_START",json.dumps(state),flush=True)
    try:
        manifest=prepare(launch["roots"],launch["result"],parent_cache=launch["parent_cache"],
                         repair_contract=launch["repair_contract"],existing_manifest=launch.get("existing_manifest"),
                         parent_clock_audit=launch.get("parent_clock_audit"),train_end_year=launch["train_end_year"],
                         validation_year=launch["validation_year"],seed=launch["seed"],horizons=launch["horizons_days"],
                         progress=progress)
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
        state.update(status="complete",stage="published",ended_unix=time.time(),manifest_sha256=sha256(root/"manifest.json"),
                     cases=manifest["cases"],training_taxa=manifest["training_taxa_with_any_target"],
                     training_taxa_by_horizon=taxa,reserved_labels_sealed=True,
                     parent_cases_accounted=manifest["parent_cache"]["accounted_parent_cases"],
                     parent_case_statuses=manifest["parent_cache"]["case_status_counts"],
                     parent_retirement_reasons=manifest["parent_cache"]["retirement_reason_counts"])
        save(job/"prepare-status.json",state)
        print("REPAIRED_PREPARE_COMPLETE",json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status="failed",ended_unix=time.time(),error_type=type(error).__name__,error=str(error))
        save(job/"prepare-status.json",state)
        raise


if __name__ == "__main__":main()
