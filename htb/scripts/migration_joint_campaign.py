"""Finite GPU queue gated on a particular verified, sealed cache producer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# Supervisor invokes this file directly, so Python initially searches scripts/.
# Resolve the project from the file rather than relying on an installed package
# or a login shell's PYTHONPATH.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):digest.update(block)
    return digest.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)


def verify_code(plan):
    for name,digest in plan["code_sha256"].items():
        if not Path(name).is_file() or sha256(name) != digest:
            raise ValueError("Frozen campaign code changed: "+name)


def producer_status(plan):
    path=Path(plan["prepare_status"])
    if not path.is_file():
        raise ValueError("Expected live cache producer status is unavailable; do not start an unverified replacement")
    state=json.loads(path.read_text())
    if state.get("launch_sha256") != plan["prepare_launch_sha256"]:
        raise ValueError("Cache producer differs from the frozen campaign dependency")
    if state.get("status") == "failed":
        raise ValueError("Cache producer failed; inspect its retained state before a new launch")
    if state.get("status") != "complete":
        if state.get("status") != "preparing" or type(state.get("pid")) is not int:
            raise ValueError("Cache producer has no recognized live state")
        try:os.kill(state["pid"],0)
        except ProcessLookupError as error:
            raise ValueError("Expected preparer process is absent without a completed attestation") from error
        return None
    return state


def bind_ready_cache(plan,state,loader=None):
    """Bind only the finished producer's exact cache; inspect train/dev only."""
    if state.get("status") != "complete" or state.get("launch_sha256") != plan["prepare_launch_sha256"]:
        raise ValueError("A completed frozen producer attestation is required")
    data=Path(plan["data"])
    if Path(state["result"]).resolve() != data.resolve():raise ValueError("Unexpected producer cache path")
    if state.get("reserved_labels_sealed") is not True:
        raise ValueError("Producer did not prove reserved target sealing")
    if state.get("parent_cases_accounted") != plan["parent_cases"]:
        raise ValueError("Producer did not account for all frozen parent origins")
    if sha256(data/"manifest.json") != state["manifest_sha256"]:
        raise ValueError("Ready cache manifest differs from its producer attestation")
    if loader is None:
        from wildcast.migration_train import load_training_data
        loader=load_training_data
    a,manifest,hashes=loader(data)
    if hashes["manifest_sha256"] != state["manifest_sha256"]:
        raise ValueError("Training loader opened a different producer manifest")
    if (manifest.get("protocol") != "wildcast-multiscale-repaired-v2"
            or manifest.get("train_end_year") != 2020 or manifest.get("validation_year") != 2021):
        raise ValueError("Corrected protocol and historical evaluation boundaries required")
    if manifest.get("physical_repair",{}).get("sha256") != plan["repair_contract_sha256"]:
        raise ValueError("Training cache differs from the approved physical/source repair")
    accounting=verify_parent_accounting(plan,data,manifest)
    available={row["sha256"] for row in manifest["sources"]}
    if not set(plan["required_observation_sha256"]) <= available:
        raise ValueError("Ready cache omits a frozen broad/bird/marine source")
    import numpy as np
    train=a["split"] == 0
    counts=a["target_mask"][train].sum(0)
    if not np.all(counts > 0) or len(manifest["species"]) < plan["minimum_training_labels"]:
        raise ValueError("Expanded cache lacks required supervised horizons or broader training labels")
    if not np.array_equal(a["horizons"],np.array([1/24,.5,1,3,7,14,30,60,90.])):
        raise ValueError("Ready training cache has the wrong horizon grid")
    named={" ".join(name.split()[:2]) for name in manifest["species"]
           if re.match(r"^[A-Z][a-z]+ [a-z][a-z-]+(?: |$)",name)}
    return {"data":str(data),**hashes,"producer_launch_sha256":state["launch_sha256"],
        "training_origins":int(train.sum()),"training_exclusion_groups":int(len(np.unique(a["individual"][train]))),
        "training_taxon_labels":len(manifest["species"]),"named_binomial_prefix_keys":len(named),
        "name_count_scope":"Retained names before authoritative synonym reconciliation; not a certified biological-species count",
        "horizons_days":a["horizons"].tolist(),"training_target_counts":counts.tolist(),
        "cross_taxon_exclusion_groups":manifest.get("cross_taxon_exclusion_groups",[]),
        "parent_accounting":accounting,"physical_repair_sha256":plan["repair_contract_sha256"],
        "reserved_targets":"remain sealed; training loader selects train/development only"}


def verify_parent_accounting(plan,data,manifest):
    """Verify conserved roles and complete retirement accounting without targets."""
    import numpy as np
    import pandas as pd
    parent=manifest["parent_cache"]
    if parent.get("case_accounting") != "parent-case-accounting.parquet":
        raise ValueError("Unexpected parent-case accounting asset")
    path=Path(data)/parent["case_accounting"]
    if sha256(path) != parent.get("case_accounting_sha256"):
        raise ValueError("Parent-case accounting bytes changed")
    table=pd.read_parquet(path)
    required={"parent_case_id","parent_canonical_animal","parent_split","parent_fresh_identity","status",
              "child_case_id","child_split","child_fresh_identity","retirement_reason"}
    if (not required <= set(table) or table.parent_case_id.tolist() != list(range(plan["parent_cases"]))
            or parent.get("accounted_parent_cases") != len(table)
            or not table.status.isin(["retained_unaffected","reassigned_repaired_owner","retired"]).all()):
        raise ValueError("Incomplete or invalid parent-case accounting")
    original=Path(parent["path"])
    if sha256(original/"manifest.json") != plan["parent_manifest_sha256"]:
        raise ValueError("Historical parent manifest changed")
    old=json.loads((original/"manifest.json").read_text())
    if sha256(original/"origins.parquet") != old.get("origins_sha256"):
        raise ValueError("Historical parent origins changed")
    origins=pd.read_parquet(original/"origins.parquet",columns=["case_id","canonical_animal","split","fresh_identity"])
    if (len(origins) != len(table) or not np.array_equal(origins.case_id,table.parent_case_id)
            or not np.array_equal(origins.canonical_animal,table.parent_canonical_animal)
            or not np.array_equal(origins.split,table.parent_split)
            or not np.array_equal(origins.fresh_identity,table.parent_fresh_identity)):
        raise ValueError("Accounting does not match frozen parent identities, roles and freshness")
    reserved=table.parent_split.isin(["test_individual","test_year"])
    if int(reserved.sum()) != plan["parent_reserved_cases"]:
        raise ValueError("Historical reserved-case accounting changed")
    active=table.status != "retired"
    if (table.loc[active,"child_case_id"].isna().any()
            or not np.array_equal(table.loc[active,"child_split"],table.loc[active,"parent_split"])
            or not table.loc[active,"child_fresh_identity"].isin([True,False]).all()
            or ((~table.loc[active,"parent_fresh_identity"].astype(bool))&table.loc[active,"child_fresh_identity"].astype(bool)).any()):
        raise ValueError("Retained/reassigned cases changed roles or promoted freshness")
    retired=table.loc[~active]
    if retired.retirement_reason.isna().any() or retired.retirement_reason.eq("").any() or retired.child_case_id.notna().any():
        raise ValueError("Every retired case needs a source-QC reason and no active child mapping")
    return {"sha256":parent["case_accounting_sha256"],"parent_cases":len(table),
            "parent_reserved_cases":int(reserved.sum()),"statuses":table.status.value_counts().to_dict(),
            "scope":"Origin identity/role/freshness accounting only; no parent or child future target arrays read"}


def verify_binding(binding):
    data=Path(binding["data"])
    if (sha256(data/"manifest.json") != binding["manifest_sha256"]
            or sha256(data/"cases.npz") != binding["cases_sha256"]):
        raise ValueError("Campaign cache changed after binding")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",required=True);p.add_argument("--gpu",required=True)
    args=p.parse_args();path=Path(args.plan);digest=sha256(path);plan=json.loads(path.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    os.environ["OMP_NUM_THREADS"]="8";os.environ["MKL_NUM_THREADS"]="8"
    verify_code(plan)
    root=Path(plan["output"])
    if root.exists():raise FileExistsError("Use a new immutable campaign root")
    root.mkdir(parents=True)
    state={"pid":os.getpid(),"gpu":args.gpu,"plan":str(path),"plan_sha256":digest,
           "started_unix":time.time(),"status":"waiting_for_verified_cache","jobs":[]}
    state_path=root/"state.json";save(state_path,state)
    deadline=time.monotonic()+plan["max_hours"]*3600
    while True:
        ready=producer_status(plan)
        if ready is not None:break
        if time.monotonic() >= deadline:
            state.update(status="campaign_wait_budget_reached",ended_unix=time.time());save(state_path,state)
            return
        time.sleep(10)
    verify_code(plan);binding=bind_ready_cache(plan,ready);save(root/"ready-cache.json",binding)
    state.update(status="running",cache=binding);save(state_path,state)
    for job in plan["jobs"]:
        if sha256(path) != digest:raise ValueError("Frozen campaign plan changed")
        verify_code(plan);verify_binding(binding)
        remaining=deadline-time.monotonic()
        if remaining <= 0:
            state["status"]="campaign_time_budget_reached";break
        out=root/job["name"]
        if out.exists():raise FileExistsError("Never overwrite an experiment directory")
        out.mkdir()
        command=[sys.executable,"-u","-m","wildcast.migration_train","--data",binding["data"],
                 "--output",str(out),*job["arguments"]]
        record={"name":job["name"],"output":str(out),"command":command,"started_unix":time.time(),"status":"running"}
        state["jobs"].append(record);save(state_path,state)
        try:
            with (out/"console.log").open("w") as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
            verify_code(plan);verify_binding(binding)
            record.update(status="complete" if result.returncode == 0 else "failed",returncode=result.returncode,ended_unix=time.time())
            if (out/"best.pt").is_file():record["best_sha256"]=sha256(out/"best.pt")
        except subprocess.TimeoutExpired:
            record.update(status="terminated_at_campaign_budget",ended_unix=time.time())
            state["status"]="campaign_time_budget_reached";save(state_path,state);break
        save(state_path,state)
    if state["status"] == "running":state["status"]="complete"
    state["ended_unix"]=time.time();save(state_path,state)
    print(json.dumps({"status":state["status"],"jobs":[{"name":j["name"],"status":j["status"]} for j in state["jobs"]]}))


if __name__ == "__main__":
    try:main()
    except Exception as error:
        # A failed dependency/check is terminal for this launch. Never silently
        # leave a dead process labelled as waiting or start another producer.
        try:
            arg=sys.argv[sys.argv.index("--plan")+1]
            plan=json.loads(Path(arg).read_text());state_path=Path(plan["output"])/"state.json"
            if state_path.is_file():
                state=json.loads(state_path.read_text())
                state.update(status="failed",ended_unix=time.time(),error_type=type(error).__name__,error=str(error))
                save(state_path,state)
        except Exception:pass
        raise
