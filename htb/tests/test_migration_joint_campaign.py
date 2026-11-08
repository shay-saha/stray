import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


spec=importlib.util.spec_from_file_location("migration_joint_campaign",Path(__file__).resolve().parents[1]/"scripts/migration_joint_campaign.py")
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def producer(tmp_path,status="preparing"):
    path=tmp_path/"prepare.json"
    state={"status":status,"launch_sha256":"a"*64,"pid":os.getpid()}
    path.write_text(json.dumps(state))
    return {"prepare_status":str(path),"prepare_launch_sha256":"a"*64},state


def test_half_published_cache_is_not_ready_until_the_specific_producer_finishes(tmp_path):
    plan,state=producer(tmp_path)
    (tmp_path/"manifest.json").write_text("half published cache")
    assert campaign.producer_status(plan) is None
    with pytest.raises(ValueError,match="completed frozen"):
        campaign.bind_ready_cache(plan,state)
    state.update(status="complete",pid=999999999)
    Path(plan["prepare_status"]).write_text(json.dumps(state))
    assert campaign.producer_status(plan) == state


@pytest.mark.parametrize("update,error", [
    ({"launch_sha256":"wrong"},"frozen campaign"),
    ({"status":"failed"},"producer failed"),
    ({"pid":999999999},"process is absent"),
])
def test_failed_or_different_dependency_never_triggers_a_replacement(tmp_path,update,error):
    plan,state=producer(tmp_path);state.update(update)
    Path(plan["prepare_status"]).write_text(json.dumps(state))
    with pytest.raises(ValueError,match=error):campaign.producer_status(plan)


def ready(tmp_path):
    plan,state=producer(tmp_path,status="complete")
    data=tmp_path/"data";data.mkdir()
    (data/"cases.npz").write_bytes(b"The production loader validates real arrays; this test isolates the binding checks.")
    (data/"manifest.json").write_text('{"fixture":"attested bytes"}')
    original=tmp_path/"parent";original.mkdir()
    origins=pd.DataFrame({"case_id":[0,1,2],"canonical_animal":["a","b","c"],
                          "split":["train","val_individual","test_year"],
                          "fresh_identity":[False,False,True]})
    origins.to_parquet(original/"origins.parquet",index=False)
    (original/"manifest.json").write_text(json.dumps({"origins_sha256":campaign.sha256(original/"origins.parquet")}))
    accounting=origins.rename(columns={"case_id":"parent_case_id","canonical_animal":"parent_canonical_animal",
                                       "split":"parent_split","fresh_identity":"parent_fresh_identity"})
    accounting["status"]=["retained_unaffected","reassigned_repaired_owner","retired"]
    accounting["child_case_id"]=[0,1,None]
    accounting["child_split"]=["train","val_individual",None]
    accounting["child_fresh_identity"]=[False,False,None]
    accounting["retirement_reason"]=[None,None,"source_owned_history_has_no_future_support"]
    accounting.to_parquet(data/"parent-case-accounting.parquet",index=False)
    plan.update(data=str(data),parent_cases=3,parent_reserved_cases=1,
                repair_contract_sha256="c"*64,parent_manifest_sha256=campaign.sha256(original/"manifest.json"),
                required_observation_sha256=["broad","bird","marine"],minimum_training_labels=23)
    state.update(result=str(data),reserved_labels_sealed=True,parent_cases_accounted=3,
                 manifest_sha256=campaign.sha256(data/"manifest.json"))
    a={"split":np.array([0,0,1]),"target_mask":np.ones((3,9),bool),"individual":np.array([0,1,2]),
       "horizons":np.array([1/24,.5,1,3,7,14,30,60,90.])}
    manifest={"protocol":"wildcast-multiscale-repaired-v2","train_end_year":2020,"validation_year":2021,
              "physical_repair":{"sha256":"c"*64},
              "parent_cache":{"path":str(original),"case_accounting":"parent-case-accounting.parquet",
                              "case_accounting_sha256":campaign.sha256(data/"parent-case-accounting.parquet"),
                              "accounted_parent_cases":3},
              "sources":[{"sha256":name} for name in ("broad","bird","marine")],
              "species":{f"Bird species{i}":i+1 for i in range(23)}}
    hashes={"manifest_sha256":state["manifest_sha256"],"cases_sha256":campaign.sha256(data/"cases.npz")}
    return plan,state,a,manifest,hashes


def test_ready_binding_is_exact_and_future_jobs_reject_changed_cache_bytes(tmp_path):
    plan,state,a,manifest,hashes=ready(tmp_path)
    binding=campaign.bind_ready_cache(plan,state,loader=lambda _: (a,manifest,hashes))
    assert binding["training_origins"] == 2
    assert binding["training_target_counts"] == [2]*9
    assert "not a certified" in binding["name_count_scope"]
    campaign.verify_binding(binding)
    with (Path(plan["data"])/"cases.npz").open("ab") as stream:stream.write(b"changed")
    with pytest.raises(ValueError,match="changed after binding"):campaign.verify_binding(binding)


@pytest.mark.parametrize("target,update,error", [
    ("state",{"reserved_labels_sealed":False},"sealing"),
    ("state",{"parent_cases_accounted":2},"all frozen parent"),
    ("manifest",{"sources":[{"sha256":"broad"}]},"omits a frozen"),
    ("manifest",{"validation_year":2022},"boundaries"),
])
def test_guard_requires_the_full_broad_provenance_and_parent_invariants(tmp_path,target,update,error):
    plan,state,a,manifest,hashes=ready(tmp_path)
    (state if target == "state" else manifest).update(update)
    with pytest.raises(ValueError,match=error):
        campaign.bind_ready_cache(plan,state,loader=lambda _: (a,manifest,hashes))


def test_standalone_launcher_imports_training_without_shell_pythonpath(tmp_path):
    plan,state,*_=ready(tmp_path)
    Path(plan["prepare_status"]).write_text(json.dumps(state))
    plan.update(code_sha256={},output=str(tmp_path/"campaign"),max_hours=.01,jobs=[])
    path=tmp_path/"plan.json";path.write_text(json.dumps(plan))
    run=subprocess.run([sys.executable,"-I",str(Path(campaign.__file__).resolve()),
                        "--plan",str(path),"--gpu","0"],cwd=tmp_path,
                       capture_output=True,text=True,timeout=30)
    # The deliberately invalid cache must reach the real training loader's hash
    # guard, even outside the repository with Python's isolated import path.
    result=json.loads((tmp_path/"campaign/state.json").read_text())
    assert run.returncode != 0
    assert result["status"] == "failed" and result["jobs"] == []
    assert result["error_type"] == "ValueError"
    assert "cases hash does not match" in result["error"]
