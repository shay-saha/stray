import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from wildcast import migration_physical_repair as repair


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def proof(tmp_path):
    t = pd.date_range("2020-06-01", periods=5, freq="D", tz="UTC")
    origins = pd.DataFrame({"case_id": range(5), "canonical_animal": ["a"]*4+["c"],
                            "origin_time": t, "split": ["train"]*4+["test_individual"],
                            "fresh_identity": [False]*4+[True], "targets_sealed": [False]*4+[True],
                            "individual": [0]*4+[1]})
    origin = np.column_stack([[10, 20, 30, 40, 50], [1]*5, t.asi8/1e9])
    np.savez(tmp_path/"cases.npz", origin=origin, y=np.zeros((5, 1, 2)),
             target_mask=np.zeros((5, 1), bool), target_latlon=np.zeros((5, 1, 2)),
             target_time=np.zeros((5, 1)))
    origins.to_parquet(tmp_path/"origins.parquet", index=False)
    meta = {"assignments": {"a": "temporal", "c": "test_individual"},
            "identity_mapping": {"a": "a", "b": "a", "c": "c"},
            "exclusion_groups": {"a": "a", "c": "c"}}
    (tmp_path/"manifest.json").write_text(json.dumps(meta))
    paths = {k: tmp_path/(k+ext) for k, ext in [("manifest", ".json"), ("cases", ".npz"), ("origins", ".parquet")]}
    parent = {"manifest": meta, "origins": origins, "freshness": {"a": False, "c": True},
              "paths": paths, "hashes": {k+"_sha256": digest(v) for k, v in paths.items()}}
    contract = {"schema": repair.SCHEMA,
                **{"parent_"+k: v for k, v in parent["hashes"].items()},
                "physical_mapping": {"a": "a", "b": "b", "c": "c"},
                "exclusion_mapping": {"a": "a", "b": "a", "c": "c"},
                "retained_physical_primitives": [], "retract_edges": [{"a": "a", "b": "b"}],
                "freshness_exclusion_animals": ["a", "b"], "raw_coordinate_ownership": [],
                "parent_memberships": [
                    {"parent_canonical": "a", "raw_ids": ["a", "b"],
                     "legacy_collapsed_raw_ids": [], "repaired_physical_components": ["a", "b"],
                     "physical_partition_changed": True, "requires_original_trajectory_recovery": False,
                     "child_exclusion_groups": ["a"], "parent_role": "temporal",
                     "parent_origin_freshness_values": [False]},
                    {"parent_canonical": "c", "raw_ids": ["c"],
                     "legacy_collapsed_raw_ids": [], "repaired_physical_components": ["c"],
                     "physical_partition_changed": False, "requires_original_trajectory_recovery": False,
                     "child_exclusion_groups": ["c"], "parent_role": "test_individual",
                     "parent_origin_freshness_values": [True]}]}
    rows = [("a", t[0], 10), ("b", t[1], 20), ("a", t[2], 30), ("b", t[2], 30), ("c", t[4], 50)]
    frame = pd.DataFrame([{"animal": n, "timestamp": time, "lat": lat, "lon": 1.,
                           "species": "bird", "study": "study", "source": "source"} for n, time, lat in rows])
    return parent, contract, frame


def test_complete_reviewed_partition_rejects_flat_star_reimport(proof):
    parent, contract, _ = proof
    repair.validate_contract(contract, parent, verify_inputs=False)
    tampered = copy.deepcopy(contract)
    tampered["physical_mapping"]["b"] = "a"
    with pytest.raises(ValueError, match="shadow alias"):
        repair.validate_contract(tampered, parent, verify_inputs=False)
    tampered["retained_physical_primitives"] = [{"a": "a", "b": "b", "evidence_source": "flat_star"}]
    with pytest.raises(ValueError, match="compressed alias star"):
        repair.validate_contract(tampered, parent, verify_inputs=False)


def test_retraction_retains_exclusion_and_parent_role_guards(proof):
    parent, contract, _ = proof
    wrong = copy.deepcopy(contract)
    wrong["exclusion_mapping"]["b"] = "b"
    with pytest.raises(ValueError, match="exclusion protection"):
        repair.validate_contract(wrong, parent, verify_inputs=False)
    wrong = copy.deepcopy(contract)
    wrong["parent_memberships"][0]["parent_role"] = "val_individual"
    with pytest.raises(ValueError, match="assignment role"):
        repair.validate_contract(wrong, parent, verify_inputs=False)
    wrong = copy.deepcopy(contract)
    wrong["parent_memberships"][0]["parent_origin_freshness_values"] = [True]
    with pytest.raises(ValueError, match="freshness"):
        repair.validate_contract(wrong, parent, verify_inputs=False)


def test_source_owned_coincidences_do_not_merge_physical_tracks(proof):
    _, contract, frame = proof
    canonical, lineage = repair.canonicalize_owned(frame, contract)
    assert set(canonical.animal) == {"a", "b", "c"}
    shared = canonical[canonical.lat == 30]
    assert set(shared.animal) == {"a", "b"}
    assert set(lineage.original_animal) == {"a", "b", "c"}
    wrong = frame.copy(); wrong.loc[0, "animal"] = "unreviewed"
    with pytest.raises(ValueError, match="Unreviewed"):
        repair.canonicalize_owned(wrong, contract)


def test_exact_origin_ownership_and_complete_case_retirement(proof):
    parent, contract, frame = proof
    _, lineage = repair.canonicalize_owned(frame, contract)
    ownership = repair.resolve_parent_origins(parent, lineage, contract)
    assert ownership.child_canonical_animal.tolist() == ["a", "b", None, None, "c"]
    assert ownership.ownership_reason.tolist() == ["", "", "ambiguous_origin_ownership", "missing_origin_owner", ""]
    child = parent["origins"].iloc[[0, 1, 4]].copy()
    child["canonical_animal"] = ["a", "b", "c"]
    child["case_id"] = [0, 1, 2]
    accounting = repair.account_parent_cases(parent, ownership, child, contract)
    assert accounting.status.tolist() == ["reassigned_repaired_owner"]*2+["retired"]*2+["retained_unaffected"]
    assert accounting.parent_case_id.tolist() == list(range(5))
    assert pd.isna(accounting.child_case_id.iloc[2])
    assert set(accounting) >= {"ownership_raw_ids", "ownership_evidence", "retirement_reason"}
    with pytest.raises(ValueError, match="each old case"):
        repair.account_parent_cases(parent, ownership.iloc[1:], child, contract)


def test_missing_repaired_support_requires_source_qc_reason(proof):
    parent, contract, frame = proof
    _, lineage = repair.canonicalize_owned(frame, contract)
    ownership = repair.resolve_parent_origins(parent, lineage, contract)
    child = parent["origins"].iloc[[4]].copy(); child["case_id"] = [0]
    with pytest.raises(ValueError, match="source-QC reason"):
        repair.account_parent_cases(parent, ownership, child, contract)
    failures = {(r.child_canonical_animal, pd.Timestamp(r.origin_time).value): "invalid_repaired_physical_past"
                for r in ownership.iloc[:2].itertuples()}
    result = repair.account_parent_cases(parent, ownership, child, contract, failures)
    assert (result.status == "retired").sum() == 4


def test_recovered_raw_ids_inherit_historical_role_without_freshness_promotion(proof):
    parent, contract, _ = proof
    contract = copy.deepcopy(contract)
    contract["physical_mapping"]["lost"] = "lost"
    contract["exclusion_mapping"]["lost"] = "a"
    first = contract["parent_memberships"][0]
    first["legacy_collapsed_raw_ids"] = ["a", "b", "lost"]
    first["repaired_physical_components"] = ["a", "b", "lost"]
    first["requires_original_trajectory_recovery"] = True
    repair.validate_contract(contract, parent, verify_inputs=False)
    pinned = repair.inherited_protections(parent, contract)
    assert pinned["ancestors"]["lost"] == {"a"}
    assert pinned["roles"]["a"] == "temporal" and pinned["freshness"]["a"] is False
    assert pinned["raw_parent"]["lost"] == "a"
    first["repaired_physical_components"].remove("lost")
    with pytest.raises(ValueError, match="recovered physical child"):
        repair.validate_contract(contract, parent, verify_inputs=False)


def test_original_recovery_is_content_bound_complete_and_nonoverlapping(proof, tmp_path):
    _, contract, _ = proof
    contract = copy.deepcopy(contract)
    contract["raw_coordinate_ownership"] = [{"input_sha256": "old", "input_animal": "collapsed",
                                           "status": "already_collapsed_needs_original_recovery"}]
    with pytest.raises(ValueError, match="incomplete"):
        repair.recovery_outputs(contract)
    recovered = tmp_path/"recovered.parquet"; recovered.write_bytes(b"synthetic source-owned bytes")
    record = {"path": str(recovered), "sha256": digest(recovered), "raw_ids": ["a", "b"],
              "source_QC_provenance": "Synthetic publisher tuple validation",
              "replaces": [{"input_sha256": "old", "input_animal": "collapsed"}]}
    contract["original_recovery"] = {"complete": True, "outputs": [record]}
    assert repair.recovery_outputs(contract) == [record]
    recovered.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        repair.recovery_outputs(contract)


def test_reserved_case_role_sealing_and_base_bytes_are_preserved(proof):
    parent, contract, frame = proof
    hashes = {k: digest(v) for k, v in parent["paths"].items()}
    _, lineage = repair.canonicalize_owned(frame, contract)
    ownership = repair.resolve_parent_origins(parent, lineage, contract)
    child = parent["origins"].iloc[[0, 1, 4]].copy()
    child["canonical_animal"] = ["a", "b", "c"]; child["case_id"] = [0, 1, 2]
    bad = child.copy(); bad.loc[bad.canonical_animal == "c", "targets_sealed"] = False
    with pytest.raises(ValueError, match="unsealed"):
        repair.account_parent_cases(parent, ownership, bad, contract)
    repair.account_parent_cases(parent, ownership, child, contract)
    assert {k: digest(v) for k, v in parent["paths"].items()} == hashes


def test_certified_origin_selection_replay_resolves_colocated_raw_owners(proof,tmp_path):
    parent,contract,frame = proof
    path=tmp_path/"unmixed.parquet"; frame.to_parquet(path,index=False)
    binding={"path":str(path),"sha256":digest(path)}
    parent["manifest"]["sources"]=[binding]
    contract=copy.deepcopy(contract)
    contract["raw_coordinate_ownership"]=[{"input_sha256":binding["sha256"],"input_animal":n,
                                          "status":"unmixed_original_source_id"} for n in ["a","b","c"]]
    _,lineage=repair.canonicalize_owned(frame,contract)
    result=repair.resolve_parent_origins(parent,lineage,contract,frozen_inputs=[(binding,frame)])
    assert result.child_canonical_animal.tolist() == ["a","b","a",None,"c"]
    assert result.selected_raw_animal.iloc[2] == "a"
    assert "selection" in result.ownership_evidence.iloc[2]
    assert contract["physical_mapping"]["a"] != contract["physical_mapping"]["b"]
    micro=frame.copy();micro["timestamp"]=micro.timestamp.dt.as_unit("us")
    parent["origins"]["origin_time"]=parent["origins"].origin_time.dt.as_unit("us")
    micro_result=repair.resolve_parent_origins(parent,lineage,contract,frozen_inputs=[(binding,micro)])
    assert micro_result.child_canonical_animal.tolist() == result.child_canonical_animal.tolist()
    # Changing a frozen origin tuple cannot make a nearby animal its owner.
    changed=frame.copy(); changed["lat"]=changed.lat.astype(float)
    changed.loc[changed.animal.eq("a"),"lat"]+=.001
    with pytest.raises(ValueError,match="exact frozen source"):
        wrong={**binding,"sha256":"wrong"}
        repair.resolve_parent_origins(parent,lineage,contract,frozen_inputs=[(wrong,changed)])


def test_collapsed_row_replay_needs_ordered_code_bound_original_row_proof(proof,tmp_path):
    parent,contract,frame=proof
    _,lineage=repair.canonicalize_owned(frame,contract)
    collapsed=frame.loc[~frame.animal.eq("b")].copy()
    path=tmp_path/"collapsed.parquet"; collapsed.to_parquet(path,index=False)
    binding={"path":str(path),"sha256":digest(path)}
    parent["manifest"]["sources"]=[binding]
    contract=copy.deepcopy(contract)
    contract["raw_coordinate_ownership"]=[
        {"input_sha256":binding["sha256"],"input_animal":"a","status":"already_collapsed_needs_original_recovery"},
        {"input_sha256":binding["sha256"],"input_animal":"c","status":"unmixed_original_source_id"}]
    absent=repair.resolve_parent_origins(parent,lineage,contract,frozen_inputs=[(binding,collapsed)])
    assert absent.ownership_reason.iloc[2] == "ambiguous_origin_ownership"
    table=tmp_path/"ownership.parquet"
    pd.DataFrame({"old_component":["a"],"timestamp":[frame.timestamp.iloc[2]],"lat":[30.],"lon":[1.],
                  "animal":["b"],"original_row_id":["publisherSHA:original_record17"]}).to_parquet(table,index=False)
    meta=tmp_path/"ownership.json"
    metadata={"ownership_table_sha256":digest(table),"all_old_tuples_reproduced_exactly":True,
              "source_order":{"originalfile.csv":0},"old_observation_sha256":binding["sha256"]}
    meta.write_text(json.dumps(metadata))
    contract["historical_selected_rows"]={"path":str(table),"sha256":digest(table),
        "metadata":{"path":str(meta),"sha256":digest(meta)}}
    with pytest.raises(ValueError,match="code-bound"):
        repair.resolve_parent_origins(parent,lineage,contract,frozen_inputs=[(binding,collapsed)])
    code=tmp_path/"historical-selector.py";code.write_text("# synthetic explicitly frozen selector\n")
    metadata["replay_code_sha256"]=digest(code)
    metadata["replay_code"]={"path":str(code),"sha256":digest(code)}
    meta.write_text(json.dumps(metadata));contract["historical_selected_rows"]["metadata"]["sha256"]=digest(meta)
    result=repair.resolve_parent_origins(parent,lineage,contract,frozen_inputs=[(binding,collapsed)])
    assert result.child_canonical_animal.iloc[2] == "b"
    assert result.selected_original_row_id.iloc[2] == "publisherSHA:original_record17"
    assert result.ownership_reason.iloc[0] == "ambiguous_origin_ownership"
