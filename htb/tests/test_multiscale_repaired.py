import json

import numpy as np
import pandas as pd
import pytest

from wildcast import migration_data as legacy
from wildcast import multiscale_repaired as repaired


def _name(role, prefix):
    import hashlib
    for i in range(1000):
        value = f"{prefix}:{i}"
        bucket = int(hashlib.sha256(f"migration-v1:1701:{value}".encode()).hexdigest()[:12], 16)%100
        if (role == "test" and bucket < 15) or (role == "train" and bucket >= 30):
            return value
    raise AssertionError("Synthetic hash bucket not found")


@pytest.fixture
def source_repair(tmp_path):
    a = _name("train", "source:study:bird:a")
    b = "source:study:bird:b"
    c = _name("test", "held:study:bird:c")
    dates = pd.date_range("2019-12-01", "2022-01-12", freq="D", tz="UTC")
    records = []
    for animal, lat, source in [(a, 10., "source"), (b, 11., "source"), (c, 30., "held")]:
        for i, time in enumerate(dates):
            # Six shared moving coordinates trigger the known old heuristic,
            # despite distinct locations over the rest of the simultaneous data.
            shared = animal == b and i < 6
            records.append({"timestamp": time, "lat": (10. if shared else lat)+i/100000,
                            "lon": 20.+i/10000, "animal": animal, "species": "bird",
                            "study": source+":study", "source": source})
    frame = pd.DataFrame(records); path = tmp_path/"unmixed-parent-source.parquet"
    frame.to_parquet(path, index=False)
    old = tmp_path/"previous.json"; old.write_text(json.dumps({"animals": {a:0}}))
    root = tmp_path/"parent"
    meta = legacy.prepare([path], root, existing_manifest=old)
    assert meta["identity_mapping"][a] == meta["identity_mapping"][b] == a
    parent = legacy.load_parent_cache(root, 2020, 2021, meta["horizons_days"], meta["stride_days"])
    risk = {a:a, b:a, c:c}
    combined = tmp_path/"combined.json"
    combined.write_text(json.dumps({"uncertainty_fold_groups": risk}))
    contract = {"schema": "wildcast-parent-physical-retraction-v1",
                **{"parent_"+key:value for key,value in parent["hashes"].items()},
                "combined_identity_sha256": legacy.sha256(combined),
                "inputs": {"combined": {"path": str(combined), "sha256": legacy.sha256(combined)}},
                "physical_mapping": {a:a, b:b, c:c}, "exclusion_mapping": risk,
                "retained_physical_primitives": [], "retract_edges": [{"a":a, "b":b}],
                "freshness_exclusion_animals": [a,b], "original_recovery": {"complete":True,"outputs":[]},
                "raw_coordinate_ownership": [
                    {"input_path":str(path), "input_sha256":legacy.sha256(path), "input_animal":n,
                     "status":"unmixed_original_source_id", "original_raw_ids":[n]} for n in [a,b,c]],
                "parent_memberships": [
                    {"parent_canonical":n, "raw_ids":([a,b] if n == a else [c]),
                     "legacy_collapsed_raw_ids":[], "repaired_physical_components":([a,b] if n == a else [c]),
                     "physical_partition_changed":n == a, "requires_original_trajectory_recovery":False,
                     "child_exclusion_groups":[risk[n]], "parent_role":meta["assignments"][n],
                     "parent_origin_freshness_values":[parent["freshness"][n]]} for n in [a,c]]}
    proof = tmp_path/"repair.json"; proof.write_text(json.dumps(contract))
    return {"path":path,"parent":parent,"root":root,"proof":proof,"contract":contract,
            "frame":frame,"a":a,"b":b,"c":c,"old":old}


def test_repaired_prepare_separates_coordinates_and_accounts_old_cases(source_repair, tmp_path):
    s = source_repair; output = tmp_path/"corrected"
    before = {k:legacy.sha256(v) for k,v in s["parent"]["paths"].items()}
    manifest = repaired.prepare([s["path"]], output, parent_cache=s["root"], repair_contract=s["proof"], existing_manifest=s["old"])
    origins = pd.read_parquet(output/"origins.parquet")
    accounting = pd.read_parquet(output/"parent-case-accounting.parquet")
    assert manifest["protocol"] == repaired.PROTOCOL
    assert manifest["identity_mapping"][s["a"]] != manifest["identity_mapping"][s["b"]]
    assert manifest["animals"][s["a"]] == manifest["animals"][s["b"]]
    assert manifest["assignments"][s["a"]] == manifest["assignments"][s["b"]] == "temporal"
    assert set(origins.canonical_animal) == {s["a"],s["b"],s["c"]}
    assert not origins.loc[origins.canonical_animal.isin([s["a"],s["b"]]),"fresh_identity"].any()
    assert accounting.parent_case_id.tolist() == list(range(len(s["parent"]["origins"])))
    assert set(accounting.status) <= {"retained_unaffected","reassigned_repaired_owner","retired"}
    assert manifest["parent_cache"]["accounted_parent_cases"] == len(accounting)
    with np.load(output/"cases.npz", allow_pickle=False) as arrays:
        assert arrays["history"].shape[1:] == (29,10) and arrays["static"].shape[1] == 22
        reserved = arrays["split"] >= 3
        assert reserved.any() and not arrays["target_mask"][reserved].any()
        assert all(not arrays[key][reserved].any() for key in ["y","target_latlon","target_time"])
        for animal in [s["a"],s["b"]]:
            ids = origins.index[origins.canonical_animal.eq(animal)].to_numpy()
            lat = arrays["origin"][ids,0]
            assert (lat < 10.1).all() if animal == s["a"] else (lat > 10.9).all()
    assert before == {k:legacy.sha256(v) for k,v in s["parent"]["paths"].items()}
    with pytest.raises(FileExistsError):
        repaired.prepare([s["path"]], output, parent_cache=s["root"], repair_contract=s["proof"])


def test_source_selector_quarantines_only_bound_raw_intervals(tmp_path):
    times = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    frame = pd.DataFrame({"animal":["a"]*5+["b"]*5+["old"]*5,"timestamp":list(times)*3,"lat":range(15)})
    manifest = {"observation_source_rules":[{"sha256":"bound", "allowed_raw_animals":["a","b"],
                 "excluded_raw_animals":["old"], "excluded_intervals":[{
                     "input_animal":"a","start_time":times[1].isoformat(),"end_time":times[3].isoformat(),
                     "closed":"both","evidence_sha256":"primary proof","reason":"ambiguous_duplicate_deployment"}]}]}
    selected = repaired.select_source_rows(frame,"bound",manifest)
    assert selected.loc[selected.animal.eq("a"),"timestamp"].tolist() == [times[0],times[4]]
    assert selected.animal.eq("b").sum() == 5 and not selected.animal.eq("old").any()
    assert len(frame) == 15
    with pytest.raises(ValueError,match="content-bound"):
        repaired.select_source_rows(frame,"changed",manifest)
    bad = frame.copy(); bad.loc[0,"animal"] = "not-reviewed"
    with pytest.raises(ValueError,match="absent its frozen"):
        repaired.select_source_rows(bad,"bound",manifest)
    interval = manifest["observation_source_rules"][0]["excluded_intervals"][0]
    interval["start_time"], interval["end_time"] = times[1].timestamp(), times[3].timestamp()
    with pytest.raises(ValueError,match="explicit nominal Unix-second"):
        repaired.select_source_rows(frame,"bound",manifest)
    interval["time_unit"] = "unix_seconds_on_nominal_axis"
    assert repaired.select_source_rows(frame,"bound",manifest).index.tolist() == selected.index.tolist()
    us=frame.copy();us["timestamp"]=us.timestamp.dt.as_unit("us")
    path=tmp_path/"microsecond-source.parquet";us.to_parquet(path,index=False)
    restored=pd.read_parquet(path)
    assert restored.timestamp.dt.unit == "us"
    assert repaired.select_source_rows(restored,"bound",manifest).index.tolist() == selected.index.tolist()


def test_reserved_timestamp_support_is_scalar_and_matches_actual_target_rule(monkeypatch):
    times = pd.date_range("2020-01-01", periods=7*24, freq="h", tz="UTC").asi8/1e9
    context = {"subday_clock_modes":{"a":"declared_utc"},"subday_clock_excluded_animals":[]}
    expected = repaired.match_targets(times,10)[1].any()
    def forbidden(*args,**kwargs):
        raise AssertionError("Reserved scalar check must not construct per-horizon target masks")
    monkeypatch.setattr(repaired,"match_targets",forbidden)
    result = repaired.has_future_timestamp(times,10,canonical_animal="a",clock_context=context)
    assert isinstance(result,bool) and result == expected
    assert repaired.has_future_timestamp(times, len(times)-1,canonical_animal="a",clock_context=context) is False
    assert repaired.has_future_timestamp(times,10,boundary=times[10]+1,canonical_animal="a",clock_context=context) is False


def test_issue_time_jump_qc_does_not_inspect_future_coordinates():
    times = np.arange(20,dtype=float)*3600
    lat = np.zeros(20); lon = np.arange(20)/1000
    a = repaired._PastJumpQC(lat,lon,times)
    changed = lat.copy(); changed[11:] = 80
    b = repaired._PastJumpQC(changed,lon,times)
    assert a.valid(10) and b.valid(10)
    assert a.done == b.done == 10
    assert a.prefix[11:].sum() == b.prefix[11:].sum() == 0
    assert a.valid(11) and not b.valid(11)


def test_frozen_native_timestamp_quarantine_applies_all_measurements_without_rounding():
    times=pd.DatetimeIndex(["2020-06-01T00:00:00.000001Z","2020-06-01T00:00:00.000001Z",
                            "2020-06-01T00:00:00.000002Z","2020-06-01T00:00:00.000001Z"]).as_unit("us")
    frame=pd.DataFrame({"animal":["a","a","a","b"],"timestamp":times,"lat":[10.,11.,12.,13.]})
    entry={"input_animal":"a","timestamps_ns":[pd.Timestamp(times[0]).value],
           "time_axis":"reported_nominal","reason":"source_measurement_ambiguity",
           "evidence_sha256":"synthetic_primary_proof","rows_in_this_occurrence":2}
    rule={"sha256":"frozen","allowed_raw_animals":["a","b"],"excluded_raw_animals":[],"excluded_native_times":[entry]}
    manifest={"observation_source_rules":[rule]}
    selected=repaired.select_source_rows(frame,"frozen",manifest)
    assert selected.index.tolist()==[2,3] and len(frame)==4
    entry["timestamps_ns"]=[pd.Timestamp(times[0]).value+1]
    with pytest.raises(ValueError,match="row count"):
        repaired.select_source_rows(frame,"frozen",manifest)
    entry["timestamps_ns"]=[float(pd.Timestamp(times[0]).value)]
    with pytest.raises(ValueError,match="integer keys"):
        repaired.select_source_rows(frame,"frozen",manifest)


def test_certified_input_record_selection_keeps_preferred_measurement_unchanged():
    frame=pd.DataFrame({"animal":["a"]*3,"timestamp":pd.to_datetime(["2020-06-01"]*3,utc=True),
                        "lat":[10.,10.000000000000002,11.]})
    record={"row_positions":[1],"row_number_basis":"zero_based_full_input_order","input_rows":3,
            "evidence_sha256":"frozen_native_record_proof","reason":"duplicate_transformed_representation"}
    rule={"sha256":"frozen","allowed_raw_animals":["a"],"excluded_raw_animals":[],"excluded_input_records":[record]}
    manifest={"observation_source_rules":[rule]}
    selected=repaired.select_source_rows(frame,"frozen",manifest)
    assert selected.index.tolist()==[0,2] and selected.lat.iloc[0]==frame.lat.iloc[0]
    with pytest.raises(ValueError,match="complete input"):
        repaired.select_source_rows(frame.iloc[::-1],"frozen",manifest)
    with pytest.raises(ValueError,match="full-input size"):
        repaired.select_source_rows(frame.iloc[:2],"frozen",manifest)
    with pytest.raises(ValueError,match="content-bound"):
        repaired.select_source_rows(frame,"different-input-sha",manifest)
    record["input_rows"]=4
    with pytest.raises(ValueError,match="full-input size"):
        repaired.select_source_rows(frame,"frozen",manifest)
    record["input_rows"]=3
    record["row_positions"]=[True]
    with pytest.raises(ValueError,match="integer positions"):
        repaired.select_source_rows(frame,"frozen",manifest)


def test_unreviewed_or_incomplete_repair_cannot_publish(source_repair,tmp_path):
    s=source_repair; bad=dict(s["contract"])
    bad["physical_mapping"]={**bad["physical_mapping"],s["b"]:s["a"]}
    proof=tmp_path/"bad.json"; proof.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="shadow alias"):
        repaired.prepare([s["path"]],tmp_path/"not-published",parent_cache=s["root"],repair_contract=proof)
    assert not (tmp_path/"not-published").exists()


def test_recovered_orphan_uses_certified_source_clock_without_upgrading_unknown_ancestor(source_repair,tmp_path):
    from wildcast.parent_clock import audit_event_clocks,make_evidence
    s=source_repair
    document=tmp_path/"clock-document.txt";document.write_text("Synthetic sources explicitly declare UTC")
    decisions={};files=[]
    for source,group in s["frame"].groupby("source"):
        raw=tmp_path/(source+"-timestamps.csv");group[["timestamp"]].to_csv(raw,index=False)
        decisions[source]={"mode":"declared_utc","documented_clock":"UTC","evidence":[{
            "path":str(document),"kind":"source_declared_utc","claim":"Synthetic UTC declaration"}]}
        files.append({"source_id":source,"path":str(raw),"sha256":legacy.sha256(raw),
                      "accepted_sensor_rows_recorded":len(group)})
    raw_audit=tmp_path/"raw-clock.json";audit_event_clocks(files,decisions,raw_audit)
    clock=tmp_path/"parent-clock.json";make_evidence(s["root"],decisions,raw_audit,clock)
    orphan="source:study:bird:previously-dropped"
    frame=pd.DataFrame({"animal":[orphan],"source":["source"]})
    context,_=repaired._clock_context(frame,s["parent"],[],[],clock,2020,2021)
    assert context["subday_clock_modes"][orphan] == "declared_utc"
    held,_=repaired._clock_context(frame,s["parent"],[{"child_animal":orphan,"parent_animal":"undocumented-old-ancestor"}],[],clock,2020,2021)
    assert orphan in held["subday_clock_excluded_animals"]
