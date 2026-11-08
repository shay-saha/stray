import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from wildcast import migration_data as legacy
from wildcast.migration_clock import export, make_policy, write_policy
from wildcast.multiscale_data import (HORIZONS, LAGS, LEGACY_LAG_COLUMNS,
                                     features, match_targets, prepare, query_features,
                                     target_tolerances, target_admission_mask)
from wildcast.multiscale_data import _cross_taxon_declarations
from wildcast.parent_clock import audit_event_clocks,make_evidence,validate_evidence


def bucket(name):
    return int(hashlib.sha256(f"migration-v1:1701:{name}".encode()).hexdigest()[:12],16)%100


def name_for(role, prefix="z-source:study:bird"):
    for i in range(1000):
        name=f"{prefix}:{i}";b=bucket(name)
        if (role == "test" and b < 15) or (role == "val" and 15 <= b < 30) or (role == "train" and b >= 30):
            return name
    raise AssertionError("No synthetic bucket")


@pytest.fixture
def frozen_parent(tmp_path):
    names={role:name_for(role) for role in ["test","val","train"]}
    names["nonfresh"]=name_for("train","z-source:study:bird:known")
    times=pd.date_range("2019-12-01","2022-01-12",freq="D",tz="UTC")+pd.Timedelta(nanoseconds=123)
    rows=[]
    for j,name in enumerate(names.values()):
        for i,t in enumerate(times):
            rows.append({"timestamp":t,"lat":20+j*10+np.sin(i/20)/100,"lon":30+i/1000,
                         "animal":name,"species":"bird","source":"z-source","study":"z-source:study"})
    frame=pd.DataFrame(rows);path=tmp_path/"parent-observations.parquet";frame.to_parquet(path,index=False)
    previous=tmp_path/"old.json";previous.write_text(json.dumps({"animals":{names["nonfresh"]:0}}))
    root=tmp_path/"frozen";manifest=legacy.prepare([path],root,existing_manifest=previous)
    return {"root":root,"manifest":manifest,"path":path,"frame":frame,"names":names,"previous":previous}


def compatible(tmp_path, frame, name="new"):
    processed=tmp_path/name/"processed";processed.mkdir(parents=True)
    qc=frame.rename(columns={"timestamp":"timestamp_nominal"}).copy()
    qc["timestamp_utc"]=qc.timestamp_nominal
    qc["timestamp_reported"]=qc.timestamp_nominal.astype(str)
    qc["time_standard_status"]="verified_UTC_or_explicit_offset"
    qc.to_parquet(processed/"observations.parquet",index=False)
    manifest=processed/"manifest.json"
    manifest.write_text(json.dumps({"schema":"wildcast-breadth-clean-v1",
                                    "observations_sha256":legacy.sha256(processed/"observations.parquet")}))
    evidence=tmp_path/(name+"-clock.txt");evidence.write_text("Synthetic source declares UTC")
    sources={str(s):{"mode":"declared_utc","documented_clock":"UTC","evidence":[{
        "path":str(evidence),"kind":"source_declared_utc","claim":"Synthetic UTC declaration"}]} for s in qc.source.unique()}
    p=make_policy([manifest],sources,minimum_horizon_days=1/24)
    pp=tmp_path/(name+"-clock.json");write_policy(pp,p)
    out=tmp_path/(name+"-compat");export(pp,out,batch_size=1000)
    return out


def short_track(name, species="Short taxon", source="new", start="2020-06-01", days=3):
    times=pd.date_range(start,periods=int(days*48)+1,freq="30min",tz="UTC")
    return pd.DataFrame({"timestamp":times,"lat":10+np.arange(len(times))/100000,"lon":50.,
                         "animal":name,"species":species,"source":source,"study":source+":study"})


def test_subday_tolerances_are_strict_and_day_rule_matches_legacy():
    np.testing.assert_array_equal(target_tolerances([1/24,.5,1]),[1/96,1/24,.125])
    times=np.array([0,75*60,13*3600,27*3600],dtype=float)
    ix,mask=match_targets(times,0,[1/24,.5,1])
    assert mask.tolist() == [True,True,True] and ix.tolist() == [1,2,3]
    times[1]=76*60
    assert match_targets(times,0,[1/24,.5,1])[1].tolist() == [False,True,True]
    sparse=np.array([0,2*3600],dtype=float)
    assert not match_targets(sparse,0,[1/24])[1].any()
    assert legacy.match_targets(sparse,0,[1/24])[1].all()
    regular=np.arange(0,100*legacy.DAY,3600.)
    a=match_targets(regular,8,[1,3,7,14,30,60,90]);b=legacy.match_targets(regular,8,[1,3,7,14,30,60,90])
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    assert match_targets([0,45*60,10_000],0,[1/24])[1].all()
    assert match_targets([0,45*60,75*60],0,[1/24])[0].tolist() == [2]


def test_history_is_strictly_causal_and_legacy_slots_are_preserved():
    dates=pd.date_range("2020-01-01",periods=40*24+1,freq="h",tz="UTC")
    times=dates.asi8/1e9;lat=np.arange(len(times))/100000+20;lon=np.full(len(times),30.)
    before=features(lat,lon,times,dates,500)
    altered_lat=lat.copy();altered_lon=lon.copy();altered_lat[501:]=80;altered_lon[501:]=-170
    after=features(altered_lat,altered_lon,times,dates,500)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    old=legacy.features(lat,lon,times,dates,500)
    np.testing.assert_array_equal(before[0][LEGACY_LAG_COLUMNS],old[0])
    np.testing.assert_array_equal(before[1][LEGACY_LAG_COLUMNS],old[1])
    np.testing.assert_array_equal(before[2][:len(legacy.STATIC_NAMES)],old[2])
    assert len(LAGS) == 27 and before[0].shape == (27,10)
    assert (before[4][before[1]] <= times[500]-LAGS[before[1]]*legacy.DAY).all()
    public=query_features([*lat[:501],"future-invalid"],[*lon[:501],"bad"],
                          [*times[:501],"unknown-clock"],[*dates[:501],"invalid-date"],500)
    for a,b in zip(before,public):np.testing.assert_array_equal(a,b)


def test_short_training_support_and_unknown_species_metric_groups(frozen_parent,tmp_path):
    training_name=name_for("train","new:study:Short taxon")
    held_a=name_for("test","held-a:study:Only A");held_b=name_for("test","held-b:study:Only B")
    extra=pd.concat([short_track(training_name),short_track(held_a,"Only A","held-a"),
                     short_track(held_b,"Only B","held-b")],ignore_index=True)
    core=compatible(tmp_path,extra)
    identity=tmp_path/"identity.json"
    identity.write_text(json.dumps({"fold_groups":{},"uncertainty_fold_groups":{n:n for n in extra.animal.unique()}}))
    base_hashes={name:legacy.sha256(frozen_parent["root"]/name) for name in ["manifest.json","cases.npz","origins.parquet"]}
    output=tmp_path/"multiscale"
    manifest=prepare([frozen_parent["path"],core],output,parent_cache=frozen_parent["root"],
                     identity_audit=identity,existing_manifest=frozen_parent["previous"])
    origins=pd.read_parquet(output/"origins.parquet");parent_map=pd.read_parquet(output/"parent-case-map.parquet")
    assert len(parent_map) == frozen_parent["manifest"]["cases"]
    assert (parent_map.parent_split == parent_map.child_split).all()
    assert manifest["species"]["Short taxon"] > 0
    assert origins.loc[origins.species.isin(["Short taxon","Only A","Only B"]),"fresh_identity"].all()
    for name,digest in base_hashes.items():assert legacy.sha256(frozen_parent["root"]/name) == digest
    with np.load(output/"cases.npz",allow_pickle=False) as a:
        selected=origins.species.eq("Short taxon").to_numpy();ids=np.flatnonzero(selected)
        assert len(ids) > 3 and a["target_mask"][ids,0].any()
        assert not a["target_mask"][ids][:,HORIZONS >= 7].any()
        assert not origins.loc[selected,"long_history_support"].any()
        assert set(origins.loc[selected,"long_migration_support_proxy"]) == {"no"}
        starts=extra.loc[extra.animal == training_name,"timestamp"].min().timestamp()
        assert a["origin"][ids[0],2]-starts >= 6*3600
        assert (np.diff(a["origin"][ids,2]) >= 6*3600).all()
        for species in ["Only A","Only B"]:
            assert (a["species"][origins.species.eq(species)] == 0).all()
        assert manifest["metric_species"]["Only A"] != manifest["metric_species"]["Only B"]
        sealed=a["split"] >= 3
        assert not a["target_mask"][sealed].any()
        for key in ["y","target_latlon","target_time"]:assert not a[key][sealed].any()
        assert origins.loc[sealed,"targets_sealed"].all()
        assert set(origins.loc[sealed,"long_migration_support_proxy"]) == {"sealed"}
        assert (a["history_time"][a["history_mask"]] <= np.broadcast_to(a["origin"][:,2,None],a["history_time"].shape)[a["history_mask"]]).all()
        assert (a["target_time"][a["target_mask"]] > np.broadcast_to(a["origin"][:,2,None],a["target_time"].shape)[a["target_mask"]]).all()
        development=a["split"] <= 2
        for split,boundary in [(0,"2021-01-01"),(1,"2022-01-01"),(2,"2022-01-01")]:
            mask=(a["split"] == split)[:,None]&a["target_mask"]
            assert (a["target_time"][mask] < pd.Timestamp(boundary,tz="UTC").timestamp()).all()
        assert a["horizons"].shape == (9,) and a["history"].shape[1:] == (27,10)
        assert a["horizons"].dtype == np.float64
        np.testing.assert_array_equal(a["horizons"],HORIZONS)
    support=pd.read_csv(output/"development-support.csv")
    row=support[(support.species == "Short taxon")&(support.horizon_days < .1)]
    assert row.cases.sum() > 0 and row.long_history_cases.sum() == 0
    audit=manifest["parent_cache"]["shared_horizon_support"]
    assert audit["parent_development_targets"] == audit["preserved_development_targets"]
    assert sum(audit["lost_development_targets"]) == 0
    with pytest.raises(FileExistsError):prepare([frozen_parent["path"],core],output,parent_cache=frozen_parent["root"])


def test_alias_and_uncertain_groups_inherit_roles_freshness_and_reserved_years(frozen_parent,tmp_path):
    test=frozen_parent["names"]["test"];temporal=frozen_parent["names"]["train"]
    alias="a-new:study:bird:alias"
    strong=frozen_parent["frame"].loc[frozen_parent["frame"].animal == test].copy()
    strong["animal"],strong["source"],strong["study"]=alias,"a-new","a-new:study"
    uncertain="new-risk:study:Alternative ray:other"
    weak=frozen_parent["frame"].loc[frozen_parent["frame"].animal == temporal].copy()
    weak["animal"],weak["source"],weak["study"]=uncertain,"new-risk","new-risk:study";weak["lat"]+=5
    weak["species"]="Alternative ray"
    core=compatible(tmp_path,pd.concat([strong,weak],ignore_index=True))
    audit=tmp_path/"reference.json";audit.write_text(json.dumps({"fold_groups":{test:alias},
        "ambiguous":[{"animals":[temporal,uncertain]}]}))
    output=tmp_path/"child"
    manifest=prepare([frozen_parent["path"],core],output,parent_cache=frozen_parent["root"],reference_audit=audit)
    assert manifest["identity_mapping"][test] == alias
    assert manifest["assignments"][alias] == "test_individual"
    assert manifest["identity_mapping"][uncertain] != manifest["identity_mapping"][temporal]
    assert manifest["animals"][uncertain] == manifest["animals"][temporal]
    assert manifest["assignments"][uncertain] == manifest["assignments"][temporal] == "temporal"
    block=next(r for r in manifest["cross_taxon_exclusion_groups"] if r["individual"] == manifest["animals"][uncertain])
    assert block == {"individual":manifest["animals"][uncertain],"canonical_animals":sorted([temporal,uncertain]),
                     "raw_species":["Alternative ray","bird"],
                     "canonical_taxa":{temporal:"bird",uncertain:"Alternative ray"},"kind":"uncertain_identity_block"}
    rows=pd.read_parquet(output/"origins.parquet")
    future=rows[(rows.canonical_animal.isin([temporal,uncertain]))&(rows.year >= 2022)]
    assert len(future) and set(future["split"]) == {"test_year"}
    assert not future.fresh_identity.any() and future.targets_sealed.all()
    known=rows[rows.canonical_animal == frozen_parent["names"]["nonfresh"]]
    assert not known.fresh_identity.any()
    links=pd.read_parquet(output/"parent-case-map.parquet")
    assert len(links) == frozen_parent["manifest"]["cases"]


def test_conflicting_parent_roles_cannot_be_hidden_in_an_uncertain_component(frozen_parent,tmp_path):
    audit=tmp_path/"bad-reference.json";audit.write_text(json.dumps({"fold_groups":{},"ambiguous":[{
        "animals":[frozen_parent["names"]["test"],frozen_parent["names"]["val"]]}]}))
    with pytest.raises(ValueError,match="Conflicting parent roles"):
        prepare([frozen_parent["path"]],tmp_path/"bad",parent_cache=frozen_parent["root"],reference_audit=audit)
    assert not (tmp_path/"bad").exists()


def test_clock_preflight_old_source_hash_and_cutoffs_are_mandatory(frozen_parent,tmp_path):
    extra=short_track(name_for("train","new:study:bird"));raw=tmp_path/"unapproved.parquet";extra.to_parquet(raw,index=False)
    with pytest.raises(ValueError,match="actual clock-compatible"):
        prepare([frozen_parent["path"],raw],tmp_path/"raw-child",parent_cache=frozen_parent["root"])
    core=compatible(tmp_path,extra)
    with pytest.raises(ValueError,match="Every frozen parent source"):
        prepare([core],tmp_path/"missing-old",parent_cache=frozen_parent["root"])
    with pytest.raises(ValueError,match="freezes calendar cutoffs"):
        prepare([frozen_parent["path"],core],tmp_path/"advanced",parent_cache=frozen_parent["root"],train_end_year=2021,validation_year=2022)
    with pytest.raises(ValueError,match="explicit new protocol"):
        legacy.prepare([frozen_parent["path"]],tmp_path/"legacy-changed",parent_cache=frozen_parent["root"],horizons=HORIZONS)


@pytest.mark.parametrize("times", [[0,1,1],[0,np.nan,2],[0,2,1]])
def test_invalid_observation_times_are_rejected(times):
    with pytest.raises(ValueError):match_targets(times,0)


def test_pure_clock_admission_canonical_exclusions_zeros_and_cutoffs():
    manifest={"subday_clock_excluded_animals":["unknown"],
              "subday_clock_modes":{"bounded":"bounded_utc_or_gps","UTC":"declared_utc"},
              "cutoff_boundaries":["2021-01-01T00:00:00Z","2022-01-01T00:00:00Z"]}
    h=[1/24,.5,1]
    assert target_admission_mask(h,"unknown",manifest).tolist() == [False,False,True]
    assert target_admission_mask(h,"not-recorded",manifest).tolist() == [False,False,True]
    origin=pd.Timestamp("2020-06-01",tz="UTC").timestamp()
    for name in ["bounded","UTC"]:
        assert target_admission_mask(h,name,manifest,origin_time=origin,target_times=[0,origin+43200,0]).tolist() == [False,True,True]
    edge=pd.Timestamp("2021-01-01",tz="UTC").timestamp()
    assert target_admission_mask(h,"bounded",manifest,origin_time=edge+1,target_times=[edge+3600,edge+43200,edge+86400]).tolist() == [False,False,True]
    assert target_admission_mask(h,"bounded",manifest,origin_time=edge-3600,target_times=[edge+1,edge+40000,edge+80000]).tolist() == [False,True,True]
    assert target_admission_mask(h,"bounded",manifest,origin_time=origin,target_times=[1e100,0,0]).tolist() == [False,False,True]


def test_parent_subday_default_cannot_be_upgraded_by_new_documented_alias(frozen_parent,tmp_path):
    original=frozen_parent["names"]["train"];alias="a-documented:study:bird:alias"
    base=frozen_parent["frame"].loc[frozen_parent["frame"].animal == original].copy()
    dense_times=pd.date_range(base.timestamp.iloc[0],periods=12*24,freq="h")
    dense=pd.DataFrame({"timestamp":dense_times,"lat":float(base.lat.iloc[0])+np.arange(len(dense_times))/100000,
                        "lon":30+np.arange(len(dense_times))/24000,"animal":alias,"species":"bird",
                        "source":"a-documented","study":"a-documented:study"})
    dense=dense.loc[~dense.timestamp.isin(base.timestamp)]
    base["animal"],base["source"],base["study"]=alias,"a-documented","a-documented:study"
    core=compatible(tmp_path,pd.concat([base,dense],ignore_index=True))
    without=tmp_path/"without-clock"
    m=prepare([frozen_parent["path"],core],without,parent_cache=frozen_parent["root"])
    assert alias in m["subday_clock_excluded_animals"]
    with np.load(without/"cases.npz") as a:assert not a["target_mask"][:,HORIZONS < 1].any()
    raw=tmp_path/"old-events.csv";frozen_parent["frame"].to_csv(raw,index=False)
    doc=tmp_path/"old-clock.txt";doc.write_text("Synthetic original source explicitly declares UTC")
    decisions={"z-source":{"mode":"declared_utc","documented_clock":"UTC","evidence":[{
        "path":str(doc),"kind":"source_declared_utc","claim":"Original source UTC declaration"}]}}
    audit=tmp_path/"raw-clock.json"
    audit_event_clocks([{"source_id":"z-source","path":str(raw),"sha256":legacy.sha256(raw),
                         "accepted_sensor_rows_recorded":len(frozen_parent["frame"])}],decisions,audit)
    proof=tmp_path/"parent-clock.json";make_evidence(frozen_parent["root"],decisions,audit,proof)
    with_clock=tmp_path/"with-clock"
    m=prepare([frozen_parent["path"],core],with_clock,parent_cache=frozen_parent["root"],parent_clock_audit=proof)
    assert alias not in m["subday_clock_excluded_animals"] and m["subday_clock_modes"][alias] == "declared_utc"
    assert any(r["kind"] == "parent_subday_clock" for r in m["audits"])
    with np.load(with_clock/"cases.npz") as a:assert a["target_mask"][:,0].any()
    assert m["parent_cache"]["parent_cases_preserved"] == frozen_parent["manifest"]["cases"]
    # The exact frozen parent bytes are still present and every old origin/day
    # target remains supported; supplemental evidence adds no UTC correction.
    assert sum(m["parent_cache"]["shared_horizon_support"]["lost_development_targets"]) == 0
    parent=legacy.load_parent_cache(frozen_parent["root"],2020,2021,legacy.np.array([1,3,7,14,30,60,90]),7.)
    validate_evidence(proof,parent)
    doc.write_text("changed original clock documentation")
    with pytest.raises(ValueError,match="evidence hash"):
        validate_evidence(proof,parent)


def test_malformed_source_clock_withholds_entire_source(tmp_path):
    raw=tmp_path/"raw.csv"
    pd.DataFrame({"timestamp":["2020-08-11 18:00:00","2020-08-11 18:00:0S"],
                  "sensor-type":["gps","gps"]}).to_csv(raw,index=False)
    doc=tmp_path/"README.txt";doc.write_text("Event timestamp units UTC or GPS time")
    decisions={"s":{"mode":"bounded_utc_or_gps","documented_clock":"UTC_or_GPS","evidence":[{
        "path":str(doc),"kind":"source_utc_or_gps","claim":"Explicit bounded clock declaration"}]}}
    report=audit_event_clocks([{"source_id":"s","path":str(raw),"sha256":legacy.sha256(raw),
                               "accepted_sensor_rows_recorded":2}],decisions,tmp_path/"audit.json")
    assert report["sources"]["s"]["status"] == "held_source_clock_anomaly"
    assert report["sources"]["s"]["counts"]["malformed_or_impossible_timestamp_rows"] == 1


def test_fast_clock_admission_matches_adapter_leap_and_split_boundaries():
    from wildcast.migration_clock import GPS_OFFSETS,_gps_interpretation,STANDARDS
    manifest={"subday_clock_modes":{"a":"bounded_utc_or_gps"},"subday_clock_excluded_animals":[],
              "cutoff_boundaries":["2021-01-01T00:00:00Z","2022-01-01T00:00:00Z"]}
    edges=[pd.Timestamp(d,tz="UTC").timestamp()+offset for d,offset in GPS_OFFSETS]
    edges.extend([pd.Timestamp(v).timestamp() for v in manifest["cutoff_boundaries"]])
    edges.append(pd.Timestamp(STANDARDS["valid_before"]).timestamp())
    offsets=[-20.,-1.,-.5,0.,.5,1.,17.,18.,19.]
    for edge in edges:
        for delta in offsets:
            stamp=edge+delta
            nominal=pd.Series(pd.to_datetime([stamp],unit="s",utc=True))
            lower,valid=_gps_interpretation(nominal)
            for boundary in manifest["cutoff_boundaries"]:
                valid &= ~(lower.lt(pd.Timestamp(boundary))&nominal.ge(pd.Timestamp(boundary))).to_numpy()
            got=target_admission_mask([1/24],"a",manifest,target_times=[stamp])[0]
            assert got == bool(valid[0])


def test_raw_species_conflicts_need_physical_resolution_or_explicit_uncertainty():
    physical=pd.DataFrame({"animal":["a","a"],"species":["Manta","Mobula alfredi"]})
    with pytest.raises(ValueError,match="canonical physical identity"):
        _cross_taxon_declarations(physical,{"a":"a"},{"a"},{"a":0})
    separate=pd.DataFrame({"animal":["a","b"],"species":["Manta","Mobula alfredi"]})
    with pytest.raises(ValueError,match="every canonical member"):
        _cross_taxon_declarations(separate,{"a":"a","b":"a"},{"a"},{"a":0,"b":0})
    assert _cross_taxon_declarations(separate,{"a":"a","b":"b"},set(),{"a":0,"b":1}) == []
