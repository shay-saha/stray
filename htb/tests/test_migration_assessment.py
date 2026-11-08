"""Synthetic assessment releases only: actual reserved wildlife targets stay sealed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildcast.data import project
from wildcast.migration_assessment import SCOPE, apply_assessment, release, sha256


@pytest.fixture
def fixture(tmp_path):
    data = tmp_path / "development"
    data.mkdir()
    animal_names = ["train", "reserved-individual", "reserved-year"]
    raw_names = ["raw-train", "different-source-alias", "raw-year"]
    starts = pd.to_datetime(["2019-01-10", "2019-02-10", "2022-01-10"], utc=True)
    rows = []
    for raw, stamp in zip(raw_names, starts):
        for day in [-8, -1, 0, 1, 3, 4]:
            rows.append({"animal": raw, "species": "Synthetic bird", "timestamp": stamp + pd.Timedelta(days=day),
                         "lat": 10. + .1 * day, "lon": 20. + .2 * day})
    observations = tmp_path / "synthetic-observations.parquet"
    pd.DataFrame(rows).to_parquet(observations, index=False)
    origin = np.column_stack([np.full(3, 10.), np.full(3, 20.), starts.asi8 / 1e9])
    a = {"history": np.zeros((3, 2, 10), "float32"), "history_mask": np.ones((3, 2), bool),
         "static": np.zeros((3, 2), "float32"), "species": np.ones(3, "int64"),
         "individual": np.arange(3), "study": np.zeros(3, "int64"), "origin": origin,
         "horizons": np.array([1., 3.], "float32"), "split": np.array([0, 3, 4], "int8"),
         "y": np.zeros((3, 2, 2), "float32"), "target_mask": np.zeros((3, 2), bool),
         "target_latlon": np.zeros((3, 2, 2)), "target_time": np.zeros((3, 2))}
    a["target_mask"][0] = True
    a["target_latlon"][0] = [[10.1, 20.2], [10.3, 20.6]]
    a["target_time"][0] = origin[0, 2] + np.array([1, 3]) * 86400
    a["y"][0] = project(a["target_latlon"][0, :, 0], a["target_latlon"][0, :, 1], 10., 20.) / 1e6
    np.savez_compressed(data / "cases.npz", **a)
    origins = pd.DataFrame({"case_id": [0, 1, 2], "canonical_animal": animal_names,
                            "origin_time": starts, "individual": [0, 1, 2], "study": ["synthetic"] * 3,
                            "split": ["train", "test_individual", "test_year"], "targets_sealed": [False, True, True]})
    origins.to_parquet(data / "origins.parquet", index=False)
    reference = tmp_path / "synthetic-reference-audit.json"
    reference.write_text('{"fold_groups": {}}\n')
    manifest = {"protocol": "wildcast-migration-v1", "cases": 3, "horizons_days": [1, 3], "unit_metres": 1e6,
                "reserved_test_targets_sealed": True, "cases_sha256": sha256(data / "cases.npz"),
                "origins_sha256": sha256(data / "origins.parquet"), "studies": {"synthetic": 0},
                "animals": dict(zip(animal_names, range(3))), "identity_mapping": dict(zip(raw_names, animal_names)),
                "sources": [{"path": str(observations), "sha256": sha256(observations)}],
                "reference_audit": str(reference), "reference_audit_sha256": sha256(reference),
                "target_rule": "Nearest actual observed future within the fixed migration tolerance"}
    (data / "manifest.json").write_text(json.dumps(manifest))
    hashes = {"manifest_sha256": sha256(data / "manifest.json"), "cases_sha256": sha256(data / "cases.npz")}
    checkpoint = tmp_path / "synthetic-checkpoint.pt"
    checkpoint.write_bytes(b"dummy weights for integrity checking, never model loading")
    selection = tmp_path / "frozen-selection.json"
    payload = {"frozen": True, "scope": SCOPE, "model_selection_protocol": {"description": "Synthetic validation only",
               "selection_splits": ["val_individual", "val_year"], "test_targets_used": False},
               "data": hashes, "checkpoints": [{"path": str(checkpoint), "sha256": sha256(checkpoint)}]}
    selection.write_text(json.dumps(payload))
    return {"data": data, "a": a, "manifest": manifest, "hashes": hashes, "selection": selection,
            "checkpoint": checkpoint, "observations": observations, "reference": reference,
            "output": tmp_path / "assessment"}


def select_change(f, fn):
    payload = json.loads(f["selection"].read_text())
    fn(payload)
    f["selection"].write_text(json.dumps(payload))


def test_synthetic_release_preserves_base_and_overlays_only_test_copy(fixture):
    f = fixture
    before = {p.name: p.read_bytes() for p in f["data"].iterdir()}
    metadata = release(f["data"], f["selection"], f["output"])
    assert metadata["cases"] == 2
    assert metadata["nonce"]
    result = apply_assessment(f["a"], f["manifest"], f["hashes"], f["output"], f["selection"])
    np.testing.assert_array_equal(result["target_latlon"][1:], [[[10.1, 20.2], [10.3, 20.6]]] * 2)
    assert result["target_mask"][1:].all()
    for key in f["a"]:
        np.testing.assert_array_equal(result[key][0], f["a"][key][0])
    assert not f["a"]["target_mask"][1:].any()
    assert {p.name: p.read_bytes() for p in f["data"].iterdir()} == before
    with pytest.raises(FileExistsError, match="immutable"):
        release(f["data"], f["selection"], f["output"])


@pytest.mark.parametrize("field,value,error", [("frozen", False, "frozen"), ("scope", "development", "scope"),
    ("model_selection_protocol", {"selection_splits": ["test_year"]}, "validation splits")])
def test_unfrozen_or_wrong_scope_selection_rejected(fixture, field, value, error):
    f = fixture
    select_change(f, lambda p: p.update({field: value}))
    with pytest.raises(ValueError, match=error):
        release(f["data"], f["selection"], f["output"])
    assert not f["output"].exists()


@pytest.mark.parametrize("name,error", [("checkpoint", "checkpoint"), ("observations", "source observations"), ("reference", "reference identity audit")])
def test_changed_inputs_rejected_before_release(fixture, name, error):
    f = fixture
    with f[name].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match=error):
        release(f["data"], f["selection"], f["output"])
    assert not f["output"].exists()


@pytest.mark.parametrize("name,error", [("checkpoint", "checkpoint"), ("observations", "source observations"), ("reference", "reference identity audit")])
def test_changed_inputs_rejected_when_applying(fixture, name, error):
    f = fixture
    release(f["data"], f["selection"], f["output"])
    with f[name].open("ab") as stream:
        stream.write(b"changed after release")
    with pytest.raises(ValueError, match=error):
        apply_assessment(f["a"], f["manifest"], f["hashes"], f["output"])


def test_overlapping_training_case_ids_rejected_even_with_updated_file_checksum(fixture):
    f = fixture
    release(f["data"], f["selection"], f["output"])
    path = f["output"] / "assessment.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["case_ids"] = np.array([0, 2])
    np.savez_compressed(path, **arrays)
    meta_path = f["output"] / "assessment.json"
    metadata = json.loads(meta_path.read_text())
    metadata["assessment_sha256"] = sha256(path)
    metadata["case_ids_sha256"] = hashlib.sha256(arrays["case_ids"].astype("<i8").tobytes()).hexdigest()
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="exactly the reserved"):
        apply_assessment(f["a"], f["manifest"], f["hashes"], f["output"])


def test_changed_supplied_array_is_rejected(fixture):
    f = fixture
    release(f["data"], f["selection"], f["output"])
    a = {k: v.copy() for k, v in f["a"].items()}
    a["y"][0, 0] += 1
    with pytest.raises(ValueError, match="Supplied base array"):
        apply_assessment(a, f["manifest"], f["hashes"], f["output"])


def test_derived_population_cache_is_ineligible(fixture):
    f = fixture
    manifest = json.loads((f["data"] / "manifest.json").read_text())
    manifest["eligible_for_final_assessment"] = False
    (f["data"] / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not eligible"):
        release(f["data"], f["selection"], f["output"])


def test_selection_fingerprint_and_optional_selection_copy_are_verified(fixture):
    f = fixture
    release(f["data"], f["selection"], f["output"])
    select_change(f, lambda p: p.update({"frozen": False}))
    with pytest.raises(ValueError, match="frozen selection"):
        apply_assessment(f["a"], f["manifest"], f["hashes"], f["output"], f["selection"])
    # Its immutable release-time copy remains independently verifiable.
    result = apply_assessment(f["a"], f["manifest"], f["hashes"], f["output"])
    assert result["target_mask"][1:].all()


def multiscale_fixture(f, offset_hours, include_day=False):
    from wildcast import multiscale_data
    rows = []
    starts = pd.to_datetime(f["a"]["origin"][:, 2], unit="s", utc=True)
    for raw, stamp in zip(f["manifest"]["identity_mapping"], starts):
        for hours in [-192, -24, 0, offset_hours, 12, *([24] if include_day else [])]:
            rows.append({"animal": raw, "species": "Synthetic bird",
                         "timestamp": stamp + pd.Timedelta(hours=hours),
                         "lat": 10. + hours * .001, "lon": 20. + hours * .002})
    # Microsecond Parquet timestamps must normalize to nanoseconds before asi8.
    observations = pd.DataFrame(rows)
    observations["timestamp"] = observations.timestamp.dt.as_unit("us")
    observations.to_parquet(f["observations"], index=False)
    a = {key: value.copy() for key, value in f["a"].items()}
    a["horizons"] = multiscale_data.HORIZONS.copy()
    a["history"] = np.zeros((3, 27, 10), "float32")
    a["history_mask"] = np.zeros((3, 27), bool)
    a["static"] = np.zeros((3, 18), "float32")
    for key, shape in {"y": (3, 9, 2), "target_latlon": (3, 9, 2), "target_time": (3, 9)}.items():
        a[key] = np.zeros(shape, dtype=a[key].dtype)
    a["target_mask"] = np.zeros((3, 9), bool)
    np.savez_compressed(f["data"] / "cases.npz", **a)
    manifest = f["manifest"].copy()
    manifest.update(protocol=multiscale_data.PROTOCOL, horizons_days=multiscale_data.HORIZONS.tolist(),
                    assessment_target_matcher="wildcast.multiscale_data.match_targets",
                    history_names=multiscale_data.HISTORY_NAMES, static_names=multiscale_data.STATIC_NAMES,
                    history_lags_days=multiscale_data.LAGS.tolist(), minimum_history_lags=2,
                    minimum_history_prior_fixes=8, minimum_history_span_days=.25,
                    minimum_observed_future_targets=1, cases_sha256=sha256(f["data"] / "cases.npz"),
                    train_end_year=2020, validation_year=2021,
                    cutoff_boundaries=["2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"],
                    subday_clock_excluded_animals=[],
                    subday_clock_modes={name: "declared_utc" for name in manifest["animals"]},
                    sources=[{"path": str(f["observations"]), "sha256": sha256(f["observations"])}])
    (f["data"] / "manifest.json").write_text(json.dumps(manifest))
    hashes = {"manifest_sha256": sha256(f["data"] / "manifest.json"),
              "cases_sha256": sha256(f["data"] / "cases.npz")}
    select_change(f, lambda p: p.update(data=hashes))
    return {**f, "a": a, "manifest": manifest, "hashes": hashes}


@pytest.mark.parametrize("offset_hours,one_hour_valid", [(1., True), (1.25, True), (1.25 + 1/3600, False)])
def test_multiscale_release_uses_one_hour_tolerance_and_one_target_minimum(fixture, offset_hours, one_hour_valid):
    f = multiscale_fixture(fixture, offset_hours)
    release(f["data"], f["selection"], f["output"])
    result = apply_assessment(f["a"], f["manifest"], f["hashes"], f["output"])
    assert result["target_mask"][1:, 0].tolist() == [one_hour_valid] * 2
    assert result["target_mask"][1:, 1].all()
    assert not result["target_mask"][1:, 2:].any()
    assert not f["a"]["target_mask"].any()
    if one_hour_valid:
        np.testing.assert_allclose(result["target_time"][1:, 0] - f["a"]["origin"][1:, 2],
                                   offset_hours * 3600, rtol=0, atol=1e-6)


def test_multiscale_sidecar_rejects_a_two_hour_fix_labelled_as_one_hour(fixture):
    f = multiscale_fixture(fixture, 1.)
    release(f["data"], f["selection"], f["output"])
    path = f["output"] / "assessment.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["target_time"][0, 0] += 3600
    np.savez_compressed(path, **arrays)
    metadata_path = f["output"] / "assessment.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["assessment_sha256"] = sha256(path)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="frozen matching rule"):
        apply_assessment(f["a"], f["manifest"], f["hashes"], f["output"])


def test_multiscale_clock_exclusion_preserves_day_target_and_rejects_override(fixture):
    f = multiscale_fixture(fixture, 1., include_day=True)
    f["manifest"]["subday_clock_excluded_animals"] = ["reserved-individual"]
    f["manifest"]["subday_clock_modes"].pop("reserved-individual")
    (f["data"] / "manifest.json").write_text(json.dumps(f["manifest"]))
    f["hashes"]["manifest_sha256"] = sha256(f["data"] / "manifest.json")
    select_change(f, lambda p: p.update(data=f["hashes"]))
    release(f["data"], f["selection"], f["output"])
    result = apply_assessment(f["a"], f["manifest"], f["hashes"], f["output"])
    assert not result["target_mask"][1, :2].any()
    assert result["target_mask"][1, 2]
    assert result["target_mask"][2, :3].all()
    path = f["output"] / "assessment.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["target_mask"][0, 0] = True
    np.savez_compressed(path, **arrays)
    metadata_path = f["output"] / "assessment.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["assessment_sha256"] = sha256(path)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="clock admission rule"):
        apply_assessment(f["a"], f["manifest"], f["hashes"], f["output"])


def repaired_fixture(f):
    from wildcast import multiscale_repaired as protocol
    f=multiscale_fixture(f,1.,include_day=True)
    count=len(f["a"]["split"])
    f["a"]["history"]=np.zeros((count,len(protocol.LAGS),len(protocol.HISTORY_NAMES)),"float32")
    f["a"]["history_mask"]=np.zeros((count,len(protocol.LAGS)),bool)
    f["a"]["static"]=np.zeros((count,len(protocol.STATIC_NAMES)),"float32")
    np.savez_compressed(f["data"]/"cases.npz",**f["a"])
    proof=f["data"].parent/"synthetic-physical-repair.json"
    proof.write_text('{"scope":"Synthetic ownership proof; no animal data"}')
    provenance={"kind":"physical_repair","path":str(proof),"sha256":sha256(proof)}
    accounting=f["data"]/"parent-case-accounting.parquet"
    pd.DataFrame({"parent_case_id":range(count),"status":["retained_unaffected"]*count}).to_parquet(accounting,index=False)
    f["manifest"].update(protocol=protocol.PROTOCOL,preparation_version=protocol.PROTOCOL,
        history_names=protocol.HISTORY_NAMES,static_names=protocol.STATIC_NAMES,history_lags_days=protocol.LAGS.tolist(),
        assessment_target_matcher="wildcast.multiscale_repaired.match_targets",physical_repair=provenance,
        audits=[provenance],parent_cache={"protocol":"wildcast-parent-physical-repair-accounted-v1",
            "physical_repair":provenance,"case_accounting":"parent-case-accounting.parquet",
            "case_accounting_sha256":sha256(accounting),"accounted_parent_cases":count,
            "case_status_counts":{"retained_unaffected":count}},
        cases_sha256=sha256(f["data"]/"cases.npz"),observation_source_rules=[{
            "sha256":sha256(f["observations"]),"allowed_raw_animals":list(f["manifest"]["identity_mapping"]),
            "excluded_raw_animals":[]}])
    refresh_repaired(f)
    return f


def refresh_repaired(f):
    (f["data"]/"manifest.json").write_text(json.dumps(f["manifest"]))
    f["hashes"]={"manifest_sha256":sha256(f["data"]/"manifest.json"),"cases_sha256":sha256(f["data"]/"cases.npz")}
    select_change(f,lambda p:p.update(data=f["hashes"]))


def test_repaired_assessment_uses_recovered_rows_instead_of_mixed_ancestry(fixture):
    f=repaired_fixture(fixture)
    good=pd.read_parquet(f["observations"])
    recovered=f["data"].parent/"recovered-original-rows.parquet"
    good.to_parquet(recovered,index=False)
    bad=good.copy()
    # Same raw IDs/times in the first source; stable tie-breaking alone would
    # pick these invalid biological associations without the ownership rule.
    later=bad.timestamp > pd.to_datetime(f["a"]["origin"][0,2],unit="s",utc=True)
    bad.loc[later,["lat","lon"]]=[55.,-120.]
    bad.to_parquet(f["observations"],index=False)
    oldhash,newhash=sha256(f["observations"]),sha256(recovered)
    names=list(f["manifest"]["identity_mapping"])
    f["manifest"].update(sources=[{"path":str(f["observations"]),"sha256":oldhash},
                                    {"path":str(recovered),"sha256":newhash}],
        observation_source_rules=[{"sha256":oldhash,"allowed_raw_animals":[],"excluded_raw_animals":names},
                                  {"sha256":newhash,"allowed_raw_animals":names,"excluded_raw_animals":[]}])
    refresh_repaired(f)
    release(f["data"],f["selection"],f["output"])
    result=apply_assessment(f["a"],f["manifest"],f["hashes"],f["output"])
    np.testing.assert_allclose(result["target_latlon"][1:,0],[[10.001,20.002]]*2)
    assert result["target_mask"][1:,:3].all()
    assert not f["a"]["target_mask"].any()


@pytest.mark.parametrize("selector", ["interval", "native_time", "record"])
def test_repaired_assessment_applies_bound_quarantine_to_microsecond_parquet(fixture,selector):
    f=repaired_fixture(fixture)
    stamp=float(f["a"]["origin"][1,2]+3600)
    rule=f["manifest"]["observation_source_rules"][0]
    rule["excluded_intervals"]=[{
        "input_animal":"different-source-alias","start_time":stamp,"end_time":stamp,
        "closed":"both","time_unit":"unix_seconds_on_nominal_axis",
        "evidence_sha256":f["manifest"]["physical_repair"]["sha256"],"reason":"ambiguous_source_owner"}]
    if selector != "interval":
        rule.pop("excluded_intervals")
        common={"evidence_sha256":f["manifest"]["physical_repair"]["sha256"],"reason":"ambiguous_source_owner"}
        if selector == "native_time":
            rule["excluded_native_times"]=[{**common,"input_animal":"different-source-alias",
                "timestamps_ns":[pd.Timestamp(stamp,unit="s",tz="UTC").value],"time_axis":"reported_nominal"}]
        else:
            source=pd.read_parquet(f["observations"])
            selected=(source.animal == "different-source-alias")&(source.timestamp == pd.Timestamp(stamp,unit="s",tz="UTC"))
            rule["excluded_input_records"]=[{**common,"input_rows":len(source),
                "row_positions":np.flatnonzero(selected).tolist(),"row_number_basis":"zero_based_full_input_order"}]
    refresh_repaired(f)
    release(f["data"],f["selection"],f["output"])
    result=apply_assessment(f["a"],f["manifest"],f["hashes"],f["output"])
    assert not result["target_mask"][1,0]
    assert result["target_mask"][1,1:3].all()
    assert result["target_mask"][2,:3].all()
