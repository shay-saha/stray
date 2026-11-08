"""Frozen cohort ancestry survives new taxa, aliases and uncertain groups."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from wildcast.migration_data import prepare, sha256


def bucket(name):
    return int(hashlib.sha256(f"migration-v1:1701:{name}".encode()).hexdigest()[:12], 16) % 100


@pytest.fixture
def parent(tmp_path):
    names = {}
    for i in range(500):
        name = f"z-source:study:bird:{i}"
        b = bucket(name)
        if b < 15: names.setdefault("test", name)
        elif b < 30: names.setdefault("val", name)
        else:
            names.setdefault("temporal", name)
            if name != names["temporal"]: names.setdefault("nonfresh", name)
    assert len(names) == 4
    dates = pd.date_range("2019-01-01", "2022-02-01", freq="D", tz="UTC") + pd.Timedelta(nanoseconds=123)
    rows = []
    for j, name in enumerate(names.values()):
        for i, time in enumerate(dates):
            rows.append({"animal": name, "species": "bird", "source": "z-source", "study": "z-source:study",
                         "timestamp": time, "lat": 20 + j * 5 + np.sin(i / 50), "lon": 30 + i / 1000})
    frame = pd.DataFrame(rows)
    observations = tmp_path / "observations.parquet"; frame.to_parquet(observations)
    old = tmp_path / "old.json"; old.write_text(json.dumps({"animals": {names["nonfresh"]: 0}}))
    data = tmp_path / "parent"
    manifest = prepare([observations], data, existing_manifest=old, horizons=(1, 3, 7))
    assert {manifest["assignments"][n] for n in names.values()} == {"test_individual", "val_individual", "temporal"}
    return {"data": data, "manifest": manifest, "frame": frame, "observations": observations, "names": names,
            "old": old, "options": {"existing_manifest": old, "horizons": (1, 3, 7)}}


def test_earlier_lexical_alias_cannot_rehash_parent_test_animal_into_training(parent, tmp_path):
    old = parent["names"]["test"]
    alias = next(f"a-new:study:bird:{i}" for i in range(500) if bucket(f"a-new:study:bird:{i}") >= 30)
    new = parent["frame"].loc[parent["frame"].animal == old].copy()
    new["animal"], new["source"], new["study"] = alias, "a-new", "a-new:study"
    path = tmp_path / "alias.parquet"; new.to_parquet(path)
    # This control documents the old failure: the canonical spelling changes
    # the hash bucket although this is the exact same moving animal.
    control = prepare([parent["observations"], path], tmp_path / "control", **parent["options"])
    assert control["identity_mapping"][old] == alias
    assert control["assignments"][alias] == "temporal"
    output = tmp_path / "child"
    child = prepare([parent["observations"], path], output, parent_cache=parent["data"], **parent["options"])
    assert child["assignments"][alias] == "test_individual"
    assert child["identity_mapping"][old] == alias
    origins = pd.read_parquet(output / "origins.parquet")
    rows = origins.loc[origins.canonical_animal == alias]
    assert set(rows["split"]) == {"test_individual"} and rows.targets_sealed.all()
    links = pd.read_parquet(output / "parent-case-map.parquet")
    assert len(links) == parent["manifest"]["cases"]
    assert (links.parent_split == links.child_split).all()
    with np.load(output / "cases.npz", allow_pickle=False) as a:
        sealed = a["split"] >= 3
        assert not a["target_mask"][sealed].any()
        for key in ("y", "target_latlon", "target_time"):
            assert not a[key][sealed].any()


def test_unrelated_new_taxa_preserve_every_old_case_and_freshness(parent, tmp_path):
    new = parent["frame"].loc[parent["frame"].animal == parent["names"]["temporal"]].copy()
    new["animal"], new["species"], new["source"], new["study"] = "new:study:newtaxon:1", "AAA new taxon", "new", "new:study"
    new["lat"] += 15
    path = tmp_path / "new-taxon.parquet"; new.to_parquet(path)
    empty_old = tmp_path / "new-old.json"; empty_old.write_text('{"animals": {}}')
    output = tmp_path / "child"
    child = prepare([parent["observations"], path], output, parent_cache=parent["data"], existing_manifest=empty_old, horizons=(1, 3, 7))
    mapping = pd.read_parquet(output / "parent-case-map.parquet")
    assert (mapping.parent_fresh_identity == mapping.child_fresh_identity).all()
    assert not mapping.loc[mapping.parent_canonical_animal == parent["names"]["nonfresh"], "child_fresh_identity"].any()
    with np.load(parent["data"] / "cases.npz", allow_pickle=False) as before, np.load(output / "cases.npz", allow_pickle=False) as after:
        for key in ("history", "history_mask", "static", "origin", "target_mask", "y", "target_latlon", "target_time", "split"):
            np.testing.assert_array_equal(before[key][mapping.parent_case_id], after[key][mapping.child_case_id])
    assert child["parent_cache"]["parent_cases_preserved"] == parent["manifest"]["cases"]
    assert sha256(output / "parent-case-map.parquet") == child["parent_cache"]["case_map_sha256"]


def test_new_earlier_observations_do_not_shift_away_frozen_origin_support(parent, tmp_path):
    old = parent["names"]["test"]
    observed = parent["frame"].loc[parent["frame"].animal == old].copy()
    earlier = observed.iloc[:12].copy()
    earlier["timestamp"] -= pd.Timedelta(days=12)
    earlier["lat"] = observed.lat.iloc[0] - np.arange(12, 0, -1) / 100
    alias = pd.concat([earlier, observed], ignore_index=True)
    alias["animal"], alias["source"], alias["study"] = "a-earlier:study:bird:1", "a-earlier", "a-earlier:study"
    path = tmp_path / "earlier.parquet"; alias.to_parquet(path)
    output = tmp_path / "child"
    child = prepare([parent["observations"], path], output, parent_cache=parent["data"], **parent["options"])
    assert child["parent_cache"]["parent_cases_preserved"] == parent["manifest"]["cases"]
    cases = pd.read_parquet(output / "parent-case-map.parquet")
    assert (cases.parent_split == cases.child_split).all()


def test_new_ambiguous_alias_is_excluded_with_parent_without_merging_coordinates(parent, tmp_path):
    old = parent["names"]["test"]
    name = "new:study:bird:ambiguous"
    new = parent["frame"].loc[parent["frame"].animal == old].copy()
    new["animal"], new["source"], new["study"] = name, "new", "new:study"
    new["lat"] += 20  # No shared GPS and no strong physical identity assertion.
    path = tmp_path / "new.parquet"; new.to_parquet(path)
    audit = tmp_path / "reference.json"
    audit.write_text(json.dumps({"fold_groups": {}, "ambiguous": [{"animals": [old, name]}]}))
    output = tmp_path / "child"
    child = prepare([parent["observations"], path], output, parent_cache=parent["data"], reference_audit=audit, **parent["options"])
    assert child["identity_mapping"][name] == name and child["identity_mapping"][old] == old
    assert child["animals"][name] == child["animals"][old]
    assert child["assignments"][name] == child["assignments"][old] == "test_individual"
    origins = pd.read_parquet(output / "origins.parquet")
    assert not origins.loc[origins.canonical_animal.isin([old, name]), "fresh_identity"].any()
    assert child["parent_cache"]["parent_freshness_suppressed_conservatively"] > 0


def test_conflicting_parent_roles_in_new_uncertain_component_fail(parent, tmp_path):
    audit = tmp_path / "conflict.json"
    audit.write_text(json.dumps({"fold_groups": {}, "ambiguous": [{"animals": [parent["names"]["test"], parent["names"]["val"]]}]}))
    output = tmp_path / "conflict"
    with pytest.raises(ValueError, match="Conflicting parent roles"):
        prepare([parent["observations"]], output, parent_cache=parent["data"], reference_audit=audit, **parent["options"])
    assert not (output / "cases.npz").exists()


def test_parent_uncertainty_groups_survive_when_child_audit_is_omitted(parent, tmp_path):
    first = parent["names"]["temporal"]
    second = "z-source:study:bird:uncertain-second"
    extra = parent["frame"].loc[parent["frame"].animal == first].copy()
    extra["animal"] = second; extra["lat"] += 20
    path = tmp_path / "more.parquet"; pd.concat([parent["frame"], extra]).to_parquet(path)
    audit = tmp_path / "uncertain.json"
    audit.write_text(json.dumps({"fold_groups": {}, "ambiguous": [{"animals": [first, second]}]}))
    frozen = tmp_path / "uncertain-parent"
    before = prepare([path], frozen, reference_audit=audit, **parent["options"])
    output = tmp_path / "child"
    after = prepare([path], output, parent_cache=frozen, **parent["options"])
    assert before["animals"][first] == before["animals"][second]
    assert after["animals"][first] == after["animals"][second]
    assert after["identity_mapping"][first] != after["identity_mapping"][second]
    assert after["assignments"][first] == before["assignments"][first]


@pytest.mark.parametrize("changed", [{"train_end_year": 2021, "validation_year": 2022}, {"release_test": True}, {"horizons": (1, 7, 30)}])
def test_parent_cutoffs_and_sealing_protocol_cannot_silently_advance(parent, tmp_path, changed):
    options = {**parent["options"], **changed}
    with pytest.raises(ValueError, match="Parent|protocol"):
        prepare([parent["observations"]], tmp_path / "child", parent_cache=parent["data"], **options)


def test_parent_origins_checksum_and_missing_cohort_guard(parent, tmp_path):
    original = parent["data"] / "origins.parquet"
    with original.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        prepare([parent["observations"]], tmp_path / "child", parent_cache=parent["data"], **parent["options"])


def test_missing_parent_animal_is_not_silently_removed(parent, tmp_path):
    subset = parent["frame"].loc[parent["frame"].animal != parent["names"]["test"]]
    path = tmp_path / "subset.parquet"; subset.to_parquet(path)
    with pytest.raises(ValueError, match="absent from child"):
        prepare([path], tmp_path / "child", parent_cache=parent["data"], **parent["options"])
