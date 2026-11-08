import json

import numpy as np
import pandas as pd
import pytest

from wildcast.migration_episodes import audit


def observations(dates, longitude, animal="animal", latitude=0.):
    return pd.DataFrame({"animal": animal, "species": "Test species", "study": "study",
                         "timestamp": pd.to_datetime(dates, utc=True), "lat": latitude, "lon": longitude})


def manifest(assignments, mapping=None):
    return {"assignments": assignments, "identity_mapping": mapping or {},
            "train_end_year": 2020, "validation_year": 2021}


def test_without_manifest_reads_only_metadata_and_never_coordinates(tmp_path):
    # No coordinate columns exist: a coordinate-reading implementation fails.
    data = observations(pd.date_range("2020-01-01", periods=150, freq="D"), 0.).drop(columns=["lat", "lon"])
    path = tmp_path / "observations.parquet"
    data.to_parquet(path)
    report = audit(path, tmp_path / "audit")
    assert not report["coordinate_audit_enabled"]
    assert report["candidate_episodes"] == 0
    assert report["coverage_sensitivity"]["declared"]["animal_years"] == 1
    years = pd.read_csv(tmp_path / "audit" / "animal-year.csv")
    assert years.observed_calendar_days.iloc[0] == 150
    assert "maximum_observed_nsd_km2" not in years
    episodes = pd.read_parquet(tmp_path / "audit" / "candidate-episodes.parquet")
    assert episodes.empty and "verified_migration" in episodes


def test_sustained_observed_shift_and_return_are_only_retrospective_proxies(tmp_path):
    dates = pd.date_range("2020-01-01", periods=70, freq="D")
    data = observations(dates, np.r_[np.zeros(20), np.full(30, 3.), np.zeros(20)])
    path = tmp_path / "observations.parquet"
    data.to_parquet(path)
    report = audit(path, tmp_path / "audit", manifest({"animal": "temporal"}))
    assert report["candidate_episodes"] == 2
    assert report["observed_return_cycle_proxies"] == 1
    episodes = pd.read_parquet(tmp_path / "audit" / "candidate-episodes.parquet")
    assert not episodes.verified_migration.any()
    assert episodes.retrospective_only.all()
    assert episodes.proxy_kind.str.contains("needs_validation").all()
    assert episodes.first_shift_observation.iloc[0] == pd.Timestamp("2020-01-21", tz="UTC")
    assert episodes.anchor_time.iloc[0] < episodes.first_shift_observation.iloc[0]
    assert episodes.shift_distance_km.iloc[0] > 300
    assert episodes.sustained_days.min() >= 14
    assert report["shift_sensitivity_counts"]["100km_14days"] == len(episodes)
    assert json.loads((tmp_path / "audit" / "report.json").read_text())["manifest_sha256"]


def test_reserved_unknown_and_post_development_coordinates_are_excluded(tmp_path):
    dates = pd.date_range("2020-01-01", periods=70, freq="D")
    longitude = np.r_[np.zeros(20), np.full(30, 3.), np.zeros(20)]
    parts = [observations(dates, longitude, "old-name"),
             observations(dates, longitude, "reserved"), observations(dates, longitude, "unknown"),
             observations(pd.date_range("2022-01-01", periods=10, freq="D"), 80., "old-name")]
    first = pd.concat(parts, ignore_index=True)
    path = tmp_path / "observations.parquet"
    first.to_parquet(path)
    meta = manifest({"canonical": "temporal", "reserved": "test_individual"}, {"old-name": "canonical"})
    result = audit(path, tmp_path / "one", meta)
    # Even nonsensical reserved coordinates cannot influence the proxy algorithm.
    first.loc[first.animal.isin(["reserved", "unknown"]), ["lat", "lon"]] = np.nan
    first.loc[first.timestamp.dt.year == 2022, "lon"] = -150
    first.to_parquet(path)
    changed = audit(path, tmp_path / "two", meta)
    assert result["candidate_episodes"] == changed["candidate_episodes"] == 2
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "one" / "candidate-episodes.parquet"),
                                  pd.read_parquet(tmp_path / "two" / "candidate-episodes.parquet"))
    assert changed["excluded"]["reserved_individual_rows"] == 70
    assert changed["excluded"]["unassigned_identity_rows"] == 70
    assert changed["excluded"]["after_development_boundary_rows"] == 10
    years = pd.read_csv(tmp_path / "two" / "animal-year.csv")
    assert set(years.canonical_animal) == {"canonical"}
    assert years.year.max() == 2020


def test_long_observation_gaps_never_create_sustained_movement_support(tmp_path):
    dates = pd.date_range("2020-01-01", periods=20, freq="D").append(pd.DatetimeIndex(["2020-02-01", "2020-06-01"]))
    data = observations(dates, np.r_[np.zeros(20), [3., 3.]])
    path = tmp_path / "observations.parquet"
    data.to_parquet(path)
    report = audit(path, tmp_path / "audit", manifest({"animal": "temporal"}))
    assert report["candidate_episodes"] == 0
    years = pd.read_csv(tmp_path / "audit" / "animal-year.csv")
    assert years.observed_calendar_days.iloc[0] == 22
    assert years.longest_observation_gap_days.iloc[0] > 100
    assert not years.coverage_heuristic_pass.iloc[0]


def test_training_episode_does_not_reach_into_validation_year(tmp_path):
    dates = pd.date_range("2020-12-01", periods=70, freq="D")
    data = observations(dates, np.r_[np.zeros(25), np.full(45, 3.)])
    path = tmp_path / "observations.parquet"
    data.to_parquet(path)
    report = audit(path, tmp_path / "audit", manifest({"animal": "temporal"}))
    assert report["candidate_episodes"] == 0
    years = pd.read_csv(tmp_path / "audit" / "animal-year.csv")
    assert set(years.scope) == {"train", "validation_year"}
    assert years.loc[years.year == 2021, "previous_year_observed"].iloc[0]


def test_dateline_distance_and_predeclared_sensitivity_are_physical(tmp_path):
    dates = pd.date_range("2020-01-01", periods=55, freq="D")
    # A 0.4-degree dateline crossing is about44km, not a globe-spanning journey.
    data = observations(dates, np.r_[np.full(20, 179.8), np.full(35, -179.8)])
    path = tmp_path / "observations.parquet"
    data.to_parquet(path)
    report = audit(path, tmp_path / "audit", manifest({"animal": "temporal"}))
    assert report["candidate_episodes"] == 0
    years = pd.read_csv(tmp_path / "audit" / "animal-year.csv")
    assert 40 < years.first_to_last_observed_distance_km.iloc[0] < 50
    assert len(report["shift_sensitivity_counts"]) == 9
    with pytest.raises(FileExistsError):
        audit(path, tmp_path / "audit", manifest({"animal": "temporal"}))


def test_entirely_reserved_input_has_empty_reviewable_outputs(tmp_path):
    path = tmp_path / "observations.parquet"
    observations(pd.date_range("2020-01-01", periods=30, freq="D"), np.nan, "reserved").to_parquet(path)
    report = audit(path, tmp_path / "audit", manifest({"reserved": "test_individual"}))
    assert report["included_rows"] == report["candidate_episodes"] == report["animal_years"] == 0
    assert report["excluded"]["reserved_individual_rows"] == 30
    assert pd.read_csv(tmp_path / "audit" / "animal-year.csv").empty
    assert pd.read_parquet(tmp_path / "audit" / "candidate-episodes.parquet").empty
