"""Synthetic timing experiments only; no real-case local fitting."""
import json

import numpy as np
import pandas as pd
import pytest

from wildcast import migration_timing as timing


@pytest.fixture
def synthetic(tmp_path):
    records, label_rows, origins, static, history = [], [], [], [], []
    for animal in range(5):
        split = "train" if animal < 3 else "val_individual" if animal == 3 else "test_individual"
        name = f"source:study:Deer:{animal}"
        if animal < 4:
            label_rows.append({"canonical_animal": name, "year": 2020, "split": split,
                               "departure_doy": 65. + animal * 10, "arrival_doy": 120. + animal * 10})
        for issue in timing.ISSUES:
            stamp = pd.Timestamp(f"2020-{issue}", tz="UTC") + pd.Timedelta(days=animal % 3)
            records.append({"case_id": len(records), "canonical_animal": name, "individual": animal,
                            "year": 2020, "split": split, "origin_time": stamp})
            origins.append([10. + animal, 20. + animal, stamp.timestamp()])
            static.append(np.arange(len(timing.STATIC)) * .01 + animal * .1)
            history.append(np.column_stack([np.full(4, animal * .1), np.arange(4) * .01, timing.LAGS]))
    frame = pd.DataFrame(records)
    a = {"origin": np.array(origins), "static": np.array(static), "history": np.array(history),
         "history_mask": np.ones((len(frame), 4), bool),
         "split": frame.split.map({"train": 0, "val_individual": 1, "test_individual": 3}).to_numpy(),
         "y": np.zeros((len(frame), 2, 2)), "target_latlon": np.zeros((len(frame), 2, 2)),
         "target_time": np.zeros((len(frame), 2)), "target_mask": np.zeros((len(frame), 2), bool)}
    a["history_time"] = a["origin"][:, 2:3] - np.array(timing.LAGS) * 86400
    data = tmp_path / "synthetic-data"
    data.mkdir()
    frame.to_parquet(data / "origins.parquet", index=False)
    manifest = {"origins_sha256": timing.sha256(data / "origins.parquet"), "static_names": list(timing.STATIC),
                "history_names": ["east_1000km", "north_1000km", "actual_age_days"], "history_lags_days": list(timing.LAGS)}
    return a, manifest, frame, pd.DataFrame(label_rows), data


def test_issue_selection_does_not_use_event_dates(synthetic):
    _, _, origins, labels, _ = synthetic
    first, _ = timing.select_issues(origins, labels)
    changed = labels.copy()
    changed["departure_doy"] = 1.
    changed["arrival_doy"] = 366.
    second, _ = timing.select_issues(origins, changed)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 16
    assert set(first.individual) == {0, 1, 2, 3}


def test_invalid_identifier_reserved_split_and_duplicate_season_rejected(synthetic):
    _, _, origins, labels, _ = synthetic
    changed = labels.copy()
    changed.loc[0, "canonical_animal"] = "unjoined-animal"
    with pytest.raises(ValueError, match="exact cache origin join"):
        timing.select_issues(origins, changed)
    changed = labels.copy()
    changed.loc[0, "split"] = "test_individual"
    with pytest.raises(ValueError, match="development splits only"):
        timing.select_issues(origins, changed)
    with pytest.raises(ValueError, match="Duplicate"):
        timing.select_issues(origins, pd.concat([labels, labels.iloc[:1]], ignore_index=True))


def test_future_targets_and_unavailable_history_cannot_change_features(synthetic):
    a, manifest, origins, labels, _ = synthetic
    issues, _ = timing.select_issues(origins, labels)
    ids = issues.case_id.to_numpy()
    a["history_mask"][:, -1] = False
    before, _ = timing.feature_matrix(a, manifest, ids)
    changed = {k: v.copy() for k, v in a.items()}
    for key in ["y", "target_latlon", "target_time"]:
        changed[key][:] = np.nan
    changed["target_mask"][:] = True
    changed["history"][:, -1] = 1e20
    changed["history_time"][:, -1] = 1e20
    after, _ = timing.feature_matrix(changed, manifest, ids)
    np.testing.assert_array_equal(before, after)
    changed["history_time"][0, 0] = changed["origin"][0, 2] + 86400
    with pytest.raises(ValueError, match="Future history"):
        timing.feature_matrix(changed, manifest, ids)
    with pytest.raises(ValueError, match="Reserved origins"):
        timing.feature_matrix(a, manifest, np.array([len(origins) - 1]))


def test_gaussian_crps_estimator_and_issue_probabilities():
    draws = np.random.default_rng(7).standard_normal(50000)
    expected = (np.sqrt(2) - 1) / np.sqrt(np.pi)
    assert abs(timing.sampled_crps(draws, 0.) - expected) < .01
    metrics = timing.score_distribution(np.array([[100., 120.]]), np.ones((1, 2)) * 10,
                                        np.array([[100., 140.]]), np.array([[100.]]),
                                        np.random.default_rng(1).standard_normal((1, 2, 256)))
    assert metrics["occurred_by_issue"][0, 0]
    assert metrics["probability_occurred_by_issue"][0, 0] == .5
    assert metrics["occurred_brier"][0, 0] == .25
    assert metrics["occurred_log_score"][0, 0] == pytest.approx(np.log(2))


def test_training_normalization_and_spread_never_fit_validation(synthetic, monkeypatch):
    a, manifest, origins, labels, _ = synthetic
    issues, _ = timing.select_issues(origins, labels)
    x, _ = timing.feature_matrix(a, manifest, issues.case_id.to_numpy())
    merged = issues.merge(labels, on=["canonical_animal", "year", "split"], validate="many_to_one")
    y = merged[["departure_doy", "arrival_doy"]].to_numpy()
    train = issues.split.to_numpy() == "train"
    calls = []
    original_fit = timing.fit_ridge
    def observed_fit(xx, yy, ww, alpha):
        calls.append(len(xx))
        return original_fit(xx, yy, ww, alpha)
    monkeypatch.setattr(timing, "fit_ridge", observed_fit)
    first = timing.fit_models(x[train], y[train], issues.loc[train])
    assert calls == [12, 8, 8, 8]
    x[~train], y[~train] = 1e12, -1e12
    second = timing.fit_models(x[train], y[train], issues.loc[train])
    for key in ("centre", "scale", "beta", "intercept", "std"):
        np.testing.assert_array_equal(first["ridge"][key], second["ridge"][key])
    with pytest.raises(ValueError, match="Only training"):
        timing.fit_models(x, y, issues)


def test_synthetic_end_to_end_outputs_and_validation_changes_preserve_fitted_model(synthetic, tmp_path, monkeypatch):
    a, manifest, origins, labels, data = synthetic
    evidence = {"provenance": {"path": "synthetic-only", "sha256": "0" * 64}}
    monkeypatch.setattr(timing, "load_cases", lambda data: (a, manifest, {"manifest_sha256": "1" * 64, "cases_sha256": "2" * 64}))
    monkeypatch.setattr(timing, "load_development_timing", lambda data: (labels, evidence))
    out = tmp_path / "fit-one"
    report = timing.run(data, out, samples=64, bootstrap=20)
    assert report["eligible_animals"] == {"train": 3, "val_individual": 1}
    assert not set(pd.read_parquet(out / "fixed-issues.parquet").split) & {"test_individual", "test_year"}
    assert set(pd.read_csv(out / "metrics.csv").stratum) >= {"all_fixed_issues", "before_event_diagnostic", "already_occurred_diagnostic"}
    labels.loc[labels.split == "val_individual", "departure_doy"] += 50
    a["static"][a["split"] == 1] = 1e8
    other = tmp_path / "fit-two"
    timing.run(data, other, samples=64, bootstrap=20)
    with np.load(out / "model.npz", allow_pickle=False) as left, np.load(other / "model.npz", allow_pickle=False) as right:
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])
    with pytest.raises(FileExistsError, match="immutable"):
        timing.run(data, out, samples=64, bootstrap=20)
