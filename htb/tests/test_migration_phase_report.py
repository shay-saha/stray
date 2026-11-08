import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildcast.data import unproject
from wildcast.migration_data import HISTORY_NAMES, LAGS
from wildcast.migration_evaluate import load_cases, sha256

spec = importlib.util.spec_from_file_location("wildcast_migration_phase_report", Path(__file__).resolve().parents[1] / "scripts/migration_phase_report.py")
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


def arrays():
    n = 9
    split = np.asarray([0, 0, 0, 0, 0, 1, 2, 2, 3], np.int8)
    individual = np.asarray([10, 10, 10, 10, 11, 20, 11, 30, 40])
    origin_time = np.asarray([pd.Timestamp("2020-01-01", tz="UTC").timestamp() + i * 7 * 86400 for i in range(5)] +
                             [pd.Timestamp("2021-01-15", tz="UTC").timestamp()] * 4)
    history = np.zeros((n, len(LAGS), 10), np.float32)
    history_mask = np.zeros((n, len(LAGS)), bool)
    history_time = np.zeros((n, len(LAGS)))
    for lag, displacement in [(7, [1, 1, 1, 1, 100, 150, 1, 1, np.nan]),
                               (30, [2, 2, 2, 2, 200, 150, 2, 1, np.nan])]:
        j = int(np.flatnonzero(LAGS == lag)[0])
        history[:, j, 0] = np.asarray(displacement) / 1000
        history[:, j, 2] = lag
        history_mask[:, j] = True
        history_time[:, j] = origin_time - lag * 86400
    horizons = np.asarray([1., 7.], np.float32)
    target_mask = np.ones((n, 2), bool); target_mask[-1] = False
    y = np.full((n, 2, 2), .01, np.float32); y[-1] = 0
    target_latlon = unproject(y.astype(float) * 1_000_000, 0., 0.)
    target_time = origin_time[:, None] + horizons * 86400; target_time[-1] = 0
    species = np.ones(n, np.int64); species[7] = 0
    metric = np.ones(n, np.int64); metric[7] = 2
    return {"history": history, "history_mask": history_mask, "history_time": history_time,
            "static": np.zeros((n, 14), np.float32), "species": species, "metric_species": metric,
            "individual": individual, "study": np.ones(n, np.int64), "split": split,
            "origin": np.column_stack([np.zeros(n), np.zeros(n), origin_time]), "y": y,
            "target_mask": target_mask, "target_latlon": target_latlon, "target_time": target_time, "horizons": horizons}


def metadata():
    return {"history_names": HISTORY_NAMES, "history_lags_days": LAGS.tolist(), "species": {"known": 1},
            "metric_species": {"known": 1, "unknown": 2}}


def fixture(root):
    data, evaluation = root / "data", root / "evaluation"
    data.mkdir(); evaluation.mkdir()
    a = arrays()
    np.savez_compressed(data / "cases.npz", **a)
    names = ["training-A"] * 4 + ["training-B", "heldout-A", "training-B", "unknown-C", "reserved"]
    split_names = {0: "train", 1: "val_individual", 2: "val_year", 3: "test_individual"}
    origins = pd.DataFrame({"case_id": np.arange(9), "canonical_animal": names,
        "origin_time": pd.to_datetime(a["origin"][:, 2], unit="s", utc=True), "year": [2020] * 5 + [2021] * 4,
        "split": [split_names[int(s)] for s in a["split"]]})
    origins["species"] = np.where(a["metric_species"] == 1, "known", "unknown")
    origins.to_parquet(data / "origins.parquet")
    m = {**metadata(), "cases_sha256": sha256(data / "cases.npz"), "origins_sha256": sha256(data / "origins.parquet")}
    (data / "manifest.json").write_text(json.dumps(m))
    _, _, hashes = load_cases(data)
    rows = []
    for model in ["seasonal", "contender"]:
        for case in [5, 6, 7]:
            for horizon in [1., 7.]:
                rows.append({"model": model, "case": case, "split": split_names[int(a["split"][case])],
                    "species": int(a["metric_species"][case]), "individual": int(a["individual"][case]), "study": 1,
                    "horizon_days": horizon, "endpoint_energy_km": 10 + case + horizon - (2 if model == "contender" else 0),
                    "radial_coverage50": case % 2 == 0, "radial_coverage90": True,
                    "displacement_proxy": "intermediate_displacement_proxy"})
    pd.DataFrame(rows).to_parquet(evaluation / "per-case.parquet", index=False)
    protocol = {**hashes, "assessment": None, "splits": ["val_individual", "val_year"], "models": ["seasonal", "contender"],
                "cases": 3, "observed_targets": 6, "samples": 64,
                "case_ids_sha256": hashlib.sha256(np.asarray([5, 6, 7], dtype="<i8").tobytes()).hexdigest()}
    (evaluation / "metrics.json").write_text(json.dumps(protocol))
    annotations = pd.DataFrame({"canonical_animal": ["training-A", "heldout-A"], "year": [2020, 2021],
                                "split": ["train", "val_individual"], "departure_doy": [10., 10.], "arrival_doy": [20., 20.]})
    return data, evaluation, annotations


def test_thresholds_are_training_only_and_physical_animal_weighted():
    a, m = arrays(), metadata()
    fitted = phase.fit_thresholds(a, m)
    assert fitted.loc[fitted.lag_days == 7, "threshold_km"].iloc[0] == pytest.approx(100.)
    assert fitted.loc[fitted.lag_days == 30, "threshold_km"].iloc[0] == pytest.approx(200.)
    # Four low-displacement origins of one animal cannot drown one other animal.
    assert fitted.training_independent_animals.tolist() == [2, 2]
    poisoned = {key: value.copy() for key, value in a.items()}
    poisoned["y"][:] = 1e9
    poisoned["history"][a["split"] != 0] = 1e9
    pd.testing.assert_frame_equal(fitted, phase.fit_thresholds(poisoned, m))


def test_causal_regimes_ignore_future_values_and_separate_missing_and_unknown_species():
    a, m = arrays(), metadata(); fitted = phase.fit_thresholds(a, m)
    first = phase.classify(a, m, [5, 6, 7], fitted)
    assert first.causal_regime.tolist() == ["high_past_mobility", "ordinary_past_mobility", "no_training_species_threshold"]
    a["y"][:] = np.nan; a["target_latlon"][:] = 1e9
    pd.testing.assert_frame_equal(first, phase.classify(a, m, [5, 6, 7], fitted))
    a["history_mask"][5, np.flatnonzero(LAGS == 30)[0]] = False
    missing = phase.classify(a, m, [5], fitted)
    assert missing.causal_regime.iloc[0] == "missing_past_history"
    with pytest.raises(ValueError, match="Reserved"):
        phase.classify(a, m, [8], fitted)


def test_author_calendar_boundaries_are_retrospective_and_reserved_are_forbidden():
    origins = pd.DataFrame({"case": range(4), "canonical_animal": "one", "year": 2021,
        "split": "val_individual", "origin_time": pd.to_datetime(["2021-01-09", "2021-01-10", "2021-01-20", "2021-01-21"], utc=True)})
    labels = pd.DataFrame({"canonical_animal": ["one"], "year": [2021], "split": ["val_individual"],
                           "departure_doy": [10.8], "arrival_doy": [20.4]})
    result = phase.author_phases(origins, labels)
    assert result.author_phase.tolist() == ["author_predeparture", "author_in_transit", "author_in_transit", "author_postarrival"]
    labels.loc[0, "split"] = "test_individual"
    with pytest.raises(ValueError, match="Reserved"):
        phase.author_phases(origins, labels)


def test_paired_bootstrap_preserves_exact_model_differences_at_each_horizon(tmp_path):
    data, evaluation, _ = fixture(tmp_path)
    _, _, frame, _, _, _ = phase.verified_development(data, evaluation)
    result = phase.paired_intervals(frame, replicates=20)
    contender = result.loc[result.model == "contender"]
    np.testing.assert_allclose(contender.delta_vs_seasonal_km, -2.)
    np.testing.assert_allclose(contender.ci95_low_km, -2.)
    np.testing.assert_allclose(contender.ci95_high_km, -2.)
    assert set(result.horizon_days) == {"all", 1., 7.}


def test_report_keeps_knowledge_and_retrospective_views_separate(tmp_path, monkeypatch):
    data, evaluation, annotations = fixture(tmp_path)
    monkeypatch.setattr(phase, "load_development_timing", lambda *args, **kwargs: (annotations, {"test": "source-scoped author dates"}))
    result = phase.report(data, evaluation, tmp_path / "report", bootstrap_replicates=10)
    assert not result["reserved_queries_or_outcomes_used"] and not result["model_selection_performed"]
    counts = pd.read_csv(tmp_path / "report/cohort-counts.csv")
    year = counts.loc[(counts.split == "val_year") & (counts.view == "individual_knowledge")]
    assert set(year.stratum) == {"seen_training_individual", "new_individual"}
    unknown = counts.loc[(counts.split == "val_year") & (counts.view == "species_knowledge") & (counts.stratum == "unknown_training_species")]
    assert unknown.origins.iloc[0] == 1
    assert result["author_annotation_cohort"]["val_individual"]["animals"] == 1
    scores = pd.read_csv(tmp_path / "report/causal-quantile-overall.csv")
    assert (scores.loc[scores.model == "contender", "delta_vs_seasonal_km"] == -2.).all()
    assert (tmp_path / "report/future-displacement-descriptive-overall.csv").exists()


@pytest.mark.parametrize("change", ["support", "cache", "reserved"])
def test_support_hash_and_reserved_guards_reject_invalid_analysis(tmp_path, change):
    data, evaluation, _ = fixture(tmp_path)
    if change == "support":
        frame = pd.read_parquet(evaluation / "per-case.parquet")
        frame.iloc[:-1].to_parquet(evaluation / "per-case.parquet", index=False)
    elif change == "cache":
        m = json.loads((evaluation / "metrics.json").read_text()); m["manifest_sha256"] = "wrong"
        (evaluation / "metrics.json").write_text(json.dumps(m))
    else:
        frame = pd.read_parquet(evaluation / "per-case.parquet")
        frame.loc[0, "split"] = "test_individual"
        frame.to_parquet(evaluation / "per-case.parquet", index=False)
    with pytest.raises(ValueError, match="support|mismatch|Reserved"):
        phase.verified_development(data, evaluation)


def frozen_rules(data, output):
    from wildcast import migration_regimes
    a, m, hashes = load_cases(data)
    rules = migration_regimes.fit_rules(a, m, hashes)
    output.mkdir()
    path = output / "rules.json"; path.write_text(json.dumps(rules))
    (output / "filter.json").write_text(json.dumps({**hashes, "rules_sha256": sha256(path)}))
    return path, hashes


def test_frozen_router_retains_unknown_global_fallback_without_refitting(tmp_path, monkeypatch):
    data, evaluation, annotations = fixture(tmp_path)
    path, hashes = frozen_rules(data, tmp_path / "rules")
    a, m, _ = load_cases(data)
    only_past = {k: a[k] for k in ("history", "history_mask", "species", "split")}
    routed, evidence = phase.frozen_routing(only_past, [5, 6, 7], hashes, path)
    assert routed.frozen_router_regime.tolist() == ["high_past_mobility", "ordinary_past_mobility", "ordinary_past_mobility"]
    assert routed.frozen_router_global_fallback.tolist() == [False, False, True]
    assert evidence["sha256"] == sha256(path) and evidence["thresholds_refitted"] is False
    monkeypatch.setattr(phase, "load_development_timing", lambda *args: (annotations, {}))
    result = phase.report(data, evaluation, tmp_path / "report", bootstrap_replicates=8, routing_rules=path)
    assert result["frozen_routing_rules"]["sha256"] == sha256(path)
    assert "migration_regimes.py" in result["code_sha256"]
    retained = pd.read_parquet(tmp_path / "report/development-origin-strata.parquet").set_index("case")
    assert retained.loc[7, "causal_regime"] == "no_training_species_threshold"
    assert retained.loc[7, "frozen_router_regime"] == "ordinary_past_mobility"
    scores = pd.read_csv(tmp_path / "report/frozen-router-overall.csv")
    np.testing.assert_allclose(scores.loc[scores.model == "contender", "delta_vs_seasonal_km"], -2.)


@pytest.mark.parametrize("change", ["hash", "cache", "causality", "unanchored"])
def test_frozen_routing_checks_hash_cache_and_causal_fit_provenance(tmp_path, change):
    data, _, _ = fixture(tmp_path)
    path, hashes = frozen_rules(data, tmp_path / "rules")
    expected = sha256(path)
    if change == "hash":
        expected = "0" * 64
    elif change in {"cache", "causality"}:
        rules = json.loads(path.read_text())
        if change == "cache": rules["cases_sha256"] = "0" * 64
        else: rules["provenance"]["uses_future_targets"] = True
        path.write_text(json.dumps(rules)); expected = sha256(path)
    else:
        (path.parent / "filter.json").unlink(); expected = None
    a, _, _ = load_cases(data)
    with pytest.raises(ValueError, match="checksum|mismatch|provenance|filter.json"):
        phase.frozen_routing(a, [5, 6, 7], hashes, path, expected)


@pytest.mark.parametrize("change", ["calibration", "splits", "prediction-species"])
def test_phase_protocol_and_calibration_cannot_change_denominators(tmp_path, change):
    data, evaluation, _ = fixture(tmp_path)
    frame = pd.read_parquet(evaluation / "per-case.parquet")
    if change == "calibration":
        frame["radial_coverage90"] = frame.radial_coverage90.astype(float)
        frame.loc[0, "radial_coverage90"] = np.nan
    elif change == "prediction-species":
        frame["prediction_species"] = 1
    else:
        path = evaluation / "metrics.json"; protocol = json.loads(path.read_text())
        protocol["splits"] = ["val_individual"]
        path.write_text(json.dumps(protocol))
    frame.to_parquet(evaluation / "per-case.parquet", index=False)
    with pytest.raises(ValueError, match="binary|splits|prediction-species"):
        phase.verified_development(data, evaluation)
