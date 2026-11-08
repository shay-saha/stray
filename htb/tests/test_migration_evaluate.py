import hashlib
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from wildcast.migration_baselines import MigrationBaselines
from wildcast.migration_evaluate import (evaluate, great_circle, load_cases,
                                        paired_animal_bootstrap, score_paths)


def corpus():
    n, length, horizons = 5, 3, np.asarray([1., 3., 7.], np.float32)
    a = {
        "history": np.zeros((n, length, 10), np.float32),
        "history_mask": np.ones((n, length), bool),
        "static": np.zeros((n, 14), np.float32),
        "species": np.ones(n, np.int64),
        "y": np.broadcast_to(np.asarray([[[.01, 0], [.03, 0], [.07, 0]]], np.float32), (n, 3, 2)).copy(),
        "target_mask": np.ones((n, 3), bool),
        "split": np.asarray([0, 0, 0, 1, 2], np.int8),
        "individual": np.asarray([10, 10, 11, 10, 10], np.int64),
        "study": np.zeros(n, np.int64),
        "origin": np.column_stack([np.zeros(n), np.zeros(n), np.asarray([1000, 1998, 2100, 2000, 2000]) * 86400.]),
        "horizons": horizons,
    }
    a["history"][:, :, 2] = [0, 1, 7]
    a["history"][:, 1:, 0] = [-.01, -.07]
    a["target_latlon"] = np.zeros((n, 3, 2), float)
    a["target_latlon"][:, :, 1] = [.09, .27, .63]
    a["target_time"] = a["origin"][:, 2:3] + horizons * 86400
    return a


def write_corpus(path, a):
    path.mkdir(exist_ok=True)
    np.savez_compressed(path / "cases.npz", **a)
    digest = hashlib.sha256((path / "cases.npz").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps({"cases_sha256": digest}))


def test_retrieval_excludes_same_identity_and_familiar_future_endpoints():
    a = corpus()
    baseline = MigrationBaselines(a)
    unseen, _ = baseline.eligible_donors(3)
    assert unseen.tolist() == [2]
    familiar, _ = baseline.eligible_donors(4)
    assert familiar.tolist() == [0]
    assert np.all(a["target_time"][familiar].max(-1) < a["origin"][4, 2])


def test_missing_route_endpoints_are_predictions_not_zero_or_nan_labels():
    a = corpus()
    a["target_mask"][0, 1] = False
    a["y"][0, 1] = np.nan
    a["target_time"][0, 1] = np.nan
    baseline = MigrationBaselines(a)
    draws, metadata = baseline.seasonal(np.asarray([4]), 20, np.random.default_rng(1))
    assert np.isfinite(draws).all()
    assert np.all(metadata["donor_ids"] == 0)
    assert metadata["completed_mask"][0, :, 1].all()
    np.testing.assert_allclose(draws[0, :, 0, 0], .01, atol=1e-7)
    np.testing.assert_allclose(draws[0, :, 2, 0], .07, atol=1e-7)
    assert np.any(abs(draws[0, :, 1]) > 0)


def test_validation_target_values_never_fit_movement_baseline():
    a = corpus()
    first = MigrationBaselines(a)
    a["y"][a["split"] != 0] = 1e8
    second = MigrationBaselines(a)
    assert first.diffusion == second.diffusion


def test_masked_dateline_geodesic_scores_ignore_missing_targets():
    origin = np.asarray([[0., 179.9, 0.]])
    target = np.asarray([[[0., -179.9], [np.nan, np.nan]]])
    mask = np.asarray([[True, False]])
    samples = np.zeros((1, 4, 2, 2))
    scored = score_paths(samples, target, mask, origin)
    expected = 2 * np.pi * 6371.0088 / 1800
    assert great_circle([0., 179.9], [0., -179.9]) == pytest.approx(expected)
    assert scored["endpoint_energy_km"][0, 0] == pytest.approx(expected)
    assert np.isnan(scored["endpoint_energy_km"][0, 1])
    assert np.isfinite(scored["joint_chordal_energy_km"]).all()
    assert scored["representative_index"].tolist() == [0]


def test_joint_score_distinguishes_dependence_with_identical_marginals():
    paths = np.zeros((1, 4, 2, 2))
    paths[0, :, :, 0] = [[.1, .1], [.1, .1], [-.1, -.1], [-.1, -.1]]
    other = paths.copy()
    other[0, :, 1, 0] *= -1
    origin = np.asarray([[0., 0., 0.]])
    truth = np.asarray([[[0., 1.], [0., 1.]]])
    mask = np.ones((1, 2), bool)
    a, b = [score_paths(p, truth, mask, origin) for p in [paths, other]]
    np.testing.assert_allclose(a["endpoint_energy_km"], b["endpoint_energy_km"])
    assert abs(a["joint_chordal_energy_km"][0] - b["joint_chordal_energy_km"][0]) > 1


def test_data_and_checkpoint_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    import torch
    write_corpus(tmp_path, corpus())
    loaded, _, _ = load_cases(tmp_path)
    assert len(loaded["split"]) == 5
    module = types.ModuleType("wildcast.migration_model")
    module.load_model = lambda *args, **kwargs: (None, {"manifest_sha256": "wrong", "cases_sha256": "wrong"})
    monkeypatch.setitem(sys.modules, "wildcast.migration_model", module)
    (tmp_path / "model").mkdir()
    torch.save({"model_kind": "wildcast_migration_joint"}, tmp_path / "model" / "best.pt")
    with pytest.raises(ValueError, match="checkpoint/data"):
        evaluate(tmp_path, tmp_path / "out", checkpoints=[tmp_path / "model" / "best.pt"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["cases_sha256"] = "wrong"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        load_cases(tmp_path)


def test_comparison_rejects_mismatched_support_and_reports_paired_difference():
    rows = []
    for case in range(3):
        for model in ["a", "b"]:
            rows.append({"model": model, "split": "val_individual", "species": 1,
                         "individual": case, "horizon_days": 7., "case": case,
                         "endpoint_energy_km": case + (model == "b") * 2})
    frame = pd.DataFrame(rows)
    report = paired_animal_bootstrap(frame, "a", replicates=20)
    assert report["splits"]["val_individual"]["b"]["ci95_km"] == pytest.approx([2., 2.])
    with pytest.raises(ValueError, match="identical"):
        paired_animal_bootstrap(frame.iloc[:-1], "a", replicates=20)


def test_baseline_evaluation_outputs_observed_cases_and_keeps_test_guard(tmp_path):
    a = corpus()
    a["target_mask"][3, 2] = False
    a["target_latlon"][3, 2] = np.nan
    a["y"][3, 2] = np.nan
    a["target_time"][3, 2] = np.nan
    write_corpus(tmp_path, a)
    report = evaluate(tmp_path, tmp_path / "evaluation", samples=4, bootstrap_replicates=5)
    assert report["cases"] == 2
    assert report["observed_targets"] == 5
    frame = pd.read_parquet(tmp_path / "evaluation" / "per-case.parquet")
    assert len(frame) == 15
    assert report["retrieval"]["sampled_points"] == 24
    assert np.isfinite(frame.endpoint_energy_km).all()
    with pytest.raises(ValueError, match="frozen"):
        evaluate(tmp_path, tmp_path / "test", splits=["test_individual"])


def test_retrieval_dispatch_supports_coupled_initial_and_best_controls(tmp_path, monkeypatch):
    import torch
    write_corpus(tmp_path, corpus())
    _, _, hashes = load_cases(tmp_path)

    class Retrieval:
        def predict_cases(self, a, batch, samples, rng, baselines):
            return baselines.predict("seasonal", batch, samples, rng)[0]

    module = types.ModuleType("wildcast.migration_retrieval_train")
    module.load_model = lambda *args, **kwargs: (Retrieval(), hashes)
    monkeypatch.setitem(sys.modules, "wildcast.migration_retrieval_train", module)
    run = tmp_path / "retrieval"
    run.mkdir()
    for name in ["initial", "best"]:
        torch.save({"model_kind": "wildcast_migration_retrieval"}, run / (name + ".pt"))
    report = evaluate(tmp_path, tmp_path / "comparison", checkpoints=[run / "initial.pt", run / "best.pt"],
                      samples=4, bootstrap_replicates=5)
    assert {"retrieval-initial", "retrieval"}.issubset(report["models"])
    frame = pd.read_parquet(tmp_path / "comparison" / "per-case.parquet")
    initial = frame.loc[frame.model == "retrieval-initial", "endpoint_energy_km"].to_numpy()
    selected = frame.loc[frame.model == "retrieval", "endpoint_energy_km"].to_numpy()
    np.testing.assert_array_equal(initial, selected)


def test_unseen_species_reporting_stays_distinct_from_unknown_prediction_token(tmp_path):
    a = corpus()
    a["species"][3:] = 0
    a["individual"][3] = 99
    write_corpus(tmp_path, a)
    names = ["known", "known", "known", "unseen bird", "unseen mammal"]
    origins = pd.DataFrame({"case_id": np.arange(5), "species": names})
    path = tmp_path / "origins.parquet"
    origins.to_parquet(path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["origins_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded, metadata, _ = load_cases(tmp_path)
    assert loaded["species"][3:].tolist() == [0, 0]
    assert loaded["metric_species"][3] != loaded["metric_species"][4]
    assert len(metadata["metric_species_names"]) == 3
    # The trainer consumes these groups solely for reporting/selection.
    from wildcast.migration_train import load_training_data
    training, _, _ = load_training_data(tmp_path)
    np.testing.assert_array_equal(training["metric_species"], loaded["metric_species"])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="origins hash"):
        load_cases(tmp_path)
