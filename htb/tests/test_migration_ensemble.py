import hashlib
import json

import numpy as np
import pytest
import torch

from wildcast.migration_ensemble import (MigrationEnsemble, fit, load_model, macro_weights,
                                          optimize_weights, physical_folds)
from wildcast.migration_evaluate import forecast_latlon
from wildcast.migration_model import MigrationModel


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class ConstantRoute:
    def __init__(self, sign):
        self.sign = sign

    def predict_cases(self, a, ids, samples, rng, baselines=None):
        shape = (len(ids), samples, len(a["horizons"]), 2)
        result = np.zeros(shape, np.float32)
        result[..., 0] = self.sign * np.arange(1, shape[2] + 1)
        return result


def test_whole_path_member_choice_preserves_mixture_law():
    model = MigrationEnsemble([ConstantRoute(-1), ConstantRoute(1)], [.3, .7])
    a = {"horizons": np.array([1., 3., 7.])}
    paths = model.predict_cases(a, np.array([0, 1]), 20000, seed=123)
    assert np.all(np.sign(paths[:, :, 0, 0]) == np.sign(paths[:, :, -1, 0]))
    assert not np.any(paths[..., 0] == 0)  # Averaging member locations would fail.
    assert np.mean(paths[:, :, 0, 0] > 0) == pytest.approx(.7, abs=.01)
    with pytest.raises(ValueError, match="sum to one"):
        MigrationEnsemble([ConstantRoute(1)], [.5])


def test_simplex_optimizer_recovers_known_energy_optimum():
    first = np.array([.5, 1.5])
    pair = np.array([[0., 2.], [2., 0.]])
    weights, score = optimize_weights(first, pair, steps=100)
    np.testing.assert_allclose(weights, [.75, .25], atol=1e-6)
    assert (weights >= 0).all() and weights.sum() == pytest.approx(1.)
    assert score < 1. / 2  # Equal mixture has score 0.5.


def test_alias_exclusion_group_cannot_cross_calibration_folds():
    a = {"individual": np.array([1, 2, 1, 3, 2, 1])}
    manifest = {"animals": {"aliasA": 1, "aliasB": 1, "other": 2, "third": 3},
                "exclusion_groups": {"aliasA": "physical-A", "aliasB": "physical-A"}}
    folds, labels = physical_folds(a, manifest, 3, seed=1701)
    assert len(np.unique(folds[a["individual"] == 1])) == 1
    assert labels[1] == "physical-A"
    permutation = np.array([5, 2, 0, 4, 1, 3])
    shuffled, _ = physical_folds({"individual": a["individual"][permutation]}, manifest, 3, 1701)
    np.testing.assert_array_equal(shuffled, folds[permutation])


def corpus(root):
    root.mkdir()
    n, horizons = 14, np.array([1., 3., 7.], np.float32)
    a = {"history": np.zeros((n, 3, 10), np.float32), "history_mask": np.ones((n, 3), bool),
         "static": np.zeros((n, 14), np.float32), "species": np.ones(n, np.int64),
         "metric_species": np.arange(n, dtype=np.int64) % 2,
         "y": np.zeros((n, 3, 2), np.float32), "target_mask": np.ones((n, 3), bool),
         "split": np.array([0] * 4 + [1] * 4 + [2] * 4 + [3, 4], np.int8),
         "individual": np.array([0, 1, 2, 3, 10, 11, 12, 13, 0, 1, 2, 3, 20, 21], np.int64),
         "study": np.zeros(n, np.int64), "horizons": horizons}
    a["history"][:, :, 2] = [0, 1, 7]; a["history"][:, :, 0] = [0., -.01, -.07]
    a["origin"] = np.column_stack((np.zeros(n), np.full(n, 179.8), 1_500_000_000 + np.arange(n) * 86400. * 20))
    a["target_time"] = a["origin"][:, 2:3] + horizons * 86400
    a["y"][:, :, 0] = horizons * .01
    a["target_latlon"] = forecast_latlon(a["y"][:, None], a["origin"])[:, 0]
    a["target_mask"][-2:] = False
    a["y"][-2:] = np.nan; a["target_latlon"][-2:] = np.nan; a["target_time"][-2:] = 0.
    np.savez_compressed(root / "cases.npz", **a)
    manifest = {"cases_sha256": hashlib.sha256((root / "cases.npz").read_bytes()).hexdigest(),
                "species": {"Animalus test": 1}, "reserved_test_targets_sealed": True,
                "animals": {f"animal-{i}": i for i in np.unique(a["individual"]).tolist()}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    hashes = {"cases_sha256": manifest["cases_sha256"],
              "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
    return a, manifest, hashes


def checkpoint(path, sign, hashes):
    model = MigrationModel(10, 14, 2, [1, 3, 7], history_length=3, width=16, layers=1,
                           components=2, variant="seasonal_static", dropout=0.)
    with torch.no_grad():
        model.mean_head.weight.zero_()
        model.mean_head.bias.copy_(torch.tensor([.01, 0., .03, 0., .07, 0.]).repeat(2) * sign)
        model.scale_head.weight.zero_(); model.scale_head.bias.fill_(-4.6)
    torch.save({"model_kind": "wildcast_migration_joint", "config": model.config,
                "model": model.state_dict(), "epoch": 0, **hashes}, path)


def test_macro_weights_keep_species_horizons_animals_and_splits_balanced(tmp_path):
    a, _, _ = corpus(tmp_path / "data")
    ids = np.flatnonzero(np.isin(a["split"], [1, 2]))
    endpoint, joint = macro_weights(a, ids)
    assert endpoint.sum() == pytest.approx(1.) and joint.sum() == pytest.approx(1.)
    for code in (1, 2):
        assert endpoint[a["split"][ids] == code].sum() == pytest.approx(.5)
    for h in range(3):
        assert endpoint[:, h].sum() == pytest.approx(1 / 3)


def test_fit_uses_validation_only_crossfits_groups_and_rejects_member_change(tmp_path):
    a, manifest, hashes = corpus(tmp_path / "data")
    paths = [tmp_path / "positive.pt", tmp_path / "negative.pt"]
    for path, sign in zip(paths, [1, -1]): checkpoint(path, sign, hashes)
    report = fit(tmp_path / "data", paths, tmp_path / "fit", samples=4, folds=3, steps=30,
                  batch_size=4, device="cpu", bootstrap_replicates=8)
    assert report["reserved_test_cases_scored"] == 0
    for fold in report["fold_reports"]:
        if "held_groups" in fold:
            assert not set(fold["fit_groups"]) & set(fold["held_groups"])
    with np.load(tmp_path / "fit/validation-costs.npz") as costs:
        assert set(a["split"][costs["case_ids"]]) == {1, 2}
        assert np.isfinite(costs["joint_first"]).all()
    model, saved = load_model(tmp_path / "fit/best.pt")
    assert saved["weights"].sum() == pytest.approx(1.)
    draws = model.predict_cases(a, np.array([4, 8]), 12, seed=3)
    assert draws.shape == (2, 12, 3, 2) and np.isfinite(draws).all()
    assert "paired_uncertainty" in report
    with paths[0].open("ab") as stream: stream.write(b"tampered")
    with pytest.raises(ValueError, match="member checksum"):
        load_model(tmp_path / "fit/best.pt")


def test_load_rejects_changed_cache(tmp_path):
    _, _, hashes = corpus(tmp_path / "data")
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for path, sign in zip(paths, [1, -1]): checkpoint(path, sign, hashes)
    fit(tmp_path / "data", paths, tmp_path / "fit", samples=2, folds=3, steps=5, device="cpu")
    with (tmp_path / "data/cases.npz").open("ab") as stream: stream.write(b"tampered")
    with pytest.raises(ValueError, match="cache"):
        load_model(tmp_path / "fit/best.pt")


def test_equal_assembly_never_reads_targets_or_fits_weights_and_cli_loads(tmp_path, monkeypatch):
    import wildcast.migration_ensemble as ensemble
    import sys
    a, _, hashes = corpus(tmp_path / "data")
    paths = [tmp_path / "positive.pt", tmp_path / "negative.pt"]
    for path, sign in zip(paths, [1, -1]): checkpoint(path, sign, hashes)
    def forbidden(*args, **kwargs):
        raise AssertionError("Uniform assembly must not inspect target arrays or fit validation weights")
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(ensemble, "optimize_weights", forbidden)
    monkeypatch.setattr(ensemble, "precompute_costs", forbidden)
    report = ensemble.assemble_equal(tmp_path / "data", paths, tmp_path / "equal", device="cpu",
                                    member_selection_scope="Fixed synthetic members; no score-based selection")
    assert report["weight_fitting_performed"] is False and report["target_arrays_read"] is False
    assert report["validation_cases_scored"] == report["reserved_test_cases_scored"] == 0
    np.testing.assert_array_equal(report["weights"], [.5, .5])
    model, saved = load_model(tmp_path / "equal/best.pt")
    assert saved["selection_version"] == ensemble.EQUAL_SELECTION_VERSION
    assert model.predict_cases(a, np.array([4, 8]), 12, seed=17).shape == (2, 12, 3, 2)
    monkeypatch.setattr(sys, "argv", ["migration_ensemble", "--data", str(tmp_path / "data"),
                        "--checkpoints", *map(str, paths), "--output", str(tmp_path / "cli-equal"),
                        "--device", "cpu", "--equal-weights"])
    ensemble.main()
    _, cli = load_model(tmp_path / "cli-equal/best.pt")
    torch.testing.assert_close(cli["weights"], saved["weights"], rtol=0, atol=0)
    assert not (tmp_path / "equal/validation-costs.npz").exists()


def test_equal_assembly_preserves_new_only_cache_and_unique_member_guards(tmp_path):
    from wildcast.migration_ensemble import assemble_equal
    _, _, hashes = corpus(tmp_path / "data")
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for path, sign in zip(paths, [1, -1]): checkpoint(path, sign, hashes)
    with pytest.raises(ValueError, match="Duplicate"):
        assemble_equal(tmp_path / "data", [paths[0], paths[0]], tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()
    assemble_equal(tmp_path / "data", paths, tmp_path / "equal")
    with pytest.raises(FileExistsError): assemble_equal(tmp_path / "data", paths, tmp_path / "equal")
    with (tmp_path / "data/cases.npz").open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="cache checksum"):
        assemble_equal(tmp_path / "data", paths, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
