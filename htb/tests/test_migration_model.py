import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from wildcast.data import unproject
from wildcast.migration_evaluate import macro_scores, score_paths
from wildcast.migration_model import (MigrationModel, load_model, masked_joint_nll,
                                      mixture_energy, sample_paths)
from wildcast.migration_train import (_physical_macro, load_training_data,
                                     physical_validation_scores, train)


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs(batch=4):
    torch.manual_seed(1)
    history = torch.randn(batch, 5, 4)
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool).expand(batch, -1).clone()
    return history, mask, torch.randn(batch, 3), torch.arange(batch) % 3


def model(variant="goal_route"):
    return MigrationModel(4, 3, 3, [1, 7, 30], history_length=5, width=16,
                          layers=1, components=3, variant=variant, dropout=0.)


def test_masked_likelihood_is_exact_observed_marginal():
    means = torch.tensor([[[[.2, -.1], [.4, .8]], [[.7, -.2], [-.1, .3]]]], dtype=torch.float64)
    base = torch.tensor([[1., .2, .4, .1], [.2, 2., .3, .2], [.4, .3, 1.5, .5], [.1, .2, .5, .8]], dtype=torch.float64)
    parameters = {"logits": torch.tensor([[.3, -.2]], dtype=torch.float64), "means": means,
                  "covariance": torch.stack((base, base * 1.3))[None]}
    target = torch.tensor([[[float("nan"), float("nan")], [.3, .5]]], dtype=torch.float64)
    mask = torch.tensor([[False, True]])
    expected = torch.distributions.MultivariateNormal(means[0, :, 1], parameters["covariance"][0, :, 2:, 2:])
    expected = -torch.logsumexp(parameters["logits"].log_softmax(-1)[0] + expected.log_prob(target[0, 1]), -1)
    torch.testing.assert_close(masked_joint_nll(parameters, target, mask)[0], expected)
    empty = masked_joint_nll(parameters, target * float("nan"), torch.zeros_like(mask))
    torch.testing.assert_close(empty, torch.zeros_like(empty), atol=1e-12, rtol=0)


def test_samples_follow_joint_covariance_and_shared_component():
    # Distinct component destinations must persist across every horizon.
    means = torch.tensor([[[[-10., -10.], [-10., -10.]], [[10., 10.], [10., 10.]]]])
    covariance = torch.eye(4)[None, None].expand(1, 2, -1, -1).clone() * .01
    covariance[..., 0, 2] = covariance[..., 2, 0] = .008
    parameters = {"logits": torch.zeros(1, 2), "means": means, "covariance": covariance}
    samples = sample_paths(parameters, 30000, torch.Generator().manual_seed(3))[0]
    assert torch.all(samples[:, 0, 0].sign() == samples[:, 1, 0].sign())
    group = samples[samples[:, 0, 0] < 0].flatten(-2)
    empirical = torch.cov(group.T)
    torch.testing.assert_close(empirical, covariance[0, 0], atol=.0005, rtol=.08)


@pytest.mark.parametrize("variant", ["seasonal_static", "history_direct", "goal_route"])
def test_masked_history_and_sample_shapes(variant):
    m = model(variant).eval()
    h, mask, static, species = inputs()
    first = m(h, mask, static, species)
    h[~mask] = float("nan")
    second = m(h, mask, static, species)
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    result = m.sample_paths(h, mask, static, species, 11, torch.Generator().manual_seed(5))
    assert result.shape == (4, 11, 3, 2)
    assert torch.isfinite(result).all()
    # Even a completely missing history has a valid prior-conditioned forecast.
    empty = m(h, torch.zeros_like(mask), static, species)
    assert all(torch.isfinite(v).all() for v in empty.values())


def test_both_proper_losses_train_probabilities_and_history():
    m = model()
    h, mask, static, species = inputs()
    target = torch.randn(4, 3, 2) * .1
    target_mask = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.bool)
    params = m(h, mask, static, species)
    energy = mixture_energy(params, target, target_mask, repeats=3, generator=torch.Generator().manual_seed(6))
    loss = energy["endpoint_energy"].mean() + energy["path_energy"].mean()
    loss.backward()
    for prefix in ("history_input", "history_encoder", "logit_head", "mean_head", "scale_head", "dependence_head"):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for name, p in m.named_parameters() if name.startswith(prefix)), prefix
    m.zero_grad()
    masked_joint_nll(m(h, mask, static, species), target, target_mask).mean().backward()
    assert m.dependence_head.weight.grad.abs().sum() > 0


def test_missing_waypoint_values_do_not_change_energy():
    m = model().eval()
    parameters = m(*inputs())
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool).expand(4, -1)
    target = torch.zeros(4, 3, 2)
    a = mixture_energy(parameters, target, mask, generator=torch.Generator().manual_seed(10))
    target[~mask] = float("nan")
    b = mixture_energy(parameters, target, mask, generator=torch.Generator().manual_seed(10))
    for key in a:
        torch.testing.assert_close(a[key], b[key])


def fixture_cache(root):
    rng = np.random.default_rng(11)
    n = 24
    split = np.repeat(np.arange(5, dtype=np.int8), [12, 4, 4, 2, 2])
    individual = np.r_[np.arange(12) % 3, [10, 10, 11, 11], [0, 0, 1, 1], [20, 20, 0, 0]]
    history = rng.normal(size=(n, 5, 4)).astype("float32")
    target_mask = np.ones((n, 3), dtype=bool)
    target_mask[split >= 3] = False
    y = rng.normal(size=(n, 3, 2)).astype("float32") * .1
    # Poisoned reserved truth demonstrates it cannot affect normalization,
    # fitting, checkpoint selection or validation metrics.
    y[split >= 3] = np.nan
    origin = np.zeros((n, 3), dtype="float64")
    target_latlon = unproject(y.astype("float64") * 1_000_000, 0., 0.)
    np.savez(root / "cases.npz", history=history, history_mask=np.ones((n, 5), bool),
             static=rng.normal(size=(n, 3)).astype("float32"), species=np.arange(n, dtype="int64") % 3,
             y=y, target_mask=target_mask, split=split, individual=individual, study=np.zeros(n, dtype="int64"),
             horizons=np.array([1, 7, 30], "float32"), origin=origin, target_latlon=target_latlon,
             target_time=np.broadcast_to(np.array([1, 7, 30]) * 86400., (n, 3)))
    (root / "manifest.json").write_text(json.dumps({"protocol": "test fixture",
        "reserved_test_targets_sealed": True,
        "cases_sha256": hashlib.sha256((root / "cases.npz").read_bytes()).hexdigest()}))


def test_trainer_excludes_reserved_targets_and_roundtrips(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    fixture_cache(data)
    a, _, hashes = load_training_data(data)
    assert len(a["y"]) == 20
    assert np.isfinite(a["y"]).all()
    assert set(a["split"]) == {0, 1, 2}
    summary = train(data, tmp_path / "run", epochs=1, batch_size=8, width=16, layers=1,
                    components=2, device="cpu", validation_repeats=2, dropout=0.)
    assert summary["epochs_completed"] == 1
    loaded, checkpoint = load_model(tmp_path / "run" / "best.pt")
    assert checkpoint["cases_sha256"] == hashes["cases_sha256"]
    assert checkpoint["manifest_sha256"] == hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    h = json.loads((tmp_path / "run" / "history.json").read_text())
    assert h[0]["epoch"] == 0
    assert set(h[1]["splits"]) == {"val_individual", "val_year"}
    assert math.isfinite(summary["best_selection"])
    assert summary["selection_version"] == "migration-physical-v2"
    assert summary["selection_units"] == "km"
    assert set(summary["code_sha256"]) == {"migration_model.py", "migration_train.py", "migration_data.py", "migration_evaluate.py"}
    assert "endpoint_energy_km" in h[1]["splits"]["val_individual"]
    assert loaded.sample_paths(*inputs(), 3).shape == (4, 3, 3, 2)


def test_training_rejects_animal_leakage(tmp_path):
    fixture_cache(tmp_path)
    with np.load(tmp_path / "cases.npz") as data:
        a = {key: data[key] for key in data.files}
    a["individual"][a["split"] == 1] = 0
    np.savez(tmp_path / "cases.npz", **a)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["cases_sha256"] = hashlib.sha256((tmp_path / "cases.npz").read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlaps training identities"):
        load_training_data(tmp_path)


def test_trainer_rejects_cache_tampering_and_unsealed_targets(tmp_path):
    fixture_cache(tmp_path)
    with (tmp_path / "cases.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_training_data(tmp_path)
    fixture_cache(tmp_path)
    with np.load(tmp_path / "cases.npz") as data:
        a = {key: data[key] for key in data.files}
    a["target_mask"][a["split"] >= 3] = True
    np.savez(tmp_path / "cases.npz", **a)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["cases_sha256"] = hashlib.sha256((tmp_path / "cases.npz").read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not sealed"):
        load_training_data(tmp_path)


def test_unknown_species_embedding_learns_shared_fallback():
    m = model().train()
    m.config["species_dropout"] = 1.
    h, mask, static, species = inputs()
    species[:] = 2
    result = m(h, mask, static, species)
    result["means"].sum().backward()
    assert m.species.weight.grad[0].abs().sum() > 0
    assert m.species.weight.grad[1:].abs().sum() == 0


def test_trainer_rejects_calendar_boundary_leakage(tmp_path):
    fixture_cache(tmp_path)
    with np.load(tmp_path / "cases.npz") as data:
        a = {key: data[key] for key in data.files}
    cutoff = datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()
    origin = np.where(a["split"] == 2, cutoff + 60 * 86400, cutoff - 120 * 86400)
    a["origin"] = np.column_stack((np.zeros(len(origin)), np.zeros(len(origin)), origin))
    a["target_time"] = origin[:, None] + a["horizons"][None] * 86400
    a["target_time"][0, -1] = cutoff  # A training endpoint in the held-out year.
    np.savez(tmp_path / "cases.npz", **a)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest.update(train_end_year=2020, validation_year=2021,
                    cases_sha256=hashlib.sha256((tmp_path / "cases.npz").read_bytes()).hexdigest())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="global year boundary"):
        load_training_data(tmp_path)


def test_physical_validation_matches_evaluator_masked_geodesic_and_joint_scores():
    rng = np.random.default_rng(55)
    draws = rng.normal(size=(3, 16, 3, 2)) * .2
    origin = np.asarray([[0., 179.9, 0.], [85., -179.9, 1.], [-40., 30., 2.]])
    truth = unproject(rng.normal(size=(3, 3, 2)) * 200_000, origin[:, None, 0], origin[:, None, 1])
    mask = np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=bool)
    truth[~mask] = np.nan
    expected = score_paths(draws, truth, mask, origin)
    endpoint, joint = physical_validation_scores(*[torch.as_tensor(x) for x in [draws, truth, mask, origin]])
    np.testing.assert_allclose(endpoint.numpy(), expected["endpoint_energy_km"], atol=1e-7, equal_nan=True)
    np.testing.assert_allclose(joint.numpy(), expected["joint_chordal_energy_km"], atol=1e-7)


def test_validation_macro_matches_evaluator_with_unequal_animals_and_horizons():
    import pandas as pd
    species = np.asarray([1, 1, 1, 2, 2])
    animals = np.asarray([10, 10, 11, 20, 21])
    mask = np.asarray([[1, 1], [1, 0], [1, 1], [1, 1], [0, 1]], bool)
    energy = np.arange(10, dtype=float).reshape(5, 2)
    joint = np.asarray([1., 3., 6., 8., 12.])
    endpoint_macro, joint_macro = _physical_macro(energy, joint, mask, species, animals)
    rows = [{"model": "one", "split": "validation", "species": species[i], "individual": animals[i],
             "horizon_days": h + 1, "endpoint_energy_km": energy[i, h]}
            for i, h in zip(*np.nonzero(mask))]
    expected, _, _ = macro_scores(pd.DataFrame(rows))
    assert endpoint_macro == pytest.approx(expected.endpoint_energy_km.iloc[0])
    # Equal animals within species: species1=(mean(1,3)+6)/2=4;
    # species2=(8+12)/2=10; then equal species gives7.
    assert joint_macro == pytest.approx(7.)
