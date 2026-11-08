import hashlib
import json

import numpy as np
import pytest
import torch

pytest.importorskip("entmax")

from wildcast.migration_evaluate import forecast_latlon
from wildcast.migration_moveformer_endpoint import (EndpointMoveFormer, causal_scale, fit_proposal,
    load_model, make_training_batch, selection_origins, train)
from wildcast.moveformer import proposal_records


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def corpus():
    n, h = 10, np.array([1., 3., 7.], np.float32)
    a = {"history": np.zeros((n, 4, 10), np.float32), "history_mask": np.ones((n, 4), bool),
         "static": np.zeros((n, 14), np.float32), "species": np.ones(n, np.int64),
         "metric_species": np.arange(n, dtype=np.int64) % 2, "y": np.zeros((n, 3, 2), np.float32),
         "target_mask": np.ones((n, 3), bool), "split": np.array([0] * 4 + [1] * 2 + [2] * 2 + [3, 4], np.int8),
         "individual": np.array([0, 1, 2, 3, 10, 11, 0, 1, 20, 0], np.int64),
         "study": np.zeros(n, np.int64), "horizons": h}
    a["history"][:, :, 2] = [0., 1., 7., 365.]
    a["history"][:, :, 0] = [0., -.01, -.07, -.3]
    a["history_mask"][:, -1] = False
    a["static"][np.isin(a["split"], [1, 2])] = 1000.
    a["origin"] = np.column_stack((np.zeros(n), np.full(n, 179.9), 1_600_000_000 + np.arange(n) * 20 * 86400.))
    a["target_time"] = a["origin"][:, 2:3] + h * 86400
    for i in range(n): a["y"][i, :, 0] = h * (.008 + .002 * (i % 4))
    a["target_latlon"] = forecast_latlon(a["y"][:, None], a["origin"])[:, 0]
    a["target_mask"][-2:] = False
    a["y"][-2:] = np.nan; a["target_latlon"][-2:] = np.nan; a["target_time"][-2:] = 0
    return a


def model(a):
    proposal, _ = fit_proposal(a)
    return EndpointMoveFormer(2, a["horizons"], history_length=4, width=16, layers=1, heads=2,
                              proposal=proposal_records(proposal), inference_candidates=8, variable_context=False)


def test_empirical_proposal_and_scale_never_fit_heldout_targets():
    a = corpus(); proposal, audit = fit_proposal(a)
    first = json.dumps(proposal_records(proposal), sort_keys=True)
    scale, heading = causal_scale(a, np.arange(10))
    np.testing.assert_allclose(scale, .01, atol=1e-8)
    a["y"][a["split"] != 0] = 1e20
    a["target_mask"][a["split"] != 0] = True
    a["history"][~a["history_mask"]] = np.nan
    second, _ = fit_proposal(a)
    assert json.dumps(proposal_records(second), sort_keys=True) == first
    new_scale, new_heading = causal_scale(a, np.arange(10))
    np.testing.assert_array_equal(new_scale, scale); np.testing.assert_array_equal(new_heading, heading)
    assert audit["training_targets"] == 12
    assert (0, 24) in proposal.records and (0, 168) in proposal.records


def test_true_positive_choice_sets_are_training_only_and_dropout_matches_proposal():
    a = corpus(); m = model(a); m.config["species_dropout"] = 1.
    batch = make_training_batch(m, a, np.array([0, 1]), np.array([0, 2]), 8, np.random.default_rng(4), "cpu")
    assert not batch["species"].any()
    scale, _ = causal_scale(a, np.array([0, 1]))
    np.testing.assert_allclose(batch["points"][:, 0].numpy(), a["y"][[0, 1], [0, 2]] / scale[:, None])
    with pytest.raises(ValueError, match="TRAINING"):
        make_training_batch(m, a, np.array([4]), np.array([0]), 8, np.random.default_rng(4), "cpu")


def test_float64_cache_horizons_are_float32_only_at_the_neural_input():
    a = corpus(); a["horizons"] = a["horizons"].astype("float64")
    m = model(a)
    batch = make_training_batch(m, a, np.array([0,1]), np.array([0,2]), 8, np.random.default_rng(4), "cpu")
    assert batch["horizon"].dtype == torch.float32
    _, loss = m.score(**batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert a["horizons"].dtype == np.float64


def test_masked_history_and_author_choice_loss_gradients():
    a = corpus(); m = model(a).eval()
    batch = make_training_batch(m, a, np.array([0, 1]), np.array([0, 2]), 8, np.random.default_rng(2), "cpu")
    first, _ = m.score(**batch)
    batch["history"][~batch["history_mask"]] = float("nan")
    second, loss = m.score(**batch)
    torch.testing.assert_close(first, second)
    loss.backward()
    for prefix in ("core.encoder", "core.cont_proj.horizon", "core.prediction_head"):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for name, p in m.named_parameters() if name.startswith(prefix)), prefix


def test_forecast_support_is_continuous_proposal_only_and_ignores_query_truth():
    a = corpus(); m = model(a)
    class Support:
        def sample(self, meta, velocity, count, rng):
            points = np.zeros((len(meta), count, 2), np.float32)
            points[..., 0] = np.arange(1, count + 1)[None] * .2
            points[..., 1] = rng.uniform(-.1, .1, (len(meta), count))
            return points
    m.proposal = Support()
    ids = np.array([4, 5])
    first = m.predict_cases(a, ids, 20, np.random.default_rng(5))
    scale, _ = causal_scale(a, ids)
    allowed = np.arange(1, 9) * .2
    for i in range(2):
        assert np.all(np.min(np.abs(first[i, :, :, 0, None] / scale[i] - allowed), axis=-1) < 1e-6)
    a["y"][ids] = 1e20; a["target_latlon"][ids] = np.nan; a["target_mask"][ids] = False
    second = m.predict_cases(a, ids, 20, np.random.default_rng(5))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 20, 3, 2)
    assert "Independent horizons" in m.upstream["evaluation"]


def test_validation_subset_is_fixed_metadata_only_and_excludes_tests():
    a = corpus()
    before = selection_origins(a, maximum=3, seed=17)
    a["y"][:] = 1e30; a["target_latlon"][:] = np.nan
    after = selection_origins(a, maximum=3, seed=17)
    np.testing.assert_array_equal(before, after)
    assert set(a["split"][before]) <= {1, 2}


def test_requested_endpoint_prefix_matches_full_forecast_without_reading_truth():
    a = corpus(); m = model(a); ids = np.array([4, 5])
    full = m.predict_cases(a, ids, 12, np.random.default_rng(4))
    for name in ("y", "target_mask", "target_latlon", "target_time"):
        del a[name]
    short = m.predict_endpoints(a, ids, horizons=a["horizons"][:2], samples=12, rng=np.random.default_rng(4))
    np.testing.assert_array_equal(short, full[:, :, :2])
    with pytest.raises(ValueError, match="frozen MoveFormer"):
        m.predict_endpoints(a, ids, horizons=[2.], samples=12)


def test_pinned_source_verifier_is_required_for_every_construction(monkeypatch):
    import wildcast.migration_moveformer_endpoint as module
    def changed():
        raise RuntimeError("Pinned source changed")
    monkeypatch.setattr(module, "pinned_author_provenance", changed)
    with pytest.raises(RuntimeError, match="Pinned source"):
        EndpointMoveFormer(2, [1, 3], width=16, layers=1, heads=2)


@pytest.mark.parametrize("short_task", [False, True])
def test_original_nce_training_and_serialization_roundtrip(tmp_path, short_task):
    a = corpus(); root = tmp_path / "data"; root.mkdir()
    np.savez_compressed(root / "cases.npz", **a)
    manifest = {"cases_sha256": hashlib.sha256((root / "cases.npz").read_bytes()).hexdigest(),
                "species": {"Animalus test": 1}, "reserved_test_targets_sealed": True}
    (root / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "fit"
    summary = train(root, output, epochs=1, batch_size=4, width=16, layers=1, heads=2,
                    candidates=8, inference_candidates=8, validation_candidates=8, validation_samples=4,
                    max_validation_origins=4, validation_batch_size=4, warmup_steps=1,
                    variable_context=False, device="cpu",
                    training_horizons_hours=[24, 168] if short_task else None,
                    neural_precision="float32" if short_task else "bf16")
    assert summary["epochs_completed"] == 1 and np.isfinite(summary["best_endpoint_selection_km"])
    loaded, ck = load_model(output / "best.pt")
    assert ck["model_kind"] == "wildcast_migration_moveformer_endpoint"
    torch.testing.assert_close(loaded.static_mean, torch.zeros_like(loaded.static_mean))
    paths = loaded.predict_cases(a, np.array([4, 6]), 5, np.random.default_rng(6))
    assert paths.shape == (2, 5, 2 if short_task else 3, 2) and np.isfinite(paths).all()
    if short_task:
        assert ck["config"]["horizons"] == [1., 7.]
        assert ck["config"]["neural_precision"] == "float32"
        assert ck["cases_sha256"] == manifest["cases_sha256"]
        np.testing.assert_array_equal(np.load(root / "cases.npz")["horizons"], [1., 3., 7.])
    ck["upstream"]["source_sha256"] = {"changed": "source"}
    torch.save(ck, output / "changed.pt")
    with pytest.raises(ValueError, match="provenance"):
        load_model(output / "changed.pt")
