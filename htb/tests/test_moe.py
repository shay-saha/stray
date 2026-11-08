import hashlib
import json

import numpy as np
import pytest
import torch

from wildcast.model import MovementTransformer, mixture_energy_score, nll
from wildcast.moe import CategoryMoE, pack_ensemble


def fixture_moe(mode="family"):
    base = MovementTransformer(3, width=32, layers=1, n_drivers=4, n_anchors=5,
                               memory_features=10, memory_length=3, context=8, student_df=4)
    config = {"base_configs": [base.config], "n_drivers": 4, "primary": 0, "hidden": 16, "mode": mode,
              "families": ["shared", "bird_one", "bird_two", "cat"], "classes": ["unknown", "Aves", "Mammalia"],
              "species_family": [0, 1, 2, 3], "species_class": [0, 1, 1, 2], "family_class": [0, 1, 1, 2]}
    model = CategoryMoE(config).eval()
    model.bases[0].load_state_dict(base.state_dict())
    rng = np.random.default_rng(1)
    a = {"x": rng.normal(size=(4, 8, 10)).astype(np.float32), "y": rng.normal(size=(4, 2)).astype(np.float32),
         "meta": np.asarray([[1, 0, 0, 0, 1], [1, 1, 0, 1, 12], [3, 2, 1, 2, 24], [3, 3, 1, 3, 168]], np.int32),
         "aux": np.zeros((4, 9), np.float64), "drivers": rng.normal(size=(4, 4)).astype(np.float32),
         "anchors": rng.normal(size=(4, 5, 2)).astype(np.float32), "memory": np.zeros((4, 3, 10), np.float32)}
    a["memory"][:, 0, 0] = 1
    return model, a


def test_zero_specialists_preserve_density_and_coupled_forecasts_across_routing_modes():
    model, a = fixture_moe()
    with torch.inference_mode():
        f, packed, species = model.encode_inputs(a, np.arange(4), "cpu")
        base = (packed[..., 0], packed[..., 1:3], packed[..., 3:5], packed[..., 5], 4)
        reference = model.sample_from_features(f, packed, species, 64, torch.Generator().manual_seed(22))
        for hidden in [0, 16]:
            for mode in ["shared", "family", "learned"]:
                candidate = CategoryMoE({**model.config, "mode": mode, "hidden": hidden}).eval()
                truth = torch.from_numpy(a["y"])
                torch.testing.assert_close(nll(candidate(f, packed, species), truth), nll(base, truth))
                draws = candidate.sample_from_features(f, packed, species, 64, torch.Generator().manual_seed(22))
                torch.testing.assert_close(draws, reference)


def test_fixed_family_gradients_only_update_selected_specialist():
    model, a = fixture_moe()
    f, base, species = model.encode_inputs(a, np.asarray([1]), "cpu")
    params = model(f, base, species)
    loss = nll(params, torch.from_numpy(a["y"][[1]])).mean()
    loss.backward()
    assert model.experts[1][-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for j, head in enumerate(model.experts) if j != 1 for p in head.parameters())
    assert all(p.grad is None for p in model.bases.parameters())


def test_learned_gate_respects_taxonomy_and_unknown_species_fallback():
    model, a = fixture_moe("learned")
    f, base, species = model.encode_inputs(a, np.arange(4), "cpu")
    params = model(f, base, species)
    usage = model.last_usage
    assert usage[0, 0] == 1 and usage[0, 1:].sum() == 0
    assert usage[1:3, 3].sum() == 0
    assert usage[3, 1:3].sum() == 0
    torch.testing.assert_close(usage.sum(-1), torch.ones(4))
    with torch.no_grad():
        model.experts[1][-1].bias[1] = .5
    loss = mixture_energy_score(model(f, base, species), torch.from_numpy(a["y"]), repeats=1).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.gate[-1].weight.grad.abs().sum() > 0


def test_category_inference_does_not_read_future_targets_or_animal_identifiers():
    model, a = fixture_moe("learned")
    with torch.inference_mode():
        first = model.predict_samples(a, np.arange(4), 8, "cpu", torch.Generator().manual_seed(9))
        a["y"][:] = 1e15
        a["aux"][:, [3, 7]] = 1e15
        a["meta"][:, 3] = 100000
        second = model.predict_samples(a, np.arange(4), 8, "cpu", torch.Generator().manual_seed(9))
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("mode", ["shared", "family", "learned"])
@pytest.mark.parametrize("hidden", [0, 16])
def test_moe_training_and_reloaded_checkpoint_forecast(tmp_path, mode, hidden):
    from wildcast.identity import case_fingerprint
    from wildcast.moe_train import train
    from wildcast.evaluate import load_model, predictions
    model, a = fixture_moe()
    # Use duplicated known-species cases to exercise nonzero specialist updates.
    a = {k: np.tile(v[1:], (4,) + (1,) * (v.ndim - 1)) for k, v in a.items()}
    a["meta"][:6, 0] = 0
    root, cache = tmp_path / "data", tmp_path / "cache"
    (root / "processed").mkdir(parents=True)
    cache.mkdir()
    np.savez_compressed(root / "processed/windows.npz", **a)
    manifest = {"windows_sha256": hashlib.sha256((root / "processed/windows.npz").read_bytes()).hexdigest()}
    (root / "processed/windows.json").write_text(json.dumps(manifest))
    ids = np.arange(len(a["meta"]))
    with torch.inference_mode():
        f, b, _ = model.encode_inputs(a, ids, "cpu")
    np.save(cache / "features.npy", f.numpy())
    np.save(cache / "base.npy", b.numpy())
    np.save(cache / "ids.npy", ids)
    torch.save({"config": model.config, "states": [m.state_dict() for m in model.bases]}, cache / "bases.pt")
    info = {**manifest, "case_fingerprint": case_fingerprint(a), "identity_audit_sha256": "test", "training_support": [],
            "files_sha256": {p: hashlib.sha256((cache / p).read_bytes()).hexdigest() for p in ["features.npy", "base.npy", "ids.npy"]}}
    (cache / "manifest.json").write_text(json.dumps(info))
    # Fixtures need positive causal physical scales for normalized training loss.
    a["aux"][:, 8] = 100
    np.savez_compressed(root / "processed/windows.npz", **a)
    manifest["windows_sha256"] = hashlib.sha256((root / "processed/windows.npz").read_bytes()).hexdigest()
    (root / "processed/windows.json").write_text(json.dumps(manifest))
    info.update(windows_sha256=manifest["windows_sha256"], case_fingerprint=case_fingerprint(a))
    (cache / "manifest.json").write_text(json.dumps(info))
    result = train(root, cache, tmp_path / "model", mode=mode, hidden=hidden, epochs=1, batch_size=4, device="cpu")
    assert np.isfinite(result["best_validation_score"])
    loaded, ck = load_model(tmp_path / "model/best.pt", "cpu")
    with torch.inference_mode():
        draws = predictions(loaded, a, ids[:3], 4, "cpu", torch.Generator().manual_seed(1))
    assert draws.shape == (3, 4, 2) and torch.isfinite(draws).all()
    assert ck["model_family"] == "wildcast_category_moe"
    from wildcast.moe_evaluate import evaluate
    report = evaluate(root, cache, [tmp_path / "model/best.pt"], tmp_path / "evaluation", samples=4, batch_size=4, device="cpu")
    assert set(report) == {"frozen-ensemble", "model"}
    evidence = json.loads((tmp_path / "evaluation/model/metrics.json").read_text())
    assert evidence["cases"] == 6
    assert sum(evidence["mean_expert_weight"].values()) == pytest.approx(1)
    assert len(evidence["distance_metrics"]) > 0


def test_small_calibration_experts_preserve_conditional_means_and_correlations():
    original, a = fixture_moe()
    model = CategoryMoE({**original.config, "hidden": 0}).eval()
    features, base, species = original.encode_inputs(a, np.arange(4), "cpu")
    with torch.no_grad():
        model.experts[1].adjustment[0] = torch.tensor([.7, -.5, .3])
    params = model(features, base, species)
    torch.testing.assert_close(params[1], base[..., 1:3])
    torch.testing.assert_close(params[3], base[..., 5])
    assert not torch.allclose(params[2][1], base[1, :, 3:5])
    torch.testing.assert_close(params[2][2:], base[2:, :, 3:5])
