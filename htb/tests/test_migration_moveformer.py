import hashlib
import json

import numpy as np
import pytest
import torch

pytest.importorskip("entmax")

from wildcast.migration_evaluate import forecast_latlon
from wildcast.migration_moveformer import MigrationMoveFormer, author_provenance, load_model, train
from wildcast.migration_retrieval_train import feature_algorithm_hash, prepare_bank


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model():
    torch.manual_seed(4)
    return MigrationMoveFormer(38, 48, 3, history_features=10, static_features=14,
                               history_length=5, candidates=8, width=16, layers=1, heads=2,
                               variable_context=True, species_dropout=0.)


def inputs():
    torch.manual_seed(3)
    h = torch.randn(3, 5, 10)
    mask = torch.ones(3, 5, dtype=torch.bool); mask[:, -2:] = False
    return h, mask, torch.randn(3, 14), torch.ones(3, dtype=torch.long), torch.randn(3, 8, 48)


def test_author_history_mask_excludes_garbage_and_empty_history_is_finite():
    m = model().eval()
    h, mask, static, species, candidate = inputs()
    first = m.logits(h, mask, static, species, candidate)
    h[~mask] = float("nan")
    second = m.logits(h, mask, static, species, candidate)
    torch.testing.assert_close(first, second)
    empty = m.logits(h, torch.zeros_like(mask), static, species, candidate)
    assert torch.isfinite(empty).all()
    m.train()
    varied = m.logits(h, mask, static, species, candidate)
    assert torch.isfinite(varied).all()


def test_energy_weight_gradients_reach_unchanged_encoder_and_selection_head():
    m = model()
    h, mask, static, species, candidate = inputs()
    logits = m.logits(h, mask, static, species, candidate)
    probabilities = logits.softmax(-1)
    # Exact discrete energy is linear/quadratic in these probabilities.
    loss = (probabilities * torch.arange(8.)).sum()
    loss.backward()
    for prefix in ("core.encoder", "core.cont_proj.history", "core.cont_proj.static",
                   "core.disc_emb.species", "core.prediction_head"):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for name, p in m.named_parameters() if name.startswith(prefix)), prefix
    assert m.upstream["pretrained_weights"] is False
    assert "replaces" in m.upstream["objective"]


def test_author_unknown_species_dropout_teaches_zero_token():
    m = model().train(); m.config["species_dropout"] = 1.
    h, mask, static, species, candidate = inputs()
    probabilities = m.logits(h, mask, static, species, candidate).softmax(-1)
    (probabilities * torch.arange(8.)).sum().backward()
    gradient = m.core.disc_emb["species"].weight.grad
    assert gradient[0].abs().sum() > 0
    assert gradient[1:].abs().sum() == 0


def test_predictions_select_one_whole_candidate_and_ignore_truth(monkeypatch):
    import wildcast.migration_retrieval_train as retrieval
    m = model().eval()
    h, mask, static, species, candidate = inputs()
    a = {"history": h.numpy(), "history_mask": mask.numpy(), "static": static.numpy(), "species": species.numpy(),
         "origin": np.zeros((3, 3)), "y": np.zeros((3, 3, 2)), "target_latlon": np.zeros((3, 3, 2))}
    paths = np.zeros((3, 8, 3, 2), np.float32)
    paths[..., 0] = np.arange(1, 9)[None, :, None] * np.array([1, 2, 3])[None, None]
    monkeypatch.setattr(retrieval, "generate_candidates", lambda a, ids, *args: (paths[ids], np.zeros((len(ids), 38)), candidate.numpy()[ids], {}))
    first = m.predict_cases(a, np.arange(3), 32, np.random.default_rng(5))
    assert np.all(first[:, :, 2, 0] == 3 * first[:, :, 0, 0])
    assert np.all(first[:, :, 1, 0] == 2 * first[:, :, 0, 0])
    a["y"][:] = 1e9; a["target_latlon"][:] = np.nan
    second = m.predict_cases(a, np.arange(3), 32, np.random.default_rng(5))
    np.testing.assert_array_equal(first, second)


def test_missing_or_changed_author_dependency_has_clear_error(tmp_path, monkeypatch):
    import wildcast.migration_moveformer as module
    monkeypatch.setattr(module, "PROJECT", tmp_path)
    with pytest.raises(RuntimeError, match="sources are required"):
        author_provenance()


def test_missing_python_dependency_has_install_instruction(monkeypatch):
    import wildcast.migration_moveformer as module
    def unavailable():
        raise ImportError("missing entmax")
    monkeypatch.setattr(module, "author_classes", unavailable)
    with pytest.raises(RuntimeError, match="moveformer extra"):
        author_provenance()


def data(root):
    root.mkdir()
    n, horizons = 8, np.array([1., 3., 7.], np.float32)
    a = {"history": np.zeros((n, 3, 10), np.float32), "history_mask": np.ones((n, 3), bool),
         "static": np.zeros((n, 14), np.float32), "species": np.ones(n, np.int64),
         "metric_species": np.zeros(n, np.int64), "y": np.zeros((n, 3, 2), np.float32),
         "target_mask": np.ones((n, 3), bool), "split": np.array([0, 0, 0, 0, 1, 2, 3, 4], np.int8),
         "individual": np.array([0, 1, 2, 3, 10, 0, 20, 0], np.int64), "study": np.zeros(n, np.int64), "horizons": horizons}
    a["history"][:, :, 2] = [0, 1, 7]; a["history"][:, :, 0] = [0., -.01, -.07]
    a["static"][np.isin(a["split"], [1, 2])] = 1000.  # Must not affect fitted normalization.
    a["origin"] = np.column_stack((np.zeros(n), np.zeros(n), 1_700_000_000 + np.arange(n) * 86400. * 20))
    a["target_time"] = a["origin"][:, 2:3] + horizons * 86400
    a["y"][:, :, 0] = horizons * .01
    a["target_latlon"] = forecast_latlon(a["y"][:, None], a["origin"])[:, 0]
    a["target_mask"][-2:] = False; a["y"][-2:] = 0; a["target_time"][-2:] = 0; a["target_latlon"][-2:] = 0
    np.savez_compressed(root / "cases.npz", **a)
    (root / "manifest.json").write_text(json.dumps({"cases_sha256": hashlib.sha256((root / "cases.npz").read_bytes()).hexdigest(),
        "species": {"Animalus test": 1}, "reserved_test_targets_sealed": True}))
    return a


def test_exact_energy_shared_trainer_and_author_checkpoint_roundtrip(tmp_path):
    a = data(tmp_path / "data")
    before = feature_algorithm_hash()
    prepare_bank(tmp_path / "data", tmp_path / "bank", candidates=8, seed=31)
    summary = train(tmp_path / "data", tmp_path / "bank", tmp_path / "fit", epochs=1,
                    batch_size=4, width=16, layers=1, heads=2, warmup_steps=1,
                    variable_context=False, device="cpu")
    assert feature_algorithm_hash() == before
    assert summary["epochs_completed"] == 1
    loaded, ck = load_model(tmp_path / "fit/best.pt")
    assert ck["model_kind"] == "wildcast_migration_moveformer"
    assert ck["upstream"]["pretrained_weights"] is False
    assert "not an official" in ck["upstream"]["scope"]
    torch.testing.assert_close(loaded.static_mean, torch.zeros_like(loaded.static_mean))
    paths = loaded.predict_cases(a, np.array([4, 5]), 8, np.random.default_rng(4))
    assert paths.shape == (2, 8, 3, 2) and np.isfinite(paths).all()
