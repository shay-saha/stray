"""Context alignment, future isolation and parent-law preservation."""
import json

import numpy as np
import pytest
import torch

from test_migration_human_pressure import sources, cache, coordinates, timestamp
from wildcast import migration_context_joint as context
from wildcast.migration_human_pressure import HumanPressure, sha256
from wildcast.migration_model import MigrationModel


def test_features_use_only_observed_positions_and_earlier_epochs(sources):
    origin = np.asarray([[*coordinates(20, 20), timestamp(2006)]] * 2)
    history = np.zeros((2, 4, 3), np.float32)
    history[..., 2] = [0, 1, 7, 365]
    a = {"origin": origin, "history": history, "history_mask": np.ones((2, 4), bool)}
    a["history_mask"][1, 1:3] = False
    with HumanPressure(sources, tile_size=16) as raster:
        values = context.features(raster, a, np.arange(2))
        assert values.shape == (2, 21)
        np.testing.assert_allclose(values[0, [0, 7, 14]], .8)
        assert not values[1, 7:].any()
        # Poison every future/truth/identity field. These are never accessed.
        a.update(y=object(), target_latlon=object(), target_time=object(), individual=object())
        np.testing.assert_array_equal(values, context.features(raster, a, np.arange(2)))
        coords, times = context.positions(a, [0, 1])
        assert np.all(times[0] <= origin[0, 2])
        assert np.isnan(coords[1, 1:]).all()


def test_controls_remove_content_without_changing_metadata():
    values = np.arange(42, dtype=np.float32).reshape(2, 21)
    availability = context.transform(values, "availability")
    np.testing.assert_array_equal(availability[:, [0, 7, 14]], 0)
    other = [i for i in range(21) if i % 7]
    np.testing.assert_array_equal(availability[:, other], values[:, other])
    assert not context.transform(values, "masked").any()
    np.testing.assert_array_equal(context.transform(values, "actual"), values)
    assert values[1, 0] == 21


def test_preparation_and_canonical_join_exclude_reserved_and_reject_tampering(cache, sources, tmp_path):
    data, a, _ = cache
    artifact = tmp_path / "context"
    metadata = context.prepare(data, sources, artifact, batch_size=2)
    assert metadata["origins"] == 3
    selected = {"static": np.zeros((2, 2), np.float32), "case_index": np.asarray([2, 0])}
    hashes = {key: metadata[key] for key in ("cases_sha256", "manifest_sha256")}
    result, evidence = context.append_features(selected, hashes, artifact, "actual")
    assert result["static"].shape == (2, 23)
    with np.load(artifact / "features.npz") as table:
        np.testing.assert_array_equal(table["case_ids"], [0, 1, 2])
        np.testing.assert_array_equal(result["static"][:, 2:], table["features"][[2, 0]])
    with pytest.raises(ValueError, match="cover"):
        context.append_features({**selected, "case_index": np.asarray([3, 0])}, hashes, artifact, "actual")
    with (artifact / "features.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        context.append_features(selected, hashes, artifact, "actual")


def test_expansion_preserves_parent_distribution_and_receives_gradients(tmp_path):
    torch.manual_seed(42)
    config = dict(history_features=3, static_features=2, n_species=3, horizons=[1., 7.],
                  history_length=4, width=16, layers=1, components=2, dropout=0., species_dropout=0., variant="history_direct")
    parent = MigrationModel(**config).eval()
    checkpoint = tmp_path / "parent.pt"
    hashes = {"manifest_sha256": "manifest", "cases_sha256": "cases"}
    torch.save({"model_kind": "wildcast_migration_joint", "config": parent.config,
                "model": parent.state_dict(), **hashes}, checkpoint)
    child = MigrationModel(**{**config, "static_features": 23}).eval()
    context.initialize_expanded(child, checkpoint, hashes, "cpu")
    history = torch.randn(3, 4, 3); mask = torch.ones((3, 4), dtype=torch.bool)
    static = torch.randn(3, 2); extra = torch.randn(3, 21); species = torch.arange(3)
    expected = parent(history, mask, static, species)
    actual = child(history, mask, torch.cat([static, extra], -1), species)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=2e-7, rtol=2e-6)
    actual["means"].square().sum().backward()
    assert child.static_encoder[0].weight.grad[:, 2:].abs().sum() > 0
    bad = {**hashes, "cases_sha256": "wrong"}
    with pytest.raises(ValueError, match="hashes"):
        context.initialize_expanded(child, checkpoint, bad, "cpu")


def test_context_training_and_inference_roundtrip(tmp_path, sources):
    from test_migration_model import fixture_cache
    from wildcast.migration_train import load_training_data, train
    torch.set_num_threads(2)
    data = tmp_path / "data"; data.mkdir(); fixture_cache(data)
    common = dict(epochs=1, batch_size=8, width=16, layers=1, components=2,
                  device="cpu", validation_samples=4, dropout=0., variant="history_direct")
    parent = tmp_path / "parent"
    train(data, parent, **common)
    table = tmp_path / "table"
    context.prepare(data, sources, table)
    child = tmp_path / "child"
    train(data, child, **common, context_features=table, context_mode="actual", init_checkpoint=parent / "best.pt", context_only=True)
    model, ck = context.load_model(child / "best.pt")
    assert ck["model_kind"] == context.KIND
    assert ck["config"]["static_features"] == 24
    assert ck["context_features"]["uses_future_targets"] is False
    parent_ck = torch.load(parent / "best.pt", weights_only=True)
    trained_ck = torch.load(child / "last.pt", weights_only=True)
    assert trained_ck["epoch"] == 1 and trained_ck["training"]["context_only"] is True
    for name, value in parent_ck["model"].items():
        trained = trained_ck["model"][name]
        if name in ("static_mean", "static_std"):
            trained = trained[:3]
        elif name == "static_encoder.0.weight":
            trained = trained[:, :3]
        torch.testing.assert_close(trained, value, rtol=0, atol=0)
    a, _, _ = load_training_data(data)
    draws = model.predict_cases(a, np.asarray([12, 13]), 8, np.random.default_rng(3))
    assert draws.shape == (2, 8, 3, 2) and np.isfinite(draws).all()
    a["target_latlon"] = object(); a["y"] = object(); a["target_mask"] = object()
    np.testing.assert_array_equal(draws, model.predict_cases(a, np.asarray([12, 13]), 8, np.random.default_rng(3)))
    model.raster.close()
