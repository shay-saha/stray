import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from wildcast.data import prepare, project
from wildcast.model import MovementTransformer
from wildcast.trajectory import (TrajectoryFlow, constrain_steps, decode_path, encode_steps,
                                 geographic_steps, project_torch, step_durations)
from wildcast.trajectory_data import observed_sequences, phases, speed_limits


def fixture_model():
    base = MovementTransformer(2, width=32, layers=1, n_drivers=4, n_anchors=5,
                               memory_features=10, memory_length=3, student_df=4, context=8)
    model = TrajectoryFlow({"backbone": base.config, "n_drivers": 4, "horizon_driver_indices": [3],
                            "speed_limits": [100., 5., 10.], "width": 32, "layers": 1,
                            "ode_steps": 2, "dropout": 0., "gps_allowance_m": 0.})
    model.backbone.load_state_dict(base.state_dict())
    x = np.zeros((2, 8, 10), np.float32)
    meta = np.asarray([[1, 1, 0, 0, 1], [1, 2, 0, 1, 24]], np.int32)
    aux = np.zeros((2, 9), np.float64)
    aux[:, :2] = [[78., 179.99], [-62., -179.99]]
    aux[:, 2] = 1577836800
    aux[:, 8] = 1000.
    memory = np.zeros((2, 3, 10), np.float32)
    memory[:, 0, 0] = 1
    a = {"x": x, "meta": meta, "aux": aux, "y": np.zeros((2, 2), np.float32),
         "drivers": np.zeros((2, 4), np.float32), "anchors": np.zeros((2, 5, 2), np.float32), "memory": memory}
    return model.eval(), a


def test_spherical_path_speed_envelope_and_dateline():
    raw = torch.full((2, 3, 4, 2), 10.)
    times = torch.tensor([[.01, .1, 1., 3.]]).expand(2, -1)
    aux = torch.zeros(2, 9, dtype=torch.float64)
    aux[:, :2] = torch.tensor([[78., 179.99], [-62., -179.99]])
    aux[:, 8] = 1000
    limits = torch.tensor([5., 10.])
    xy, coords = decode_path(raw, times, aux, limits, 0.)
    prev = torch.cat([aux[:, None, None, :2].expand(-1, 3, 1, -1), coords[:, :, :-1]], 2)
    lengths = project_torch(coords, prev).norm(dim=-1)
    bounds = limits[:, None] * step_durations(times) * 3600
    assert torch.all(lengths <= bounds[:, None] * 1.000001)
    assert torch.all(coords[..., 1] >= -180) and torch.all(coords[..., 1] < 180)
    assert torch.isfinite(xy).all()


def test_irregular_duration_encoding_roundtrip_and_zero_gradient():
    times = torch.tensor([[.1, 2., 24.]])
    physical = torch.tensor([[[1., 2.], [300., -500.], [2000., 1000.]]])
    scale, speed = torch.tensor([500.]), torch.tensor([5.])
    raw = encode_steps(physical, times, scale, speed, 0.)
    got = constrain_steps(raw, times, scale, speed, 0.)
    torch.testing.assert_close(got, physical)
    aux = torch.zeros(1, 9, dtype=torch.float64)
    aux[:, :2] = torch.tensor([[55., 179.99]])
    aux[:, 8] = 500
    z = torch.zeros(1, 2, 3, 2, requires_grad=True)
    xy, coords = decode_path(z, times, aux, speed, 0.)
    xy.sum().backward()
    assert torch.isfinite(z.grad).all()
    torch.testing.assert_close(coords, aux[:, None, None, :2].expand_as(coords))


def test_history_context_and_inference_never_read_future_labels():
    model, a = fixture_model()
    with torch.inference_mode():
        ctx = model.observed_context(a, np.arange(2), "cpu")
        first = model.predict_samples(a, np.arange(2), 3, "cpu", torch.Generator().manual_seed(2))
        a["y"][:] = 1e9
        a["aux"][:, 3] = 1e15  # future observation timestamp
        a["aux"][:, 7] = 1e9   # future observation row
        a["drivers"][:, 3] = 1e6  # horizon-dependent driver must be removed
        torch.testing.assert_close(model.observed_context(a, np.arange(2), "cpu"), ctx)
        second = model.predict_samples(a, np.arange(2), 3, "cpu", torch.Generator().manual_seed(2))
    torch.testing.assert_close(first, second)


def test_joint_flow_training_gradients_and_state_probabilities():
    model, a = fixture_model()
    model.train()
    context = model.observed_context(a, np.arange(2), "cpu")
    times = torch.tensor([[1., 4., 12.], [2., 6., 6.]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    aux = torch.from_numpy(a["aux"])
    velocity = aux[:, 4:6].float()
    anchors = torch.from_numpy(a["anchors"])
    species = torch.tensor([1, 2])
    prior = model.prior(context, times, mask, aux, velocity, anchors, species, 2, torch.Generator().manual_seed(3))
    tokens, states = model.prepare_condition(context, times, aux, mask, prior, species)
    torch.testing.assert_close(states.sum(-1), torch.ones(2, 3))
    assert torch.all(states >= 0)
    vector = model.vector_field(prior, prior, torch.tensor([.3, .7]), tokens, states, mask)
    positions, _ = decode_path(prior + vector, times, aux, model.speed_limits[species], 0.)
    loss = ((vector - 1).square() * mask[:, None, :, None]).mean() + positions.square().mean() + states[..., 0].mean()
    loss.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is None for p in model.backbone.parameters())


def test_future_sequences_preserve_actual_targets_and_purge_phase_boundaries(tmp_path):
    p = tmp_path / "processed"
    p.mkdir()
    f = pd.DataFrame({"animal": "one", "species": "species", "study": "study", "segment": 0,
                      "timestamp": pd.date_range("2020-01-01", periods=400, freq="h", tz="UTC"),
                      "lat": np.linspace(10, 10.1, 400), "lon": 179.9})
    f.to_parquet(p / "observations.parquet", index=False)
    manifest = prepare(tmp_path, context=8, stride=8)
    a = dict(np.load(p / "windows.npz"))
    seq = observed_sequences(f, a, manifest)
    phase = phases(f, manifest)
    for row, window in enumerate(seq["window_ids"]):
        valid = seq["mask"][row]
        future = seq["future_rows"][row, valid]
        origin = int(a["aux"][window, 6])
        assert np.all(future > origin)
        assert np.all(phase[future] == phase[origin])
        np.testing.assert_allclose(seq["times"][row, valid], (future - origin).astype(float))
        expected = project(f.lat.to_numpy()[future], f.lon.to_numpy()[future], f.lat.iloc[origin], f.lon.iloc[origin]) / a["aux"][window, 8]
        np.testing.assert_allclose(seq["future_xy"][row, valid], expected, rtol=1e-6, atol=1e-5)
        same = (a["aux"][:, 6] == origin) & (a["meta"][:, 0] == phase[origin])
        assert set(a["aux"][same, 7].astype(int)) <= set(future)
    assert set(a["meta"][seq["window_ids"], 0]) <= {0, 1, 3}
    val = np.flatnonzero(a["meta"][:, 0] == 1)
    excluded = observed_sequences(f, a, manifest, excluded_windows=[int(val[0])])
    assert a["aux"][val[0], 6] not in a["aux"][excluded["window_ids"], 6]


def test_speed_envelope_cannot_learn_from_future_tracks():
    f = pd.DataFrame({"animal": "one", "species": "species", "segment": 0,
                      "timestamp": pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC"),
                      "lat": np.linspace(10, 10.1, 100), "lon": 20.})
    manifest = {"animal_assignments": {"one": "temporal"}, "species": {"species": 1}}
    first, _ = speed_limits(f, manifest)
    f.loc[65:, "lat"] = np.linspace(-70, 70, 35)
    second, _ = speed_limits(f, manifest)
    assert first == second


def test_joint_forecast_rejects_reversed_or_zero_time_queries():
    model, a = fixture_model()
    times = torch.tensor([[1., .5], [1., 2.]])
    with pytest.raises(ValueError, match="strictly increasing"):
        model.sample_paths(model.observed_context(a, np.arange(2), "cpu"), times, torch.ones_like(times, dtype=torch.bool),
                           torch.from_numpy(a["aux"]), torch.zeros(2, 2), torch.from_numpy(a["anchors"]), torch.tensor([1, 2]))


def test_shared_experience_masks_same_animal_aliases_and_other_classes():
    model, a = fixture_model()
    context = model.observed_context(a, np.arange(2), "cpu")
    model.bank_keys = context[:1].repeat(3, 1)
    model.bank_values = torch.randn(3, 16)
    model.bank_species = torch.tensor([1, 1, 2])
    model.bank_identity = torch.tensor([10, 10, 20])
    model.species_classes = torch.tensor([0, 1, 2])
    # Both same-identity entries are excluded, including an alias. The remaining
    # other-class entry is also excluded: no uniform fallback to forbidden data.
    value = model.retrieve_experience(context[:1], torch.tensor([1]), torch.tensor([10]))
    torch.testing.assert_close(value, torch.zeros_like(value))
    with pytest.raises(ValueError, match="canonical query identity"):
        model.retrieve_experience(context[:1], torch.tensor([1]), None)
    # Changing forbidden experiences cannot affect an allowed query's output.
    before = model.retrieve_experience(context[:1], torch.tensor([1]), torch.tensor([30]))
    model.bank_values[2] = 1e9
    after = model.retrieve_experience(context[:1], torch.tensor([1]), torch.tensor([30]))
    torch.testing.assert_close(before, after)


def test_species_adaptation_changes_only_selected_species_condition():
    model, a = fixture_model()
    assert model.species_adapter.weight.abs().sum() == 0
    with torch.no_grad():
        model.species_adapter.weight[1, 32:] = 1
    context = model.observed_context(a, np.arange(2), "cpu")
    times = torch.tensor([[1., 12.], [1., 12.]])
    aux = torch.from_numpy(a["aux"])
    prior = torch.zeros(2, 2, 2, 2)
    tokens, _ = model.prepare_condition(context, times, aux, torch.ones_like(times, dtype=torch.bool), prior, torch.tensor([1, 2]))
    with torch.no_grad():
        model.species_adapter.weight.zero_()
    base_tokens, _ = model.prepare_condition(context, times, aux, torch.ones_like(times, dtype=torch.bool), prior, torch.tensor([1, 2]))
    torch.testing.assert_close(tokens[1], base_tokens[1])
    torch.testing.assert_close(tokens[0] - base_tokens[0], torch.ones_like(tokens[0]))


def test_copula_query_context_matches_original_horizon_specific_model():
    from wildcast.drivers import daylength
    model, a = fixture_model()
    model.config["future_photoperiod"] = {"index": 3, "median": 0., "iqr": 1.}
    times = torch.tensor([[1., 12., 24., 168.]]).expand(2, -1)
    aux = torch.from_numpy(a["aux"])
    with torch.inference_mode():
        context = model.observed_context(a, np.arange(2), "cpu")
        actual, h = model.horizon_context(context, times, aux)
        doy = pd.to_datetime(a["aux"][:, 2], unit="s", utc=True).dayofyear.to_numpy()
        light = daylength(a["aux"][:, 0, None], doy[:, None] + times.numpy() / 24) / 24
        dr = torch.zeros(8, 4)
        dr[:, 3] = torch.from_numpy(light.flatten().astype(np.float32))
        expected = model.backbone.context_features(torch.from_numpy(a["x"]).repeat_interleave(4, 0),
                    torch.from_numpy(a["meta"][:, 1]).long().repeat_interleave(4), h,
                    aux[:, 4:6].float().repeat_interleave(4, 0), dr,
                    torch.from_numpy(a["anchors"]).repeat_interleave(4, 0),
                    torch.from_numpy(a["memory"]).repeat_interleave(4, 0))
    torch.testing.assert_close(actual, expected)


def test_ensemble_member_is_chosen_once_per_entire_path():
    from wildcast.trajectory import JointTrajectoryEnsemble
    class ConstantPaths(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
        def sample_paths(self, context, times, mask, aux, velocity, anchors, species, count, generator, **kwargs):
            points = torch.full((len(times), count, times.shape[1], 2), self.value)
            return {"xy": points, "latlon": points, "latent_steps": points,
                    "regime_probabilities": torch.ones(len(times), times.shape[1], 3) / 3}
    ensemble = JointTrajectoryEnsemble({"members": []})
    ensemble.members = torch.nn.ModuleList([ConstantPaths(0.), ConstantPaths(1.)])
    times = torch.tensor([[1., 12., 24.]])
    result = ensemble.sample_paths([None, None], times, torch.ones_like(times, dtype=torch.bool),
                                  torch.zeros(1, 9), torch.zeros(1, 2), torch.zeros(1, 5, 2),
                                  torch.ones(1, dtype=torch.long), 32, torch.Generator().manual_seed(7))
    assert set(result["xy"].unique().tolist()) == {0., 1.}
    assert torch.all(result["xy"][:, :, 0:1] == result["xy"])


@pytest.mark.parametrize("objective", ["flow", "energy"])
def test_training_checkpoint_reload_and_joint_export_end_to_end(tmp_path, objective):
    from wildcast.identity import case_fingerprint
    from wildcast.trajectory_train import train
    from wildcast.trajectory_evaluate import evaluate, forecast
    model, original = fixture_model()
    model.config["identity_groups"] = [0, 1]
    model.identity_groups = torch.tensor([0, 1])
    root = tmp_path / "data"
    processed = root / "processed"
    processed.mkdir(parents=True)
    a = {k: np.tile(v, (8,) + (1,) * (v.ndim - 1)) for k, v in original.items()}
    a = {k: np.repeat(v, 4, axis=0) for k, v in a.items()}
    a["meta"][:32, 0] = 0
    a["meta"][32:, 0] = 1
    a["aux"][:, 6] = np.repeat(np.arange(16), 4)
    a["meta"][:, 4] = np.tile([1, 12, 24, 168], 16)
    np.savez_compressed(processed / "windows.npz", **a)
    manifest = {"windows_sha256": hashlib.sha256((processed / "windows.npz").read_bytes()).hexdigest(),
                "observations_sha256": "synthetic", "species": {"one": 1, "two": 2}, "protocol": "Synthetic integration test"}
    (processed / "windows.json").write_text(json.dumps(manifest))
    cache = tmp_path / "cache"
    cache.mkdir()
    ids = np.arange(0, 64, 4)
    np.save(cache / "context.npy", model.observed_context(a, ids, "cpu").numpy())
    times = np.tile([1., 12., 24., 168.], (16, 1)).astype(np.float32)
    np.savez_compressed(cache / "sequences.npz", window_ids=ids, times=times,
                        mask=np.ones_like(times, bool), future_xy=np.zeros((16, 4, 2), np.float32),
                        future_rows=np.zeros((16, 4), int))
    torch.save({"config": model.config, "model": model.state_dict()}, cache / "initial.pt")
    info = {"windows_sha256": manifest["windows_sha256"], "case_fingerprint": case_fingerprint(a),
            "identity_audit_sha256": "synthetic", "context_sha256": hashlib.sha256((cache / "context.npy").read_bytes()).hexdigest(),
            "sequences_sha256": hashlib.sha256((cache / "sequences.npz").read_bytes()).hexdigest()}
    (cache / "manifest.json").write_text(json.dumps(info))
    out = tmp_path / "trained"
    result = train(root, cache, out, epochs=1, batch_size=4, width=32, layers=1,
                   validation_origins=4, validation_samples=2, ode_steps=1, device="cpu", objective=objective, training_samples=2)
    assert np.isfinite(result["best_validation_score"])
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({**manifest, "excluded_windows": []}))
    report = evaluate(root, out / "best.pt", tmp_path / "evaluation", audit, samples=3, batch_size=4,
                      max_origins=4, export_origins=2, device="cpu")
    assert report["physical_audit"]["steps_outside_envelope"] == 0
    assert {r["horizon_hours"] for r in report["distance_metrics"]} == {1, 12, 24, 168}
    exported = json.loads((tmp_path / "evaluation/trajectories.json").read_text())
    assert len(exported["origins"]) == 2
    for origin in exported["origins"]:
        for path in origin["paths_latlon"]:
            np.testing.assert_allclose(path[0], origin["origin_latlon"])
    forecast(root, out / "best.pt", tmp_path / "forecast.json", window_ids=[32],
             times=[.5, 3., 24.], samples=3, device="cpu")
    prediction = json.loads((tmp_path / "forecast.json").read_text())
    assert prediction["future_observations_included"] is False
    assert prediction["origins"][0]["requested_times_hours"] == [.5, 3., 24.]
    assert {0., .5, 1., 2., 3., 24.} <= set(prediction["origins"][0]["times_hours"])
    assert "observed_future" not in prediction["origins"][0]
