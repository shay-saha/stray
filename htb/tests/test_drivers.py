import json

import numpy as np
import pandas as pd
import pytest
import torch

from wildcast.drivers import WEATHER_VARS, WeatherLookup, astronomy, daylength, history_features
from wildcast.model import MovementTransformer, nll, sample, mixture_energy_score


def test_photoperiod_tracks_hemisphere_and_polar_seasons():
    assert daylength(55, 172) > daylength(55, 355)
    assert daylength(-55, 172) < daylength(-55, 355)
    assert daylength(89, 172) == pytest.approx(24)
    assert daylength(89, 355) == pytest.approx(0)
    assert daylength(0, 80) == pytest.approx(12)
    assert np.isfinite(list(astronomy(89, 179, 1609459200, 168).values())).all()


def test_memory_is_invariant_to_future_coordinates_and_phase_prefix():
    lat = 10 + np.arange(200) * .0001
    lon = 20 + np.sin(np.arange(200) * .1) * .01
    t = np.arange(200) * 3600.
    a, anchors = history_features(lat, lon, t, 120, 60, 100.)
    lat2, lon2 = lat.copy(), lon.copy()
    lat2[:60] = -70; lon2[:60] = -130
    lat2[121:] = 80; lon2[121:] = 120
    b, anchors2 = history_features(lat2, lon2, t, 120, 60, 100.)
    assert a == b
    np.testing.assert_allclose(anchors, anchors2)
    assert a["past_7d_present"] == 0


def test_weather_does_not_use_origin_day_or_future_days(tmp_path):
    folder = tmp_path / "environment/weather"; folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"grid_degrees": 1.}))
    days = pd.date_range("2020-01-01", periods=10, tz="UTC")
    daily = {"time": [t.strftime("%Y-%m-%d") for t in days]}
    daily.update({v: list(np.arange(10, dtype=float)) for v in WEATHER_VARS})
    path = folder / "10_20.json"
    path.write_text(json.dumps({"response": {"daily": daily}}))
    origin = pd.Timestamp("2020-01-06T12:00Z").timestamp()
    a = WeatherLookup(tmp_path, lag_days=2).features(10.2, 20.2, origin)
    for v in WEATHER_VARS:
        daily[v][4:] = [999999.] * 6
    path.write_text(json.dumps({"response": {"daily": daily}}))
    b = WeatherLookup(tmp_path, lag_days=2).features(10.2, 20.2, origin)
    assert a == b
    missing = WeatherLookup(tmp_path).features(-5, -5, origin)
    assert all(value == 0 for value in missing.values())
    assert a["weather_1d_temperature_2m_mean_present"] == 1


def test_migration_model_gradients_and_required_inputs():
    m = MovementTransformer(3, width=32, layers=1, n_drivers=4, driver_indices=[0, 2, 3, 5], n_anchors=5, student_df=4)
    x = torch.randn(4, 32, 10); dr = torch.randn(4, 6); anc = torch.randn(4, 5, 2)
    args = (x, torch.tensor([0, 1, 2, 3]), torch.tensor([1., 12., 24., 168.]), torch.randn(4, 2))
    with pytest.raises(ValueError, match="drivers"):
        m(*args)
    with pytest.raises(ValueError, match="anchors"):
        m(*args, dr)
    params = m(*args, dr, anc)
    loss = nll(params, torch.randn(4, 2)).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert sample(params, 64).shape == (4, 64, 2)


def test_stratified_energy_trains_mixture_weights_without_discrete_sampling():
    logits = torch.tensor([[0., 0.]], requires_grad=True)
    params = (logits, torch.tensor([[[0., 0.], [2., 0.]]]), torch.zeros(1, 2, 2), torch.zeros(1, 2))
    loss = mixture_energy_score(params, torch.tensor([[0., 0.]]))
    assert loss.item() == pytest.approx(.5)
    loss.mean().backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0


def test_device_batches_preserve_validation_order_and_sampler_indices():
    from wildcast.train import DeviceBatches
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.arange(10), torch.arange(10) * 2)
    values = list(DeviceBatches(ds, 4, "cpu"))
    torch.testing.assert_close(torch.cat([v[0] for v in values]), torch.arange(10))
    sampled = list(DeviceBatches(ds, 3, "cpu", sampler=[8, 1, 1, 6]))
    torch.testing.assert_close(torch.cat([v[0] for v in sampled]), torch.tensor([8, 1, 1, 6]))


def test_deployment_traits_do_not_use_future_measurements(tmp_path):
    from wildcast.traits import TraitLookup
    folder = tmp_path / "raw/source"; folder.mkdir(parents=True)
    (folder / "source.json").write_text(json.dumps({"metadata": {"dc.rights.uri": ["https://creativecommons.org/licenses/by/4.0/"]}}))
    (folder / "README.txt").write_text("animal mass: mass at deployment\nUnits: grams\n")
    pd.DataFrame([{"animal-id": "A", "deploy-on-date": "2020-01-01", "animal-mass": 10000, "animal-sex": "f", "animal-life-stage": "adult"},
                  {"animal-id": "A", "deploy-on-date": "2021-01-01", "animal-mass": 50000, "animal-sex": "f", "animal-life-stage": "adult"}]).to_csv(folder / "reference-data.csv", index=False)
    obs = pd.DataFrame({"animal": ["source:study:species:A"]})
    lookup = TraitLookup(tmp_path, tmp_path, obs, [])
    before = lookup.features(obs.animal[0], "species", pd.Timestamp("2019-01-01", tz="UTC").timestamp())
    after = lookup.features(obs.animal[0], "species", pd.Timestamp("2020-06-01", tz="UTC").timestamp())
    assert before["trait_capture_mass_present"] == 0
    assert after["trait_capture_log_mass_kg"] == pytest.approx(np.log1p(10))
    assert after["trait_sex_female"] == 1


def test_sparse_memory_never_reads_future_fixes():
    from wildcast.memory import sparse_memory
    lat = 10 + np.arange(300) * .001
    lon = 20 + np.arange(300) * .001
    t = np.arange(300) * 3600.
    a = sparse_memory(lat, lon, t, np.asarray([120]), np.asarray([100.]))
    lat[121:] = -80; lon[121:] = -170
    b = sparse_memory(lat, lon, t, np.asarray([120]), np.asarray([100.]))
    np.testing.assert_array_equal(a, b)
    assert a[0, 0, 0] == 1
    assert a[0, -1, 0] == 0
    assert np.all(a[0, -1] == 0)


def test_sparse_memory_attention_masks_missing_history():
    m = MovementTransformer(2, width=32, layers=1, n_drivers=4, n_anchors=5, memory_features=10, memory_length=25, student_df=4)
    x = torch.randn(3, 32, 10); dr = torch.randn(3, 4); anc = torch.randn(3, 5, 2)
    memory = torch.zeros(3, 25, 10); memory[:, 0, 0] = 1
    args = (x, torch.tensor([0, 1, 2]), torch.tensor([1., 24., 168.]), torch.zeros(3, 2), dr, anc)
    with pytest.raises(ValueError, match="memory"):
        m(*args)
    loss = nll(m(*args, memory), torch.randn(3, 2)).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_zero_gated_memory_preserves_pretrained_forecast():
    kwargs = dict(n_species=2, width=32, layers=1, n_drivers=4, n_anchors=5, student_df=4)
    base = MovementTransformer(**kwargs).eval()
    enriched = MovementTransformer(**kwargs, memory_features=10, memory_length=25, memory_fusion="residual").eval()
    enriched.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(3, 32, 10); dr = torch.randn(3, 4); anc = torch.randn(3, 5, 2)
    args = (x, torch.tensor([0, 1, 2]), torch.tensor([1., 24., 168.]), torch.zeros(3, 2), dr, anc)
    mem = torch.zeros(3, 25, 10); mem[:, 0, 0] = 1
    with torch.inference_mode():
        p = base(*args); q = enriched(*args, mem)
    for left, right in zip(p[:4], q[:4]):
        torch.testing.assert_close(left, right)


def test_horizon_and_displacement_regime_adapters_preserve_initial_forecast():
    kwargs = dict(n_species=2, width=32, layers=1, n_drivers=4, n_anchors=5, components=8, student_df=4)
    base = MovementTransformer(**kwargs).eval()
    adapted = MovementTransformer(**kwargs, horizon_adapter=True, regime_head=True).eval()
    adapted.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(4,32,10); dr = torch.randn(4,4); anc = torch.randn(4,5,2)
    args = (x,torch.tensor([1,2,1,2]),torch.tensor([1.,12.,24.,168.]),torch.randn(4,2),dr,anc)
    with torch.inference_mode():
        p,q = base(*args),adapted(*args)
    for left,right in zip(p[:4],q[:4]):torch.testing.assert_close(left,right)
    adapted.zero_grad();params=adapted(*args)
    loss=nll(params,torch.randn(4,2)).mean()+torch.nn.functional.cross_entropy(adapted.last_regime_logits,torch.tensor([0,1,2,1]))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in adapted.parameters() if p.grad is not None)
    assert adapted.regime_head.weight.grad.abs().sum()>0
