import json

import numpy as np
import pandas as pd
import pytest
import torch

from wildcast.data import SPLITS, prepare, project, unproject
from wildcast.model import energy_score, nll, sample
from wildcast.sources import download


def test_geography_roundtrip_across_dateline_and_at_high_latitude():
    lat = np.asarray([1., 85., -72., 0.])
    lon = np.asarray([-179.9, -179.8, 179.9, 0.])
    lat0 = np.asarray([1., 84.9, -72.1, 0.])
    lon0 = np.asarray([179.9, 179.8, -179.9, 0.])
    xy = project(lat, lon, lat0, lon0)
    back = unproject(xy, lat0, lon0)
    np.testing.assert_allclose(back[:, 0], lat, atol=1e-8)
    np.testing.assert_allclose((back[:, 1] - lon + 180) % 360 - 180, 0, atol=1e-8)
    assert np.linalg.norm(xy[0]) < 23_000


def test_energy_score_matches_exact_discrete_two_sample_value():
    # Samples (0,0), (2,0), target (1,0): accuracy=1,
    # off-diagonal mean pair distance=2, so unbiased ES=0.
    x = torch.tensor([[[0., 0.], [2., 0.]]])
    y = torch.tensor([[1., 0.]])
    assert energy_score(x, y).item() == pytest.approx(0)
    assert energy_score(torch.zeros(1, 8, 2), torch.tensor([[3., 4.]])).item() == pytest.approx(5)
    with pytest.raises(ValueError):
        energy_score(x[:, :1], y)


def test_mixture_sampling_covariance_and_likelihood():
    p = (torch.zeros(1, 1), torch.tensor([[[2., -1.]]]), torch.tensor([[[3., 2.]]]), torch.tensor([[.6]]))
    draws = sample(p, 80_000, torch.Generator().manual_seed(1))[0].numpy()
    np.testing.assert_allclose(draws.mean(0), [2., -1.], atol=.04)
    np.testing.assert_allclose(np.cov(draws.T), [[9., 3.6], [3.6, 4.]], atol=.13)
    assert nll(p, torch.tensor([[2., -1.]])).item() < nll(p, torch.tensor([[20., 10.]])).item()


def test_split_raw_fixes_are_disjoint_and_individuals_stay_held_out(tmp_path):
    rows = []
    for animal in range(12):
        for i in range(300):
            rows.append({"animal": f"animal-{animal}", "study": "study-a", "source": "source-a", "species": "test species",
                         "segment": 1, "lat": 10 + animal * .01 + i * .00001, "lon": 20 + i * .00001,
                         "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=i)})
    f = pd.DataFrame(rows)
    processed = tmp_path / "processed"; processed.mkdir()
    f.to_parquet(processed / "observations.parquet", index=False)
    manifest = prepare(tmp_path, context=8, stride=3, horizons=[1, 12], seed=42)
    a = np.load(processed / "windows.npz")
    meta, aux = a["meta"], a["aux"]
    used = {}
    animals = {}
    for split, code in SPLITS.items():
        ix = np.flatnonzero(meta[:, 0] == code)
        assert len(ix), split
        fixes = set()
        for j in ix:
            origin, target = aux[j, 6:8].astype(int)
            fixes.update(range(origin - 7, origin + 1)); fixes.add(target)
        used[split] = fixes
        animals[split] = set(meta[ix, 3])
    for left in SPLITS:
        for right in SPLITS:
            if left != right:
                assert not used[left] & used[right], (left, right)
    for held in ["val_individual", "test_individual"]:
        for other in SPLITS:
            if held != other:
                assert not animals[held] & animals[other]
    assert np.all(a["aux"][:, 8] >= 25)


def test_download_refuses_bad_checksum_and_removes_partial_file(tmp_path):
    class Response:
        headers = {"Content-Length": "3"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, _): yield b"abc"
    class Session:
        def get(self, *args, **kwargs): return Response()
    out = tmp_path / "download.csv"
    with pytest.raises(ValueError, match="checksum"):
        download(Session(), "https://example.test/data", out, 100, ("md5", "incorrect"))
    assert not out.exists()
    assert not list(tmp_path.glob("*.part"))


def test_heavy_tails_do_not_give_extreme_moves_exploding_gaussian_loss():
    gaussian = (torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.ones(1, 1, 2), torch.zeros(1, 1))
    student = (*gaussian, 4)
    target = torch.tensor([[10000., 10000.]])
    assert torch.isfinite(nll(student, target)).all()
    assert nll(student, target).item() < 100
    assert nll(gaussian, target).item() > 1e6
    # Fixed generators reproduce the full multivariate Student draw, including
    # its shared radial multiplier (separate univariate t draws would be wrong).
    a = sample(student, 100, torch.Generator().manual_seed(7))
    b = sample(student, 100, torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)


def test_analogue_queries_are_rotation_equivariant():
    from wildcast.baselines import AnalogueForecaster, rotate
    rng = np.random.default_rng(42)
    xy = rng.normal(size=(100, 32, 2)).cumsum(1)
    xy -= xy[:, -1:]
    step = np.diff(xy, axis=1, prepend=xy[:, :1])
    x = np.zeros((100, 32, 10), dtype=np.float32)
    x[..., :2] = np.sign(xy) * np.log1p(np.abs(xy))
    x[..., 2:4] = np.sign(step) * np.log1p(np.abs(step))
    v = step[:, -1]
    y = rng.normal(size=(100, 2))
    meta = np.tile([0, 1, 0, 0, 12], (100, 1))
    model = AnalogueForecaster().fit(x, y, meta, v)
    angle = .73
    q = x[:8].copy()
    qxy, qstep = rotate(xy[:8], angle), rotate(step[:8], angle)
    q[..., :2] = np.sign(qxy) * np.log1p(np.abs(qxy))
    q[..., 2:4] = np.sign(qstep) * np.log1p(np.abs(qstep))
    a = model.sample(x[:8], meta[:8], v[:8], 32, np.random.default_rng(0))
    b = model.sample(q, meta[:8], rotate(v[:8], angle), 32, np.random.default_rng(0))
    np.testing.assert_allclose(b, rotate(a, angle), atol=1e-5)


def test_ensemble_samples_all_members_at_exact_requested_budget(monkeypatch):
    import wildcast.evaluate as evaluation
    def constant_draws(model, a, ix, count, device, generator):
        return torch.full((len(ix), count, 2), float(model))
    monkeypatch.setattr(evaluation, "predictions", constant_draws)
    draws = evaluation.ensemble_predictions([0, 1, 2], {}, np.arange(400), 64, "cpu", torch.Generator().manual_seed(2))
    assert draws.shape == (400, 64, 2)
    proportions = torch.bincount(draws[..., 0].long().flatten(), minlength=3) / (400 * 64)
    torch.testing.assert_close(proportions, torch.full((3,), 1 / 3), atol=.015, rtol=0)
