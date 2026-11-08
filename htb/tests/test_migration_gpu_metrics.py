import numpy as np
import pytest
import torch

from wildcast import migration_gpu_metrics as backend
from wildcast.migration_evaluate import (forecast_latlon, score_paths,
                                        spherical_xyz)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("samples", [2, 64])
@pytest.mark.parametrize("scale", [0., .000005, .05])
def test_accelerated_scores_preserve_support_points_and_route_choices(monkeypatch, samples, scale):
    rng = np.random.default_rng(17)
    origin = np.array([[0., 179.999, 0.], [88., -179.999, 0.], [-60., 73., 0.]])
    paths = rng.normal(0, scale, (3, samples, 9, 2))
    target = forecast_latlon(rng.normal(0, .01, (3, 1, 9, 2)), origin)[:, 0]
    mask = rng.random((3, 9)) > .3
    mask[:, 0] = True
    target[~mask] = np.nan
    expected = score_paths(paths, target, mask, origin)
    actual_backend = backend.distribution_scores
    monkeypatch.setattr(backend, "distribution_scores",
                        lambda c, y, m, o, device: actual_backend(c, y, m, o, device="cpu"))
    actual = score_paths(paths, target, mask, origin, device="cuda")
    np.testing.assert_allclose(actual["endpoint_energy_km"], expected["endpoint_energy_km"], atol=1e-7, rtol=1e-9)
    # The legacy Earth-centred dot-product formula can lose centimetres for
    # repeated/near-repeated paths. The translated score avoids cancellation.
    np.testing.assert_allclose(actual["joint_chordal_energy_km"], expected["joint_chordal_energy_km"], atol=.0003, rtol=1e-8)
    for name in ["point_error_km", "representative_error_km", "radial_coverage50", "radial_coverage90",
                 "representative_index", "representative_latlon"]:
        np.testing.assert_array_equal(actual[name], expected[name])
    if scale == 0:
        truth = np.where(mask[..., None], target, origin[:, None, :2])
        squared = ((spherical_xyz(truth) - spherical_xyz(origin[:, :2])[:, None]) ** 2).sum(-1)
        analytic = np.sqrt((squared * mask).sum(-1) / mask.sum(-1))
        np.testing.assert_allclose(actual["joint_chordal_energy_km"], analytic, atol=1e-8, rtol=1e-10)
