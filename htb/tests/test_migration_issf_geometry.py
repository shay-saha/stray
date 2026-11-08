"""Numerical parity for accelerated iSSF geometry and candidate covariates."""
import numpy as np
import pytest
import torch

from test_migration_issf import arrays, model
from wildcast import migration_issf as reference
from wildcast import migration_issf_accelerated as accelerated
from wildcast import migration_issf_geometry as device_geometry


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cell_quadrature_parity_at_regular_and_parameter_boundary_cases(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(2)
    parameters = np.asarray([[2., 500., .7, 70.], [.05, 10000., -.7, 100.],
                             [10000., 1., 0., 0.], [1., 1e-4, .5, 1.]])
    cells = np.asarray([100., 100., 1000., 1.])
    bearings = np.asarray([1.2, -.9, 2.4, .3]); known = np.asarray([True, False, True, True])
    expected = reference.gamma_vm_grid(parameters, cells, bearings, known, 17)
    result = device_geometry.gamma_vm_grid(parameters, torch.tensor(cells, device=device), bearings, known, 17)
    np.testing.assert_allclose(result.cpu(), expected, rtol=2e-12, atol=3e-9)
    np.testing.assert_allclose(result.exp().sum((-1, -2)).cpu(), 1., atol=2e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("variant", ["linear", "mlp"])
def test_complete_iSSF_distribution_and_feature_parity(variant):
    a = arrays(); m = model(variant, a).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, .03)
    ids = np.asarray([0, 1, 2, 3]); heads = np.asarray([0, 1, 2, 0])
    expected = m.distributions(a, ids, heads)
    fast = accelerated.ISSFShort(**m.config).cuda().eval()
    fast.load_state_dict(m.state_dict())
    result = fast.distributions(a, ids, heads)
    for actual, wanted in zip(result, expected):
        torch.testing.assert_close(actual.cpu(), wanted, rtol=3e-7, atol=3e-7)
    result[0].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in fast.parameters())
