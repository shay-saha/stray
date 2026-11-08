import math

import pytest
import torch

from wildcast.migration_model import MigrationModel, load_model, masked_joint_nll, sample_paths


@pytest.fixture(autouse=True)
def few_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model(**kwargs):
    return MigrationModel(10, 14, 2, [1 / 24, .5, 1, 3, 7, 14, 30, 60, 90],
                          history_length=3, width=16, layers=1, components=2,
                          variant="seasonal_static", dropout=0, **kwargs).eval()


def inputs(n=2):
    return (torch.zeros(n, 3, 10), torch.ones(n, 3, dtype=torch.bool),
            torch.zeros(n, 14), torch.ones(n, dtype=torch.long))


def test_historical_default_is_bitwise_unchanged_and_config_compatible():
    torch.manual_seed(17)
    old = model()
    torch.manual_seed(17)
    explicit = model(covariance_jitter=1e-8)
    assert "covariance_jitter" not in old.config
    left, right = old(*inputs()), explicit(*inputs())
    for name in left:
        assert torch.equal(left[name], right[name])
    a = sample_paths(left, 64, torch.Generator().manual_seed(11))
    b = sample_paths(right, 64, torch.Generator().manual_seed(11))
    assert torch.equal(a, b)


def test_subdaily_noise_floor_has_declared_physical_size():
    m = model(min_scale=5e-6, covariance_jitter=1e-12)
    with torch.no_grad():
        m.mean_head.weight.zero_(); m.mean_head.bias.zero_()
        m.scale_head.weight.zero_(); m.scale_head.bias.fill_(-100.)
        m.dependence_head.weight.zero_(); m.dependence_head.bias.zero_()
    parameters = m(*inputs(1))
    expected_metres = math.sqrt((5e-6) ** 2 + 1e-12) * 1_000_000
    assert expected_metres == pytest.approx(5.0990195)
    draws = sample_paths(parameters, 40000, torch.Generator().manual_seed(91)).detach().numpy()
    assert draws[0, :, 0].std(0) * 1_000_000 == pytest.approx([expected_metres] * 2, rel=.015)
    assert parameters["covariance"][0, 0, 0, 0].item() == pytest.approx(2.6e-11, rel=1e-6)


def test_small_scale_missing_marginal_nll_and_gradients_are_finite():
    m = model(min_scale=5e-6, covariance_jitter=1e-12)
    with torch.no_grad():
        m.scale_head.weight.zero_(); m.scale_head.bias.fill_(math.log(math.expm1(1e-5)))
    parameters = m(*inputs())
    target = parameters["means"][:, 0].detach().clone()
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[:, [0, 2, 8]] = True
    target[~mask] = torch.nan
    loss = masked_joint_nll(parameters, target, mask).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.scale_head.bias.grad.abs().sum() > 0


def test_explicit_precision_roundtrips_in_checkpoint(tmp_path):
    m = model(min_scale=5e-6, covariance_jitter=1e-12)
    path = tmp_path / "best.pt"
    torch.save({"model_kind": "wildcast_migration_joint", "config": m.config,
                "model": m.state_dict()}, path)
    restored, _ = load_model(path)
    assert restored.config == m.config
    for name, value in m(*inputs()).items():
        assert torch.equal(value, restored(*inputs())[name])


@pytest.mark.parametrize("kwargs", [{"min_scale": 0}, {"min_scale": float("nan")},
                                    {"covariance_jitter": -1}, {"covariance_jitter": float("inf")}])
def test_invalid_numerical_floors_are_rejected(kwargs):
    with pytest.raises(ValueError, match="finite and positive"):
        model(**kwargs)
