import math

import pytest
import torch

from wildcast.migration_model import (MigrationModel, VARIANTS, load_model, masked_joint_nll,
                                      mixture_energy, sample_paths)


HORIZONS = [1., 3., 7., 14., 30., 60., 90.]


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model(variant="residence_route"):
    return MigrationModel(10, 3, 3, HORIZONS, history_length=4, width=16, layers=1,
                          components=8, variant=variant, dropout=0.)


def inputs(batch=2):
    torch.manual_seed(19)
    return (torch.randn(batch, 4, 10), torch.ones(batch, 4, dtype=torch.bool),
            torch.randn(batch, 3), torch.arange(batch) % 3)


def test_exact_residence_prefix_and_monotone_destination_progress():
    m = model()
    raw = torch.zeros(2, 8, 8)
    raw[..., 0] = 1.
    mean, departure, duration = m._residence_means(raw, return_timing=True)
    assert torch.equal(mean[:, 0], torch.zeros_like(mean[:, 0]))
    assert torch.all(departure[:, 0] > 90) and torch.all(departure[:, 1] == 0)
    before = torch.tensor(HORIZONS)[None, None] <= departure[..., None]
    assert torch.equal(mean[before], torch.zeros_like(mean[before]))
    assert torch.all(mean[:, 1:, 1:, 0] >= mean[:, 1:, :-1, 0])
    torch.testing.assert_close(mean[:, 1:, -1, 0], torch.ones_like(mean[:, 1:, -1, 0]))
    assert torch.all(mean[:, 1, 0, 0] > 0)
    assert torch.all(duration > 0)


def test_already_moving_short_duration_and_delayed_timings_receive_gradients():
    m = model()
    raw = m.mean_head.bias.detach().reshape(1, 8, 8).clone()
    raw[..., 0] = 1.; raw.requires_grad_()
    mean, departure, duration = m._residence_means(raw, return_timing=True)
    assert duration[0, 1].detach().item() == pytest.approx(7., abs=1e-5)
    assert departure[0, 1] == 0
    loss = mean[:, 1, 0, 0].sum() + mean[:, 2:, :, 0].sum()
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[0, 1, 3].abs() > 0  # Already moving duration can learn.
    assert raw.grad[0, 2:, 2].abs().sum() > 0
    assert raw.grad[0, 2:, 3].abs().sum() > 0
    assert raw.grad[0, 0].abs().sum() == 0  # Residence mean is anchored by design.


def test_timing_inside_an_unobserved_gap_is_not_identified():
    m = model()
    raw = torch.zeros(1, 8, 8)
    raw[..., 0] = 1.
    minimum = 1 / 90
    departure = 45 / 90
    raw[0, 2, 2] = math.log((departure / (1 - minimum)) / (1 - departure / (1 - minimum)))
    fraction = (7 / 90 - minimum) / (1 - departure - minimum)
    raw[0, 2, 3] = math.log(fraction / (1 - fraction))
    raw.requires_grad_()
    mean = m._residence_means(raw)
    mean[0, 2, :, 0].sum().backward()
    # All available horizons are <=30 or >=60 days, while this mean departs
    # at45 and arrives at52. No observed mean constrains the timing inside.
    torch.testing.assert_close(raw.grad[0, 2, 2:4], torch.zeros(2), atol=0, rtol=0)
    assert raw.grad[0, 2, 0] > 0


def test_covariance_and_parameter_budget_match_goal_control_exactly():
    torch.manual_seed(7); goal = model("goal_route").eval()
    residence = model().eval(); residence.load_state_dict(goal.state_dict())
    assert sum(p.numel() for p in goal.parameters()) == sum(p.numel() for p in residence.parameters())
    assert goal.config["min_scale"] == residence.config["min_scale"] == .0005
    left, right = goal(*inputs()), residence(*inputs())
    torch.testing.assert_close(left["covariance"], right["covariance"], atol=0, rtol=0)
    torch.testing.assert_close(left["logits"], right["logits"], atol=0, rtol=0)
    assert torch.equal(right["means"][:, 0], torch.zeros_like(right["means"][:, 0]))


def test_joint_likelihood_sampling_energy_and_checkpoint_are_consistent(tmp_path):
    torch.manual_seed(8); m = model().eval(); x = inputs()
    parameters = m(*x)
    y, mask = torch.randn(2, 7, 2) * .1, torch.ones(2, 7, dtype=torch.bool)
    mask[:, 2] = False; y[~mask] = float("nan")
    loss = masked_joint_nll(parameters, y, mask).mean()
    energy = mixture_energy(parameters, y, mask, repeats=2, generator=torch.Generator().manual_seed(5))
    (loss + energy["endpoint_energy"].mean() + energy["path_energy"].mean()).backward()
    for prefix in ("history_encoder", "mean_head", "scale_head", "dependence_head", "logit_head"):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for name, p in m.named_parameters() if name.startswith(prefix)), prefix
    # Isolate the residence component: it still has uncertainty, rather than
    # pretending to be an exact stationary trajectory.
    chosen = {key: value[:1].detach().clone() for key, value in parameters.items()}
    chosen["logits"].fill_(-60); chosen["logits"][:, 0] = 60
    draws = sample_paths(chosen, 20000, torch.Generator().manual_seed(9))[0].flatten(-2)
    assert draws.abs().sum() > 0
    torch.testing.assert_close(torch.cov(draws.T), chosen["covariance"][0, 0], atol=.0007, rtol=.08)
    observed = torch.randn(1, 7, 2) * .1
    full = masked_joint_nll(chosen, observed, torch.ones(1, 7, dtype=torch.bool))
    expected = -torch.distributions.MultivariateNormal(chosen["means"][0, 0].flatten(),
                                                       chosen["covariance"][0, 0]).log_prob(observed.flatten())
    torch.testing.assert_close(full[0], expected)
    checkpoint = tmp_path / "residence.pt"
    torch.save({"model_kind": "wildcast_migration_joint", "config": m.config, "model": m.state_dict()}, checkpoint)
    loaded, _ = load_model(checkpoint)
    before = m.sample_paths(*x, 12, torch.Generator().manual_seed(4))
    after = loaded.sample_paths(*x, 12, torch.Generator().manual_seed(4))
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert "residence_route" in VARIANTS
