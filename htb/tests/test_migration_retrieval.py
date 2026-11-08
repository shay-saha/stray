import hashlib
import json

import numpy as np
import pytest
import torch

from wildcast.migration_baselines import MigrationBaselines
from wildcast.migration_evaluate import forecast_latlon, great_circle
from wildcast.migration_retrieval_train import (RetrievalReweighter, discrete_energy,
    generate_candidates, load_bank, load_model, prepare_bank, route_costs, train_bank)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def corpus():
    n, h = 10, np.array([1., 3., 7.], "float32")
    a = {"history": np.zeros((n, 4, 10), "float32"), "history_mask": np.ones((n, 4), bool),
         "static": np.zeros((n, 14), "float32"), "species": np.ones(n, "int64"),
         "y": np.zeros((n, 3, 2), "float32"), "target_mask": np.ones((n, 3), bool),
         "split": np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4], "int8"),
         "individual": np.array([10, 10, 11, 11, 12, 12, 20, 10, 30, 10], "int64"),
         "study": np.zeros(n, "int64"), "horizons": h}
    a["origin"] = np.column_stack((np.zeros(n), np.zeros(n), np.array([1000, 1100, 1000, 1100, 1000, 1100, 1500, 1500, 2000, 2000]) * 86400.))
    a["target_time"] = a["origin"][:, 2:3] + h * 86400.
    a["history"][:, :, 2] = [0., 1., 3., 7.]
    a["history"][:, :, 0] = [0., -.01, -.03, -.07]
    for i in range(n):
        a["y"][i, :, 0] = h * (.005 + .005 * (i % 3))
        a["y"][i, :, 1] = h * (.002 * (i % 2))
    a["target_mask"][2, 1] = False
    a["target_latlon"] = forecast_latlon(a["y"][:, None], a["origin"])[:, 0]
    a["target_mask"][a["split"] >= 3] = False
    a["y"][a["split"] >= 3] = np.nan
    a["target_latlon"][~a["target_mask"]] = np.nan
    return a


def write_data(root, a):
    root.mkdir()
    np.savez_compressed(root / "cases.npz", **a)
    (root / "manifest.json").write_text(json.dumps({"cases_sha256": hashlib.sha256((root / "cases.npz").read_bytes()).hexdigest(),
        "species": {"Animalus test": 1}, "reserved_test_targets_sealed": True}))


def test_candidates_ignore_query_future_and_entire_training_identity():
    a = corpus()
    ids = np.array([0, 6, 7])
    paths, query, features, info = generate_candidates(a, ids, 12, 42)
    assert not np.isin(info["donor_ids"][0], [0, 1]).any()
    end = np.where(a["target_mask"], a["target_time"], -np.inf).max(-1)
    assert np.all(end[info["donor_ids"][2]] < a["origin"][7, 2])
    altered = {k: v.copy() for k, v in a.items()}
    # Entire training identity is removed even from the fallback diffusion fit.
    altered["y"][[0, 1]] = 1e6
    p, q, f, _ = generate_candidates(altered, np.array([0]), 12, 42)
    np.testing.assert_array_equal(p[0], paths[0]); np.testing.assert_array_equal(q[0], query[0]); np.testing.assert_array_equal(f[0], features[0])
    # Neither validation targets nor their observation masks influence inputs.
    altered = {k: v.copy() for k, v in a.items()}
    altered["y"][6:] = 1e6; altered["target_latlon"][6:] = -80.; altered["target_mask"][6:] = False
    p, q, f, _ = generate_candidates(altered, ids[1:], 12, 42)
    np.testing.assert_array_equal(p, paths[1:]); np.testing.assert_array_equal(q, query[1:]); np.testing.assert_array_equal(f, features[1:])


def test_candidate_support_does_not_depend_on_batching():
    a = corpus(); ids = np.array([0, 2, 6, 7])
    whole = generate_candidates(a, ids, 8, 123)[0]
    separate = np.concatenate([generate_candidates(a, np.array([i]), 8, 123)[0] for i in ids])
    np.testing.assert_array_equal(whole, separate)


def test_zero_head_preserves_exact_empirical_prior_and_gradients_reweight():
    m = RetrievalReweighter(4, 6, 3, candidates=8, hidden=16, dropout=0.)
    q, c, s = torch.randn(3, 4), torch.randn(3, 8, 6), torch.ones(3, dtype=torch.long)
    weights = m(q, c, s)
    torch.testing.assert_close(weights, torch.full_like(weights, 1 / 8), atol=0, rtol=0)
    # Candidate cost, not true-future coordinates, drives residual weights.
    cost = torch.arange(8.).expand(3, -1)
    (weights * cost).sum().backward()
    assert m.network[-1].weight.grad.abs().sum() > 0
    opt = torch.optim.Adam(m.parameters(), lr=.01); opt.step()
    assert not torch.equal(m(q, c, s), weights)


def test_unknown_species_dropout_trains_shared_fallback_once():
    m = RetrievalReweighter(4, 6, 3, candidates=8, hidden=16, dropout=0., species_dropout=1.)
    # The first residual update opens the path to the species embedding.
    torch.nn.init.normal_(m.network[-1].weight, std=.1)
    q, c, s = torch.randn(3, 4), torch.randn(3, 8, 6), torch.full((3,), 2, dtype=torch.long)
    (m(q, c, s) * torch.arange(8.)).sum().backward()
    assert m.species.weight.grad[0].abs().sum() > 0
    assert m.species.weight.grad[1:].abs().sum() == 0


def test_exact_discrete_energy_matches_two_atom_formula_and_missing_masks():
    paths = torch.tensor([[[[.1, 0.], [.2, 0.]], [[-.1, 0.], [-.2, 0.]]]], dtype=torch.float64)
    origin = torch.tensor([[0., 179.9, 0.]], dtype=torch.float64)
    target = torch.tensor([[[0., 179.9], [float("nan"), float("nan")]]], dtype=torch.float64)
    mask = torch.tensor([[True, False]])
    costs = route_costs(paths, origin, target, mask)
    w = torch.tensor([[.5, .5]], dtype=torch.float64, requires_grad=True)
    scores = discrete_energy(w, costs, mask)
    assert scores["endpoint_energy_km"].item() == pytest.approx(50., abs=1e-8)
    expected = .5 * (costs["joint_first"][0, 0] + costs["joint_first"][0, 1]) - .25 * costs["joint_pair"][0, 0, 1]
    torch.testing.assert_close(scores["joint_energy_km"][0], expected)
    scores["joint_energy_km"].sum().backward()
    assert torch.isfinite(w.grad).all()


def test_bank_integrity_training_roundtrip_and_no_reserved_cases(tmp_path):
    data = tmp_path / "data"; write_data(data, corpus())
    bank = tmp_path / "bank"
    summary = prepare_bank(data, bank, candidates=8, seed=33)
    assert summary["reserved_test_cases_prepared"] == 0
    a, _, _, b, _ = load_bank(data, bank)
    assert set(a["split"][b["case_ids"]]) == {0, 1, 2}
    output = tmp_path / "fit"
    result = train_bank(data, bank, output, epochs=1, hidden=16, batch_size=4, device="cpu", dropout=0.)
    assert result["epochs_completed"] == 1
    initial, ck = load_model(output / "initial.pt")
    pred = initial.predict_cases(a, np.array([6, 7]), 5, np.random.default_rng(3))
    assert pred.shape == (2, 5, 3, 2) and np.isfinite(pred).all()
    first = initial.predict_cases(a, np.array([6]), 5, np.random.default_rng(3))
    a["target_latlon"][6] = np.nan; a["y"][6] = 1e10
    second = initial.predict_cases(a, np.array([6]), 5, np.random.default_rng(3))
    np.testing.assert_array_equal(first, second)
    history = json.loads((output / "history.json").read_text())
    assert history[0]["epoch"] == 0 and history[0]["splits"]["val_individual"]["mean_kl_from_prior"] == 0
    assert ck["bank_sha256"] == summary["bank_sha256"]
    with (bank / "bank.npz").open("ab") as stream: stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_bank(data, bank)
