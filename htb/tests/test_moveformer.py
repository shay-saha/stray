import numpy as np
import pytest
import torch

from wildcast.moveformer import LocalMoveFormer, candidate_features, load_proposal, proposal_records
from wildcast.baselines import CorrelatedRandomWalk


def inputs():
    return (torch.randn(3, 8, 10), torch.tensor([1, 2, 1]), torch.tensor([1., 24., 168.]),
            torch.randn(3, 2), torch.randn(3, 4))


def test_author_selection_logits_are_equivariant_to_candidate_permutation():
    m = LocalMoveFormer(2, n_drivers=4, width=32, layers=1, heads=2).eval()
    args = inputs(); q = torch.randn(3, 11, 2); order = torch.randperm(11)
    with torch.inference_mode():
        first, _ = m(*args, q); second, _ = m(*args, q[:, order])
    torch.testing.assert_close(first[:, order], second)


def test_author_forecasts_do_not_read_or_append_future_targets():
    from wildcast.evaluate import predictions
    y = np.asarray([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], np.float32)
    meta = np.zeros((4, 5)); meta[:, 1] = 1; meta[:, 4] = 1
    proposal = CorrelatedRandomWalk().fit(y, meta, np.ones((4, 2)))
    m = LocalMoveFormer(2, n_drivers=4, width=32, layers=1, heads=2,
                       proposal=proposal_records(proposal), inference_candidates=32).eval()
    a = {"x": np.random.default_rng(1).normal(size=(4, 8, 10)).astype(np.float32),
         "y": y.copy(), "meta": meta, "aux": np.zeros((4, 9)), "drivers": np.zeros((4, 4), np.float32)}
    with torch.inference_mode():
        first = predictions(m, a, np.arange(4), 16, "cpu", torch.Generator().manual_seed(123))
        a["y"][:] = 1e9
        second = predictions(m, a, np.arange(4), 16, "cpu", torch.Generator().manual_seed(123))
    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()


def test_moveformer_candidate_features_are_finite_for_rest_and_zero_velocity():
    out = candidate_features(torch.zeros(3, 7, 2), torch.zeros(3, 2))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_moveformer_checkpoint_roundtrip_uses_same_author_network(tmp_path):
    from wildcast.evaluate import load_model
    m = LocalMoveFormer(2, n_drivers=4, width=32, layers=1, heads=2).eval()
    path = tmp_path / "author.pt"
    torch.save({"model_family": m.family, "config": m.config, "model": m.state_dict()}, path)
    loaded, _ = load_model(path, "cpu")
    args = inputs(); q = torch.randn(3, 11, 2)
    with torch.inference_mode():
        torch.testing.assert_close(m(*args, q)[0], loaded(*args, q)[0])


def test_empirical_proposal_roundtrip_and_scale_without_invented_zero_atom():
    from wildcast.moveformer import EmpiricalStepProposal
    rng = np.random.default_rng(5)
    y = rng.normal(size=(100,2)).astype(np.float32)
    meta = np.zeros((100,5)); meta[:,1]=1; meta[:,4]=24
    p = EmpiricalStepProposal().fit(y,meta,np.ones((100,2)))
    q = load_proposal(proposal_records(p))
    first = p.sample(meta[:10],np.ones((10,2)),100,np.random.default_rng(7))
    second = q.sample(meta[:10],np.ones((10,2)),100,np.random.default_rng(7))
    np.testing.assert_array_equal(first,second)
    assert np.all(np.linalg.norm(first,axis=-1)>0)
    assert np.linalg.norm(first,axis=-1).max() <= np.linalg.norm(y,axis=-1).max()+1e-5


def test_variable_context_attention_has_finite_training_gradients():
    m = LocalMoveFormer(2,n_drivers=4,width=32,layers=1,heads=2,variable_context=True)
    _,loss = m(*inputs(),torch.randn(3,11,2))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
