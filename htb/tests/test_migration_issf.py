"""Density-ratio choices, isotropic missing-heading control, and matched support."""
import json
import math
import shutil

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from test_migration_deepssf import arrays, write_cache, single_cpu_thread, protocol, repaired_protocol
from wildcast import migration_deepssf as support
from wildcast import migration_issf as issf
from wildcast.data import unproject


def model(variant="linear", a=None):
    a = arrays() if a is None else a
    geometry = support.fit_geometry(a, window=17)
    proposal = issf.fit_proposal(a, geometry)
    return issf.ISSFShort(geometry, proposal, variant=variant, window=17, hidden=8)


def test_train_only_kernel_fit_is_invariant_to_poisoned_heldout_outcomes():
    a = arrays(); geometry = support.fit_geometry(a, window=17)
    first = issf.fit_proposal(a, geometry)
    changed = {k: v.copy() for k, v in a.items()}
    changed["y"][2:] = np.nan; changed["target_mask"][2:] = True
    assert issf.fit_proposal(changed, geometry) == first
    assert first["fit_split"] == "train"
    assert np.isfinite(first["parameters"]).all()
    a["history"][:] = 0
    no_headings = issf.fit_proposal(a, geometry)
    assert np.all(np.asarray(no_headings["parameters"])[..., 3] == 0)


def test_declared_cross_taxon_kernel_weights_preserve_global_ids():
    a = arrays(); a["individual"][1] = 10; a["y"][1, :3, 0] = .01
    geometry = support.fit_geometry(a, window=17, allowed_cross_taxon_exclusion_ids={10})
    with pytest.raises(ValueError, match="declared uncertainty"):
        issf.fit_proposal(a, geometry)
    first = issf.fit_proposal(a, geometry, allowed_cross_taxon_exclusion_ids={10})
    a["split"][2], a["individual"][2], a["species"][2], a["metric_species"][2] = 0, 10, 2, 1
    a["y"][2], a["target_mask"][2] = a["y"][1], a["target_mask"][1]
    second = issf.fit_proposal(a, geometry, allowed_cross_taxon_exclusion_ids={10})
    np.testing.assert_allclose(first["parameters"], second["parameters"], rtol=1e-10, atol=1e-10)
    assert a["individual"][:3].tolist() == [10, 10, 10]


def test_discrete_gamma_VM_kernel_is_normalized_positive_and_isotropic_without_heading():
    parameters = np.asarray([[2., 500., .7, 70.]])
    unknown = issf.gamma_vm_grid(parameters, [100.], [1.2], [False], 17)[0]
    uniform_angles = issf.gamma_vm_grid(np.asarray([[2., 500., 0., 0.]]), [100.], [0.], [True], 17)[0]
    np.testing.assert_allclose(unknown, uniform_angles, atol=1e-12)
    np.testing.assert_allclose(unknown, np.rot90(unknown), atol=1e-12)
    assert np.exp(unknown).sum() == pytest.approx(1., abs=1e-12)
    assert np.isfinite(unknown).all() and np.exp(unknown).min() > 0
    known = issf.gamma_vm_grid(parameters, [100.], [1.2], [True], 17)[0]
    assert not np.allclose(known, np.rot90(known))


@pytest.mark.parametrize("variant", ["linear", "mlp", "movement"])
def test_zero_utility_reproduces_proposal_and_uniform_conditional_choices(variant):
    a = arrays(); m = model(variant).eval()
    rows, heads, columns = support.observed_pairs(a, [0])
    log_p, log_q, _, _ = m.distributions(a, rows, heads)
    torch.testing.assert_close(log_p, log_q, rtol=0, atol=1e-8)
    loss, logits = issf.conditional_loss(m, a, rows, heads, columns, np.random.default_rng(17), negatives=100)
    torch.testing.assert_close(logits, torch.zeros_like(logits), rtol=0, atol=1e-8)
    torch.testing.assert_close(loss, torch.full_like(loss, math.log(101)), rtol=0, atol=1e-8)
    # Outside-window observations are appended, not filtered or copied as zero.
    a["y"][0, 0] = [10., 0.]
    outside, _ = issf.conditional_loss(m, a, [0], [0], [0], np.random.default_rng(4), negatives=4)
    assert outside.shape == (1,) and torch.isfinite(outside).all()


@pytest.mark.parametrize("variant", ["linear", "mlp"])
def test_no_previous_heading_cannot_create_an_artificial_bearing_zero_preference(variant):
    a = arrays(); a["history"][:] = 0
    m = model(variant, a).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, .03)
    log_p, log_q, _, _ = m.distributions(a, np.asarray([0]), np.asarray([0]))
    np.testing.assert_allclose(log_q[0].numpy(), np.rot90(log_q[0].numpy()), atol=1e-10)
    np.testing.assert_allclose(log_p[0].detach().numpy(), np.rot90(log_p[0].detach().numpy()), atol=1e-8)


def test_continuous_density_agrees_with_shared_geographic_cell_tail_law():
    grid = torch.full((3, 17, 17), -math.log(17**2), dtype=torch.float64)
    points = torch.tensor([[0., 0.], [5000., 0.], [math.pi * 6371008.8 / 2, 0.]], dtype=torch.float64)
    cell, tail = torch.full((3,), 100.), torch.full((3,), 1000.)
    nll, _ = support.mixture_nll(grid, points, cell, tail, geographic=True)
    density = issf.point_log_density(grid, points[:, None], cell, tail, geographic=True)[:, 0]
    torch.testing.assert_close(density, -nll, rtol=0, atol=1e-12)
    # Geographic Jacobians cancel exactly in density ratios used for choices.
    tilted = grid.clone(); tilted[:, 4, 8] += 2; tilted -= torch.logsumexp(tilted.flatten(1), -1)[:, None, None]
    ratio_plane = issf.point_log_density(tilted, points[:, None], cell, tail) - issf.point_log_density(grid, points[:, None], cell, tail)
    ratio_sphere = issf.point_log_density(tilted, points[:, None], cell, tail, True) - issf.point_log_density(grid, points[:, None], cell, tail, True)
    torch.testing.assert_close(ratio_plane, ratio_sphere, rtol=0, atol=1e-12)


def test_sampler_matches_integrated_nonuniform_cell_and_tail_probabilities():
    cell, scale, window = 100., 1000., 17
    grid = issf.gamma_vm_grid(np.asarray([[2., 500., .3, 1.]]), [cell], [.7], [True], window)
    draws = issf.sample_grid(grid, [cell], [scale], 60_000, np.random.default_rng(2))[0]
    half = window * cell / 4
    centres = (np.arange(window) - window//2) * cell
    overlap = np.maximum(0, np.minimum(centres + cell/2, half) - np.maximum(centres - cell/2, -half)) / cell
    raster_probability = np.sum(np.exp(grid[0]) * overlap[:, None] * overlap[None])
    z = 1-(1+support.MAX_RADIUS**2/(support.TAIL_DF*scale**2))**(-support.TAIL_DF/2)
    cdf = lambda r: (1-(1+r*r/(support.TAIL_DF*scale**2))**(-support.TAIL_DF/2))/z
    tail_probability = 4/math.pi * quad(lambda angle:cdf(half/math.cos(angle)),0,math.pi/4)[0]
    expected = .9*raster_probability + .1*tail_probability
    assert (np.abs(draws).max(-1) < half).mean() == pytest.approx(expected, abs=.006)
    assert np.linalg.norm(draws, axis=-1).max() < support.MAX_RADIUS


def test_predictors_ignore_all_future_keys_and_masked_history_poison():
    a = arrays(); m = model("mlp").eval()
    class Safe(dict):
        def __getitem__(self, key):
            if key in {"y", "target_mask", "target_time", "target_latlon"}:
                raise AssertionError("Predictor read a future label")
            return super().__getitem__(key)
    first = m.predict_endpoints(Safe(a), [2,3], samples=8, rng=np.random.default_rng(9))
    changed = {k:v.copy() for k,v in a.items()}
    changed["history"][~changed["history_mask"]] = np.nan
    for name in ("y", "target_time", "target_latlon"):
        changed[name][:] = np.inf
    second = m.predict_endpoints(Safe(changed), [2,3], samples=8, rng=np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="no 90-day route"):
        m.predict_cases(Safe(a), [2], 4)
    with pytest.raises(ValueError, match="TRAIN"):
        issf.conditional_loss(m,a,[2],[0],[0],np.random.default_rng(1))


def test_pooled_augmentation_changes_kernel_geometry_tail_footprint_and_utility(monkeypatch):
    a = arrays(); a["y"][0,:3,0] = .1
    m = model("linear", a)
    monkeypatch.setattr(torch,"rand",lambda count,device:torch.zeros(count,device=device))
    with torch.no_grad():
        m.species_weights.weight[0,0] = .2
        m.species_weights.weight[2,0] = -.5
    log_p, log_q, cell, tail = m.distributions(a,[1],[0])
    angle, known = issf.heading(a,np.asarray([1]))
    pooled = m.proposal_parameters[[0],[0]].numpy()
    expected = issf.gamma_vm_grid(pooled,m.cell_metres[[0],[0]].numpy(),angle,known,17)
    np.testing.assert_allclose(log_q.numpy(),expected,atol=1e-12)
    torch.testing.assert_close(cell,m.cell_metres[[0],[0]])
    torch.testing.assert_close(tail,m.tail_scale_metres[[0],[0]])
    feature = m._features(a,np.asarray([1]),np.asarray([0]),pooled,cell,angle,known)
    raw = log_q.flatten(1) + .2*feature[...,0]
    torch.testing.assert_close(log_p.flatten(1),torch.log_softmax(raw,-1),rtol=0,atol=1e-8)


@pytest.mark.parametrize("variant", ["linear", "mlp", "movement"])
@pytest.mark.parametrize("schema", [protocol, repaired_protocol])
def test_finite_synthetic_fit_and_checkpoint_reload(tmp_path, variant, schema):
    root = tmp_path/"data"; a = write_cache(root, schema)
    a["y"][1,:3,0] = [.002,.0001,.003]
    a["target_latlon"] = unproject(a["y"].astype(float)*1_000_000,0.,0.)
    a["target_latlon"][~a["target_mask"]] = 0
    np.savez_compressed(root/"cases.npz",**a)
    manifest=json.loads((root/"manifest.json").read_text());manifest["cases_sha256"]=support.sha256(root/"cases.npz")
    (root/"manifest.json").write_text(json.dumps(manifest))
    result=issf.train(root,tmp_path/"run",variant=variant,epochs=1,batch_size=8,window=17,hidden=8,validation_samples=4,negatives=8)
    assert math.isfinite(result["best_endpoint_energy_km"])
    m,ck=issf.load_model(tmp_path/"run/last.pt")
    assert ck["epoch"] == (0 if variant=="movement" else 1)
    assert ck["reserved_targets_selected"] is False
    assert ck["cache_protocol"] == schema.PROTOCOL
    assert ck["feature_schema"]["history_lags_days"] == schema.LAGS.tolist()
    assert ck["feature_schema"]["static_names"] == schema.STATIC_NAMES
    if schema is repaired_protocol:
        assert {"multiscale_repaired.py", "migration_physical_repair.py", "migration_repaired_features.py"} <= ck["fitting_code_sha256"].keys()
        assert a["history"].shape[1:] == (29, 10) and a["static"].shape[1] == 22
    assert ck["supported_horizons_days"] == [1/24,.5,1.]
    if variant!="movement":
        initial=torch.load(tmp_path/"run/initial.pt",weights_only=True)
        assert any(not torch.equal(initial["model"][key],value) for key,value in ck["model"].items() if "weight" in key)
    before=m.predict_endpoints(a,[2,3],samples=6,rng=np.random.default_rng(2))
    shutil.copy(tmp_path/"run/last.pt",tmp_path/"relocated.pt")
    relocated,_=issf.load_model(tmp_path/"relocated.pt")
    np.testing.assert_array_equal(before,relocated.predict_endpoints(a,[2,3],samples=6,rng=np.random.default_rng(2)))
    ck["model"]["proposal_parameters"] *= 2
    torch.save(ck,tmp_path/"wrong.pt")
    with pytest.raises(ValueError,match="sampling buffers"):
        issf.load_model(tmp_path/"wrong.pt")
