import hashlib
import json

import numpy as np
import pytest
import torch

from wildcast.full_moe import FullNetworkMoE
from wildcast.identity import case_fingerprint
from wildcast.model import MovementTransformer, nll, sample


def fixture_data():
    rng = np.random.default_rng(4)
    count = 48
    a = {"x": rng.normal(size=(count, 8, 10)).astype(np.float32),
         "y": rng.normal(size=(count, 2)).astype(np.float32),
         "meta": np.zeros((count, 5), np.int32), "aux": np.zeros((count, 9), np.float64),
         "drivers": rng.normal(size=(count, 4)).astype(np.float32),
         "anchors": rng.normal(size=(count, 5, 2)).astype(np.float32),
         "memory": rng.normal(size=(count, 3, 10)).astype(np.float32)}
    a["meta"][:, 0] = np.repeat([0, 1, 3], 16)
    a["meta"][:, 1] = 1 + np.arange(count) % 2
    a["meta"][:, 2] = np.arange(count) % 2
    a["meta"][:, 3] = np.arange(count) % 12
    a["meta"][:, 4] = np.tile([1, 12, 24, 168], 12)
    a["aux"][:, 8] = 1000
    a["memory"][:, :, 0] = 1
    base = MovementTransformer(2, width=32, layers=1, n_drivers=4, n_anchors=5,
                               memory_features=10, memory_length=3, context=8, student_df=4)
    config = {"families": ["shared", "Birdidae", "Catidae"], "species_family": [0, 1, 2],
              "experts": [base.config] * 3, "routing": "family", "gate_driver_indices": [0, 1],
              "conditional_gate": True, "animal_folds": {str(i): i % 3 for i in range(12)}}
    return a, config


def test_complete_experts_have_independent_storage_and_update_encoders():
    a, config = fixture_data()
    model = FullNetworkMoE(config).eval()
    # Equal initialization still must leave separate storage and optimizers.
    for expert in model.experts[1:]:
        expert.load_state_dict(model.experts[0].state_dict())
    assert len({expert.encoder.layers[0].linear1.weight.data_ptr() for expert in model.experts}) == 3
    ids = np.asarray([0, 2, 4])  # Birdidae only
    before = model.experts[1].encoder.layers[0].linear1.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.experts[1].parameters(), lr=.001)
    loss = nll(model.expert_parameters(a, ids, "cpu"), torch.from_numpy(a["y"][ids])).mean()
    loss.backward()
    for prefix in ["input.", "encoder.", "species.", "driver_encoder.", "memory_encoder.", "memory_attention.", "head."]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in model.experts[1].named_parameters() if name.startswith(prefix))
    assert all(p.grad is None for i in [0, 2] for p in model.experts[i].parameters())
    optimizer.step()
    assert not torch.equal(before, model.experts[1].encoder.layers[0].linear1.weight)


def test_complete_moe_routes_to_entire_model_and_ignores_future():
    a, config = fixture_data()
    model = FullNetworkMoE(config).eval()
    ids = np.asarray([0, 2, 4])
    with torch.inference_mode():
        routed = model.expert_parameters(a, ids, "cpu")
        direct = model.experts[1](torch.from_numpy(a["x"][ids]), torch.tensor(a["meta"][ids, 1]),
                                 torch.tensor(a["meta"][ids, 4], dtype=torch.float32), torch.tensor(a["aux"][ids, 4:6], dtype=torch.float32),
                                 torch.from_numpy(a["drivers"][ids]), torch.from_numpy(a["anchors"][ids]), torch.from_numpy(a["memory"][ids]))
        for left, right in zip(routed[:4], direct[:4]):
            torch.testing.assert_close(left, right)
        first = model.predict_samples(a, ids, 8, "cpu", torch.Generator().manual_seed(9))
        a["y"][:] = 1e10; a["aux"][:, [3, 7]] = 1e10
        second = model.predict_samples(a, ids, 8, "cpu", torch.Generator().manual_seed(9))
        torch.testing.assert_close(first, second)


def test_crossfit_routing_uses_heldout_gate_and_unknown_species_global():
    a, config = fixture_data()
    model = FullNetworkMoE({**config, "routing": "gated"}).eval()
    with torch.no_grad():
        for i, gate in enumerate(model.gates):
            gate.logits.fill_(i - 1)
    ids = np.arange(4)
    weights = model.weights(a, ids, "cpu")
    torch.testing.assert_close(weights[:, 1], torch.tensor([-1., 0., 1., -1.]).sigmoid())
    a["meta"][0, 1] = 0
    assert model.weights(a, ids, "cpu")[0, 1] == 0
    a["meta"][:, 3] = 99999
    torch.testing.assert_close(model.weights(a, ids, "cpu")[1:, 1], torch.full((3,), .5))


def test_full_training_assembly_and_gate_calibration(tmp_path):
    from wildcast.train import train
    from wildcast.full_moe import assemble
    from wildcast.full_moe_train import calibrate
    from wildcast.evaluate import load_model
    a, config = fixture_data()
    root = tmp_path / "data"; (root / "processed").mkdir(parents=True)
    np.savez_compressed(root / "processed/windows.npz", **a)
    manifest = {"species": {"Birdus test": 1, "Catus test": 2}, "animals": {f"animal{i}": i for i in range(12)},
                "studies": {"study0": 0, "study1": 1}, "context": 8, "n_anchors": 5, "protocol": "fixture",
                "driver_names": ["ecology_0", "ecology_1", "ecology_2", "ecology_3"], "observations_sha256": "fixture",
                "windows_sha256": hashlib.sha256((root / "processed/windows.npz").read_bytes()).hexdigest()}
    (root / "processed/windows.json").write_text(json.dumps(manifest))
    taxonomy = tmp_path / "taxonomy"; taxonomy.mkdir()
    for name, family, cls in [("Birdus test", "Birdidae", "Aves"), ("Catus test", "Catidae", "Mammalia")]:
        (taxonomy / (name.replace(" ", "_") + ".json")).write_text(json.dumps({"response": {"canonicalName": name, "family": family, "class": cls}}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"observations_sha256": "fixture", "windows_sha256": manifest["windows_sha256"],
                                "case_fingerprint": case_fingerprint(a), "excluded_windows": [], "fold_groups": {}}))
    arguments = dict(epochs=1, batch_size=8, width=32, layers=1, components=8, anchors=True,
                     long_memory=True, identity_audit=audit, taxonomy_root=taxonomy, device="cpu", stratified_energy=True)
    global_dir = tmp_path / "global"
    train(root, global_dir, **arguments)
    specialists = {}
    for family in ["Birdidae", "Catidae"]:
        destination = tmp_path / family
        train(root, destination, init_checkpoint=str(global_dir / "best-trained.pt"), scope_family=family, **arguments)
        proof = json.loads((destination / "parameter-change-audit.json").read_text())
        assert proof["selected_checkpoint_changed_tensors_by_module"]["encoder."] > 0
        specialists[family] = str(destination / "best-trained.pt")
    paths = tmp_path / "specialists.json"; paths.write_text(json.dumps(specialists))
    bundle = tmp_path / "bundle"
    result = assemble(root, global_dir / "best-trained.pt", paths, bundle, audit, taxonomy)
    assert result["complete_networks"] == 3
    calibrated = tmp_path / "calibrated"
    calibrate(root, bundle / "best.pt", calibrated, audit, samples=4, batch_size=8, steps=2, device="cpu")
    model, _ = load_model(calibrated / "best.pt", "cpu")
    with torch.inference_mode():
        draws = model.predict_samples(a, np.arange(4), 8, "cpu", torch.Generator().manual_seed(1))
    assert draws.shape == (4, 8, 2) and torch.isfinite(draws).all()
    from wildcast.full_moe_evaluate import evaluate
    experiment_file = tmp_path / "experiments.json"
    experiment_file.write_text(json.dumps({"whole-specialists": {"data": str(root), "checkpoints": [str(bundle / "best.pt")]}}))
    scores = evaluate(str(root), experiment_file, tmp_path / "comparison", audit, samples=4, batch_size=8, device="cpu")
    assert set(scores["whole-specialists"]) == {"val_temporal", "val_individual"}
    import pandas as pd
    measured = pd.read_parquet(tmp_path / "comparison/whole-specialists/per-window.parquet")
    assert len(measured) == 32 and (measured.point_error_m >= 0).all()
    # Preserve the strongest shared forecaster while the complete family models
    # specialize; calibrating the outer gate must not update either source.
    from wildcast.router import RoutedForecast, export_analogue
    from wildcast.baselines import AnalogueForecaster
    fit = a["meta"][:, 0] == 0
    analogue = AnalogueForecaster().fit(a["x"][fit], a["y"][fit], a["meta"][fit], a["aux"][fit, 4:6])
    prior_config = {"base_configs": [config["experts"][0]], "crossfit": True, "n_features": 23,
                    "driver_indices": [0, 1], "conditional": False, "animal_names": manifest["animals"], "fold_groups": {}}
    prior = RoutedForecast(prior_config, export_analogue(analogue)).eval()
    shared = tmp_path / "strong-shared.pt"
    torch.save({"model_family": prior.family, "config": prior.config, "analogue": prior.analogue_records,
                "model": prior.state_dict(), "manifest": manifest, "epoch": 1,
                "training": {"seed": 42, "parameters": sum(p.numel() for p in prior.parameters())}}, shared)
    shared_bundle = tmp_path / "shared-bundle"
    assemble(root, global_dir / "best-trained.pt", paths, shared_bundle, audit, taxonomy,
             shared_checkpoint=shared, shared_data=root)
    shared_calibrated = tmp_path / "shared-calibrated"
    calibrate(root, shared_bundle / "best.pt", shared_calibrated, audit, samples=4, batch_size=8, steps=2, device="cpu")
    retained, _ = load_model(shared_calibrated / "best.pt", "cpu")
    for key, value in prior.state_dict().items():
        torch.testing.assert_close(retained.shared_fallback.state_dict()[key], value, rtol=0, atol=0)
    unknown = {key: value.copy() for key, value in a.items()}
    unknown["meta"][:, 1] = 0
    with torch.inference_mode():
        direct = retained.shared_draws(unknown, np.arange(4), 8, "cpu", torch.Generator().manual_seed(11))
        mixed = retained.predict_samples(unknown, np.arange(4), 8, "cpu", torch.Generator().manual_seed(11))
    torch.testing.assert_close(direct, mixed)
    # Reject the exact failure mode from the previous experiment: epoch zero.
    ck = torch.load(specialists["Birdidae"], weights_only=True)
    ck["epoch"] = 0
    torch.save(ck, specialists["Birdidae"])
    with pytest.raises(ValueError, match="post-update"):
        assemble(root, global_dir / "best-trained.pt", paths, bundle, audit, taxonomy)
