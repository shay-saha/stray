"""Validation-only cross-fitted gating AFTER full specialist fine-tuning."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .full_moe import FullExpertGate, gate_inputs
from .model import sample
from .router import mixture_terms, weighted_energy
from .sources import dump
from .train import load_windows


def calibrate(root, checkpoint, output, identity_audit, samples=32, batch_size=256, steps=300,
              lr=.01, seed=2026, device="auto", table_only=False):
    from .evaluate import load_model
    from .identity import verify_audit
    if samples < 2:
        raise ValueError("At least two independent draw indices required")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(seed)
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest = load_windows(root)
    audit = json.loads(Path(identity_audit).read_text())
    verify_audit(audit, manifest, a)
    model, ck = load_model(checkpoint, device)
    if ck["model_family"] != "wildcast_full_network_moe" or ck["manifest"]["windows_sha256"] != manifest["windows_sha256"]:
        raise ValueError("Expected complete independently trained experts on this schema")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.flatnonzero(np.isin(a["meta"][:, 0], [1, 3]))
    ids = ids[~np.isin(ids, audit["excluded_windows"])]
    generator = torch.Generator(device=device).manual_seed(seed)
    terms_a, terms_d, inputs = [], [], []
    model.requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            take = ids[start:start + batch_size]
            global_draws = model.shared_draws(a, take, samples, device, generator)
            family_draws = sample(model.expert_parameters(a, take, device, specialist=True), samples, generator)
            accuracy, pair = mixture_terms(torch.stack([global_draws, family_draws], 1), torch.as_tensor(a["y"][take], device=device))
            terms_a.append(accuracy.cpu().numpy()); terms_d.append(pair.cpu().numpy())
            inputs.append(gate_inputs(a, take, model.config["gate_driver_indices"], device).cpu().numpy())
    accuracy, pair, features = np.concatenate(terms_a), np.concatenate(terms_d), np.concatenate(inputs)
    scales = a["aux"][ids, 8]
    distance = np.linalg.norm(a["y"][ids], axis=-1) * scales
    _, group, counts = np.unique(a["meta"][ids][:, [0, 2, 4]], axis=0, return_inverse=True, return_counts=True)
    denominators = np.asarray([max(distance[group == k].mean(), 1.) for k in range(len(counts))])
    weight = scales / denominators[group] / counts[group]
    weight *= len(ids) / len(counts)
    folds = np.asarray([model.folds[int(i)] for i in a["meta"][ids, 3]])
    family = np.asarray(model.config["species_family"])[a["meta"][ids, 1]]
    np.savez_compressed(out / "gate-bank.npz", ids=ids, accuracy=accuracy, pair=pair,
                        features=features, folds=folds, family=family, weights=weight)
    accuracy, pair, features = [torch.as_tensor(v, device=device) for v in [accuracy, pair, features]]
    weight = torch.as_tensor(weight, device=device, dtype=torch.float32)
    families = torch.as_tensor(family, device=device, dtype=torch.long)
    horizon = torch.as_tensor(a["meta"][ids, 4], device=device, dtype=torch.float32)
    states, probabilities, histories = [], np.zeros((len(ids), 2), np.float32), []
    for fold in range(3):
        fit = torch.as_tensor(np.flatnonzero(folds != fold), device=device)
        assessment = torch.as_tensor(np.flatnonzero(folds == fold), device=device)
        if not len(fit) or not len(assessment):
            raise ValueError("Every gate needs distinct fitting and held-out animals")
        gate = FullExpertGate(len(model.config["families"]), features.shape[-1], not table_only).to(device)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=.01)
        history = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            probability = gate(features[fit], families[fit], horizon[fit])
            es = weighted_energy(probability, accuracy[fit], pair[fit])
            loss = (es * weight[fit]).mean() + .001 * gate.logits.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite gate calibration")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step % 50 == 0 or step == steps - 1:
                history.append({"step": step + 1, "fit_loss": float(loss.detach())})
        with torch.inference_mode():
            probabilities[assessment.cpu().numpy()] = gate(features[assessment], families[assessment], horizon[assessment]).cpu().numpy()
        states.append(copy.deepcopy(gate.state_dict()))
        histories.append({"fold": fold, "fit_cases": len(fit), "heldout_cases": len(assessment), "history": history})
        print(f"Complete-expert gate fold {fold}: fitted on {len(fit)}, held out {len(assessment)}", flush=True)
    model.config = {**model.config, "routing": "gated", "conditional_gate": not table_only}
    model.gates = nn.ModuleList([FullExpertGate(len(model.config["families"]), features.shape[-1], not table_only).to(device) for _ in range(3)])
    for gate, state in zip(model.gates, states):
        gate.load_state_dict(state)
    training = {**ck["training"], "gate_steps": steps, "gate_lr": lr, "gate_seed": seed,
                "source_complete_moe_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                "gate_protocol": "Physical-animal 3-fold validation calibration; checkpoints already selected on validation. Development result, not independent test."}
    torch.save({**ck, "model": model.state_dict(), "config": model.config, "training": training}, out / "best.pt")
    with torch.inference_mode():
        scores = weighted_energy(torch.as_tensor(probabilities, device=device), accuracy, pair).cpu().numpy()
    report = {"output": str(out), "cases": len(ids), "folds": histories,
              "mean_specialist_weight": float(probabilities[:, 1].mean()),
              "crossfit_macro_energy_32draws": float(np.mean(scores * weight.cpu().numpy())),
              "scope": training["gate_protocol"],
              "source_complete_moe_sha256": training["source_complete_moe_sha256"]}
    dump(out / "gate-fit.json", report)
    return report
