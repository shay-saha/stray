"""Mixture of independently fine-tuned COMPLETE movement transformers.

Each specialist owns its input/history/memory/species/driver/head weights. No
encoder tensors are shared by reference. Gate calibration is a later stage,
after all specialists have been trained; it does not replace expert training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import MovementTransformer, sample
from .moe import taxonomy_groups
from .sources import dump


def identity_fold(name):
    return int(hashlib.sha256(("full-moe-v1:" + name).encode()).hexdigest()[:8], 16) % 3


def gate_inputs(a, ids, driver_indices, device):
    drivers = torch.as_tensor(a["drivers"][ids][:, driver_indices], device=device)
    x = torch.as_tensor(a["x"][ids], device=device)
    h = torch.as_tensor(a["meta"][ids, 4], device=device, dtype=torch.float32)
    return torch.cat([drivers, x[:, -1], x.mean(1), h.log1p()[:, None]], -1)


class FullExpertGate(nn.Module):
    def __init__(self, families, features, conditional=True):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(families, 4))
        self.conditional = conditional
        if conditional:
            self.network = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, family, horizon):
        anchors = torch.tensor([1., 12., 24., 168.], device=features.device)
        slot = (horizon[:, None] - anchors).abs().argmin(-1)
        score = self.logits[family, slot]
        if self.conditional:
            score = score + self.network(features).squeeze(-1).float()
        p = score.sigmoid()
        p = torch.where(family > 0, p, torch.zeros_like(p))
        return torch.stack([1 - p, p], -1)


class FullNetworkMoE(nn.Module):
    family = "wildcast_full_network_moe"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([MovementTransformer(**c) for c in config["experts"]])
        self.register_buffer("species_family", torch.tensor(config["species_family"], dtype=torch.long))
        self.gates = nn.ModuleList([FullExpertGate(len(config["families"]), len(config.get("gate_driver_indices", [])) + 21,
                                                 config.get("conditional_gate", True)) for _ in range(3)])
        self.folds = {int(k): int(v) for k, v in config.get("animal_folds", {}).items()}
        self.shared_fallback = None
        if config.get("shared_fallback"):
            from .router import RoutedForecast
            fallback = config["shared_fallback"]
            self.shared_fallback = RoutedForecast(fallback["config"], fallback["analogue"])

    def weights(self, a, ids, device):
        species = torch.as_tensor(a["meta"][ids, 1], device=device, dtype=torch.long)
        family = self.species_family[species]
        if self.config.get("routing", "family") == "family":
            specialist = (family > 0).float()
            return torch.stack([1 - specialist, specialist], -1)
        features = gate_inputs(a, ids, self.config["gate_driver_indices"], device)
        horizon = torch.as_tensor(a["meta"][ids, 4], device=device, dtype=torch.float32)
        candidates = torch.stack([gate(features, family, horizon) for gate in self.gates], 1)
        fold = torch.as_tensor([self.folds.get(int(i), -1) for i in a["meta"][ids, 3]], device=device)
        heldout = candidates[torch.arange(len(ids), device=device), fold.clamp_min(0)]
        return torch.where((fold >= 0)[:, None], heldout, candidates.mean(1))

    def expert_parameters(self, a, ids, device, specialist=True):
        x = torch.as_tensor(a["x"][ids], device=device)
        species = torch.as_tensor(a["meta"][ids, 1], device=device, dtype=torch.long)
        horizon = torch.as_tensor(a["meta"][ids, 4], device=device, dtype=torch.float32)
        velocity = torch.as_tensor(a["aux"][ids, 4:6], device=device, dtype=torch.float32)
        drivers = torch.as_tensor(a["drivers"][ids], device=device)
        anchors = torch.as_tensor(a["anchors"][ids], device=device)
        memory = torch.as_tensor(a["memory"][ids], device=device)
        family = self.species_family[species] if specialist else torch.zeros_like(species)
        parameters = None
        degree = None
        for i, expert in enumerate(self.experts):
            take = torch.nonzero(family == i, as_tuple=True)[0]
            if not len(take):
                continue
            values = expert(x[take], species[take], horizon[take], velocity[take], drivers[take], anchors[take], memory[take])
            if parameters is None:
                parameters = [torch.empty((len(ids), *v.shape[1:]), device=device, dtype=torch.float32) for v in values[:4]]
                degree = values[4] if len(values) == 5 else None
            for dest, source in zip(parameters, values[:4]):
                dest[take] = source.float()
        return (*parameters, degree) if degree else tuple(parameters)

    def shared_draws(self, a, ids, count, device, generator):
        if self.shared_fallback is not None:
            return self.shared_fallback.predict_samples(a, ids, count, device, generator)
        return sample(self.expert_parameters(a, ids, device, specialist=False), count, generator)

    def predict_samples(self, a, ids, count, device, generator):
        if self.config.get("routing", "family") == "family":
            return sample(self.expert_parameters(a, ids, device), count, generator)
        weights = self.weights(a, ids, device)
        if self.shared_fallback is not None:
            shared = self.shared_draws(a, ids, count, device, generator)
            specialist = sample(self.expert_parameters(a, ids, device, specialist=True), count, generator)
            choose_specialist = torch.rand((len(ids), count), device=device, generator=generator) < weights[:, 1:2]
            return torch.where(choose_specialist[..., None], specialist, shared)
        global_params = self.expert_parameters(a, ids, device, specialist=False)
        family_params = self.expert_parameters(a, ids, device, specialist=True)
        if len(global_params) != len(family_params) or (len(global_params) == 5 and global_params[4] != family_params[4]):
            raise ValueError("Whole-model experts use incompatible forecast distributions")
        logits = torch.cat([global_params[0].log_softmax(-1) + weights[:, :1].clamp_min(1e-30).log(),
                            family_params[0].log_softmax(-1) + weights[:, 1:].clamp_min(1e-30).log()], -1)
        values = [logits, *[torch.cat([global_params[j], family_params[j]], 1) for j in [1, 2, 3]]]
        return sample((*values, global_params[4]) if len(global_params) == 5 else tuple(values), count, generator)


def assemble(root, global_checkpoint, specialists, output, identity_audit, taxonomy_root="data/combined-v2/environment/taxonomy",
             shared_checkpoint=None, shared_data=None):
    from .train import load_windows
    from .identity import verify_audit
    a, manifest = load_windows(root)
    audit = json.loads(Path(identity_audit).read_text())
    verify_audit(audit, manifest, a)
    grouping = taxonomy_groups(manifest, taxonomy_root)
    paths = {None: global_checkpoint, **json.loads(Path(specialists).read_text())}
    loaded = []
    records = []
    for family in [None, *grouping["families"][1:]]:
        path = Path(paths[family])
        ck = torch.load(path, map_location="cpu", weights_only=True)
        training = ck["training"]
        if ck["manifest"]["windows_sha256"] != manifest["windows_sha256"]:
            raise ValueError("Complete experts use different data schemas")
        if ck["epoch"] < 1 or training.get("freeze_backbone") or training["parameters"] != training["trainable_parameters"]:
            raise ValueError("Every complete expert must be post-update and fully trainable")
        if training.get("scope_family") != family:
            raise ValueError("Expert checkpoint does not match its declared family")
        grad = json.loads((path.parent / "gradient-audit.json").read_text())
        change = json.loads((path.parent / "parameter-change-audit.json").read_text())
        for key in ["input.", "encoder.", "memory_encoder.", "memory_attention.", "head."]:
            if not grad[key]["all_trainable"] or not grad[key]["nonzero_gradient_tensors"] or not change["changed_tensors_by_module"][key]:
                raise ValueError("Missing evidence of whole-encoder training: " + key)
            selected_changes = change.get("selected_checkpoint_changed_tensors_by_module")
            if selected_changes is not None and not selected_changes[key]:
                raise ValueError("Selected checkpoint has unchanged encoder weights: " + key)
        loaded.append(ck)
        records.append({"family": family or "global", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "epoch": ck["epoch"], "training_cases": training["training_cases"],
                        "trainable_parameters": training["trainable_parameters"], "gradient_audit": grad, "parameter_change_audit": change})
    animal_folds = {}
    for name, i in manifest["animals"].items():
        animal_folds[str(i)] = identity_fold(audit["fold_groups"].get(name, name))
    selected_drivers = [i for i, name in enumerate(manifest["driver_names"]) if name.startswith("ecology_") or
                        name.startswith(("photoperiod", "season_", "weather_1d_", "weather_7d_", "vegetation_"))]
    config = {**grouping, "experts": [ck["config"] for ck in loaded], "routing": "family",
              "gate_driver_indices": selected_drivers, "animal_folds": animal_folds, "conditional_gate": True}
    fallback_record = None
    if shared_checkpoint:
        from .identity import case_fingerprint
        if not shared_data:
            raise ValueError("A shared fallback requires its exact source dataset")
        prior_data, prior_manifest = load_windows(shared_data)
        prior = torch.load(shared_checkpoint, map_location="cpu", weights_only=True)
        if prior.get("model_family") != "wildcast_routed_ensemble" or not prior["config"].get("crossfit"):
            raise ValueError("Shared fallback must be an audited cross-fitted routed model")
        if prior["manifest"]["windows_sha256"] != prior_manifest["windows_sha256"] or prior_manifest["species"] != manifest["species"]:
            raise ValueError("Shared fallback source schema differs")
        if case_fingerprint(prior_data) != case_fingerprint(a):
            raise ValueError("Shared fallback uses other forecast cases")
        for key in ["anchors", "memory"]:
            if not np.array_equal(prior_data[key], a[key]):
                raise ValueError("Shared fallback observed context changed")
        if not np.array_equal(prior_data["drivers"], a["drivers"][:, :prior_data["drivers"].shape[1]]):
            raise ValueError("Shared fallback driver prefix changed")
        if prior["config"].get("fold_groups", {}) != audit["fold_groups"]:
            raise ValueError("Shared fallback physical identity groups differ")
        config["shared_fallback"] = {"config": prior["config"], "analogue": prior["analogue"]}
        fallback_record = {"path": str(shared_checkpoint), "sha256": hashlib.sha256(Path(shared_checkpoint).read_bytes()).hexdigest(),
                           "feature_prefix_verified": True, "family": prior["model_family"]}
    model = FullNetworkMoE(config)
    for expert, ck in zip(model.experts, loaded):
        expert.load_state_dict(ck["model"])
    if fallback_record:
        model.shared_fallback.load_state_dict(prior["model"])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    training = {"seed": 42, "parameters": sum(p.numel() for p in model.parameters()), "expert_records": records,
                "scope": "Independently fine-tuned complete networks. Family validation selects each post-update checkpoint; no fresh-test selection.",
                "identity_audit_sha256": hashlib.sha256(Path(identity_audit).read_bytes()).hexdigest(),
                "shared_fallback": fallback_record}
    torch.save({"model_family": model.family, "model": model.state_dict(), "config": config, "training": training,
                "manifest": manifest, "epoch": 1, "val_score": None}, out / "best.pt")
    dump(out / "expert-audits.json", training)
    return {"output": str(out), "complete_networks": len(loaded), "parameters": training["parameters"]}
