"""Family-specialized probabilistic experts around a frozen movement ensemble.

This is an endpoint MoE, not a language-model token router. Specialist heads
learn different forecast distributions while retaining shared history/memory
encoders. Learned routing activates two same-class experts (including a shared
fallback); fixed family routing activates exactly one. Species zero falls back
to the shared expert. Expert parameters never average endpoint coordinates.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import MovementTransformer
from .sources import dump


def taxonomy_groups(manifest, taxonomy_root):
    records, sources = {}, {}
    for name, species in manifest["species"].items():
        p = Path(taxonomy_root) / (name.replace(" ", "_") + ".json")
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        taxon = r.get("response", {})
        if taxon.get("canonicalName") != name:
            raise ValueError("Taxonomy does not exactly match the trajectory species")
        if taxon.get("family") and taxon.get("class"):
            records[species] = taxon
            sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    families = ["shared", *sorted({r["family"] for r in records.values()})]
    classes = ["unknown", *sorted({r["class"] for r in records.values()})]
    species_family = [0] * (len(manifest["species"]) + 1)
    species_class = [0] * len(species_family)
    family_class = [0] * len(families)
    for species, r in records.items():
        f, c = families.index(r["family"]), classes.index(r["class"])
        species_family[species], species_class[species], family_class[f] = f, c, c
    return {"families": families, "classes": classes, "species_family": species_family,
            "species_class": species_class, "family_class": family_class,
            "taxonomy_sources_sha256": sources}


def pack_ensemble(params):
    if not params or any(len(p) != 5 or p[4] != params[0][4] for p in params):
        raise ValueError("Endpoint ensemble requires a common Student-t degree of freedom")
    logits = torch.cat([p[0].float().log_softmax(-1) - math.log(len(params)) for p in params], 1)
    mu, sigma, rho = [torch.cat([p[k].float() for p in params], 1) for k in (1, 2, 3)]
    return torch.cat([logits[..., None], mu, sigma, rho[..., None]], -1)


class CalibrationExpert(nn.Module):
    def __init__(self, components):
        super().__init__()
        self.adjustment = nn.Parameter(torch.zeros(components, 3))

    def forward(self, features):
        z = torch.zeros_like(self.adjustment[:, :1])
        # Category-specific mixture weights and marginal uncertainty only;
        # preserve conditional means and correlations from the backbone.
        delta = torch.cat([self.adjustment[:, :1], z, z, self.adjustment[:, 1:3], z], -1)
        return delta.flatten()[None].expand(len(features), -1)


class CategoryMoE(nn.Module):
    family = "wildcast_category_moe"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bases = nn.ModuleList([MovementTransformer(**c) for c in config["base_configs"]])
        for p in self.bases.parameters():
            p.requires_grad_(False)
        self.primary = config.get("primary", 1)
        self.components = sum(c["components"] for c in config["base_configs"])
        self.student_df = config["base_configs"][0]["student_df"]
        self.n_features = self.bases[self.primary].head[0].normalized_shape[0] + config["n_drivers"]
        self.mode = config.get("mode", "family")
        count = len(config["families"]) if self.mode != "shared" else 1
        hidden = config.get("hidden", 96)
        self.experts = nn.ModuleList([CalibrationExpert(self.components) if hidden == 0 else
                    nn.Sequential(nn.LayerNorm(self.n_features), nn.Linear(self.n_features, hidden),
                    nn.GELU(), nn.Linear(hidden, self.components * 6)) for _ in range(count)])
        for head in self.experts:
            if isinstance(head, nn.Sequential):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        self.register_buffer("species_family", torch.tensor(config["species_family"], dtype=torch.long))
        self.register_buffer("species_class", torch.tensor(config["species_class"], dtype=torch.long))
        self.register_buffer("family_class", torch.tensor(config["family_class"], dtype=torch.long))
        if self.mode == "learned":
            self.gate = nn.Sequential(nn.LayerNorm(self.n_features), nn.Linear(self.n_features, 48), nn.GELU(), nn.Linear(48, count))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        self.last_regularization = None
        self.last_usage = None

    def train(self, mode=True):
        super().train(mode)
        self.bases.eval()
        return self

    def encode_inputs(self, a, ids, device):
        """This path deliberately never accesses y, future time or future row IDs."""
        x = torch.as_tensor(a["x"][ids], device=device)
        species = torch.as_tensor(a["meta"][ids, 1], device=device, dtype=torch.long)
        horizon = torch.as_tensor(a["meta"][ids, 4], device=device, dtype=torch.float32)
        velocity = torch.as_tensor(a["aux"][ids, 4:6], device=device, dtype=torch.float32)
        drivers = torch.as_tensor(a["drivers"][ids], device=device)
        anchors = torch.as_tensor(a["anchors"][ids], device=device)
        memory = torch.as_tensor(a["memory"][ids], device=device)
        params, features = [], None
        with torch.no_grad():
            for j, model in enumerate(self.bases):
                context = model.context_features(x, species, horizon, velocity, drivers, anchors, memory)
                params.append(model.parameters_from_context(context, horizon, velocity, anchors))
                if j == self.primary:
                    features = torch.cat([context.float(), drivers.float()], -1)
        return features, pack_ensemble(params), species

    def routing(self, features, species):
        b = len(features)
        if self.mode == "shared":
            return torch.zeros((b, 1), device=features.device, dtype=torch.long), torch.ones((b, 1), device=features.device)
        family = self.species_family[species]
        if self.mode == "family":
            return family[:, None], torch.ones((b, 1), device=features.device)
        initial = torch.zeros((b, len(self.experts)), device=features.device)
        initial[:, 0] = 1
        initial.scatter_add_(1, family[:, None], torch.full((b, 1), 2., device=features.device))
        valid = self.species_class[species, None] == self.family_class
        valid &= self.species_class[species, None] != 0
        valid[:, 0] = True
        logits = (initial + self.gate(features).float()).masked_fill(~valid, -1e9)
        values, selected = logits.topk(min(2, len(self.experts)), dim=-1)
        return selected, values.softmax(-1)

    def parameters_and_routing(self, features, base, species):
        selected, weights = self.routing(features, species)
        deltas = []
        for slot in range(selected.shape[1]):
            delta = torch.zeros((len(features), self.components * 6), device=features.device)
            for expert, head in enumerate(self.experts):
                ids = torch.nonzero(selected[:, slot] == expert, as_tuple=True)[0]
                if len(ids):
                    delta[ids] = head(features[ids]).float()
            deltas.append(delta.reshape(len(features), self.components, 6))
        raw = torch.stack(deltas, 1)
        base = base[:, None].float()
        logits = base[..., 0] + 4 * torch.tanh(raw[..., 0] / 4)
        mu = base[..., 1:3] + 2 * base[..., 3:5] * torch.tanh(raw[..., 1:3] / 2)
        sigma = base[..., 3:5] * torch.exp(2 * torch.tanh(raw[..., 3:5] / 2))
        rho = .95 * torch.tanh(torch.atanh((base[..., 5] / .95).clamp(-.99999, .99999)) + .5 * raw[..., 5].tanh())
        log_weight = torch.where(weights > 0, weights.clamp_min(1e-30).log(), torch.full_like(weights, -torch.inf))
        logits = logits.log_softmax(-1) + log_weight[..., None]
        self.last_regularization = (raw.square().mean((-1, -2)) * weights).sum(-1).mean()
        usage = torch.zeros((len(features), len(self.experts)), device=features.device)
        self.last_usage = usage.scatter_add(1, selected, weights)
        params = (logits.flatten(1, 2), mu.flatten(1, 2), sigma.flatten(1, 2), rho.flatten(1, 2), self.student_df)
        return params, selected, weights

    def forward(self, features, base, species):
        return self.parameters_and_routing(features, base, species)[0]

    def sample_from_features(self, features, base, species, count, generator=None):
        params, selected, weights = self.parameters_and_routing(features, base, species)
        # Hierarchical common random numbers make the zero-residual forecast
        # identical across routing modes, rather than only equal in distribution.
        b, slots, k = len(features), selected.shape[1], self.components
        expert_u = torch.rand((b, count), device=features.device, generator=generator)
        component_u = torch.rand((b, count), device=features.device, generator=generator)
        choice = (expert_u[..., None] > weights.cumsum(-1)[:, None]).sum(-1).clamp_max(slots - 1)
        logits = params[0].reshape(b, slots, k)
        conditional = torch.where(weights[..., None] > 0, logits, torch.zeros_like(logits)).softmax(-1)
        probs = conditional.gather(1, choice[..., None].expand(-1, -1, k))
        component = (component_u[..., None] > probs.cumsum(-1)).sum(-1).clamp_max(k - 1)
        index = choice * k + component
        mu = params[1].gather(1, index[..., None].expand(-1, -1, 2))
        sigma = params[2].gather(1, index[..., None].expand(-1, -1, 2))
        rho = params[3].gather(1, index)
        eps = torch.randn((b, count, 2), device=features.device, generator=generator)
        z = torch.stack([eps[..., 0], rho * eps[..., 0] + (1 - rho.square()).sqrt() * eps[..., 1]], -1)
        chi = torch.randn((b, count, self.student_df), device=features.device, generator=generator).square().sum(-1).clamp_min(1e-8)
        return mu + sigma * z / (chi / self.student_df).sqrt()[..., None]

    def predict_samples(self, a, ids, count, device, generator):
        return self.sample_from_features(*self.encode_inputs(a, ids, device), count, generator)


def prepare(root, checkpoints, output, identity_audit, taxonomy_root="data/combined-v2/environment/taxonomy", batch_size=512, device="auto"):
    from .train import load_windows
    from .identity import verify_audit, case_fingerprint
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest = load_windows(root)
    audit = json.loads(Path(identity_audit).read_text())
    verify_audit(audit, manifest, a)
    ckpts = [torch.load(p, map_location="cpu", weights_only=True) for p in checkpoints]
    for ck in ckpts:
        for key in ("species", "animals", "context", "observations_sha256"):
            if ck["manifest"][key] != manifest[key]:
                raise ValueError(f"Base ensemble schema differs: {key}")
        if ck["manifest"]["driver_names"] != manifest["driver_names"][:ck["config"]["n_drivers"]]:
            raise ValueError("Base ensemble driver ordering differs")
    groups = taxonomy_groups(manifest, taxonomy_root)
    config = {**groups, "base_configs": [ck["config"] for ck in ckpts], "n_drivers": len(manifest["driver_names"]),
              "primary": min(1, len(ckpts) - 1), "hidden": 96, "mode": "family"}
    model = CategoryMoE(config).to(device).eval()
    for base, ck in zip(model.bases, ckpts):
        base.load_state_dict(ck["model"])
    ids = np.flatnonzero(np.isin(a["meta"][:, 0], [0, 1, 3]))
    ids = ids[~np.isin(ids, audit["excluded_windows"])]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float32, shape=(len(ids), model.n_features))
    base_params = np.lib.format.open_memmap(out / "base.npy", mode="w+", dtype=np.float32, shape=(len(ids), model.components, 6))
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            selected = ids[start:start + batch_size]
            f, base, _ = model.encode_inputs(a, selected, device)
            features[start:start + len(selected)] = f.cpu().numpy()
            base_params[start:start + len(selected)] = base.cpu().numpy()
            if start % (batch_size * 40) == 0:
                print(f"MoE frozen contexts {start}/{len(ids)}", flush=True)
    features.flush()
    base_params.flush()
    np.save(out / "ids.npy", ids)
    torch.save({"config": config, "states": [m.cpu().state_dict() for m in model.bases]}, out / "bases.pt")
    support = []
    for i, name in enumerate(groups["families"]):
        selected = ids[(a["meta"][ids, 0] == 0) & (np.asarray(groups["species_family"])[a["meta"][ids, 1]] == i)]
        animal_names = {v: k for k, v in manifest["animals"].items()}
        physical = {audit["fold_groups"].get(animal_names[int(v)], animal_names[int(v)]) for v in a["meta"][selected, 3]}
        support.append({"family": name, "train_windows": len(selected), "physical_training_animals": len(physical)})
    report = {"source_data": str(root), "config": config, "windows_sha256": manifest["windows_sha256"],
              "case_fingerprint": case_fingerprint(a), "cases": len(ids), "training_support": support,
              "identity_audit_sha256": hashlib.sha256(Path(identity_audit).read_bytes()).hexdigest(),
              "base_checkpoints": [{"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in checkpoints],
              "files_sha256": {p: hashlib.sha256((out / p).read_bytes()).hexdigest() for p in ("features.npy", "base.npy", "ids.npy")},
              "scope": "Frozen causal inputs and base distributions for train/validation only. No test forecasts or future targets used as features."}
    dump(out / "manifest.json", report)
    return report
