"""Experimental residual reweighting of whole seasonal migration routes.

The support is a deterministic finite bank drawn from the existing seasonal
baseline. The zero-initialized head preserves its *empirical bank* distribution,
which approximates rather than equals the baseline's infinite sampling law.
Query future observations affect scores only, never candidates or their features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .migration_baselines import MigrationBaselines, _features, _velocity


EARTH_KM = 6371.0088
SELECTION_VERSION = "migration-retrieval-physical-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_algorithm_hash():
    # Training/optional context changes do not change the base candidate bank.
    source = "\n".join(inspect.getsource(f) for f in (query_features, candidate_features, generate_candidates))
    return hashlib.sha256(source.encode()).hexdigest()


def _load_data(data):
    from .migration_evaluate import load_cases
    return load_cases(data)


def query_features(a, ids):
    """Only origin-time predictors: static information and observed past fixes."""
    history, mask = a["history"][ids], a["history_mask"][ids]
    features = [a["static"][ids].astype(np.float32)]
    for lag in (1., 7., 30., 90., 180., 365.):
        distance = np.abs(history[:, :, 2] - lag)
        distance[~mask] = np.inf
        chosen = distance.argmin(-1)
        rows = np.arange(len(ids))
        valid = np.isfinite(distance[rows, chosen])
        selected = history[rows, chosen]
        # An explicit mask distinguishes missing memory from the origin.
        features.append(np.column_stack((np.where(valid[:, None], selected[:, :2], 0),
                                         np.where(valid, selected[:, 2] / 365., 0), valid)))
    return np.column_stack(features).astype(np.float32)


def candidate_features(a, ids, paths, info):
    """Route geometry and donor provenance, with no query future or mask."""
    n, count, horizons, _ = paths.shape
    steps = np.diff(np.concatenate((np.zeros((n, count, 1, 2)), paths), axis=2), axis=2)
    length = np.linalg.norm(steps, axis=-1)
    radial = np.linalg.norm(paths, axis=-1)
    velocity = _velocity(a, ids)
    velocity /= np.maximum(np.linalg.norm(velocity, axis=-1, keepdims=True), 1e-8)
    direction = steps / np.maximum(length[..., None], 1e-8)
    alignment = (direction * velocity[:, None, None]).sum(-1)
    turning = (direction[:, :, 1:] * direction[:, :, :-1]).sum(-1)
    total = length.sum(-1)
    dt = np.diff(np.r_[0., a["horizons"]])
    donor_ids = info["donor_ids"]
    similarity = np.full((n, count), 25., dtype=np.float32)
    same_species = np.zeros((n, count), dtype=np.float32)
    query = _features(a, ids)
    for row in range(n):
        valid = donor_ids[row] >= 0
        if valid.any():
            donor = donor_ids[row, valid]
            similarity[row, valid] = np.linalg.norm(_features(a, donor) - query[row], axis=-1)
            same_species[row, valid] = a["species"][donor] == a["species"][ids[row]]
    summaries = np.stack((total, radial[..., -1] / np.maximum(total, 1e-8),
                           (length / dt).max(-1), turning.mean(-1) if horizons > 1 else np.zeros_like(total),
                           similarity, same_species), axis=-1)
    result = np.concatenate((paths.reshape(n, count, -1), radial, length, alignment,
                             info["completed_mask"].astype(np.float32), summaries), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite migration route features")
    return result.astype(np.float32)


def generate_candidates(a, ids, candidates, seed, baselines=None):
    """Per-case seeds make support independent of batching and model weights."""
    baselines = baselines or MigrationBaselines(a)
    count, horizons = len(ids), len(a["horizons"])
    paths = np.empty((count, candidates, horizons, 2), dtype=np.float32)
    donors = np.empty((count, candidates), dtype=np.int64)
    completed = np.empty((count, candidates, horizons), dtype=bool)
    for row, case in enumerate(ids):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(case)]))
        original_diffusion = baselines.diffusion
        if a["split"][case] == 0:
            # Cross-fit the fallback noise as well as excluding route donors.
            # Otherwise a training query's future could affect its own bank
            # through the shared diffusion estimate even with donor exclusion.
            cache = getattr(baselines, "_leave_identity_diffusion", {})
            identity, species = int(a["individual"][case]), int(a["species"][case])
            key = (identity, species)
            if key not in cache:
                remaining = baselines.train[a["individual"][baselines.train] != identity]
                diffusion = dict(original_diffusion)
                for group in (-1, species):
                    selected = remaining if group == -1 else remaining[a["species"][remaining] == group]
                    mask = a["target_mask"][selected]
                    variance = (a["y"][selected] ** 2).sum(-1) / np.maximum(a["horizons"], .01)
                    values = variance[mask]
                    diffusion[group] = max(float(np.median(values)) / 2, 1e-6) if len(values) else diffusion.get(-1, 1e-4) if group != -1 else 1e-4
                cache[key] = diffusion
                baselines._leave_identity_diffusion = cache
            baselines.diffusion = cache[key]
        try:
            sampled, info = baselines.seasonal(np.asarray([case]), candidates, rng)
        finally:
            baselines.diffusion = original_diffusion
        paths[row], donors[row], completed[row] = sampled[0], info["donor_ids"][0], info["completed_mask"][0]
    info = {"donor_ids": donors, "completed_mask": completed}
    return paths, query_features(a, ids), candidate_features(a, ids, paths, info), info


def prepare_bank(data, output, candidates=64, seed=1701):
    if candidates < 2:
        raise ValueError("At least two whole-route candidates required")
    a, manifest, hashes = _load_data(data)
    if not manifest.get("reserved_test_targets_sealed", False):
        raise ValueError("Prepare learning banks only while the test targets remain sealed")
    out = Path(output)
    if (out / "manifest.json").exists():
        raise FileExistsError("Use a new path to preserve a frozen route bank")
    out.mkdir(parents=True, exist_ok=True)
    ids = np.flatnonzero(np.isin(a["split"], [0, 1, 2]) & a["target_mask"].any(-1))
    if not len(ids) or not np.any(a["split"][ids] == 0):
        raise ValueError("Route bank requires training and development cases")
    fitted = MigrationBaselines(a, manifest)
    arrays = {"case_ids": ids}
    all_paths, all_query, all_candidate, all_donors, all_completed = [], [], [], [], []
    for start in range(0, len(ids), 64):
        paths, query, features, info = generate_candidates(a, ids[start:start + 64], candidates, seed, fitted)
        # Baseline methods define eligibility. Assert it explicitly here rather
        # than relying only on the scorer to notice an accidental leakage.
        for row, case in enumerate(ids[start:start + 64]):
            donors = info["donor_ids"][row]
            valid = donors >= 0
            if valid.any():
                if not np.all(a["split"][donors[valid]] == 0):
                    raise ValueError("Nontraining route donor")
                if a["split"][case] in (0, 1) and np.any(a["individual"][donors[valid]] == a["individual"][case]):
                    raise ValueError("Query physical identity appears in its route bank")
                if a["split"][case] == 2 and any(fitted.end_by_id[int(d)] >= a["origin"][case, 2] for d in donors[valid]):
                    raise ValueError("Year validation donor future extends past query origin")
        all_paths.append(paths); all_query.append(query); all_candidate.append(features)
        all_donors.append(info["donor_ids"]); all_completed.append(info["completed_mask"])
    arrays.update(paths=np.concatenate(all_paths), query_features=np.concatenate(all_query),
                  candidate_features=np.concatenate(all_candidate), donor_ids=np.concatenate(all_donors),
                  completed_mask=np.concatenate(all_completed), base_weights=np.full((len(ids), candidates), 1 / candidates, np.float32))
    np.savez_compressed(out / "bank.npz", **arrays)
    summary = {"protocol": "wildcast-migration-route-bank-v1", "candidates": int(candidates), "seed": int(seed),
               "cases": int(len(ids)), "case_counts": {str(s): int(np.sum(a["split"][ids] == s)) for s in (0, 1, 2)},
               "query_features": int(arrays["query_features"].shape[-1]),
               "candidate_features": int(arrays["candidate_features"].shape[-1]),
               "reserved_test_cases_prepared": 0, "fallback_routes": int((arrays["donor_ids"] < 0).sum()),
               "bank_sha256": sha256(out / "bank.npz"), **hashes,
               "baseline_code_sha256": sha256(Path(__file__).with_name("migration_baselines.py")),
               "feature_code_sha256": sha256(__file__),
               "feature_algorithm_sha256": feature_algorithm_hash(),
               "initial_law": "Uniform finite empirical bank of seasonal whole-route draws; approximation of the baseline sampling law",
               "eligibility": "Only training donors; train/individual queries exclude entire physical identity; training noise diffusion also leaves query identity out; year donor futures end before query origin",
               "features": "Origin static and observed history; candidate geometry, donor similarity/species and donor completion masks. No query future or target mask."}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_bank(data, bank):
    a, manifest, hashes = _load_data(data)
    root = Path(bank)
    metadata = json.loads((root / "manifest.json").read_text())
    for key, value in hashes.items():
        if metadata.get(key) != value:
            raise ValueError("Route bank/data " + key + " mismatch")
    if metadata.get("bank_sha256") != sha256(root / "bank.npz"):
        raise ValueError("Route bank hash mismatch")
    if metadata.get("baseline_code_sha256") != sha256(Path(__file__).with_name("migration_baselines.py")):
        raise ValueError("Seasonal baseline code differs from the frozen route bank")
    if metadata.get("feature_algorithm_sha256") != feature_algorithm_hash():
        raise ValueError("Route-bank feature algorithm differs from its frozen source")
    with np.load(root / "bank.npz", allow_pickle=False) as source:
        b = {key: source[key] for key in source.files}
    ids = b["case_ids"]
    if np.any(~np.isin(a["split"][ids], [0, 1, 2])) or len(np.unique(ids)) != len(ids):
        raise ValueError("Route bank contains reserved/duplicate query cases")
    if b["paths"].shape[:2] != b["candidate_features"].shape[:2] or b["paths"].shape[2:] != (len(a["horizons"]), 2):
        raise ValueError("Route bank schema mismatch")
    if not np.isfinite(b["paths"]).all() or not np.isfinite(b["query_features"]).all() or not np.isfinite(b["candidate_features"]).all():
        raise ValueError("Route bank predictions/features must be finite")
    return a, manifest, hashes, b, metadata


class RetrievalReweighter(nn.Module):
    def __init__(self, query_features, candidate_features, n_species, candidates=64, seed=1701,
                 hidden=64, dropout=.1, max_logit=3., species_dropout=.1, context=None):
        super().__init__()
        self.config = dict(query_features=int(query_features), candidate_features=int(candidate_features),
                           n_species=int(n_species), candidates=int(candidates), seed=int(seed), hidden=int(hidden),
                           dropout=float(dropout), max_logit=float(max_logit), species_dropout=float(species_dropout),
                           context=context)
        self._route_context = None
        self.register_buffer("query_mean", torch.zeros(query_features))
        self.register_buffer("query_std", torch.ones(query_features))
        self.register_buffer("candidate_mean", torch.zeros(candidate_features))
        self.register_buffer("candidate_std", torch.ones(candidate_features))
        self.species = nn.Embedding(n_species, 8)
        self.network = nn.Sequential(nn.Linear(query_features + candidate_features + 8, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, query, candidate, species, base_weights=None):
        if self.training and self.config["species_dropout"]:
            species = torch.where(torch.rand(species.shape, device=species.device) < self.config["species_dropout"], 0, species)
        q = ((query - self.query_mean) / self.query_std).clamp(-20, 20)
        c = ((candidate - self.candidate_mean) / self.candidate_std).clamp(-20, 20)
        if self.config["context"]:
            offset = candidate.shape[-1] - 132
            # Normalize observed embedding values only. Missing vectors stay
            # zero after normalization, with their availability flags explicit.
            c = torch.cat((c[..., :offset],
                           c[..., offset:offset + 64] * (candidate[..., offset + 128:offset + 129] > 0),
                           c[..., offset + 64:offset + 128] * (candidate[..., offset + 131:offset + 132] > 0),
                           c[..., offset + 128:]), dim=-1)
        s = self.species(species)
        joined = torch.cat((q[:, None].expand(-1, c.shape[1], -1), c,
                            s[:, None].expand(-1, c.shape[1], -1)), dim=-1)
        bound = self.config["max_logit"]
        delta = bound * torch.tanh(self.network(joined).squeeze(-1) / bound)
        if base_weights is not None:
            delta = delta + base_weights.clamp_min(1e-20).log()
        return delta.softmax(-1)

    @torch.no_grad()
    def predict_cases(self, a, ids, samples, rng=None, baselines=None):
        """Evaluator interface; all future-free support is regenerated by case ID."""
        if samples < 1:
            raise ValueError("At least one route sample required")
        self.eval()
        from .migration_candidate_cache import cached_candidates
        paths, query, features, _ = cached_candidates(a, np.asarray(ids), self.config["candidates"],
                                                    self.config["seed"], baselines)
        if self.config["context"]:
            from .migration_route_context import RouteContext
            spec = self.config["context"]
            if self._route_context is None:
                self._route_context = RouteContext(spec["fields_root"], expected=spec["fingerprint"])
            extra = (np.zeros((*paths.shape[:2], 132), np.float32) if spec.get("mode") == "masked"
                     else self._route_context.features(a["origin"][ids], paths))
            if spec.get("mode") == "availability":
                extra[..., :128] = 0.
            features = np.concatenate((features, extra), axis=-1)
        device = self.query_mean.device
        weights = self(torch.as_tensor(query, device=device), torch.as_tensor(features, device=device),
                       torch.as_tensor(a["species"][ids], dtype=torch.long, device=device)).cpu().numpy()
        rng = rng or np.random.default_rng(2026)
        chosen = np.stack([rng.choice(paths.shape[1], samples, p=w.astype(float) / w.sum(dtype=float)) for w in weights])
        return paths[np.arange(len(ids))[:, None], chosen]


def _xyz_from_latlon(latlon):
    lat, lon = torch.deg2rad(latlon).unbind(-1)
    return EARTH_KM * torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()), -1)


def route_costs(paths, origin, target_latlon, target_mask):
    """Exact discrete-law costs, on observed waypoint support only.

    Coordinates use a spherical exponential map. Endpoint costs are great-circle
    km; joint costs are Euclidean distances between stacked Cartesian paths.
    """
    mask = target_mask.bool()
    if not mask.any(-1).all():
        raise ValueError("Each scored route origin needs an observed target")
    lat, lon = torch.deg2rad(origin[:, :2]).unbind(-1)
    zero, one = torch.zeros_like(lat), torch.ones_like(lat)
    unit = torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()), -1)
    east = torch.stack((-lon.sin(), lon.cos(), zero), -1)
    north = torch.stack((-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos()), -1)
    xy = paths * 1000. / EARTH_KM
    angle = xy.norm(dim=-1)
    displacement = xy[..., :1] * east[:, None, None] + xy[..., 1:] * north[:, None, None]
    predicted = EARTH_KM * (angle.cos()[..., None] * unit[:, None, None] +
                            torch.sinc(angle / math.pi)[..., None] * displacement)
    clean = torch.where(mask[..., None], target_latlon, origin[:, None, :2])
    truth = _xyz_from_latlon(clean)
    chord = (predicted - truth[:, None]).norm(dim=-1)
    first = 2 * EARTH_KM * torch.asin((chord / (2 * EARTH_KM)).clamp(0, 1))
    endpoint_pair = torch.cdist(predicted.transpose(1, 2), predicted.transpose(1, 2),
                                compute_mode="donot_use_mm_for_euclid_dist")
    endpoint_pair = 2 * EARTH_KM * torch.asin((endpoint_pair / (2 * EARTH_KM)).clamp(0, 1))
    factor = (mask / mask.sum(-1, keepdim=True)).sqrt()[..., None]
    embed = (predicted * factor[:, None]).flatten(-2)
    observed = (truth * factor).flatten(-2)
    return {"endpoint_first": first.transpose(1, 2), "endpoint_pair": endpoint_pair,
            "joint_first": (embed - observed[:, None]).norm(dim=-1),
            "joint_pair": torch.cdist(embed, embed, compute_mode="donot_use_mm_for_euclid_dist")}


def discrete_energy(weights, costs, target_mask):
    """The exact proper score of the finite categorical route distribution.

    There is no finite-sample diagonal correction: each route is an atom of the
    predicted law, and all weighted atom pairs (including zero diagonals) count.
    """
    endpoint = (costs["endpoint_first"] * weights[:, None]).sum(-1)
    endpoint -= .5 * torch.einsum("bi,bhij,bj->bh", weights, costs["endpoint_pair"], weights)
    joint = (weights * costs["joint_first"]).sum(-1)
    joint -= .5 * torch.einsum("bi,bij,bj->b", weights, costs["joint_pair"], weights)
    mask = target_mask.bool()
    endpoint_mean = (endpoint * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return {"endpoint_by_horizon": endpoint, "endpoint_energy_km": endpoint_mean,
            "joint_energy_km": joint}


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    if ck.get("model_kind") != "wildcast_migration_retrieval":
        raise ValueError("Not a migration route-reweighting checkpoint")
    model = RetrievalReweighter(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def _batch(a, b, rows, device):
    ids = b["case_ids"][rows]
    result = {key: torch.as_tensor(b[key][rows], device=device) for key in
              ("paths", "query_features", "candidate_features", "base_weights")}
    result.update({key: torch.as_tensor(a[key][ids], device=device) for key in
                   ("history", "history_mask", "static", "species", "origin", "target_latlon", "target_mask")})
    # Costs do not differentiate candidate locations, only their weights. Float
    # precision here matches the geometry of stored route samples to ~metres.
    result["origin"] = result["origin"].float()
    result["target_latlon"] = result["target_latlon"].float()
    return result


def _weights(model, batch):
    if hasattr(model, "weights_from_batch"):
        return model.weights_from_batch(batch)
    return model(batch["query_features"], batch["candidate_features"], batch["species"], batch["base_weights"])


def _macro(values, species, animals):
    return float(np.mean([np.mean([values[(species == s) & (animals == animal)].mean()
                                   for animal in np.unique(animals[species == s])]) for s in np.unique(species)]))


@torch.no_grad()
def validate_bank(model, a, b, device, batch_size=128):
    model.eval()
    splits = {}
    metric_species = a.get("metric_species", a["species"])
    for code, name in ((1, "val_individual"), (2, "val_year")):
        rows = np.flatnonzero(a["split"][b["case_ids"]] == code)
        if not len(rows):
            continue
        endpoint, joint, kl = [], [], []
        for start in range(0, len(rows), batch_size):
            batch = _batch(a, b, rows[start:start + batch_size], device)
            weights = _weights(model, batch)
            costs = route_costs(batch["paths"], batch["origin"], batch["target_latlon"], batch["target_mask"])
            scores = discrete_energy(weights, costs, batch["target_mask"])
            endpoint.append(scores["endpoint_by_horizon"].cpu().numpy())
            joint.append(scores["joint_energy_km"].cpu().numpy())
            kl.append((weights * (weights.clamp_min(1e-20).log() - batch["base_weights"].log())).sum(-1).cpu().numpy())
        ids = b["case_ids"][rows]
        endpoint = np.concatenate(endpoint)
        by_horizon = []
        for h in range(len(a["horizons"])):
            good = a["target_mask"][ids, h]
            if good.any():
                by_horizon.append(_macro(endpoint[good, h], metric_species[ids[good]], a["individual"][ids[good]]))
        splits[name] = {"endpoint_energy_km": float(np.mean(by_horizon)),
                        "joint_energy_km": _macro(np.concatenate(joint), metric_species[ids], a["individual"][ids]),
                        "mean_kl_from_prior": float(np.concatenate(kl).mean()), "cases": int(len(ids))}
    if not splits:
        raise ValueError("No development cases available for route reweighting")
    selection = float(np.mean([.5 * (v["endpoint_energy_km"] + v["joint_energy_km"]) for v in splits.values()]))
    return {"selection_km": selection, "splits": splits}


def train_bank(data, bank, output, epochs=30, batch_size=128, hidden=64, lr=.001,
               kl_weight=.02, weight_decay=.01, patience=6, seed=42, device="auto", dropout=.1, context=None,
               context_mode="actual", model_class="residual", width=128, layers=6, heads=8,
               warmup_steps=1000, variable_context=True, species_dropout=.1):
    if epochs < 0 or batch_size < 1 or patience < 1 or kl_weight < 0:
        raise ValueError("Invalid route-reweighting training options")
    if not 0 <= species_dropout <= 1:
        raise ValueError("species_dropout must lie between zero and one")
    if context_mode not in ("actual", "masked", "availability") or (context_mode != "actual" and not context):
        raise ValueError("context_mode must be actual/masked/availability; controls require a context cache")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest, hashes, b, metadata = load_bank(data, bank)
    context_spec, context_manifest = None, None
    if context:
        from .migration_route_context import load_context
        features, context_manifest = load_context(hashes, bank, metadata, b["case_ids"], context)
        if features.shape[:2] != b["candidate_features"].shape[:2]:
            raise ValueError("Route context candidate dimensions differ from its bank")
        if context_mode == "masked":
            features = np.zeros_like(features)
        elif context_mode == "availability":
            features = features.copy(); features[..., :128] = 0.
        b["candidate_features"] = np.concatenate((b["candidate_features"], features), axis=-1)
        context_spec = {"fields_root": context_manifest["field_provenance"]["fields_root"],
                        "fingerprint": context_manifest["context_fingerprint"], "mode": context_mode}
    train_rows = np.flatnonzero(a["split"][b["case_ids"]] == 0)
    if not len(train_rows):
        raise ValueError("No training route-bank cases")
    n_species = max(manifest.get("species", {}).values(), default=int(a["species"].max())) + 1
    if model_class == "residual":
        model = RetrievalReweighter(b["query_features"].shape[-1], b["candidate_features"].shape[-1],
                                    n_species, metadata["candidates"], metadata["seed"], hidden, dropout,
                                    context=context_spec, species_dropout=species_dropout).to(device)
    elif model_class == "moveformer":
        from .migration_moveformer import MigrationMoveFormer
        model = MigrationMoveFormer(b["query_features"].shape[-1], b["candidate_features"].shape[-1], n_species,
                                     history_features=a["history"].shape[-1], static_features=a["static"].shape[-1],
                                     history_length=a["history"].shape[1], candidates=metadata["candidates"], seed=metadata["seed"],
                                     width=width, layers=layers, heads=heads, variable_context=variable_context,
                                     context=context_spec, species_dropout=species_dropout).to(device)
    else:
        raise ValueError("model_class must be residual or moveformer")
    with torch.no_grad():
        q = b["query_features"][train_rows]
        c = b["candidate_features"][train_rows].reshape(-1, b["candidate_features"].shape[-1])
        for prefix, features in (("query", q), ("candidate", c)):
            mean, std = features.mean(0), np.maximum(features.std(0), 1e-3)
            if prefix == "candidate" and context:
                offset = features.shape[-1] - 132
                for left, right, available in ((offset, offset + 64, offset + 128),
                                                (offset + 64, offset + 128, offset + 131)):
                    present = features[:, available] > 0
                    mean[left:right] = features[present, left:right].mean(0) if present.any() else 0.
                    std[left:right] = np.maximum(features[present, left:right].std(0), 1e-3) if present.any() else 1.
                mean[offset + 128:] = 0.; std[offset + 128:] = 1.
            getattr(model, prefix + "_mean").copy_(torch.as_tensor(mean, device=device))
            getattr(model, prefix + "_std").copy_(torch.as_tensor(std, device=device))
        if hasattr(model, "set_history_normalization"):
            ids = b["case_ids"][train_rows]
            observed = a["history"][ids][a["history_mask"][ids]]
            if not len(observed):
                raise ValueError("MoveFormer requires observed training history")
            static = a["static"][ids]
            model.set_history_normalization(observed.mean(0), np.maximum(observed.std(0), 1e-3),
                                             static.mean(0), np.maximum(static.std(0), 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs), eta_min=lr * .1)
    rng = np.random.default_rng(seed)
    train_ids = b["case_ids"][train_rows]
    _, inv, counts = np.unique(a["individual"][train_ids], return_inverse=True, return_counts=True)
    probabilities = 1. / counts[inv]; probabilities /= probabilities.sum()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    config = {"scope": ("Local validation development; author-core whole-route candidate selection"
                        if model_class == "moveformer" else "Local validation development; residual weighting of a fixed training-only route bank"),
              "selection_version": SELECTION_VERSION, "context": context_manifest, "context_mode": context_mode,
              "model_class": model_class, "warmup_steps": warmup_steps if model_class == "moveformer" else 0,
              "species_dropout": float(species_dropout),
              "species_dropout_rule": "Model training forward replaces species with unknown token zero at this probability; validation and prediction disable dropout. Applied once, not again in the trainer.",
              "data": str(data), "bank": str(bank), "epochs": epochs, "batch_size": batch_size,
              "kl_weight": kl_weight, "weight_decay": weight_decay, "lr": lr, "seed": seed,
              "patience": patience, "device": str(device), "model": model.config,
              "selection": "Equal endpoint great-circle and joint Cartesian energy; equal split weight; species/physical-animal macro averages",
              **hashes, "bank_sha256": metadata["bank_sha256"], "bank_manifest_sha256": sha256(Path(bank) / "manifest.json"),
              "code_sha256": sha256(__file__)}
    if hasattr(model, "upstream"):
        config["upstream"] = model.upstream
        config["adapter_code_sha256"] = sha256(Path(__file__).with_name("migration_moveformer.py"))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    history, best, best_epoch, stale, global_step = [], float("inf"), 0, 0, 0
    started = time.monotonic()
    for epoch in range(epochs + 1):
        total, batches = 0., 0
        if epoch:
            model.train()
            selected = rng.choice(train_rows, len(train_rows), replace=True, p=probabilities)
            for start in range(0, len(selected), batch_size):
                batch = _batch(a, b, selected[start:start + batch_size], device)
                if model_class == "moveformer":
                    warmup = min(1., (global_step + 1) / max(1, warmup_steps))
                    decay = .1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, epochs)))
                    for group in optimizer.param_groups:
                        group["lr"] = lr * warmup * decay
                optimizer.zero_grad(set_to_none=True)
                weights = _weights(model, batch)
                costs = route_costs(batch["paths"], batch["origin"], batch["target_latlon"], batch["target_mask"])
                scores = discrete_energy(weights, costs, batch["target_mask"])
                kl = (weights * (weights.clamp_min(1e-20).log() - batch["base_weights"].log())).sum(-1)
                loss = (.5 * (scores["endpoint_energy_km"] + scores["joint_energy_km"]) / 1000. + kl_weight * kl).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite route-reweighting loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite route-reweighting gradients")
                optimizer.step(); total += float(loss.detach()); batches += 1; global_step += 1
            if model_class == "residual":
                scheduler.step()
        metrics = validate_bank(model, a, b, device, batch_size)
        if not np.isfinite(metrics["selection_km"]):
            raise FloatingPointError("Nonfinite route-reweighting validation score")
        row = {"epoch": epoch, "train_loss": total / batches if batches else None,
               "elapsed_seconds": time.monotonic() - started, **metrics}
        history.append(row); (out / "history.json").write_text(json.dumps(history, indent=2))
        improved = metrics["selection_km"] < best - 1e-4
        if improved:
            best, best_epoch, stale = metrics["selection_km"], epoch, 0
        elif epoch:
            stale += 1
        checkpoint = {"model_kind": getattr(model, "model_kind", "wildcast_migration_retrieval"), "model": model.state_dict(),
                      "selection_version": SELECTION_VERSION,
                      "config": model.config, "epoch": epoch, "metrics": metrics, "manifest": manifest,
                      "training": config, "optimizer": optimizer.state_dict(), **hashes,
                      "bank_sha256": metadata["bank_sha256"], "bank_manifest_sha256": config["bank_manifest_sha256"]}
        if hasattr(model, "upstream"):
            checkpoint["upstream"] = model.upstream
        if not epoch:
            torch.save(checkpoint, out / "initial.pt")
        if improved:
            torch.save(checkpoint, out / "best.pt")
        torch.save(checkpoint, out / "last.pt")
        print(json.dumps({"epoch": epoch, "selection_km": metrics["selection_km"], "best_km": best,
                          "train_loss": row["train_loss"], "seconds": round(row["elapsed_seconds"], 1)}), flush=True)
        if epoch and stale >= patience:
            break
    result = {"selected_epoch": best_epoch, "best_selection_km": best,
              "selection_version": SELECTION_VERSION,
              "initial_selection_km": history[0]["selection_km"], "epochs_completed": history[-1]["epoch"],
              "parameters": sum(p.numel() for p in model.parameters()), **hashes,
              "scope": "Development result; initial.pt: " + getattr(model, "initial_law", "exact uniform empirical-bank control")}
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data", required=True); prepare.add_argument("--output", required=True)
    prepare.add_argument("--candidates", type=int, default=64); prepare.add_argument("--seed", type=int, default=1701)
    fit = sub.add_parser("train")
    for name in ("data", "bank", "output"):
        fit.add_argument("--" + name, required=True)
    for name, default in (("epochs", 30), ("batch-size", 128), ("hidden", 64), ("patience", 6), ("seed", 42)):
        fit.add_argument("--" + name, type=int, default=default)
    for name, default in (("lr", .001), ("kl-weight", .02), ("weight-decay", .01), ("dropout", .1), ("species-dropout", .1)):
        fit.add_argument("--" + name, type=float, default=default)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--context", help="Prepared previous-year route-context cache; omit for matched control")
    fit.add_argument("--context-mode", choices=("actual", "masked", "availability"), default="actual",
                     help="Actual fields or zero fields/masks with identical input shape and parameter count")
    args = vars(parser.parse_args()); command = args.pop("command")
    (prepare_bank if command == "prepare" else train_bank)(**args)


if __name__ == "__main__":
    main()
