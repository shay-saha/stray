"""Conditional iSSF Linear/MLP and Gamma/von-Mises short movement controls.

The primary SSF recipe is adapted to the same explicit continuous cell/tail
support as migration_deepssf, not presented as official MoveBench code. Choice
training uses log(p_model/q_negative), including the fixed tail. Every observed
training endpoint remains eligible; validation truth is only used for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import optimize, special
from torch import nn

from . import migration_deepssf as support
from .data import R

MODEL_KIND = "wildcast_migration_issf_short"
SELECTION_VERSION = "migration-issf-short-endpoint-physical-v1"
VARIANTS = ("linear", "mlp", "movement")
HORIZONS = support.SUPPORTED_HORIZONS
QUADRATURE = 4
PRIMARY_RECIPES = {"iSSF": "https://doi.org/10.1111/2041-210X.12528",
                   "implementation_recipe": "https://arxiv.org/html/2609.15780v1#A5",
                   "negative_sampling": "https://arxiv.org/html/2609.15780v1#A6"}


def code_hashes():
    return {name: support.sha256(Path(__file__).with_name(name)) for name in
            ("migration_issf.py", "migration_deepssf.py", "migration_human_pressure.py", "data.py")}


def heading(a, ids):
    """No heading for missing or exactly zero previous displacement."""
    scalars, angle = support.predictor_scalars(a, ids, np.full(len(ids), 1/24))
    h, mask = a["history"][ids], a["history_mask"][ids]
    age = np.where(mask & (h[..., 2] > 0), h[..., 2], np.inf)
    chosen = age.argmin(-1)
    moved = np.linalg.norm(h[np.arange(len(ids)), chosen, :2], axis=-1) > 0
    present = (scalars[:, -1] > 0) & moved
    return np.where(present, angle[:, 0], 0.), present


def fit_proposal(a, geometry, *, allowed_cross_taxon_exclusion_ids=()):
    """Weighted TRAIN-only Gamma MLE and circular-moment von-Mises fit.

    A 0.1m radial floor permits exact zero fixes in the Gamma fit; it is a
    modelling floor, not sensor accuracy. Shape/concentration bounds regularize
    degenerate tracks. Prediction becomes isotropic without a past heading.
    """
    from .migration_train import sampling_probabilities
    columns = support._horizon_indices(HORIZONS, a["horizons"])
    n_species = len(geometry["cell_metres"])
    result = np.empty((n_species, len(HORIZONS), 4), dtype=float)
    counts = np.zeros((n_species, len(HORIZONS)), dtype=int)
    for head, column in enumerate(columns):
        ids = np.flatnonzero((a["split"] == 0) & a["target_mask"][:, column])
        if not len(ids):
            raise ValueError("Every supported horizon requires observed training movement")
        values = a["y"][ids, column].astype(float) * 1_000_000
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=-1) >= support.MAX_RADIUS):
            raise ValueError("Invalid training movement in the spherical AEQD domain")
        prior_angle, known = heading(a, ids)
        radial = np.maximum(np.linalg.norm(values, axis=-1), .1)
        turn = np.arctan2(values[:, 1], values[:, 0]) - prior_angle
        # A zero-length observed endpoint has no observed turn angle.
        known &= np.linalg.norm(values, axis=-1) > 0
        for token in range(n_species):
            selected = np.ones(len(ids), bool) if token == 0 else a["species"][ids] == token
            if not selected.any():
                result[token, head] = result[0, head]
                continue
            chosen = ids[selected]
            weights = sampling_probabilities(a, chosen, "species_animal",
                allowed_cross_taxon_exclusion_ids=allowed_cross_taxon_exclusion_ids)
            x = radial[selected]
            mean, logmean = np.sum(weights * x), np.sum(weights * np.log(x))
            discrepancy = max(0., math.log(mean) - logmean)
            equation = lambda k: math.log(k) - special.digamma(k) - discrepancy
            shape = (.05 if equation(.05) < 0 else 10_000. if equation(10_000.) > 0
                     else optimize.brentq(equation, .05, 10_000.))
            scale = mean / shape
            angle_weights = weights * known[selected]
            if angle_weights.sum() == 0:
                mu, kappa = 0., 0.
            else:
                vector = np.sum(angle_weights * np.exp(1j * turn[selected])) / angle_weights.sum()
                mu, concentration = float(np.angle(vector)), float(abs(vector))
                ratio = lambda k: special.i1e(k) / special.i0e(k) - concentration
                kappa = 0. if concentration < 1e-10 else 100. if ratio(100.) < 0 else optimize.brentq(ratio, 0., 100.)
            result[token, head], counts[token, head] = [shape, scale, mu, kappa], len(chosen)
    return {"parameters": result.tolist(), "training_observations": counts.tolist(),
            "fit_split": "train", "parameter_names": ["gamma_shape", "gamma_scale_metres", "turn_mu", "turn_kappa"],
            "radial_fit_floor_metres": .1, "gamma_shape_bounds": [.05, 10_000.], "kappa_bounds": [0., 100.],
            "heading_rule": "Missing or zero past displacement sets kappa and turning covariate to zero",
            "quadrature": QUADRATURE, "central_cell_angular_quadrature": 256}


def gamma_vm_grid(parameters, cell_metres, bearing, known, window):
    """Positive discrete approximation to Gamma(r)*VM(turn)/r cell masses.

    Noncentral cells use fixed4x4 area quadrature. The central square uses a
    radial Gamma CDF integrated over256 angles, avoiding a singular r=0 point
    evaluation. Resulting finite masses are explicitly normalized. Prediction
    is exactly this discretized law, not a claim of exact continuous integration.
    """
    parameters = np.asarray(parameters, float)
    outputs = []
    row, col = np.indices((window, window)); middle = window // 2
    offsets = (np.arange(QUADRATURE) + .5) / QUADRATURE - .5
    ox, oy = np.meshgrid(offsets, offsets)
    for p, cell, direction, available in zip(parameters, cell_metres, bearing, known):
        shape, scale, mu, kappa = p
        if not available:
            kappa, direction, mu = 0., 0., 0.
        x = (col[..., None] - middle + ox.ravel()) * cell
        y = (middle - row[..., None] + oy.ravel()) * cell
        radial = np.hypot(x, y)
        angle = np.arctan2(y, x) - direction - mu
        log_density = ((shape - 2) * np.log(radial) - radial / scale - special.gammaln(shape) - shape * math.log(scale)
                       + kappa * np.cos(angle) - math.log(2 * math.pi) - math.log(special.i0e(kappa)) - abs(kappa))
        log_mass = special.logsumexp(log_density, -1) + 2 * math.log(cell) - math.log(QUADRATURE ** 2)
        theta = (np.arange(256) + .5) / 256 * (2 * math.pi)
        limit = cell / 2 / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
        angular = np.exp(kappa * np.cos(theta - direction - mu) - math.log(2 * math.pi) - math.log(special.i0e(kappa)) - abs(kappa))
        central = np.sum(special.gammainc(shape, limit / scale) * angular) * (2 * math.pi / 256)
        log_mass[middle, middle] = math.log(max(central, np.finfo(float).tiny))
        outputs.append(log_mass - special.logsumexp(log_mass))
    return np.asarray(outputs)


def point_log_density(log_grid, points, cell, tail_scale, geographic=False):
    """Normalized cell-uniform/tail density, for any number of candidate points."""
    points = points.double(); cell = cell.double(); window = log_grid.shape[-1]
    row = torch.floor(-points[..., 1] / cell[:, None] + window / 2).long()
    col = torch.floor(points[..., 0] / cell[:, None] + window / 2).long()
    inside = (row >= 0) & (row < window) & (col >= 0) & (col < window)
    picked = log_grid[torch.arange(len(points), device=points.device)[:, None], row.clamp(0, window - 1), col.clamp(0, window - 1)]
    raster = torch.where(inside, picked - 2 * cell[:, None].log() + math.log1p(-support.TAIL_WEIGHT), -torch.inf)
    tail = support.tail_log_density(points, tail_scale[:, None]) + math.log(support.TAIL_WEIGHT)
    result = torch.logaddexp(raster, tail)
    if geographic:
        result = result - torch.log(torch.sinc(points.norm(dim=-1) / (math.pi * R)))
    return result


def sample_grid(log_grid, cell, tail_scale, samples, rng):
    log_grid, cell, tail_scale = np.asarray(log_grid), np.asarray(cell), np.asarray(tail_scale)
    window = log_grid.shape[-1]; center = window // 2
    result = support.sample_tail(tail_scale, samples, rng)
    tail = rng.random((len(cell), samples)) < support.TAIL_WEIGHT
    for i in range(len(cell)):
        selected = np.flatnonzero(~tail[i])
        p = np.exp(log_grid[i].ravel()); p /= p.sum()
        index = rng.choice(window * window, len(selected), p=p)
        row, col = np.divmod(index, window)
        jitter = rng.uniform(-.5, .5, (len(selected), 2))
        result[i, selected] = np.column_stack([col - center + jitter[:, 0], center - row + jitter[:, 1]]) * cell[i]
    return result


class ISSFShort(nn.Module):
    def __init__(self, geometry, proposal, variant="linear", window=101, hidden=32,
                 context_mode="masked", fields_root=None, fields_fingerprint=None):
        super().__init__()
        if variant not in VARIANTS or window < 17 or window > 101 or window % 2 != 1 or not 8 <= hidden <= 128:
            raise ValueError("Invalid bounded iSSF variant/architecture")
        if context_mode not in ("masked", "actual", "availability") or (context_mode != "masked" and (not fields_root or not fields_fingerprint)):
            raise ValueError("Actual/availability context requires frozen local GHSL sources")
        if tuple(geometry["supported_horizons_days"]) != HORIZONS or proposal.get("fit_split") != "train":
            raise ValueError("iSSF requires frozen TRAIN geometry/proposal and supported horizons")
        cell, tail = np.asarray(geometry["cell_metres"], float), np.asarray(geometry["tail_scale_metres"], float)
        params = np.asarray(proposal["parameters"], float)
        if (params.shape != (*cell.shape, 4) or tail.shape != cell.shape or cell.shape[1:] != (3,)
                or not np.isfinite(params).all() or np.any(params[..., :2] <= 0) or np.any(params[..., 3] < 0)
                or not np.isfinite(cell).all() or np.any(cell <= 0) or np.any(cell * window / 2 * math.sqrt(2) >= support.MAX_RADIUS)
                or not np.isfinite(tail).all() or np.any(tail <= 0)):
            raise ValueError("Invalid frozen iSSF proposal or cell/tail geometry")
        if variant == "movement" and context_mode != "masked":
            raise ValueError("Zero-utility movement control has no environmental input")
        self.config = dict(geometry=geometry, proposal=proposal, variant=variant, window=window, hidden=hidden,
                           context_mode=context_mode, fields_root=str(fields_root) if fields_root else None, fields_fingerprint=fields_fingerprint)
        self.supported_horizons = HORIZONS
        self.register_buffer("cell_metres", torch.tensor(cell, dtype=torch.float64))
        self.register_buffer("tail_scale_metres", torch.tensor(tail, dtype=torch.float64))
        self.register_buffer("proposal_parameters", torch.tensor(params, dtype=torch.float64))
        # Base: movement3, GHSL6, cyclic time4, heading-present1. Linear adds
        # movement/environment, time/environment and time/movement interactions.
        self.feature_size = 14 + (18 + 24 + 12 if variant == "linear" else 0)
        if variant == "linear":
            self.global_weights = nn.Parameter(torch.zeros(3, self.feature_size))
            self.species_weights = nn.Embedding(len(cell), 3 * self.feature_size)
            nn.init.zeros_(self.species_weights.weight)
        elif variant == "mlp":
            self.species_embedding = nn.Embedding(len(cell), 4)
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(18, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)) for _ in range(3)])
            for head in self.heads:
                nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        self._raster = None

    def close(self):
        if self._raster is not None:
            self._raster.close(); self._raster = None

    def distributions(self, a, ids, heads):
        ids, heads = np.asarray(ids, int), np.asarray(heads, int)
        if ids.shape != heads.shape or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(a["origin"])) or np.any(heads < 0) or np.any(heads >= 3):
            raise ValueError("Invalid endpoint origins/heads")
        tokens = np.asarray(a["species"])[ids].astype(int)
        if np.any(tokens < 0):
            raise ValueError("Negative species token")
        tokens = np.where(tokens < len(self.cell_metres), tokens, 0)
        device = self.cell_metres.device
        if self.training and self.config["variant"] != "movement":
            tokens = np.where((torch.rand(len(tokens), device=device) < .1).cpu().numpy(), 0, tokens)
        cell, tail = self.cell_metres[tokens, heads], self.tail_scale_metres[tokens, heads]
        parameters = self.proposal_parameters[tokens, heads].detach().cpu().numpy()
        bearing, known = heading(a, ids)
        base = gamma_vm_grid(parameters, cell.detach().cpu().numpy(), bearing, known, self.config["window"])
        log_q = torch.tensor(base, dtype=torch.float64, device=device)
        if self.config["variant"] == "movement":
            return log_q, log_q, cell, tail
        features = self._features(a, ids, heads, parameters, cell, bearing, known)
        utility = torch.empty((len(ids), self.config["window"] ** 2), dtype=torch.float32, device=device)
        tokens_t = torch.as_tensor(tokens, device=device)
        for h in np.unique(heads):
            where = np.flatnonzero(heads == h)
            if self.config["variant"] == "linear":
                weights = self.global_weights[h] + self.species_weights(tokens_t[where]).reshape(len(where), 3, -1)[:, h]
                utility[where] = (features[where] * weights[:, None]).sum(-1)
            else:
                embedded = self.species_embedding(tokens_t[where])[:, None].expand(-1, features.shape[1], -1)
                utility[where] = self.heads[h](torch.cat([features[where], embedded], -1)).squeeze(-1)
        raw = log_q.flatten(1) + utility.double()
        log_p = torch.log_softmax(raw, -1).reshape_as(log_q)
        return log_p, log_q, cell, tail

    def _features(self, a, ids, heads, parameters, cell, bearing, known):
        window = self.config["window"]; center = window // 2
        row, col = np.indices((window, window))
        unit = np.stack([col - center, center - row], -1).reshape(-1, 2)
        xy = unit[None] * cell.detach().cpu().numpy()[:, None, None]
        radius = np.linalg.norm(xy, axis=-1)
        theta = np.arctan2(xy[..., 1], xy[..., 0])
        shape, scale = parameters[:, 0:1], parameters[:, 1:2]
        movement = np.stack([radius / (shape * scale),
            (np.log(np.maximum(radius, .1)) - special.digamma(shape) - np.log(scale)) / np.sqrt(special.polygamma(1, shape)),
            np.cos(theta - bearing[:, None]) * known[:, None]], -1)
        movement[..., 2] = np.where(radius > 0, movement[..., 2], 0)
        scalar, _ = support.predictor_scalars(a, ids, np.asarray(HORIZONS)[heads])
        temporal = np.broadcast_to(scalar[:, None, :4], (*radius.shape, 4))
        if self.config["context_mode"] == "masked":
            environment = np.zeros((*radius.shape, 6), "float32")
        else:
            if self._raster is None:
                self._raster = support.GHSLRaster(self.config["fields_root"], expected=self.config["fields_fingerprint"])
            patch = self._raster.patches(a["origin"][ids], cell.detach().cpu().numpy(), window, self.config["context_mode"])
            environment = patch.reshape(len(ids), 6, -1).transpose(0, 2, 1)
        base = [movement, environment, temporal, np.broadcast_to(known[:, None, None], (*radius.shape, 1))]
        if self.config["variant"] == "linear":
            for left, right in ((movement, environment), (temporal, environment), (temporal, movement)):
                base.append((left[..., :, None] * right[..., None, :]).reshape(*radius.shape, -1))
        features = np.clip(np.concatenate(base, -1), -40, 40).astype("float32")
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite origin/candidate iSSF features")
        return torch.tensor(features, device=self.cell_metres.device)

    @torch.no_grad()
    def predict_endpoints(self, a, ids, horizons=HORIZONS, samples=64, rng=None):
        if self.training or samples < 2:
            raise ValueError("Prediction requires eval mode and at least two samples")
        ids = np.asarray(ids, int); heads = support._horizon_indices(horizons)
        rng = rng if rng is not None else np.random.default_rng(2026)
        output = np.empty((len(ids), samples, len(heads), 2), dtype=float)
        for h, head in enumerate(heads):
            log_p, _, cell, tail = self.distributions(a, ids, np.full(len(ids), head))
            output[:, :, h] = sample_grid(log_p.cpu().numpy(), cell.cpu().numpy(), tail.cpu().numpy(), samples, rng) / 1_000_000
        return output

    def predict_cases(self, a, ids, samples, rng=None, baselines=None, seed=2026):
        return self.predict_endpoints(a, ids, a["horizons"], samples, rng if rng is not None else np.random.default_rng(seed))


def conditional_loss(model, a, ids, heads, columns, rng, negatives=100):
    ids, heads, columns = np.asarray(ids, int), np.asarray(heads, int), np.asarray(columns, int)
    if (negatives < 1 or not np.isin(a["split"][ids], [0]).all() or not a["target_mask"][ids, columns].all()
            or not np.array_equal(a["horizons"][columns], np.asarray(HORIZONS)[heads])):
        raise ValueError("Conditional choices require observed TRAIN targets at the known requested horizon")
    log_p, log_q, cell, tail = model.distributions(a, ids, heads)
    negative = sample_grid(log_q.detach().cpu().numpy(), cell.detach().cpu().numpy(), tail.detach().cpu().numpy(), negatives, rng)
    observed = a["y"][ids, columns].astype(float) * 1_000_000
    points = torch.tensor(np.concatenate([negative, observed[:, None]], 1), device=cell.device)
    logits = point_log_density(log_p, points, cell, tail) - point_log_density(log_q, points, cell, tail)
    if not torch.isfinite(logits).all():
        raise ValueError("Invalid continuous conditional-choice density ratio")
    loss = -torch.log_softmax(logits, -1)[:, -1]
    return loss, logits


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    if ck.get("model_kind") != MODEL_KIND or ck.get("code_sha256") != code_hashes():
        raise ValueError("Wrong iSSF kind or changed prediction dependency")
    if ck.get("primary_recipes") != PRIMARY_RECIPES or ck.get("supported_horizons_days") != list(HORIZONS):
        raise ValueError("Changed iSSF recipe or supported horizons")
    for key in ("manifest_sha256", "cases_sha256"):
        value = ck.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("iSSF checkpoint lacks frozen data hashes")
    model = ISSFShort(**ck["config"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    for name, expected in (("cell_metres", model.config["geometry"]["cell_metres"]),
                           ("tail_scale_metres", model.config["geometry"]["tail_scale_metres"]),
                           ("proposal_parameters", model.config["proposal"]["parameters"])):
        if not torch.equal(getattr(model, name), torch.tensor(expected, dtype=torch.float64, device=device)):
            raise ValueError("iSSF sampling buffers differ from the frozen geometry/proposal policy")
    return model, ck


def validate(model, a, batch_size=16, samples=64, seed=2026, origin_ids=None):
    metrics = support.validate(model, a, batch_size, samples, seed, origin_ids=origin_ids)
    metrics["selection_version"] = SELECTION_VERSION
    for key in ("split_metrics", "horizon_metrics"):
        for row in metrics[key]:
            row["model"] = "issf-" + model.config["variant"]
    return metrics


def train(data, output, variant="linear", sources=None, context_mode="masked", epochs=20,
          batch_size=16, window=101, hidden=32, lr=.001, weight_decay=.001, seed=2026,
          negatives=100, validation_samples=64, patience=5, device="cpu", train_pairs_per_epoch=0, validation_origins=0):
    """Explicit finite developer fit; movement variant only fits TRAIN moments."""
    from .migration_train import load_training_data, sampling_probabilities
    from .migration_protocol import cross_taxon_exclusion_ids
    out = Path(output)
    if out.exists():
        raise FileExistsError("Use a new immutable iSSF experiment directory")
    if epochs < 0 or batch_size < 1 or negatives < 1 or validation_samples < 2 or patience < 1 or train_pairs_per_epoch < 0 or validation_origins < 0:
        raise ValueError("Invalid finite iSSF budget")
    a, manifest, hashes = load_training_data(data)
    schema = support.training_schema(manifest, a)
    panel = support.validation_panel(a, validation_origins) if validation_origins else None
    allowed = cross_taxon_exclusion_ids(manifest)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    geometry = support.fit_geometry(a, window=window, allowed_cross_taxon_exclusion_ids=allowed)
    proposal = fit_proposal(a, geometry, allowed_cross_taxon_exclusion_ids=allowed)
    fingerprint, field_evidence = None, None
    if context_mode != "masked":
        fields = support.GHSLRaster(sources)
        fingerprint, field_evidence = fields.fingerprint, fields.provenance
        fields.close()
    model = ISSFShort(geometry, proposal, variant=variant, window=window, hidden=hidden,
                     context_mode=context_mode, fields_root=sources, fields_fingerprint=fingerprint).to(device)
    rows, heads, columns = support.observed_pairs(a, [0])
    origins, inverse, count = np.unique(rows, return_inverse=True, return_counts=True)
    probability = sampling_probabilities(a, origins, "species_animal",
        allowed_cross_taxon_exclusion_ids=allowed)[inverse] / count[inverse]
    probability /= probability.sum()
    effective_epochs = 0 if variant == "movement" else epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) if effective_epochs else None
    scope = {"model_kind": MODEL_KIND, "config": model.config, **hashes,
        "primary_recipes": PRIMARY_RECIPES, "code_sha256": code_hashes(),
        "cache_protocol": manifest["protocol"],
        "feature_schema": {"history_names": schema.HISTORY_NAMES, "static_names": schema.STATIC_NAMES,
                           "history_lags_days": schema.LAGS.tolist()},
        "fitting_code_sha256": support.fitting_code_hashes(manifest),
        "supported_horizons_days": list(HORIZONS), "independent_horizon_draws": True,
        "selection_version": SELECTION_VERSION, "official_MoveBench_reproduction": False,
        "scope": "Local iSSF score-function controls with explicit discretized continuous cell/tail support; no author code or pretrained weights are used",
        "author_recipe": "Train-fitted Gamma lengths/von-Mises turns; movement length/loglength/cos-turn, environmental and cyclic-time covariates; linear interactions or MLP; sampled conditional choices",
        "adaptations": ["Global plus taxon-specific linear effects, or taxon-embedded horizon-specific MLPs, instead of unverified official per-study configurations",
            "All finite TRAIN outcomes retained with a0.1m radial Gamma-fitting floor and explicit parameter bounds",
            "TRAIN-only geometry, fixed finite cell quadrature, uniform within-cell density and fixed10% exact truncated Student-t tail",
            "Choice logits log(p_model/q_negative) keep this mixed continuous law coherent; negatives are drawn from zero-utility cell/tail proposal",
            "No heading or zero past displacement forces isotropic proposal and zero turn feature",
            "10% pooled augmentation changes geometry, kernel, tail, field footprint and utility token together",
            "Movement features use TRAIN-fitted Gamma theoretical moments, GHSL known physical units, cyclic time, and predeclared feature clipping±40; no fitted heldout transforms",
            "No additional history/goal/memory augmentation is mixed into this standard reference family"],
        "choice_negatives": negatives, "positive_alternative": "Last; observed training endpoint only",
        "geography": "AEQD area density; geographic pushforward uses division by sin(r/R)/(r/R). This Jacobian cancels in conditional density ratios.",
        "context": field_evidence, "context_caveat": "GHSL R2023A historical reconstructions remain retrospective; exact issue-prior epoch and explicit missingness, not operational historical availability or hard barriers",
        "selection": "Geodesic short endpoint energy, equal physical/exclusion group within actual species then horizon and available validation split; point errors and radial90 coverage alongside",
        "reserved_targets_selected": False,
        "training_pairs": len(rows), "training_origins": len(origins),
        "train_case_ids_sha256": hashlib.sha256(np.asarray(a["case_index"][origins], dtype="<i8").tobytes()).hexdigest(),
        "budget": dict(requested_epochs=epochs, effective_epochs=effective_epochs, batch_size=batch_size,
                       lr=lr, weight_decay=weight_decay, negatives=negatives, validation_samples=validation_samples, seed=seed, patience=patience,
                       training_pairs_per_epoch=train_pairs_per_epoch or len(rows), validation_origins=validation_origins),
        "validation_panel_case_ids_sha256": hashlib.sha256(np.asarray(a.get("case_index", np.arange(len(a["split"])))[panel], dtype="<i8").tobytes()).hexdigest() if panel is not None else None,
        "training_budget_rule": "All TRAIN endpoints retain their sampling probability; a declared draw budget limits optimizer steps"}
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(scope, indent=2, allow_nan=False) + "\n")
    history, best, stale, began = [], math.inf, 0, time.monotonic()
    try:
        for epoch in range(effective_epochs + 1):
            losses = []
            if epoch:
                model.train()
                order = rng.choice(len(rows), train_pairs_per_epoch or len(rows), replace=True, p=probability)
                for start in range(0, len(order), batch_size):
                    selected = order[start:start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = conditional_loss(model, a, rows[selected], heads[selected], columns[selected], rng, negatives)
                    value = loss.mean(); value.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite iSSF conditional-choice gradient")
                    optimizer.step(); losses.append(float(value.detach()))
            metrics = validate(model, a, batch_size, validation_samples, seed, origin_ids=panel)
            if not math.isfinite(metrics["selection"]):
                raise ValueError("Nonfinite iSSF physical validation score")
            history.append({"epoch": epoch, "train_conditional_choice_loss": float(np.mean(losses)) if losses else None,
                            "elapsed_seconds": time.monotonic() - began, **metrics})
            (out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
            ck = {**scope, "model": model.state_dict(), "epoch": epoch, "metrics": metrics}
            if metrics["selection"] < best:
                best, stale = metrics["selection"], 0; torch.save(ck, out / "best.pt")
            else:
                stale += 1
            torch.save(ck, out / "last.pt")
            if epoch == 0:
                torch.save(ck, out / "initial.pt")
            print(json.dumps({"model": "iSSF-" + variant, "epoch": epoch, "endpoint_energy_km": metrics["selection"],
                              "best_km": best, "seconds": round(history[-1]["elapsed_seconds"], 1)}), flush=True)
            if stale >= patience:
                break
    finally:
        model.close()
    return {"output": str(out), "epochs_observed": len(history), "best_endpoint_energy_km": best, **hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="linear")
    parser.add_argument("--sources"); parser.add_argument("--context-mode", choices=["actual", "availability", "masked"], default="masked")
    for name, default in (("epochs", 20), ("batch-size", 16), ("window", 101), ("hidden", 32), ("seed", 2026), ("negatives", 100), ("validation-samples", 64)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-pairs-per-epoch", type=int, default=0)
    parser.add_argument("--validation-origins", type=int, default=0)
    print(json.dumps(train(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
