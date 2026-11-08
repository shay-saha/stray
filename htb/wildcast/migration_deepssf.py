"""Pinned author-core deepSSF for local 1h/12h/1d endpoint comparisons.

This is an explicit task adaptation, not the unreleased MoveBench recipe. The
unchanged author CNN predicts categorical raster cells; uniform within-cell
jitter and a fixed 10% continuous Student-t tail define the complete law. All
observed outcomes, including outside-window outcomes, remain in its likelihood.
Endpoints at different horizons are independent, not a learned migration route.
Only bounded, shared local GHSL fields are read; no per-case crops are stored.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import R, unproject

PROJECT = Path(__file__).resolve().parent.parent
AUTHOR_COMMIT = "4eef65e021de8384b570b079bd34fa3f90d949da"
AUTHOR_HASHES = {"model.py": "c0b794dfd1708a6e6a1ef928e30f776fa9933d2e9102f34377933ba295f678c5",
                 "utils.py": "62a262b988d24cd1d103df032f73c24fc6d64a3729ab127e84feb73d54c01e54",
                 "LICENSE": "d7d214d8f94e17b2893bf907115eb5f746bbda3c9426ec56379b3def9b4aa6e2"}
SUPPORTED_HORIZONS = (1 / 24, .5, 1.)
MODEL_KIND = "wildcast_migration_deepssf_short"
SELECTION_VERSION = "migration-deepssf-short-endpoint-physical-v1"
TAIL_WEIGHT, TAIL_DF, MAX_RADIUS = .1, 3., math.pi * R
RASTER_NAMES = ["ghsl_native_built_fraction", "ghsl_native_available", "ghsl_epoch_age_25years",
                "ghsl_prior_epoch_present", "ghsl_release_prior_calendar_year", "ghsl_retrospective_reconstruction"]
_AUTHOR_SCOPE = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def author_core():
    """Compile exact pinned classes, replacing only their package import wiring."""
    global _AUTHOR_SCOPE
    root = PROJECT / "third_party/deepssf"
    if any(not (root / name).is_file() or sha256(root / name) != digest for name, digest in AUTHOR_HASHES.items()):
        raise ValueError("Pinned MIT deepSSF author source/licence checksum differs")
    if _AUTHOR_SCOPE is None:
        scope = {"__name__": "wildcast.deepssf_author", "torch": torch}
        helpers = ast.parse((root / "utils.py").read_text())
        get_device = [node for node in helpers.body if isinstance(node, ast.FunctionDef) and node.name == "get_device"]
        if len(get_device) != 1:
            raise ValueError("Pinned author get_device definition is unavailable")
        exec(compile(ast.Module(body=get_device, type_ignores=[]), str(root / "utils.py"), "exec"), scope)
        tree = ast.parse((root / "model.py").read_text())
        # Everything else, including every class/method body, is unchanged.
        tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "deepssf.utils")]
        exec(compile(tree, str(root / "model.py"), "exec"), scope)
        _AUTHOR_SCOPE = scope
    return _AUTHOR_SCOPE


def provenance():
    author_core()  # Verify even if its imported class objects are already cached.
    return {"repository": "https://github.com/swforrest/deepSSF_package", "commit": AUTHOR_COMMIT,
            "version": "0.3.2", "licence": "MIT", "source_sha256": dict(AUTHOR_HASHES),
            "pretrained_weights": False,
            "adapter": "Unchanged author model classes; package get_device import wired from the pinned helper. Global horizon-conditioned fit, learned species scalar, GHSL rasters, and continuous cell/tail law are task adaptations.",
            "official_MoveBench_reproduction": False}


def prediction_code_hashes():
    return {name: sha256(Path(__file__).with_name(name)) for name in
            ("migration_deepssf.py", "data.py", "migration_human_pressure.py")}


def training_schema(manifest, arrays):
    """Validate either frozen multiscale generation without guessing dimensions."""
    from .migration_protocol import schema_for
    if manifest.get("protocol") not in (
            "wildcast-multiscale-observed-v1", "wildcast-multiscale-repaired-v2"):
        raise ValueError("Short endpoint fitting requires a known frozen multiscale protocol")
    return schema_for(manifest, arrays)


def fitting_code_hashes(manifest):
    """Pin the selected feature/ownership schema as well as the shared trainer."""
    from .migration_protocol import protocol_code_files
    files = [*protocol_code_files(manifest), Path(__file__).with_name("migration_train.py")]
    return {path.name: sha256(path) for path in files}


def _horizon_indices(horizons, supported=SUPPORTED_HORIZONS):
    values = np.asarray(horizons, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("Requested horizons must be finite, nonempty and strictly increasing")
    indices = []
    for value in values:
        matches = np.flatnonzero(np.asarray(supported, dtype=float) == value)
        if len(matches) != 1:
            raise ValueError("deepSSF short adapter supports only the declared 1h/12h/1d horizons; no 90-day route fallback")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[min(np.searchsorted(np.cumsum(weights), quantile * weights.sum()), len(values) - 1)])


def fit_geometry(a, window=101, quantile=.99, minimum_half_width=1000., maximum_half_width=5_000_000.,
                 *, allowed_cross_taxon_exclusion_ids=()):
    """Training-only, animal/species-balanced scales; unseen species use row zero.

    No validation values or reserved target arrays are indexed. Supports and
    tails use all valid training outcomes, including long-distance outcomes.
    """
    if window < 17 or window > 101 or window % 2 != 1 or not .5 <= quantile < 1:
        raise ValueError("Choose an odd 17..101 raster window and a predeclared support quantile")
    if not 0 < minimum_half_width <= maximum_half_width < MAX_RADIUS / math.sqrt(2):
        raise ValueError("Raster support must remain strictly within the AEQD antipodal disc")
    indices = _horizon_indices(SUPPORTED_HORIZONS, a["horizons"])
    train = np.flatnonzero(a["split"] == 0)
    if not len(train):
        raise ValueError("Training-only support fitting requires training origins")
    species = np.asarray(a["species"])[train]
    if np.any(species < 0):
        raise ValueError("Invalid training species token")
    n_species = int(species.max(initial=0)) + 1
    width, tail = np.empty((n_species, len(indices))), np.empty((n_species, len(indices)))
    counts = np.zeros((n_species, len(indices)), dtype=int)
    actual_species = np.asarray(a.get("metric_species", a["species"]))[train]
    identities = np.asarray(a["individual"])[train]
    for head, horizon in enumerate(indices):
        observed = np.asarray(a["target_mask"])[train, horizon].astype(bool)
        selected = train[observed]
        if not len(selected):
            raise ValueError("Every declared short horizon needs actual training targets")
        values = np.asarray(a["y"])[selected, horizon].astype(float) * 1_000_000.
        if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=-1) >= MAX_RADIUS):
            raise ValueError("Training endpoints must lie in the finite spherical AEQD domain")
        physical, labels = identities[observed], actual_species[observed]
        groups = pd.DataFrame({"animal": physical, "species": labels})
        taxa_per_group = groups.groupby("animal").species.nunique()
        conflicts = set(taxa_per_group.index[taxa_per_group > 1].tolist())
        if conflicts - set(allowed_cross_taxon_exclusion_ids):
            raise ValueError("Physical training identities have undeclared conflicting species labels")
        animals_per_species = groups.drop_duplicates().groupby("species").size().to_dict()
        observations_per_pair = groups.groupby(["species", "animal"]).size().to_dict()
        weights = np.asarray([1 / observations_per_pair[(s, i)] / animals_per_species[s] for i, s in zip(physical, labels)])
        radial, square = np.linalg.norm(values, axis=-1), np.max(np.abs(values), axis=-1)
        median_factor = math.sqrt(TAIL_DF * (2 ** (2 / TAIL_DF) - 1))
        for token in range(n_species):
            take = np.ones(len(values), bool) if token == 0 else species[observed] == token
            if not take.any():
                width[token, head], tail[token, head] = width[0, head], tail[0, head]
                continue
            radius = np.clip(_weighted_quantile(square[take], weights[take], quantile), minimum_half_width, maximum_half_width)
            width[token, head] = 2 * radius / window
            tail[token, head] = max(100., _weighted_quantile(radial[take], weights[take], .5) / median_factor)
            counts[token, head] = int(take.sum())
    return {"cell_metres": width.tolist(), "tail_scale_metres": tail.tolist(), "training_observations": counts.tolist(),
            "supported_horizons_days": list(SUPPORTED_HORIZONS), "support_quantile": float(quantile),
            "minimum_half_width_metres": float(minimum_half_width), "maximum_half_width_metres": float(maximum_half_width),
            "fit_split": "train", "weighting": "Equal physical/exclusion group within actual species, then species; row zero is pooled training fallback",
            "allowed_cross_taxon_exclusion_ids": sorted(map(int, allowed_cross_taxon_exclusion_ids)),
            "cross_taxon_rule": "Only verified manifest-declared uncertain-alias blocks may span taxa. Weight their within-species pairs separately; preserve their original dependence/exclusion ID."}


def predictor_scalars(a, ids, horizons):
    """Clock/horizon/prior heading only; no query target/mask access."""
    ids = np.asarray(ids, dtype=int)
    horizons = np.asarray(horizons, dtype=float)
    if horizons.shape != ids.shape or not np.isfinite(horizons).all() or np.any(horizons <= 0):
        raise ValueError("One known requested horizon is required per predictor origin")
    origin = np.asarray(a["origin"])[ids]
    history, mask = np.asarray(a["history"])[ids], np.asarray(a["history_mask"])[ids].astype(bool)
    if (origin.shape != (len(ids), 3) or history.ndim != 3 or history.shape[-1] < 3
            or mask.shape != history.shape[:2] or not np.isfinite(origin).all()
            or np.any(np.abs(origin[:, 0]) > 90) or np.any(np.abs(origin[:, 1]) > 180)
            or not np.isfinite(history[mask]).all()):
        raise ValueError("Only finite observed GPS predictors are accepted")
    age = np.where(mask & (history[..., 2] > 0), history[..., 2], np.inf)
    previous = age.argmin(-1)
    known = np.isfinite(age[np.arange(len(ids)), previous])
    displacement = -history[np.arange(len(ids)), previous, :2]
    bearing = np.where(known, np.arctan2(displacement[:, 1], displacement[:, 0]), 0.)
    dates = pd.to_datetime(origin[:, 2], unit="s", utc=True)
    hour = dates.hour.to_numpy() + dates.minute.to_numpy() / 60 + dates.second.to_numpy() / 3600
    day = dates.dayofyear.to_numpy()
    scalar = np.column_stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                              np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25),
                              horizons * 24, known]).astype("float32")
    return scalar, bearing[:, None].astype("float32")


class GHSLRaster:
    """On-demand issue-frozen fields, using the existing verified bounded LRU."""

    def __init__(self, sources, expected=None):
        from .migration_human_pressure import HumanPressure
        class LocalSources(HumanPressure):
            def _path(self, raw):
                # Portable closure keeps sources.json byte-exact. Its original
                # absolute paths remain provenance; local copies are located by
                # basename and then undergo the same archive/raster hash proof.
                local = self.root / Path(raw).name
                if local.is_file() and local.resolve().is_relative_to(self.root.resolve()):
                    return local
                return super()._path(raw)
        self.context = LocalSources(sources, max_tiles=8, expected=expected,
                                    native_cache_bytes=16 * 1024**3)
        self.fingerprint, self.provenance = self.context.fingerprint, self.context.provenance

    def close(self):
        self.context.close()

    def patches(self, origins, cell_metres, window, mode="actual"):
        if mode not in ("actual", "availability", "masked"):
            raise ValueError("Choose actual/availability/masked raster context")
        origins, cell_metres = np.asarray(origins), np.asarray(cell_metres)
        output = np.zeros((len(origins), len(RASTER_NAMES), window, window), dtype="float32")
        if mode == "masked":
            return output
        row, col = np.indices((window, window)); center = window // 2
        # Vectorize only the current bounded batch. The issue timestamp and
        # sampled native containing cells are identical to per-origin lookup.
        xy = np.stack([col - center, center - row], -1)[None] * cell_metres[:, None, None, None]
        coordinates = unproject(xy, origins[:, None, None, 0], origins[:, None, None, 1])
        fields = self.context.lookup(coordinates, origins[:, 2], native_only=True)
        available = fields["available"][..., 0]
        content = np.where(available, fields["values"][..., 0], 0.) if mode == "actual" else np.zeros_like(available, float)
        channels = [content, available, fields["epoch_age_years"] / 25, fields["epoch"] >= 0,
                    fields["release_prior_calendar_year"], fields["retrospective_reconstruction"]]
        return np.stack(channels, axis=1).astype("float32")


def tail_log_density(y_metres, scale_metres):
    """Density w.r.t. AEQD square metres, normalized inside r < pi*R."""
    y = y_metres.double(); scale = scale_metres.double()
    r2 = y.square().sum(-1)
    z = -torch.expm1(-TAIL_DF / 2 * torch.log1p(MAX_RADIUS ** 2 / (TAIL_DF * scale.square())))
    logp = -math.log(2 * math.pi) - 2 * scale.log() - (TAIL_DF + 2) / 2 * torch.log1p(r2 / (TAIL_DF * scale.square())) - z.log()
    return torch.where(r2 < MAX_RADIUS ** 2, logp, -torch.inf)


def sample_tail(scale_metres, samples, rng):
    """Exact inverse-CDF disc truncation; no clipped or rejected point masses."""
    scale = np.asarray(scale_metres, dtype=float)[:, None]
    if samples < 1 or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Tail sampling requires positive finite scales and sample count")
    z = -np.expm1(-TAIL_DF / 2 * np.log1p(MAX_RADIUS ** 2 / (TAIL_DF * scale ** 2)))
    probability = rng.random((len(scale), samples)) * z
    radius = scale * np.sqrt(TAIL_DF * np.expm1(-2 / TAIL_DF * np.log1p(-probability)))
    angle = rng.uniform(-np.pi, np.pi, radius.shape)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], -1)


def mixture_nll(log_cells, y_metres, cell_metres, tail_scale_metres, geographic=True):
    """Continuous cell/tail NLL; outside-window outcomes contribute tail loss.

    AEQD area dA_plane maps to spherical area J*dA_plane with
    J=sin(r/R)/(r/R). Geographic density is plane density/J, so geographic
    NLL = plane NLL + log(J). The Jacobian is fixed w.r.t. model parameters.
    Sampling in the plane and unprojecting already implements this pushforward.
    """
    window = log_cells.shape[-1]
    if log_cells.ndim != 3 or log_cells.shape[1] != window:
        raise ValueError("Expected a square probability grid per endpoint")
    if not torch.isfinite(log_cells).all() or not torch.all(torch.logsumexp(log_cells.flatten(1), -1).abs() < 1e-4):
        raise ValueError("Raster log probabilities must be finite and normalized")
    y = y_metres.double(); cell = cell_metres.double()
    row = torch.floor(-y[:, 1] / cell + window / 2).long()
    col = torch.floor(y[:, 0] / cell + window / 2).long()
    inside = (row >= 0) & (row < window) & (col >= 0) & (col < window)
    picked = log_cells[torch.arange(len(y), device=y.device), row.clamp(0, window - 1), col.clamp(0, window - 1)].double()
    raster = torch.where(inside, picked - 2 * cell.log() + math.log1p(-TAIL_WEIGHT), -torch.inf)
    tail = tail_log_density(y, tail_scale_metres) + math.log(TAIL_WEIGHT)
    loss = -torch.logaddexp(raster, tail)
    if geographic:
        radius = y.norm(dim=-1)
        jacobian = torch.sinc(radius / (math.pi * R))
        loss = loss + torch.log(jacobian)
    if not torch.isfinite(loss).all():
        raise ValueError("Endpoint lies outside the declared spherical AEQD support or model density is invalid")
    return loss, inside


class DeepSSFShort(nn.Module):
    """Author CNN with declared local geometry, species scalar and mixture law."""

    def __init__(self, geometry, window=101, channels=8, hidden=64, embedding=4, dropout=.2,
                 context_mode="masked", fields_root=None, fields_fingerprint=None):
        super().__init__()
        if window < 17 or window > 101 or window % 2 != 1:
            raise ValueError("Use an odd bounded raster window of 17..101 pixels")
        if context_mode not in ("masked", "actual", "availability"):
            raise ValueError("Choose masked/actual/availability context")
        if context_mode != "masked" and (not fields_root or not fields_fingerprint):
            raise ValueError("Nonmasked context requires a frozen GHSL source fingerprint")
        if not 1 <= channels <= 32 or not 8 <= hidden <= 256 or not 0 <= embedding <= 16 or not 0 <= dropout < 1:
            raise ValueError("deepSSF network exceeds declared architecture bounds")
        expected = (len(geometry["cell_metres"]), len(SUPPORTED_HORIZONS))
        cell = np.asarray(geometry["cell_metres"], dtype=float)
        tail = np.asarray(geometry["tail_scale_metres"], dtype=float)
        if (tuple(geometry.get("supported_horizons_days", [])) != SUPPORTED_HORIZONS
                or cell.shape != expected or tail.shape != expected or not np.isfinite(cell).all()
                or not np.isfinite(tail).all() or np.any(cell <= 0) or np.any(tail <= 0)
                or np.any(cell * window / 2 * math.sqrt(2) >= MAX_RADIUS)):
            raise ValueError("Invalid frozen training-only raster/tail geometry")
        self.config = dict(geometry=geometry, window=int(window), channels=int(channels), hidden=int(hidden),
                           embedding=int(embedding), dropout=float(dropout), context_mode=context_mode,
                           fields_root=str(fields_root) if fields_root else None, fields_fingerprint=fields_fingerprint)
        self.supported_horizons = tuple(SUPPORTED_HORIZONS)
        self.register_buffer("cell_metres", torch.tensor(cell, dtype=torch.float64))
        self.register_buffer("tail_scale_metres", torch.tensor(tail, dtype=torch.float64))
        self.species_embedding = nn.Embedding(expected[0], embedding) if embedding else None
        source = author_core()
        scalars = 6 + embedding
        params = source["ModelParams"](dict(batch_size=1, image_dim=window, pixel_size=1.,
            dim_in_nonspatial_to_grid=scalars, dense_dim_in_nonspatial=scalars, dense_dim_hidden=hidden,
            dense_dim_in_all=source["flattened_conv_dim"](window, 3, channels),
            input_channels=len(RASTER_NAMES) + scalars, output_channels=channels, kernel_size=3,
            stride=1, kernel_size_mp=2, stride_mp=2, padding=1, num_movement_params=12,
            dropout=dropout, n_conv_layers_hab=4, n_conv_layers_move=3, device="cpu"))
        self.core = source["ConvJointModel"](params)
        self._raster = None

    def close(self):
        if self._raster is not None:
            self._raster.close(); self._raster = None

    def _tokens(self, a, ids):
        tokens = np.asarray(a["species"])[ids].astype(int)
        if np.any(tokens < 0):
            raise ValueError("Species token must be nonnegative")
        return np.where(tokens < len(self.cell_metres), tokens, 0)

    def inputs(self, a, ids, head_indices):
        ids, heads = np.asarray(ids, int), np.asarray(head_indices, int)
        if (ids.ndim != 1 or ids.shape != heads.shape or np.any(ids < 0) or np.any(ids >= len(a["origin"]))
                or np.any(heads < 0) or np.any(heads >= len(SUPPORTED_HORIZONS))):
            raise ValueError("One supported endpoint head required per origin")
        tokens = self._tokens(a, ids)
        device = self.cell_metres.device
        if self.training:
            # Train the actual unknown-species law, including its pooled spatial
            # geometry and field footprint, not merely embedding row zero.
            pooled = (torch.rand(len(tokens), device=device) < .1).cpu().numpy()
            tokens = np.where(pooled, 0, tokens)
        cell = self.cell_metres[tokens, heads]
        scalar, bearing = predictor_scalars(a, ids, np.asarray(SUPPORTED_HORIZONS)[heads])
        if self.config["context_mode"] == "masked":
            patches = np.zeros((len(ids), len(RASTER_NAMES), self.config["window"], self.config["window"]), "float32")
        else:
            if self._raster is None:
                self._raster = GHSLRaster(self.config["fields_root"], expected=self.config["fields_fingerprint"])
            patches = self._raster.patches(a["origin"][ids], cell.detach().cpu().numpy(), self.config["window"], self.config["context_mode"])
        return (torch.as_tensor(patches, device=device), torch.as_tensor(scalar, device=device),
                torch.as_tensor(bearing, device=device), torch.as_tensor(tokens, device=device)), cell, self.tail_scale_metres[tokens, heads]

    def forward(self, patches, scalars, bearing, species):
        if self.species_embedding is not None:
            scalars = torch.cat([scalars, self.species_embedding(species)], -1)
        raw = self.core((patches, scalars, bearing)).sum(-1)
        return torch.log_softmax(raw.flatten(1), -1).reshape_as(raw)

    @torch.no_grad()
    def predict_endpoints(self, a, ids, horizons=SUPPORTED_HORIZONS, samples=64, rng=None):
        """Independent endpoint draws [case,sample,requested horizon,east/north]."""
        if samples < 2:
            raise ValueError("At least two predictive samples are required")
        heads, ids = _horizon_indices(horizons), np.asarray(ids, dtype=int)
        if ids.ndim != 1 or not len(ids):
            raise ValueError("Nonempty one-dimensional origin IDs required")
        if self.training:
            raise ValueError("Set the deepSSF model to eval before prediction")
        rng = rng if rng is not None else np.random.default_rng(2026)
        result = np.empty((len(ids), samples, len(heads), 2), dtype="float64")
        window, center = self.config["window"], self.config["window"] // 2
        for h, head in enumerate(heads):
            inputs, cell, tail_scale = self.inputs(a, ids, np.full(len(ids), head))
            probabilities = self(*inputs).exp().double().cpu().numpy().reshape(len(ids), -1)
            cell = cell.cpu().numpy()
            draws = sample_tail(tail_scale.cpu().numpy(), samples, rng)
            tail = rng.random((len(ids), samples)) < TAIL_WEIGHT
            for row in range(len(ids)):
                selected = np.flatnonzero(~tail[row])
                if not len(selected):
                    continue
                p = probabilities[row] / probabilities[row].sum()
                index = rng.choice(window * window, len(selected), p=p)
                rr, cc = np.divmod(index, window)
                jitter = rng.uniform(-.5, .5, (len(selected), 2))
                draws[row, selected] = np.column_stack([cc - center + jitter[:, 0], center - rr + jitter[:, 1]]) * cell[row]
            result[:, :, h] = draws / 1_000_000.
        return result

    def predict_cases(self, a, ids, samples, rng=None, baselines=None, seed=2026):
        # A full nine-horizon migration cache must use the explicit endpoint API.
        return self.predict_endpoints(a, ids, a["horizons"], samples, rng if rng is not None else np.random.default_rng(seed))


def observed_pairs(a, split_codes):
    codes = tuple(split_codes)
    if not codes or not set(codes) <= {0, 1, 2}:
        raise ValueError("deepSSF development fitting/scoring never selects reserved outcomes")
    columns = _horizon_indices(SUPPORTED_HORIZONS, a["horizons"])
    ids = np.flatnonzero(np.isin(a["split"], codes))
    rows, heads = np.nonzero(a["target_mask"][ids][:, columns])
    return ids[rows], heads, columns[heads]


def loss_for_pairs(model, a, ids, heads, columns):
    """Known training/development endpoints enter the loss, never predictors."""
    ids, heads, columns = np.asarray(ids, int), np.asarray(heads, int), np.asarray(columns, int)
    if (ids.shape != heads.shape or ids.shape != columns.shape or not np.isin(a["split"][ids], [0, 1, 2]).all()
            or not a["target_mask"][ids, columns].all()
            or not np.array_equal(a["horizons"][columns], np.asarray(SUPPORTED_HORIZONS)[heads])):
        raise ValueError("Loss needs observed train/development targets matching the requested horizon; reserved truth is excluded")
    inputs, cell, scale = model.inputs(a, ids, heads)
    target = torch.as_tensor(a["y"][ids, columns].astype(float) * 1_000_000., device=cell.device)
    return mixture_nll(model(*inputs), target, cell, scale)


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    if ck.get("model_kind") != MODEL_KIND or ck.get("author") != provenance():
        raise ValueError("Wrong deepSSF kind or pinned author provenance")
    if ck.get("adapter_sha256") != sha256(__file__):
        raise ValueError("deepSSF adapter changed after checkpoint creation")
    if ck.get("prediction_code_sha256") != prediction_code_hashes():
        raise ValueError("deepSSF prediction dependency changed after checkpoint creation")
    if ck.get("tail") != {"weight": TAIL_WEIGHT, "degrees_of_freedom": TAIL_DF, "disc_radius_metres": MAX_RADIUS, "scale_fit": "train only"}:
        raise ValueError("deepSSF checkpoint tail policy differs from its sampling law")
    for key in ("manifest_sha256", "cases_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(ck.get(key, ""))):
            raise ValueError("deepSSF checkpoint requires frozen reference hashes")
    model = DeepSSFShort(**ck["config"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    # Geometry buffers and serialized policy must describe the same sampling law.
    for name in ("cell_metres", "tail_scale_metres"):
        expected = torch.tensor(model.config["geometry"][name], dtype=torch.float64, device=device)
        if not torch.equal(getattr(model, name), expected):
            raise ValueError("deepSSF geometry state differs from its frozen policy")
    return model, ck


def validation_panel(a, maximum=0, seed=1701, horizons=SUPPORTED_HORIZONS):
    """Fixed split/taxon/exclusion-balanced metadata panel; no target values."""
    columns = _horizon_indices(horizons, a["horizons"])
    ids = np.flatnonzero(np.isin(a["split"], [1,2]) & a["target_mask"][:, columns].any(-1))
    if maximum <= 0 or len(ids) <= maximum:
        return ids
    rng = np.random.default_rng(seed)
    species = a.get("metric_species", a["species"])
    blocks = {}
    for case in ids:
        key = int(a["split"][case]), int(species[case]), int(a["individual"][case])
        blocks.setdefault(key, []).append(int(case))
    queues = [rng.permutation(blocks[key]) for key in sorted(blocks)]
    order = rng.permutation(len(queues))
    result, depth = [], 0
    while len(result) < maximum:
        for q in order:
            if depth < len(queues[q]):
                result.append(int(queues[q][depth]))
                if len(result) == maximum:
                    break
        depth += 1
    return np.asarray(sorted(result), dtype="int64")


def validate(model, a, batch_size=16, samples=64, seed=2026, origin_ids=None):
    from .migration_evaluate import macro_scores, score_paths
    model.eval(); frames = []
    columns = _horizon_indices(SUPPORTED_HORIZONS, a["horizons"])
    rng = np.random.default_rng(seed)
    for split, name in ((1, "val_individual"), (2, "val_year")):
        ids = np.flatnonzero((a["split"] == split) & a["target_mask"][:, columns].any(-1))
        if origin_ids is not None:
            ids = ids[np.isin(ids, origin_ids)]
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            paths = model.predict_endpoints(a, batch, samples=samples, rng=rng)
            scores = score_paths(paths, a["target_latlon"][batch][:, columns], a["target_mask"][batch][:, columns], a["origin"][batch], device=str(model.cell_metres.device))
            for j, horizon in enumerate(SUPPORTED_HORIZONS):
                good = a["target_mask"][batch, columns[j]]
                frames.append(pd.DataFrame({"model": "deepSSF-short", "split": name, "species": a.get("metric_species", a["species"])[batch][good],
                    "individual": a["individual"][batch][good], "horizon_days": horizon,
                    "endpoint_energy_km": scores["endpoint_energy_km"][good, j], "point_error_km": scores["point_error_km"][good, j],
                    "radial_coverage90": scores["radial_coverage90"][good, j]}))
    if not frames or not any(len(f) for f in frames):
        raise ValueError("Observed development endpoints are required")
    overall, _, horizons = macro_scores(pd.concat(frames, ignore_index=True))
    return {"selection": float(overall.endpoint_energy_km.mean()), "selection_units": "km",
            "selection_version": SELECTION_VERSION, "split_metrics": overall.to_dict("records"),
            "horizon_metrics": horizons.to_dict("records"), "samples": samples, "seed": seed}


def train(data, output, sources=None, context_mode="masked", epochs=20, batch_size=16,
          window=101, channels=8, hidden=64, embedding=4, lr=.0003, seed=2026,
          validation_samples=64, patience=5, device="cpu", train_pairs_per_epoch=0, validation_origins=0):
    """Finite developer-invoked fit; only frozen train/val labels are selected."""
    from .migration_train import load_training_data, sampling_probabilities
    from .migration_protocol import cross_taxon_exclusion_ids
    out = Path(output)
    if out.exists():
        raise FileExistsError("Use a new immutable deepSSF experiment directory")
    if epochs < 0 or batch_size < 1 or validation_samples < 2 or patience < 1 or train_pairs_per_epoch < 0 or validation_origins < 0:
        raise ValueError("Invalid finite training/validation budget")
    a, manifest, hashes = load_training_data(data)
    schema = training_schema(manifest, a)
    panel = validation_panel(a, validation_origins) if validation_origins else None
    allowed_cross_taxon = cross_taxon_exclusion_ids(manifest)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    geometry = fit_geometry(a, window=window, allowed_cross_taxon_exclusion_ids=allowed_cross_taxon)
    fingerprint, field_provenance = None, None
    if context_mode != "masked":
        context = GHSLRaster(sources)
        fingerprint, field_provenance = context.fingerprint, context.provenance
        context.close()
    model = DeepSSFShort(geometry, window=window, channels=channels, hidden=hidden, embedding=embedding,
                         context_mode=context_mode, fields_root=sources, fields_fingerprint=fingerprint).to(device)
    rows, heads, columns = observed_pairs(a, [0])
    if not len(rows):
        raise ValueError("No observed training endpoints")
    origins, inverse, count = np.unique(rows, return_inverse=True, return_counts=True)
    probability = sampling_probabilities(a, origins, "species_animal",
        allowed_cross_taxon_exclusion_ids=allowed_cross_taxon)[inverse] / count[inverse]
    probability /= probability.sum()
    author = provenance()
    scope = {"model_kind": MODEL_KIND, "scope": "Local author-core short endpoints; no official MoveBench reproduction or 90-day route claim",
        "independent_horizon_draws": True, "supported_horizons_days": list(SUPPORTED_HORIZONS),
        "selection_version": SELECTION_VERSION, "selection": "Geodesic endpoint energy; physical animal/species/horizon macro, then equal available validation splits",
        "loss": "Continuous AEQD cell-uniform plus fixed tail likelihood, converted to spherical-area density with exact projection Jacobian",
        "tail": {"weight": TAIL_WEIGHT, "degrees_of_freedom": TAIL_DF, "disc_radius_metres": MAX_RADIUS, "scale_fit": "train only"},
        "outcomes": "All observed short endpoints retained, including outside raster and missing-GHSL cases",
        "context": field_provenance, "raster_channels": RASTER_NAMES,
        "context_caveat": "Native 1km containing-cell GHSL R2023A fractions sampled on a local output grid; finer grid spacing adds no measured spatial resolution. Prior nominal epoch is retrospective, not historical operational availability.",
        "scalar_features": "UTC-compatible clock-axis hour/day cycles, requested horizon hours, past-heading availability and learned species scalar; no future realized sampling interval",
        "unknown_species_training": "10% of training pairs use pooled token-zero geometry, tail scale, field footprint and species scalar together; inference uses exact declared token or pooled fallback",
        "geometry": geometry, "author": author, "config": model.config,
        "cache_protocol": manifest["protocol"],
        "feature_schema": {"history_names": schema.HISTORY_NAMES, "static_names": schema.STATIC_NAMES,
                           "history_lags_days": schema.LAGS.tolist()},
        "adapter_sha256": sha256(__file__), "prediction_code_sha256": prediction_code_hashes(),
        "fitting_code_sha256": fitting_code_hashes(manifest), **hashes,
        "training_pairs": len(rows), "training_origins": len(origins),
        "train_case_ids_sha256": hashlib.sha256(np.asarray(a["case_index"][origins], dtype="<i8").tobytes()).hexdigest(),
        "budget": dict(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, validation_samples=validation_samples, patience=patience,
                       training_pairs_per_epoch=train_pairs_per_epoch or len(rows), validation_origins=validation_origins),
        "validation_panel_case_ids_sha256": hashlib.sha256(np.asarray(a.get("case_index", np.arange(len(a["split"])))[panel], dtype="<i8").tobytes()).hexdigest() if panel is not None else None,
        "validation_panel_rule": "Fixed metadata-only split/taxon/exclusion round-robin seed1701; final comparisons use common held-out cases",
        "training_budget_rule": "Every TRAIN endpoint stays eligible under the unchanged sampling law; a declared optimizer-step budget limits draws per epoch"}
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(scope, indent=2, allow_nan=False) + "\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, stale, history, began = math.inf, 0, [], time.monotonic()
    try:
        for epoch in range(epochs + 1):
            total, outside = [], 0
            if epoch:
                model.train()
                order = rng.choice(len(rows), train_pairs_per_epoch or len(rows), replace=True, p=probability)
                for start in range(0, len(order), batch_size):
                    chosen = order[start:start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    losses, inside = loss_for_pairs(model, a, rows[chosen], heads[chosen], columns[chosen])
                    loss = losses.mean(); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite deepSSF training gradient")
                    optimizer.step(); total.append(float(loss.detach())); outside += int((~inside).sum())
            metrics = validate(model, a, batch_size, validation_samples, seed, origin_ids=panel)
            record = {"epoch": epoch, "train_nll_per_square_metre": float(np.mean(total)) if total else None,
                      "training_draws_outside_raster": outside, "elapsed_seconds": time.monotonic() - began, **metrics}
            history.append(record)
            (out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
            ck = {**scope, "model": model.state_dict(), "epoch": epoch, "metrics": metrics}
            if metrics["selection"] < best:
                best, stale = metrics["selection"], 0
                torch.save(ck, out / "best.pt")
            else:
                stale += 1
            torch.save(ck, out / "last.pt")
            if epoch == 0:
                torch.save(ck, out / "initial.pt")
            print(json.dumps({"model": "deepSSF", "epoch": epoch, "endpoint_energy_km": metrics["selection"],
                              "best_km": best, "seconds": round(record["elapsed_seconds"], 1)}), flush=True)
            if stale >= patience:
                break
    finally:
        model.close()
    return {"output": str(out), "best_endpoint_energy_km": best, "epochs_observed": len(history), **hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--sources"); parser.add_argument("--context-mode", choices=["actual", "availability", "masked"], default="masked")
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=101); parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-pairs-per-epoch", type=int, default=0)
    parser.add_argument("--validation-origins", type=int, default=0)
    print(json.dumps(train(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
