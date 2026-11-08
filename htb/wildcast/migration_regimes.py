"""Frozen past-mobility rules and whole-path migration expert routing.

High past displacement is a causal predictor regime, not a biological label.
Routing never reads query identity, calendar year, phase or future targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .migration_evaluate import load_cases
from .migration_model import load_model as load_joint, sample_paths
from .sources import dump

KIND = "wildcast_migration_regime"
PROTOCOL = "wildcast-migration-causal-regimes-v1"
LAGS = (7., 30.)
QUANTILE = .8
SCALE_GRID = (1., 1.5, 2., 3., 5.)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weighted_quantile(values, weights, quantile=QUANTILE):
    values, weights = np.asarray(values), np.asarray(weights, float)
    if not len(values) or weights.shape != values.shape or np.any(weights <= 0) or not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("Weighted mobility quantile needs finite values and positive weights")
    order = np.argsort(values, kind="stable")
    cross = np.searchsorted(np.cumsum(weights[order]), quantile * weights.sum(), side="left")
    return float(values[order[min(int(cross), len(values) - 1)]])


def _schema(a, manifest):
    names = manifest["history_names"]
    history_lags = np.asarray(manifest["history_lags_days"], float)
    indices = {key: names.index(value) for key, value in
               (("east", "east_1000km"), ("north", "north_1000km"), ("age", "actual_age_days"))}
    lag_indices = []
    for lag in LAGS:
        found = np.flatnonzero(history_lags == lag)
        if len(found) != 1:
            raise ValueError("Causal mobility lag missing or duplicated")
        lag_indices.append(int(found[0]))
    return {"history_shape": list(a["history"].shape[1:]), "history_indices": indices,
            "lag_indices": lag_indices, "lags_days": list(LAGS), "distance_multiplier_km": 1000.}


def past_mobility(a, ids, rules):
    """Only history, its mask, and predictor species are accessed at runtime."""
    ids = np.asarray(ids, dtype="int64")
    history, masks = np.asarray(a["history"]), np.asarray(a["history_mask"])
    if list(history.shape[1:]) != rules["history_shape"] or masks.shape != history.shape[:2]:
        raise ValueError("Runtime mobility history schema differs from frozen rules")
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(history)):
        raise ValueError("Invalid mobility query indices")
    feature = rules["history_indices"]
    distances = np.full((len(ids), 2), np.nan)
    available = np.zeros((len(ids), 2), bool)
    for k, (lag, j) in enumerate(zip(LAGS, rules["lag_indices"])):
        valid = masks[ids, j].astype(bool)
        value = history[ids, j]
        selected = value[valid][:, [feature["east"], feature["north"], feature["age"]]]
        if not np.isfinite(selected).all() or np.any(selected[:, 2] < lag - 1e-5):
            raise ValueError("Mobility history is invalid or later than its declared past lag")
        distances[valid, k] = np.linalg.norm(selected[:, :2], axis=-1) * rules["distance_multiplier_km"]
        available[:, k] = valid
    return distances, available


def fit_rules(a, manifest, hashes=None):
    """Per-predictor-species TRAIN quantiles; balanced global unknown fallback."""
    ids = np.flatnonzero(a["split"] == 0)
    if not len(ids):
        raise ValueError("Training observations are required for mobility rules")
    rules = {"protocol": PROTOCOL, "quantile": QUANTILE, **_schema(a, manifest), **(hashes or {})}
    distances, available = past_mobility(a, ids, rules)
    species, groups = a["species"][ids], a["individual"][ids]
    actual_species = a.get("metric_species", a["species"])[ids]
    thresholds, evidence = {}, []
    for taxon in sorted(np.unique(species)):
        cutoffs = []
        for k, lag in enumerate(LAGS):
            selected = (species == taxon) & available[:, k]
            animals = groups[selected]
            threshold = None
            if selected.any():
                _, inverse, counts = np.unique(animals, return_inverse=True, return_counts=True)
                threshold = weighted_quantile(distances[selected, k], 1. / counts[inverse])
            cutoffs.append(threshold)
            evidence.append({"predictor_species": int(taxon), "lag_days": lag, "threshold_km": threshold,
                             "training_origins": int(selected.sum()), "training_physical_groups": len(np.unique(animals))})
        thresholds[str(int(taxon))] = cutoffs
    global_thresholds = []
    for k, lag in enumerate(LAGS):
        selected = available[:, k]
        if not selected.any():
            raise ValueError("No training history for a required global mobility lag")
        values, taxon, animal = distances[selected, k], actual_species[selected], groups[selected]
        weights = np.zeros(len(values))
        for species_id in np.unique(taxon):
            rows = np.flatnonzero(taxon == species_id)
            _, inverse, counts = np.unique(animal[rows], return_inverse=True, return_counts=True)
            weights[rows] = 1. / (len(counts) * counts[inverse])
        global_thresholds.append(weighted_quantile(values, weights))
    rules.update({"thresholds_km": thresholds, "global_thresholds_km": global_thresholds,
                  "training_evidence": evidence, "predictor_species": manifest.get("species", {}),
                  "provenance": {"causal": True, "threshold_fit_split": "train", "uses_future_targets": False,
                      "inputs": ["history", "history_mask", "species"],
                      "rule": "Both observed7d/30d fixes must exist; high iff either past displacement is strictly above its TRAIN80th percentile. Species thresholds weight physical groups equally; global unknown/unavailable-species fallback weights actual species then physical groups equally. Missing history routes to base."},
                  "observed_utc": datetime.now(timezone.utc).isoformat(), "code_sha256": sha256(__file__)})
    return rules


def classify(a, ids, rules):
    if rules.get("protocol") != PROTOCOL or rules.get("quantile") != QUANTILE or rules.get("lags_days") != list(LAGS):
        raise ValueError("Unsupported causal mobility rule protocol")
    distances, available = past_mobility(a, ids, rules)
    species = np.asarray(a["species"])[np.asarray(ids, dtype="int64")]
    complete, high, fallback = available.all(-1), np.zeros(len(species), bool), np.zeros(len(species), bool)
    global_thresholds = np.asarray(rules["global_thresholds_km"], float)
    if global_thresholds.shape != (2,) or not np.isfinite(global_thresholds).all() or np.any(global_thresholds < 0):
        raise ValueError("Invalid frozen global mobility thresholds")
    for row, taxon in enumerate(species):
        if not complete[row]:
            continue
        local = rules["thresholds_km"].get(str(int(taxon))) if int(taxon) != 0 else None
        cutoff = np.asarray([global_thresholds[k] if local is None or local[k] is None else local[k] for k in range(2)], float)
        if not np.isfinite(cutoff).all() or np.any(cutoff < 0):
            raise ValueError("Invalid frozen species mobility threshold")
        fallback[row] = local is None or any(v is None for v in local)
        high[row] = bool(np.any(distances[row] > cutoff))
    regime = np.where(~complete, "missing_past_history", np.where(high, "high_past_mobility", "ordinary_past_mobility"))
    return {"high": high, "missing": ~complete, "regime": regime, "global_fallback": fallback,
            "past_displacement_km": distances}


def prepare(data, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Mobility filter artifacts are immutable; choose a new output")
    a, manifest, hashes = load_cases(data)
    rules = fit_rules(a, manifest, hashes)
    dev = np.flatnonzero(np.isin(a["split"], [0, 1, 2]))
    result = classify(a, dev, rules)
    high = np.zeros(len(a["split"]), bool)
    high[dev] = result["high"]
    train_filter = high & (a["split"] == 0)
    val_filter = high & np.isin(a["split"], [1, 2])
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "rules.json", rules)
    np.savez_compressed(output / "filters.npz", train_filter=train_filter, val_filter=val_filter)
    metadata = {**hashes, "filters_sha256": sha256(output / "filters.npz"), "rules_sha256": sha256(output / "rules.json"),
                "provenance": rules["provenance"], "cases": len(high),
                "selected_train_origins": int(train_filter.sum()), "selected_validation_origins": int(val_filter.sum()),
                "selected_validation_by_split": {str(s): int(np.sum(high & (a["split"] == s))) for s in [1, 2]},
                "development_regime_counts": {str(name): int(np.sum(result["regime"] == name)) for name in np.unique(result["regime"])},
                "reserved_selected": int(np.sum((train_filter | val_filter) & (a["split"] >= 3))),
                "code_sha256": sha256(__file__), "scope": "Causal past-mobility training/development subset, not verified migration states."}
    dump(output / "filter.json", metadata)
    return metadata


def scale_joint_parameters(parameters, scale):
    """Scale each component's full conditional covariance, preserving its mean/logit."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Joint uncertainty scale must be finite and positive")
    return {**parameters, "covariance": parameters["covariance"] * float(scale) ** 2}


def _draw_joint(model, a, ids, samples, rng, device, scale=1.):
    generator = torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**63 - 1)))
    with torch.inference_mode():
        inputs = [torch.as_tensor(a[key][ids], device=device) for key in ("history", "history_mask", "static", "species")]
        parameters = model(*inputs)
        if scale != 1.:
            parameters = scale_joint_parameters(parameters, scale)
        return sample_paths(parameters, samples, generator).detach().cpu().numpy()


class MigrationRegime:
    def __init__(self, base, rules, active=None, active_fallback="seasonal", active_scale=1., device="cpu", manifest=None):
        if active_fallback not in {"base", "seasonal"} or float(active_scale) not in SCALE_GRID:
            raise ValueError("Unsupported active fallback or non-predeclared covariance scale")
        if active is None and active_scale != 1.:
            raise ValueError("Covariance calibration requires a joint active expert")
        self.base, self.active, self.rules = base, active, rules
        self.active_fallback, self.active_scale, self.device = active_fallback, float(active_scale), str(device)
        self.manifest = manifest
        self.identical_active = active is base
        if active is not None and not self.identical_active and type(base) is type(active) and getattr(base, "config", None) == getattr(active, "config", None):
            left, right = base.state_dict(), active.state_dict()
            self.identical_active = (left.keys() == right.keys() and
                                     all(torch.equal(left[key], right[key]) for key in left))

    def predict_cases(self, a, ids, samples, rng=None, baselines=None):
        if samples < 2:
            raise ValueError("At least two whole-route samples are required")
        ids = np.asarray(ids, dtype="int64")
        rng = rng or np.random.default_rng(2026)
        routing = classify(a, ids, self.rules)
        # Draw base for the full batch first: ordinary/missing cases retain the
        # same realization as an otherwise identical base-only router control.
        result = _draw_joint(self.base, a, ids, samples, rng, self.device)
        high = np.flatnonzero(routing["high"])
        if not len(high) or (self.active is None and self.active_fallback == "base") or (self.identical_active and self.active_scale == 1.):
            return result
        if self.active is not None:
            replacement = _draw_joint(self.active, a, ids[high], samples, rng, self.device, self.active_scale)
        else:
            if baselines is None:
                from .migration_baselines import MigrationBaselines
                baselines = MigrationBaselines(a, self.manifest)
            replacement, _ = baselines.predict("seasonal", ids[high], samples, rng)
        if replacement.shape != result[high].shape or not np.isfinite(replacement).all():
            raise ValueError("Active expert did not return finite complete route samples")
        result[high] = replacement
        return result


def _load_member(path, device, hashes, expected=None):
    path = Path(path)
    before = sha256(path)
    if expected is not None and before != expected:
        raise ValueError("Regime member checksum mismatch")
    probe = torch.load(path, map_location="cpu", weights_only=True)
    if probe.get("model_kind") != "wildcast_migration_joint":
        raise ValueError("Regime base/active members must be joint migration models")
    for key, value in hashes.items():
        if probe.get(key) != value:
            raise ValueError("Regime member/cache " + key + " mismatch")
    model, metadata = load_joint(path, device)
    if sha256(path) != before:
        raise ValueError("Regime member changed during loading")
    return model, {"path": str(path.resolve()), "sha256": before, "model_kind": metadata["model_kind"], "epoch": metadata.get("epoch")}


def assemble(data, rules, base, output, active=None, active_fallback="seasonal", active_scale=1., device="cpu", calibration=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Regime checkpoints are immutable; choose a new output")
    _, manifest, hashes = load_cases(data)
    rules_path = Path(rules)
    if rules_path.is_dir():
        rules_path = rules_path / "rules.json"
    rule_hash = sha256(rules_path)
    payload = json.loads(rules_path.read_text())
    for key, value in hashes.items():
        if payload.get(key) != value:
            raise ValueError("Regime rules/cache " + key + " mismatch")
    base_model, base_record = _load_member(base, device, hashes)
    active_model, active_record = (None, None) if active is None else _load_member(active, device, hashes)
    calibration_evidence = None
    if calibration is not None:
        path = Path(calibration)
        if path.is_dir():
            path = path / "scale.json"
        calibrated = json.loads(path.read_text())
        if (calibrated.get("protocol") != "wildcast-migration-active-covariance-final-v1" or
                calibrated.get("family_selected") is not True or active_record is None):
            raise ValueError("Final scale requires an explicitly selected calibration family and joint active expert")
        for key, value in {**hashes, "rules_sha256": rule_hash, "checkpoint_sha256": active_record["sha256"]}.items():
            if calibrated.get(key) != value:
                raise ValueError("Calibration/active expert provenance mismatch: " + key)
        selected = float(calibrated["active_scale"])
        if active_scale != 1. and active_scale != selected:
            raise ValueError("Declared scale conflicts with finalized calibration")
        active_scale = selected
        calibration_evidence = {"path": str(path.resolve()), "sha256": sha256(path), "metadata": calibrated}
    MigrationRegime(base_model, payload, active_model, active_fallback, active_scale, device, manifest)
    checkpoint = {"model_kind": KIND, "selection_version": PROTOCOL, "data": str(Path(data).resolve()),
                  "manifest": manifest, **hashes, "rules": payload, "rules_path": str(rules_path.resolve()),
                  "rules_sha256": rule_hash, "base": base_record, "active": active_record,
                  "active_fallback": active_fallback, "active_scale": float(active_scale),
                  "calibration": calibration_evidence,
                  "sampling_rule": "One deterministic past-only regime per origin; each replacement is an entire joint route. No location or horizon averaging.",
                  "uncertainty_rule": "Covariance scale multiplies each joint component covariance by scale squared; logits/means/component identity are preserved.",
                  "scope": "Local development. Causal high mobility is not a verified biological migration state.", "code_sha256": sha256(__file__)}
    if sha256(rules_path) != rule_hash:
        raise ValueError("Regime rules changed during assembly")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, output / "best.pt")
    dump(output / "artifact-manifest.json", {"checkpoint_sha256": sha256(output / "best.pt"), **hashes,
                                            "rules_sha256": rule_hash, "base": base_record, "active": active_record})
    return checkpoint


def load_model(checkpoint, device="cpu"):
    before = sha256(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck.get("model_kind") != KIND:
        raise ValueError("Not a joint migration regime checkpoint")
    root = Path(ck["data"])
    hashes = {"manifest_sha256": sha256(root / "manifest.json"), "cases_sha256": sha256(root / "cases.npz")}
    for key, value in hashes.items():
        if ck.get(key) != value or ck["rules"].get(key) != value:
            raise ValueError("Regime checkpoint/cache " + key + " mismatch")
    if sha256(ck["rules_path"]) != ck["rules_sha256"] or json.loads(Path(ck["rules_path"]).read_text()) != ck["rules"]:
        raise ValueError("Regime frozen rules checksum mismatch")
    if ck.get("calibration"):
        evidence = ck["calibration"]
        if sha256(evidence["path"]) != evidence["sha256"] or json.loads(Path(evidence["path"]).read_text()) != evidence["metadata"]:
            raise ValueError("Regime calibration checksum mismatch")
        if float(evidence["metadata"]["active_scale"]) != ck["active_scale"]:
            raise ValueError("Regime scale differs from finalized calibration")
        expected = {**hashes, "rules_sha256": ck["rules_sha256"],
                    "checkpoint_sha256": ck["active"]["sha256"] if ck.get("active") else None}
        if (evidence["metadata"].get("family_selected") is not True or
                any(evidence["metadata"].get(key) != value for key, value in expected.items())):
            raise ValueError("Regime calibration/active expert provenance mismatch")
    base, _ = _load_member(ck["base"]["path"], device, hashes, ck["base"]["sha256"])
    active = None
    if ck.get("active"):
        active, _ = _load_member(ck["active"]["path"], device, hashes, ck["active"]["sha256"])
    model = MigrationRegime(base, ck["rules"], active, ck["active_fallback"], ck["active_scale"], device, ck.get("manifest"))
    if sha256(checkpoint) != before:
        raise ValueError("Regime checkpoint changed during loading")
    return model, ck


def scaled_joint_bank(parameters, samples, scales=SCALE_GRID, generator=None):
    """Couple component identity and joint noise across all covariance scales."""
    if samples < 2 or any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("Invalid coupled covariance sample bank")
    means = parameters["means"]
    batch, _, horizons, _ = means.shape
    components = torch.multinomial(parameters["logits"].softmax(-1), samples, replacement=True, generator=generator)
    rows = torch.arange(batch, device=means.device)[:, None]
    chosen = means.flatten(-2)[rows, components]
    cholesky = torch.linalg.cholesky(parameters["covariance"])[rows, components]
    noise = torch.randn(batch, samples, 2 * horizons, dtype=means.dtype, device=means.device, generator=generator)
    residual = torch.einsum("bsij,bsj->bsi", cholesky, noise)
    return [(chosen + float(scale) * residual).reshape(batch, samples, horizons, 2) for scale in scales]


def _coverage(draws, truth, mask, origins):
    from .migration_evaluate import forecast_latlon, spherical_xyz, great_circle
    coords = forecast_latlon(draws, origins)
    xyz = spherical_xyz(coords).mean(1)
    means = np.stack([np.rad2deg(np.arctan2(xyz[..., 2], np.hypot(xyz[..., 0], xyz[..., 1]))),
                      np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))], -1)
    empty = np.linalg.norm(xyz, axis=-1) < 1e-10
    means[empty] = coords[:, 0][empty]
    radii = great_circle(coords, means[:, None])
    error = great_circle(truth, means)
    return [((error <= np.quantile(radii, q, axis=1)) & mask) for q in (.5, .9)]


def _scale_objectives(a, ids, endpoint, joint):
    from .migration_ensemble import macro_weights
    ew, jw = macro_weights(a, ids)
    return .5 * ((np.nan_to_num(endpoint) * ew[None]).sum((1, 2)) + (joint * jw[None]).sum(1))


def _validate_costs(a, ids, endpoint, joint):
    ids = np.asarray(ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or
            len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(a["split"]))):
        raise ValueError("Calibration needs unique valid integer origin indices")
    if np.any(~np.isin(a["split"][ids], [1, 2])):
        raise ValueError("Covariance calibration uses validation origins only")
    mask = a["target_mask"][ids]
    if not mask.any(-1).all():
        raise ValueError("Calibration origins must have observed development targets")
    if endpoint.shape != (len(SCALE_GRID), len(ids), len(a["horizons"])) or joint.shape != (len(SCALE_GRID), len(ids)):
        raise ValueError("Calibration scale cost dimensions differ")
    if not np.isfinite(endpoint[:, mask]).all() or not np.isfinite(joint).all():
        raise ValueError("Calibration observed scores must be finite")
    return ids.astype("int64", copy=False)


def crossfit_scales(a, manifest, ids, endpoint, joint, folds=3, seed=1701):
    """Choose scales without any held-fold targets entering that fold's fit."""
    from .migration_ensemble import physical_folds
    ids = _validate_costs(a, ids, endpoint, joint)
    if folds != 3:
        raise ValueError("Covariance calibration uses exactly three physical folds on validation only")
    assignments, _ = physical_folds(a, manifest, folds, seed)
    assignment = assignments[ids]
    selected = np.zeros(len(ids), dtype="int64")
    records = []
    for fold in range(folds):
        held, train = assignment == fold, assignment != fold
        if not held.any():
            continue
        if not train.any():
            raise ValueError("Insufficient physical validation groups for cross-fitting")
        objectives = _scale_objectives(a, ids[train], endpoint[:, train], joint[:, train])
        choice = int(np.argmin(objectives))
        selected[held] = choice
        records.append({"fold": fold, "fit_origins": int(train.sum()), "held_origins": int(held.sum()),
                        "fit_physical_groups": len(np.unique(a["individual"][ids[train]])),
                        "held_physical_groups": len(np.unique(a["individual"][ids[held]])),
                        "scale": SCALE_GRID[choice], "fit_objectives_km": objectives.tolist()})
    return selected, assignment, records


def calibrate(data, checkpoint, rules, output, samples=32, batch_size=64, seed=1701, device="cpu"):
    """Evaluate the fixed scale family on validation HIGH; do not finalize a scale."""
    from .migration_train import physical_validation_scores
    from .migration_ensemble import macro_weights
    output = Path(output)
    if output.exists():
        raise FileExistsError("Calibration outputs are immutable; choose a new directory")
    if samples < 2 or batch_size < 1:
        raise ValueError("Positive calibration batch size and at least two samples required")
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest, hashes = load_cases(data)
    rules_path = Path(rules)
    if rules_path.is_dir():
        rules_path = rules_path / "rules.json"
    rule_hash, payload = sha256(rules_path), json.loads(rules_path.read_text())
    for key, value in hashes.items():
        if payload.get(key) != value:
            raise ValueError("Calibration rules/cache " + key + " mismatch")
    dev = np.flatnonzero(np.isin(a["split"], [1, 2]))
    ids = dev[classify(a, dev, payload)["high"] & a["target_mask"][dev].any(-1)]
    if not len(ids):
        raise ValueError("No observed validation HIGH cases for calibration")
    model, evidence = _load_member(checkpoint, device, hashes)
    shape = (len(SCALE_GRID), len(ids), len(a["horizons"]))
    endpoint, joint = np.empty(shape), np.empty(shape[:2])
    coverage50, coverage90 = np.empty(shape, bool), np.empty(shape, bool)
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        with torch.inference_mode():
            inputs = [torch.as_tensor(a[key][batch], device=device) for key in ("history", "history_mask", "static", "species")]
            generator = torch.Generator(device=device).manual_seed(seed + start)
            bank = scaled_joint_bank(model(*inputs), samples, generator=generator)
            truth = torch.as_tensor(a["target_latlon"][batch], device=device)
            mask = torch.as_tensor(a["target_mask"][batch], device=device)
            origin = torch.as_tensor(a["origin"][batch], device=device)
            for j, draws in enumerate(bank):
                e, p = physical_validation_scores(draws, truth, mask, origin)
                endpoint[j, start:start + len(batch)] = e.cpu().numpy()
                joint[j, start:start + len(batch)] = p.cpu().numpy()
                c50, c90 = _coverage(draws.cpu().numpy(), a["target_latlon"][batch], a["target_mask"][batch], a["origin"][batch])
                coverage50[j, start:start + len(batch)] = c50
                coverage90[j, start:start + len(batch)] = c90
    selected, assignment, fold_records = crossfit_scales(a, manifest, ids, endpoint, joint, seed=seed)
    ew, jw = macro_weights(a, ids)
    summaries = {}
    for label, choices in (("unscaled", np.zeros(len(ids), int)), ("crossfit", selected)):
        e, p = endpoint[choices, np.arange(len(ids))], joint[choices, np.arange(len(ids))]
        summaries[label] = {"endpoint_energy_km": float((np.nan_to_num(e) * ew).sum()),
                            "joint_chordal_energy_km": float((p * jw).sum()),
                            "radial_coverage50": float((coverage50[choices, np.arange(len(ids))] * ew).sum()),
                            "radial_coverage90": float((coverage90[choices, np.arange(len(ids))] * ew).sum())}
    for row in summaries.values():
        row["selection_objective_km"] = .5 * (row["endpoint_energy_km"] + row["joint_chordal_energy_km"])
    if sha256(checkpoint) != evidence["sha256"] or sha256(rules_path) != rule_hash:
        raise ValueError("Calibration member/rules changed while scoring")
    for name, key in (("manifest.json", "manifest_sha256"), ("cases.npz", "cases_sha256")):
        if sha256(Path(data) / name) != hashes[key]:
            raise ValueError("Calibration source cache changed while scoring")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "validation-costs.npz", case_ids=ids, scales=np.asarray(SCALE_GRID),
                        endpoint_energy_km=endpoint, joint_energy_km=joint, coverage50=coverage50,
                        coverage90=coverage90, folds=assignment, crossfit_scale_index=selected)
    report = {"protocol": "wildcast-migration-active-covariance-crossfit-v1", "data": str(Path(data).resolve()), **hashes,
              "rules_path": str(rules_path.resolve()), "rules_sha256": rule_hash, "checkpoint": evidence,
              "scale_grid": SCALE_GRID, "samples": samples, "batch_size": batch_size, "seed": seed,
              "validation_high_origins": len(ids), "validation_physical_groups": len(np.unique(a["individual"][ids])),
              "validation_case_ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
              "folds": fold_records, "scores": summaries, "scale_grid_objectives_km": _scale_objectives(a, ids, endpoint, joint).tolist(),
              "crossfit_minus_unscaled_objective_km": summaries["crossfit"]["selection_objective_km"] - summaries["unscaled"]["selection_objective_km"],
              "final_scale_fitted": False, "family_selected": False, "reserved_cases_scored": 0,
              "code_sha256": sha256(__file__), "costs_sha256": sha256(output / "validation-costs.npz"),
              "scope": "Conditional covariance development calibration on causal HIGH validation cases only.",
              "limitations": ["Expert checkpoint selection already used validation; scale cross-fitting is not an independent test.",
                              "Scales share fixed component/noise draws; finite Monte Carlo and physical-fold uncertainty remain.",
                              "Radial coverage is about the spherical sample mean, not a highest-density region.",
                              "The family must be selected before explicitly finalizing an all-validation scale; no automatic promotion.",
                              "Ordinary/missing queries keep the base forecast; this is conditional mathematical uncertainty calibration, not biological state discovery."]}
    dump(output / "comparison.json", report)
    return report


def finalize_scale(calibration, output):
    """Explicitly select this family, then fit one scale using all validation HIGH."""
    calibration, output = Path(calibration), Path(output)
    if output.exists():
        raise FileExistsError("Finalized scales are immutable; choose a new output")
    report = json.loads((calibration / "comparison.json").read_text())
    if report.get("protocol") != "wildcast-migration-active-covariance-crossfit-v1":
        raise ValueError("Unsupported covariance calibration protocol")
    a, manifest, hashes = load_cases(report["data"])
    for key, value in hashes.items():
        if report.get(key) != value:
            raise ValueError("Final calibration/cache provenance mismatch")
    if sha256(report["checkpoint"]["path"]) != report["checkpoint"]["sha256"] or sha256(report["rules_path"]) != report["rules_sha256"]:
        raise ValueError("Final calibration expert/rules changed")
    path = calibration / "validation-costs.npz"
    if sha256(path) != report["costs_sha256"]:
        raise ValueError("Final calibration cost checksum differs")
    with np.load(path, allow_pickle=False) as bank:
        ids = _validate_costs(a, bank["case_ids"], bank["endpoint_energy_km"], bank["joint_energy_km"])
        rules = json.loads(Path(report["rules_path"]).read_text())
        dev = np.flatnonzero(np.isin(a["split"], [1, 2]))
        expected = dev[classify(a, dev, rules)["high"] & a["target_mask"][dev].any(-1)]
        if not np.array_equal(ids, expected) or not np.array_equal(bank["scales"], SCALE_GRID):
            raise ValueError("Final scale costs do not match the causal validation HIGH cohort/grid")
        if (report.get("validation_high_origins") != len(ids) or
                report.get("validation_case_ids_sha256") != hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()):
            raise ValueError("Final calibration origin identity evidence differs")
        # Use the same frozen manifest labels used for physical/exclusion folds.
        choices, assignment, records = crossfit_scales(a, manifest, ids, bank["endpoint_energy_km"], bank["joint_energy_km"], seed=report["seed"])
        if (not np.array_equal(assignment, bank["folds"]) or not np.array_equal(choices, bank["crossfit_scale_index"]) or
                records != report["folds"]):
            raise ValueError("Final calibration physical-fold evidence differs")
        objective = _scale_objectives(a, ids, bank["endpoint_energy_km"], bank["joint_energy_km"])
    choice = int(np.argmin(objective))
    result = {"protocol": "wildcast-migration-active-covariance-final-v1", "family_selected": True,
              "active_scale": SCALE_GRID[choice], **hashes, "rules_sha256": report["rules_sha256"],
              "checkpoint_sha256": report["checkpoint"]["sha256"],
              "comparison_path": str((calibration / "comparison.json").resolve()),
              "comparison_sha256": sha256(calibration / "comparison.json"), "costs_sha256": report["costs_sha256"],
              "fit_split": "all_validation_HIGH_after_explicit_family_selection", "objective_grid_km": objective.tolist(),
              "independent_result": False, "reserved_cases_scored": 0}
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "scale.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--data", required=True); prepare_parser.add_argument("--output", required=True)
    assemble_parser = commands.add_parser("assemble")
    for name in ("data", "rules", "base", "output"):
        assemble_parser.add_argument("--" + name, required=True)
    assemble_parser.add_argument("--active"); assemble_parser.add_argument("--active-fallback", choices=("base", "seasonal"), default="seasonal")
    assemble_parser.add_argument("--active-scale", type=float, default=1.)
    assemble_parser.add_argument("--device", default="cpu")
    assemble_parser.add_argument("--calibration")
    calibrate_parser = commands.add_parser("calibrate")
    for name in ("data", "checkpoint", "rules", "output"):
        calibrate_parser.add_argument("--" + name, required=True)
    for name, default in (("samples", 32), ("batch-size", 64), ("seed", 1701)):
        calibrate_parser.add_argument("--" + name, type=int, default=default)
    calibrate_parser.add_argument("--device", default="cpu")
    final_parser = commands.add_parser("finalize-scale")
    final_parser.add_argument("--calibration", required=True); final_parser.add_argument("--output", required=True)
    args = vars(p.parse_args()); command = args.pop("command")
    handlers = {"prepare": prepare, "assemble": assemble, "calibrate": calibrate, "finalize-scale": finalize_scale}
    result = handlers[command](**args)
    fields = ("model_kind", "protocol", "selected_train_origins", "selected_validation_origins",
              "reserved_selected", "validation_high_origins", "crossfit_minus_unscaled_objective_km",
              "family_selected", "active_scale", "reserved_cases_scored")
    print(json.dumps({k: result[k] for k in fields if k in result}, indent=2))


if __name__ == "__main__":
    main()
