"""Auditable, masked migration evaluation with physical-animal uncertainty.

Endpoint distances use great-circle kilometres. Joint energy uses the Euclidean
embedding of complete masked spherical paths, which gives a proper joint score;
it is explicitly a chordal score, not a great-circle path score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import unproject
from .migration_baselines import MigrationBaselines
from .sources import dump

SPLITS = {"train": 0, "val_individual": 1, "val_year": 2,
          "test_individual": 3, "test_year": 4}
EARTH_KM = 6371.0088


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metric_species(a, manifest, root):
    """Separate biological reporting groups from an unknown-species input token."""
    if "metric_species" in a:
        groups = np.asarray(a["metric_species"], dtype=np.int64)
        mapping = manifest.get("metric_species", manifest.get("species", {}))
    elif manifest.get("origins_sha256"):
        path = Path(root) / "origins.parquet"
        if not path.exists() or sha256(path) != manifest["origins_sha256"]:
            raise ValueError("Migration origins hash does not match the frozen manifest")
        origins = pd.read_parquet(path, columns=["case_id", "species"])
        if len(origins) != len(a["species"]) or not np.array_equal(origins.case_id.to_numpy(), np.arange(len(origins))):
            raise ValueError("Migration origin species groups do not align with case IDs")
        names = sorted(origins.species.unique())
        mapping = {name: i for i, name in enumerate(names)}
        groups = origins.species.map(mapping).to_numpy(dtype=np.int64)
    else:
        groups = np.asarray(a["species"], dtype=np.int64)
        mapping = manifest.get("species", {})
    if groups.shape != np.asarray(a["species"]).shape:
        raise ValueError("Migration reporting species dimensions differ")
    return groups, {int(v): str(k) for k, v in mapping.items()}


def load_cases(data):
    root = Path(data)
    manifest_path, cases_path = root / "manifest.json", root / "cases.npz"
    manifest = json.loads(manifest_path.read_text())
    actual = sha256(cases_path)
    expected = manifest.get("cases_sha256", manifest.get("files_sha256", {}).get("cases.npz"))
    if expected != actual:
        raise ValueError("Migration cases hash does not match the frozen manifest")
    with np.load(cases_path, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    required = {"history", "history_mask", "static", "species", "y", "target_mask", "split",
                "individual", "study", "origin", "target_latlon", "target_time", "horizons"}
    if required - a.keys():
        raise ValueError("Missing migration arrays: " + ", ".join(sorted(required - a.keys())))
    if manifest.get("protocol", "").startswith("wildcast-multiscale-"):
        from .migration_protocol import schema_for
        schema_for(manifest, a)
    n, h = len(a["split"]), len(a["horizons"])
    if a["y"].shape != (n, h, 2) or a["target_mask"].shape != (n, h):
        raise ValueError("Migration target dimensions differ")
    if a["target_latlon"].shape != (n, h, 2) or a["target_time"].shape != (n, h):
        raise ValueError("Migration target coordinate/time dimensions differ")
    if a["origin"].shape != (n, 3) or a["history_mask"].shape != a["history"].shape[:2]:
        raise ValueError("Migration origin/history dimensions differ")
    if np.any(np.diff(a["horizons"]) <= 0) or np.any(a["horizons"] <= 0):
        raise ValueError("Migration horizons must be positive and strictly increasing")
    for key in ["y", "target_latlon"]:
        if not np.isfinite(a[key][a["target_mask"]]).all():
            raise ValueError("Observed migration targets must be finite")
    if not np.isfinite(a["target_time"][a["target_mask"]]).all():
        raise ValueError("Observed migration target times must be finite")
    if np.any(a["target_time"][a["target_mask"]] <= np.broadcast_to(a["origin"][:, 2:3], (n, h))[a["target_mask"]]):
        raise ValueError("Migration target observations must occur after their origins")
    a["metric_species"], metric_names = load_metric_species(a, manifest, root)
    manifest = {**manifest, "metric_species_names": metric_names}
    return a, manifest, {"manifest_sha256": sha256(manifest_path), "cases_sha256": actual}


def great_circle(left, right):
    """Broadcast latitude/longitude arrays to great-circle distance in km."""
    left, right = np.deg2rad(left), np.deg2rad(right)
    dlat = left[..., 0] - right[..., 0]
    dlon = (left[..., 1] - right[..., 1] + np.pi) % (2 * np.pi) - np.pi
    hav = np.sin(dlat / 2) ** 2 + np.cos(left[..., 0]) * np.cos(right[..., 0]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arctan2(np.sqrt(np.clip(hav, 0, 1)), np.sqrt(np.clip(1 - hav, 0, 1)))


def spherical_xyz(latlon):
    lat, lon = np.moveaxis(np.deg2rad(latlon), -1, 0)
    return EARTH_KM * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def forecast_latlon(paths, origin):
    return unproject(np.asarray(paths, dtype=float) * 1_000_000,
                     origin[:, None, None, 0], origin[:, None, None, 1])


def score_paths(paths, target_latlon, target_mask, origin, *, device="cpu"):
    """Score exactly the observed support; masked target values are ignored.

    The unbiased finite-sample energy estimator can be slightly negative. No
    minimum-over-samples error is used as an accuracy or selection metric.
    """
    paths = np.asarray(paths, dtype=float)
    mask = np.asarray(target_mask, dtype=bool)
    if paths.ndim != 4 or paths.shape[-1] != 2 or paths.shape[0] != len(mask) or paths.shape[2] != mask.shape[1]:
        raise ValueError("Expected migration paths [case,sample,horizon,2]")
    if paths.shape[1] < 2 or not np.isfinite(paths).all():
        raise ValueError("At least two finite path samples required")
    if np.any(~mask.any(-1)):
        raise ValueError("Each scored origin needs at least one observed target")
    coords = forecast_latlon(paths, origin)
    truth = np.where(mask[..., None], target_latlon, origin[:, None, :2])
    n, samples, horizons, _ = coords.shape
    energy = np.full((n, horizons), np.nan)
    means = np.empty((n, horizons, 2))
    coverage50, coverage90 = np.zeros((n, horizons), bool), np.zeros((n, horizons), bool)
    medoid_cost = np.zeros((n, samples))
    accelerated = str(device).startswith("cuda")
    if accelerated:
        from .migration_gpu_metrics import distribution_scores
        energy, joint, medoid_cost = distribution_scores(coords, truth, mask, origin, device)
    for horizon in range(horizons):
        value = coords[:, :, horizon]
        if not accelerated:
            pair = great_circle(value[:, :, None], value[:, None, :])
            accuracy = great_circle(value, truth[:, None, horizon]).mean(1)
            energy[:, horizon] = np.where(mask[:, horizon], accuracy - pair.sum((1, 2)) / (2 * samples * (samples - 1)), np.nan)
            medoid_cost += pair.mean(-1)
        xyz_mean = spherical_xyz(value).mean(1)
        norm = np.linalg.norm(xyz_mean, axis=-1)
        means[:, horizon, 0] = np.rad2deg(np.arctan2(xyz_mean[:, 2], np.hypot(xyz_mean[:, 0], xyz_mean[:, 1])))
        means[:, horizon, 1] = np.rad2deg(np.arctan2(xyz_mean[:, 1], xyz_mean[:, 0]))
        means[norm < 1e-10, horizon] = value[norm < 1e-10, 0]
        radii = great_circle(value, means[:, None, horizon])
        error = great_circle(truth[:, horizon], means[:, horizon])
        coverage50[:, horizon] = (error <= np.quantile(radii, .5, axis=1)) & mask[:, horizon]
        coverage90[:, horizon] = (error <= np.quantile(radii, .9, axis=1)) & mask[:, horizon]
    representative = medoid_cost.argmin(1)
    representative_coords = coords[np.arange(n), representative]
    representative_error = np.where(mask, great_circle(representative_coords, truth), np.nan)
    point_error = np.where(mask, great_circle(means, truth), np.nan)
    # A proper energy score on the stacked, masked Cartesian path embedding.
    if not accelerated:
        weights = np.sqrt(mask / mask.sum(-1, keepdims=True))[..., None]
        embedding = (spherical_xyz(coords) * weights[:, None]).reshape(n, samples, -1)
        observed = (spherical_xyz(truth) * weights).reshape(n, -1)
        accuracy = np.linalg.norm(embedding - observed[:, None], axis=-1).mean(1)
        norms = (embedding ** 2).sum(-1)
        pair2 = norms[:, :, None] + norms[:, None, :] - 2 * np.einsum("bsd,btd->bst", embedding, embedding)
        pair = np.sqrt(np.maximum(pair2, 0))
        # Exact diagonal zero avoids cancellation noise for identical samples.
        pair[:, np.arange(samples), np.arange(samples)] = 0
        joint = accuracy - pair.sum((1, 2)) / (2 * samples * (samples - 1))
    return {"endpoint_energy_km": energy, "joint_chordal_energy_km": joint,
            "point_error_km": point_error, "representative_error_km": representative_error,
            "radial_coverage50": coverage50, "radial_coverage90": coverage90,
            "representative_index": representative, "representative_latlon": representative_coords}


def macro_scores(frame):
    """Equal physical-animal weight within species, then species and horizon."""
    columns = [c for c in ["endpoint_energy_km", "point_error_km", "representative_error_km", "radial_coverage50",
                           "radial_coverage90", "within_1km", "within_10km", "within_100km"] if c in frame]
    animal = frame.groupby(["model", "split", "species", "individual", "horizon_days"], observed=True)[columns].mean()
    species = animal.groupby(["model", "split", "species", "horizon_days"], observed=True).mean()
    horizon = species.groupby(["model", "split", "horizon_days"], observed=True).mean()
    overall = horizon.groupby(["model", "split"], observed=True).mean()
    return overall.reset_index(), species.reset_index(), horizon.reset_index()


def paired_animal_bootstrap(frame, reference, seed=2026, replicates=300):
    """Resample physical animals within species, preserving horizons and origins.

    The bootstrap does not pretend to measure geographic or species-transfer
    uncertainty. A species with one evaluated animal has no animal variance.
    """
    if (frame.groupby(["split", "individual"]).species.nunique() > 1).any():
        from . import migration_uncertainty
        result = migration_uncertainty.paired_global_cluster_bootstrap(frame, reference, seed, replicates)
        return {**result, "code_sha256": sha256(migration_uncertainty.__file__)}
    keys = ["split", "species", "individual", "horizon_days", "case"]
    if frame.duplicated(["model", *keys]).any():
        raise ValueError("Duplicate scored migration model/case/horizon")
    pivot = frame.pivot(index=keys, columns="model", values="endpoint_energy_km")
    if reference not in pivot or pivot.isna().any().any():
        raise ValueError("Paired comparisons require identical origins and target masks")
    rng = np.random.default_rng(seed)
    result = {}
    for split, group in pivot.reset_index().groupby("split", observed=True):
        models = list(pivot.columns)
        animal = group.groupby(["species", "individual", "horizon_days"], observed=True)[models].mean()
        horizons = sorted(animal.index.get_level_values("horizon_days").unique())
        blocks = []
        for _, block in animal.groupby(level="species", observed=True):
            identities = block.index.get_level_values("individual").unique().tolist()
            values = np.zeros((len(identities), len(horizons), len(models)))
            present = np.zeros((len(identities), len(horizons)))
            ai, hi = {v: i for i, v in enumerate(identities)}, {v: i for i, v in enumerate(horizons)}
            for (_, identity, horizon), row in block.iterrows():
                values[ai[identity], hi[horizon]] = row.to_numpy()
                present[ai[identity], hi[horizon]] = 1
            blocks.append((values, present))
        draws = []
        for _ in range(replicates):
            horizon_sums = np.zeros((len(horizons), len(models)))
            species_counts = np.zeros(len(horizons))
            for values, present in blocks:
                weights = rng.multinomial(len(values), np.full(len(values), 1 / len(values)))
                counts = weights @ present
                sums = np.einsum("a,ahm->hm", weights, values)
                good = counts > 0
                horizon_sums[good] += sums[good] / counts[good, None]
                species_counts[good] += 1
            good = species_counts > 0
            means = (horizon_sums[good] / species_counts[good, None]).mean(0)
            draws.append(means - means[models.index(reference)])
        values = np.asarray(draws)
        observed = animal.groupby(level=["species", "horizon_days"]).mean().groupby(level="horizon_days").mean().mean()
        result[str(split)] = {model: {"delta_km": float(observed[model] - observed[reference]),
                                      "ci95_km": np.quantile(values[:, i], [.025, .975]).tolist()}
                              for i, model in enumerate(models)}
    return {"reference": reference, "replicates": replicates,
            "unit": "physical animal within species; retain all origins and horizons", "splits": result}


def evaluate(data, output, checkpoints=None, baselines=("stationary", "movement", "seasonal"),
             splits=("val_individual", "val_year"), samples=64, seed=2026, batch_size=64,
             device="cpu", allow_test=False, bootstrap_replicates=300, reference=None,
             assessment=None, selection=None):
    if samples < 2:
        raise ValueError("At least two samples required")
    a, manifest, hashes = load_cases(data)
    codes = [SPLITS[s] if isinstance(s, str) else int(s) for s in splits]
    if any(s not in SPLITS.values() or s == 0 for s in codes):
        raise ValueError("Evaluate held-out migration splits only")
    if any(s >= 3 for s in codes) and not allow_test:
        raise ValueError("Test labels require an explicitly frozen final assessment")
    assessment_evidence = None
    if assessment is not None:
        if not any(s >= 3 for s in codes):
            raise ValueError("Do not open reserved truth for development-only evaluation")
        from .migration_assessment import apply_assessment
        a = apply_assessment(a, manifest, hashes, assessment, selection=selection)
        assessment_path = Path(assessment)
        if assessment_path.is_dir(): assessment_path = assessment_path / "assessment.json"
        assessment_evidence = json.loads(assessment_path.read_text())
        selected = {row["sha256"] for row in assessment_evidence["checkpoints"]}
        if any(sha256(path) not in selected for path in checkpoints or []):
            raise ValueError("Final assessment accepts only checkpoints frozen in its selection")
    ids = np.flatnonzero(np.isin(a["split"], codes) & a["target_mask"].any(-1))
    if not len(ids):
        raise ValueError("No observed migration cases in the requested splits")
    forecasters = []
    checkpoint_evidence = []
    for checkpoint in checkpoints or []:
        import torch
        loaded_hash = sha256(checkpoint)
        probe = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if probe.get("model_kind") == "wildcast_migration_environment_gate":
            from .migration_environment_gate import load_model
        elif probe.get("model_kind") == "wildcast_migration_context_joint":
            from .migration_context_joint import load_model
        elif probe.get("model_kind") == "wildcast_migration_retrieval":
            from .migration_retrieval_train import load_model
        elif probe.get("model_kind") == "wildcast_migration_ensemble":
            from .migration_ensemble import load_model
        elif probe.get("model_kind") == "wildcast_migration_moveformer":
            from .migration_moveformer import load_model
        elif probe.get("model_kind") == "wildcast_migration_moveformer_endpoint":
            from .migration_moveformer_endpoint import load_model
        elif probe.get("model_kind") == "wildcast_migration_regime":
            from .migration_regimes import load_model
        else:
            from .migration_model import load_model
        del probe
        model, ck = load_model(checkpoint, device=device)
        if sha256(checkpoint) != loaded_hash:
            raise ValueError("Migration checkpoint changed during loading; evaluate a frozen snapshot")
        for key, value in hashes.items():
            if ck.get(key) != value:
                raise ValueError("Migration checkpoint/data " + key + " mismatch")
        label = Path(checkpoint).parent.name + ("-" + Path(checkpoint).stem if Path(checkpoint).stem != "best" else "")
        if any(label == name for name, *_ in forecasters) or label in baselines:
            raise ValueError("Migration model names must be unique")
        forecasters.append((label, model, str(checkpoint)))
        checkpoint_evidence.append({"name": label, "path": str(checkpoint), "sha256": loaded_hash,
                                    "model_kind": ck.get("model_kind"), "selected_epoch": ck.get("epoch"),
                                    "selection_version": ck.get("selection_version", ck.get("training", {}).get("selection_version", "legacy-or-unspecified")),
                                    "code_sha256": ck.get("code_sha256", ck.get("training", {}).get("code_sha256"))})
    fitted = MigrationBaselines(a, manifest)
    names = [*baselines, *[name for name, *_ in forecasters]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide uniquely named migration forecasts")
    rows, joint_rows, exports = [], [], []
    retrieval = {"fallback_routes": 0, "completed_points": 0, "sampled_points": 0}
    split_names = {value: key for key, value in SPLITS.items()}
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        predictions = []
        # Every baseline starts with the same seed per batch: movement noise is
        # coupled across stationary/movement/retrieval comparisons.
        for name in baselines:
            draws, info = fitted.predict(name, batch, samples, np.random.default_rng(seed + start))
            if "completed_mask" in info:
                retrieval["fallback_routes"] += info["fallback_routes"]
                retrieval["completed_points"] += int(info["completed_mask"].sum())
                retrieval["sampled_points"] += int(info["completed_mask"].size)
            predictions.append((name, draws))
        for name, model, _ in forecasters:
            import torch
            if hasattr(model, "predict_cases"):
                draws = model.predict_cases(a, batch, samples, np.random.default_rng(seed + start), baselines=fitted)
                predictions.append((name, draws))
                continue
            generator = torch.Generator(device=device).manual_seed(seed + start)
            with torch.inference_mode():
                tensors = [torch.as_tensor(a[key][batch], device=device) for key in ["history", "history_mask", "static", "species"]]
                draws = model.sample_paths(*tensors, samples=samples, generator=generator).detach().cpu().numpy()
            predictions.append((name, draws))
        for name, draws in predictions:
            scores = score_paths(draws, a["target_latlon"][batch], a["target_mask"][batch], a["origin"][batch], device=device)
            for j, case in enumerate(batch):
                valid = a["target_mask"][case]
                observed_disp = great_circle(a["target_latlon"][case, valid], a["origin"][case, :2])
                proxy = "large_displacement_proxy" if observed_disp.max() >= 100 else "local_displacement_proxy" if observed_disp.max() < 10 else "intermediate_displacement_proxy"
                base = {"model": name, "case": int(case), "split": split_names[int(a["split"][case])],
                        "individual": int(a["individual"][case]), "species": int(a.get("metric_species", a["species"])[case]),
                        "prediction_species": int(a["species"][case]),
                        "study": int(a["study"][case]), "displacement_proxy": proxy}
                joint_rows.append({**base, "observed_horizons": int(valid.sum()),
                                   "joint_chordal_energy_km": float(scores["joint_chordal_energy_km"][j])})
                for h in np.flatnonzero(valid):
                    error = float(scores["point_error_km"][j, h])
                    rows.append({**base, "horizon_days": float(a["horizons"][h]),
                                 "actual_elapsed_days": float((a["target_time"][case, h] - a["origin"][case, 2]) / 86400),
                                 "target_timing_error_hours": float((a["target_time"][case, h] - a["origin"][case, 2]) / 3600 - a["horizons"][h] * 24),
                                 **{key: float(scores[key][j, h]) for key in ["endpoint_energy_km", "point_error_km", "representative_error_km"]},
                                 **{key: bool(scores[key][j, h]) for key in ["radial_coverage50", "radial_coverage90"]},
                                 "within_1km": error <= 1, "within_10km": error <= 10, "within_100km": error <= 100})
                if len(exports) < 24 * len(names):
                    exports.append({**base, "origin_latlon": a["origin"][case, :2].tolist(),
                                    "representative_index": int(scores["representative_index"][j]),
                                    "representative_latlon": scores["representative_latlon"][j].tolist(),
                                    "horizons_days": a["horizons"].tolist(),
                                    "observed_future": [{"latlon": a["target_latlon"][case, h].tolist(),
                                                         "timestamp": float(a["target_time"][case, h])} for h in np.flatnonzero(valid)]})
    frame = pd.DataFrame(rows)
    joint = pd.DataFrame(joint_rows)
    overall, species, horizon = macro_scores(frame)
    reference = reference or ("seasonal" if "seasonal" in names else names[0])
    paired = paired_animal_bootstrap(frame, reference, seed, bootstrap_replicates)
    joint_for_pairing = joint.rename(columns={"joint_chordal_energy_km": "endpoint_energy_km"}).assign(horizon_days=0.)
    paired_joint = paired_animal_bootstrap(joint_for_pairing, reference, seed, bootstrap_replicates)
    timing = frame.loc[frame.model == names[0]].groupby(["split", "horizon_days"]).target_timing_error_hours.agg(
        cases="size", mean_hours="mean", minimum_hours="min", maximum_hours="max").reset_index()
    # Quantiles are descriptive and aggregated separately from proper scores.
    errors = frame.groupby(["model", "split"]).point_error_km.quantile([.5, .9]).unstack().reset_index()
    errors.columns = ["model", "split", "median_point_error_km", "p90_point_error_km"]
    joint_macro = joint.groupby(["model", "split", "species", "individual"]).joint_chordal_energy_km.mean().groupby(
        ["model", "split", "species"]).mean().groupby(["model", "split"]).mean().reset_index()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "per-case.parquet", index=False)
    joint.to_parquet(out / "joint-per-origin.parquet", index=False)
    overall.to_csv(out / "scores.csv", index=False)
    species.to_csv(out / "per-species.csv", index=False)
    horizon.to_csv(out / "per-horizon.csv", index=False)
    timing.to_csv(out / "target-timing.csv", index=False)
    errors.to_csv(out / "distance-quantiles.csv", index=False)
    joint_macro.to_csv(out / "joint-scores.csv", index=False)
    strata = frame.groupby(["model", "split", "displacement_proxy", "species", "individual", "horizon_days"]).endpoint_energy_km.mean().groupby(
        ["model", "split", "displacement_proxy", "species", "horizon_days"]).mean().groupby(
        ["model", "split", "displacement_proxy", "horizon_days"]).mean().reset_index()
    strata.to_csv(out / "displacement-proxies.csv", index=False)
    protocol = {"scope": "Frozen local migration final assessment" if assessment_evidence else "Local migration development; no official benchmark or biological migration-label claim",
                "assessment": {"sha256": sha256(assessment_path), "selection_sha256": assessment_evidence["frozen_selection_sha256"]} if assessment_evidence else None,
                "identity_unit": manifest.get("identity_unit", "Recorded physical-animal grouping under the source audit"),
                **hashes, "data": str(data), "samples": samples, "seed": seed, "batch_size": batch_size, "device": device,
                "cases": len(ids), "observed_targets": len(frame) // len(names),
                "splits": [split_names[c] for c in codes], "models": names,
                "metric_species_names": manifest.get("metric_species_names", {}),
                "checkpoints": checkpoint_evidence,
                "retrieval": retrieval, "scores": overall.to_dict("records"), "joint_scores": joint_macro.to_dict("records"),
                "paired_animal_bootstrap": paired, "paired_joint_animal_bootstrap": paired_joint,
                "code_sha256": {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name("migration_baselines.py")]},
                "case_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
                "distribution_scoring_backend": "torch float64, origin-translated Cartesian joint geometry" if str(device).startswith("cuda") else "numpy float64",
                "notes": ["All forecasts use exactly the same origins and observed-target masks.",
                          "Endpoint energy and point errors use great-circle km; joint energy uses a masked stacked spherical Cartesian embedding in km.",
                          "Joint score divides Cartesian squared distance by the number of observed horizons; scores on different observation patterns need separate interpretation.",
                          "Energy estimates use off-diagonal sample pairs and may be slightly negative.",
                          "Macro aggregation gives equal weight to physical animals within species, then species, then horizons.",
                          "Median/p90 errors are descriptive pooled observed-target quantiles.",
                          "Radial coverage is around the spherical sample mean, not highest-density-region coverage.",
                          "Representative route is the sample medoid across all requested horizons, selected without truth.",
                          "Displacement strata use observed displacement thresholds and are not verified migratory or behavioural labels.",
                          "Seasonal retrieval selects a training-only whole donor route; missing donor endpoints are stochastic forecast bridges or movement continuations, never observations.",
                          "Retrieval uses nominal horizon support; target-timing.csv records actual observed timing deviations.",
                          "Unseen-individual retrieval excludes every same-identity donor. Year retrieval permits only donors whose complete observed future precedes the query origin.",
                          "Point forecasts and route exports are evaluation overlays; observed futures must not be inputs to inference."]}
    if str(device).startswith("cuda"):
        backend = Path(__file__).with_name("migration_gpu_metrics.py")
        protocol["code_sha256"][backend.name] = sha256(backend)
    dump(out / "metrics.json", protocol)
    dump(out / "representative-routes.json", {"scope": "Evaluation exports containing observed-future overlays", "routes": exports})
    return protocol
