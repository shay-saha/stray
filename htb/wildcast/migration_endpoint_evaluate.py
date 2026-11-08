"""Matched endpoint comparisons without changing the frozen multiscale cache.

Short-only references use an explicit endpoint API. Reports contain neither
joint route scores nor trajectory overlays. Predictors receive training
experience and the current query batch's observed history, never held-out truth.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .migration_baselines import MigrationBaselines
from .migration_evaluate import SPLITS, load_cases, macro_scores, paired_animal_bootstrap, score_paths, sha256
from .migration_inference import LOADERS
from .migration_protocol import schema_for
from .sources import dump

LOADERS = {**LOADERS, "wildcast_migration_deepssf_short": ".migration_deepssf",
           "wildcast_migration_issf_short": ".migration_issf",
           "wildcast_migration_issf_short_accelerated": ".migration_issf_accelerated"}
TARGETS = {"y", "target_mask", "target_latlon", "target_time"}
PREDICTORS = {"history", "history_mask", "static", "species", "metric_species", "individual", "study",
              "origin", "history_latlon", "history_time", "case_index"}


def horizon_columns(stored, requested):
    requested = np.asarray(requested, dtype=float)
    if (requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all()
            or np.any(requested <= 0) or np.any(np.diff(requested) <= 0)):
        raise ValueError("Choose positive, finite, increasing endpoint horizons")
    columns = []
    for value in requested:
        found = np.flatnonzero(np.asarray(stored) == value)
        if len(found) != 1:
            raise ValueError("Requested endpoint horizon is absent from the exact frozen grid")
        columns.append(int(found[0]))
    return np.asarray(columns, dtype=int)


def predictor_view(a):
    """Retain training experience; zero every held-out row before adding queries."""
    train = a["split"] == 0
    view = {"split": a["split"].copy(), "horizons": a["horizons"].copy()}
    for key in TARGETS | PREDICTORS:
        if key in a:
            view[key] = np.zeros_like(a[key])
            view[key][train] = a[key][train]
    return view


def _load(path, hashes, device):
    path = Path(path)
    digest = sha256(path)
    probe = torch.load(path, map_location="cpu", weights_only=True)
    kind = probe.get("model_kind")
    if kind not in LOADERS:
        raise ValueError("Unsupported endpoint reference kind: " + str(kind))
    if any(probe.get(key) != value for key, value in hashes.items()):
        raise ValueError("Endpoint checkpoint/data hash mismatch")
    del probe
    model, metadata = importlib.import_module(LOADERS[kind], package="wildcast").load_model(path, device=device)
    if sha256(path) != digest or any(metadata.get(key) != value for key, value in hashes.items()):
        raise ValueError("Endpoint checkpoint changed or loaded different data provenance")
    return model, {"path": str(path), "sha256": digest, "model_kind": kind,
                   "selection_version": metadata.get("selection_version"),
                   "supported_horizons_days": metadata.get("supported_horizons_days"),
                   "independent_horizon_draws": metadata.get("independent_horizon_draws"),
                   "code_sha256": metadata.get("code_sha256"),
                   "adapter_sha256": metadata.get("adapter_sha256"),
                   "prediction_code_sha256": metadata.get("prediction_code_sha256"),
                   "author": metadata.get("author")}


def evaluate(data, output, checkpoints=(), baselines=("stationary", "movement", "seasonal"),
             horizons_days=(1/24, .5, 1), splits=("val_individual", "val_year"),
             samples=64, seed=2026, batch_size=32, device="cpu", bootstrap_replicates=300,
             reference=None, assessment=None, selection=None, max_origins=0):
    if samples < 2 or batch_size < 1 or bootstrap_replicates < 1:
        raise ValueError("Use at least two samples and positive batch/bootstrap counts")
    output, data = Path(output), Path(data)
    if output.exists():
        raise FileExistsError("Endpoint reports are immutable; choose a new output directory")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    a, manifest, hashes = load_cases(data)
    schema_for(manifest, a, legacy_default=True)
    columns = horizon_columns(a["horizons"], horizons_days)
    horizons = a["horizons"][columns].copy()
    codes = [SPLITS[value] if isinstance(value, str) and value in SPLITS else value for value in splits]
    if not codes or any(code not in (1, 2, 3, 4) for code in codes):
        raise ValueError("Evaluate recognized held-out splits only")
    final = any(code >= 3 for code in codes)
    if final != (assessment is not None):
        raise ValueError("Reserved splits require frozen assessment; development must not open final truth")
    if max_origins < 0 or (max_origins and final):
        raise ValueError("A bounded origin panel is for development only")
    assessment_evidence = None
    if final:
        from .migration_assessment import apply_assessment
        a = apply_assessment(a, manifest, hashes, assessment, selection=selection)
        path = Path(assessment)
        path = path / "assessment.json" if path.is_dir() else path
        assessment_evidence = json.loads(path.read_text())
        chosen = {row["sha256"] for row in assessment_evidence["checkpoints"]}
        if any(sha256(path) not in chosen for path in checkpoints):
            raise ValueError("Final endpoint comparison requires its frozen selected checkpoints")
    ids = np.flatnonzero(np.isin(a["split"], codes) & a["target_mask"][:, columns].any(-1))
    if max_origins:
        from .migration_deepssf import validation_panel
        panel = validation_panel(a, max_origins, horizons=tuple(horizons))
        ids = ids[np.isin(ids, panel)]
    if not len(ids):
        raise ValueError("No actual observed endpoints at the requested horizons/splits")
    forecasts = []
    for path in checkpoints:
        model, evidence = _load(path, hashes, device)
        label = Path(path).parent.name + ("-" + Path(path).stem if Path(path).stem != "best" else "")
        forecasts.append((label, model, evidence))
    names = [*baselines, *[name for name, _, _ in forecasts]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Provide uniquely named endpoint references")
    reference = reference or ("seasonal" if "seasonal" in names else names[0])
    if reference not in names:
        raise ValueError("Paired reference must be among the evaluated models")
    view = predictor_view(a)
    fitted = MigrationBaselines(view, manifest)
    previous = np.empty(0, dtype=int)
    train_identities = set(a["individual"][a["split"] == 0])
    rows = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start+batch_size]
        for key in PREDICTORS & view.keys():
            view[key][previous] = 0
            view[key][batch] = a[key][batch]
        previous = batch
        predictions = []
        for name in baselines:
            paths, _ = fitted.predict(name, batch, samples, np.random.default_rng(seed+start))
            predictions.append((name, paths[:, :, columns]))
        for name, model, _ in forecasts:
            if hasattr(model, "predict_endpoints"):
                paths = model.predict_endpoints(view, batch, horizons=horizons,
                                                samples=samples, rng=np.random.default_rng(seed+start))
            elif hasattr(model, "predict_cases"):
                paths = model.predict_cases(view, batch, samples, np.random.default_rng(seed+start), baselines=fitted)
                paths = np.asarray(paths)[:, :, columns]
            else:
                with torch.inference_mode():
                    inputs = [torch.as_tensor(view[key][batch], device=device)
                              for key in ("history", "history_mask", "static", "species")]
                    paths = model.sample_paths(*inputs, samples=samples,
                            generator=torch.Generator(device=device).manual_seed(seed+start)).detach().cpu().numpy()[:, :, columns]
            predictions.append((name, paths))
        for name, paths in predictions:
            paths = np.asarray(paths)
            if paths.shape != (len(batch), samples, len(columns), 2) or not np.isfinite(paths).all():
                raise ValueError("Endpoint reference returned invalid samples")
            scores = score_paths(paths, a["target_latlon"][batch][:, columns],
                                 a["target_mask"][batch][:, columns], a["origin"][batch], device=device)
            for row, case in enumerate(batch):
                for local in np.flatnonzero(a["target_mask"][case, columns]):
                    column = columns[local]
                    elapsed = (a["target_time"][case, column]-a["origin"][case, 2])/86400
                    error = float(scores["point_error_km"][row, local])
                    rows.append({"model": name, "case": int(case), "split": next(k for k,v in SPLITS.items() if v == a["split"][case]),
                        "species": int(a["metric_species"][case]), "prediction_species": int(a["species"][case]),
                        "individual": int(a["individual"][case]), "study": int(a["study"][case]),
                        "identity_in_training": bool(a["individual"][case] in train_identities),
                        "horizon_days": float(horizons[local]), "actual_elapsed_days": float(elapsed),
                        "target_timing_error_hours": float((elapsed-horizons[local])*24),
                        "endpoint_energy_km": float(scores["endpoint_energy_km"][row, local]),
                        "point_error_km": error, "radial_coverage50": bool(scores["radial_coverage50"][row, local]),
                        "radial_coverage90": bool(scores["radial_coverage90"][row, local]),
                        "within_1km": error <= 1, "within_10km": error <= 10, "within_100km": error <= 100})
    if (sha256(data/"manifest.json") != hashes["manifest_sha256"] or sha256(data/"cases.npz") != hashes["cases_sha256"]
            or any(sha256(e["path"]) != e["sha256"] for _, _, e in forecasts)):
        raise ValueError("Frozen input changed during endpoint evaluation")
    frame = pd.DataFrame(rows)
    overall, species, horizon = macro_scores(frame)
    paired = paired_animal_bootstrap(frame, reference, seed, bootstrap_replicates)
    point_frame = frame.drop(columns="endpoint_energy_km").rename(columns={"point_error_km":"endpoint_energy_km"})
    paired_point = paired_animal_bootstrap(point_frame, reference, seed, bootstrap_replicates)
    support = frame.loc[frame.model == names[0]].groupby(["split", "horizon_days"]).agg(
        cases=("case", "size"), identities=("individual", "nunique"), species=("species", "nunique"),
        minimum_elapsed_days=("actual_elapsed_days", "min"), maximum_elapsed_days=("actual_elapsed_days", "max")).reset_index()
    cohorts = []
    for known, group in frame.groupby("identity_in_training"):
        result, _, _ = macro_scores(group)
        result["identity_in_training"] = known
        cohorts.append(result)
    protocol = {"scope":"Matched endpoint-only final assessment" if final else "Matched local endpoint-only development; no official benchmark claim",
        "data":str(data), **hashes, "horizons_days":horizons.tolist(), "stored_cache_horizons_days":a["horizons"].tolist(),
        "cases":len(ids), "observed_targets":len(frame)//len(names), "models":names,
        "checkpoints":[e for _,_,e in forecasts], "samples":samples, "seed":seed, "batch_size":batch_size,
        "scores":overall.to_dict("records"), "paired_energy_deltas":paired, "paired_mean_location_error_deltas":paired_point,
        "case_ids_sha256":hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest(),
        "origin_panel_maximum":max_origins,
        "origin_panel_rule":"Fixed split/taxon/exclusion metadata round-robin seed1701; no target-value or model-error selection" if max_origins else "All eligible held-out origins",
        "assessment_selection_sha256":assessment_evidence["frozen_selection_sha256"] if final else None,
        "metric_species_names":manifest.get("metric_species_names",{}),
        "code_sha256":{name:sha256(Path(__file__).with_name(name)) for name in
                       ("migration_endpoint_evaluate.py","migration_evaluate.py","migration_baselines.py",
                        "migration_protocol.py","multiscale_data.py","migration_clock.py","migration_uncertainty.py")},
        "notes":["Every reference uses the identical frozen origins, actual-target masks and selected horizons.",
                 "No evaluation outcome is filtered by predicted displacement, raster support or missing context.",
                 "Point error is the great-circle miss of the spherical sample mean; energy and radial coverage are separate metrics.",
                 "Aggregation gives equal physical/exclusion identity weight within species, then species and horizon.",
                 "Bootstrap resamples recorded identities within species; it does not establish independent geographic or species transfer.",
                 "Predictors receive training experience and current query histories; all held-out target arrays and other held-out histories are zero.",
                 "This report contains no joint migration-route metric. Short independent endpoint references do not claim seasonal route skill."]}
    if str(device).startswith("cuda"):
        protocol["code_sha256"]["migration_gpu_metrics.py"] = sha256(Path(__file__).with_name("migration_gpu_metrics.py"))
        protocol["distribution_scoring_backend"] = "torch float64, origin-translated Cartesian joint geometry"
    output.mkdir(parents=True,exist_ok=False)
    frame.to_parquet(output/"per-case.parquet",index=False)
    overall.to_csv(output/"scores.csv",index=False)
    species.to_csv(output/"per-species.csv",index=False)
    horizon.to_csv(output/"per-horizon.csv",index=False)
    support.to_csv(output/"observed-support.csv",index=False)
    pd.concat(cohorts,ignore_index=True).to_csv(output/"identity-cohorts.csv",index=False)
    protocol["output_sha256"] = {name:sha256(output/name) for name in
        ("per-case.parquet","scores.csv","per-species.csv","per-horizon.csv","observed-support.csv","identity-cohorts.csv")}
    dump(output/"metrics.json",protocol)
    return protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True);p.add_argument("--output",required=True)
    p.add_argument("--checkpoints",nargs="*",default=[])
    p.add_argument("--baselines",nargs="*",default=["stationary","movement","seasonal"])
    p.add_argument("--horizons-hours",nargs="+",type=float,default=[1,12,24])
    p.add_argument("--splits",nargs="+",default=["val_individual","val_year"])
    for name,default in (("samples",64),("seed",2026),("batch-size",32),("bootstrap-replicates",300)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--device",default="auto");p.add_argument("--reference")
    p.add_argument("--assessment");p.add_argument("--selection")
    p.add_argument("--max-origins",type=int,default=0)
    args=vars(p.parse_args());args["horizons_days"]=np.asarray(args.pop("horizons_hours"))/24
    result=evaluate(**args)
    print(json.dumps({k:result[k] for k in ("cases","observed_targets","models","horizons_days")}))


if __name__ == "__main__":
    main()
