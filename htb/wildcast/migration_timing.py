"""Finite, development-only prediction of author-defined spring migration dates.

Fixed calendar issue times avoid a truth-selected at-risk cohort. Dates are
author NSD delineations; this small study does not establish biological states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .migration_annotations import DEVELOPMENT, WORKBOOK, load_development_timing
from .migration_evaluate import load_cases
from .sources import dump

ISSUES = ("03-01", "04-01", "05-01", "06-01")
EVENTS = ("departure", "arrival")
STATIC = ("origin_unit_x", "origin_unit_y", "origin_unit_z",
          "velocity_1d_east_1000km_per_day", "velocity_1d_north_1000km_per_day",
          "velocity_7d_east_1000km_per_day", "velocity_7d_north_1000km_per_day",
          "log1p_observed_span_days", "memory365_available", "daylength_fraction")
LAGS = (7., 30., 90., 365.)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_issues(origins, labels, issue_dates=ISSUES, tolerance_days=7.):
    """Select actual issue origins using only ID, year, split and calendar time."""
    if tuple(issue_dates) != ISSUES or tolerance_days != 7.:
        raise ValueError("This finite experiment fixes March–June issues and a seven-day window")
    if not set(labels.split) <= DEVELOPMENT:
        raise ValueError("Timing annotations must contain development splits only")
    if labels.duplicated(["canonical_animal", "year"]).any():
        raise ValueError("Duplicate or ambiguous timing animal-year join")
    origins = origins.copy()
    origins["origin_time"] = pd.to_datetime(origins.origin_time, utc=True)
    out, missing = [], Counter()
    for label in labels.itertuples(index=False):
        group = origins.loc[(origins.canonical_animal == label.canonical_animal) &
                            (origins.year == label.year) & (origins.split == label.split)]
        if group.empty:
            raise ValueError("Timing annotation ID/year/split has no exact cache origin join")
        group = group.sort_values(["origin_time", "case_id"], kind="stable")
        for month_day in issue_dates:
            anchor = pd.Timestamp(f"{int(label.year)}-{month_day}", tz="UTC")
            eligible = group.loc[(group.origin_time >= anchor) &
                                 (group.origin_time <= anchor + pd.Timedelta(days=tolerance_days))]
            if eligible.empty:
                missing[f"{label.split}:{month_day}"] += 1
                continue
            row = eligible.iloc[0]
            out.append({"case_id": int(row.case_id), "canonical_animal": label.canonical_animal,
                        "individual": int(row.individual), "year": int(label.year), "split": label.split,
                        "issue_anchor": month_day, "origin_time": row.origin_time,
                        "issue_doy": float(row.origin_time.dayofyear) +
                            (row.origin_time.hour * 3600 + row.origin_time.minute * 60 + row.origin_time.second + row.origin_time.microsecond / 1e6) / 86400})
    result = pd.DataFrame(out)
    if len(result) and result.duplicated(["case_id"]).any():
        raise ValueError("Fixed timing issues unexpectedly reuse an origin")
    return result, dict(missing)


def feature_matrix(a, manifest, ids):
    """Use only observed history/static context; never forecast targets or labels."""
    ids = np.asarray(ids, dtype="int64")
    if np.any(~np.isin(a["split"][ids], [0, 1, 2])):
        raise ValueError("Reserved origins cannot enter the timing development experiment")
    static_names = manifest["static_names"]
    history_names = manifest["history_names"]
    missing = set(STATIC) - set(static_names)
    if missing:
        raise ValueError(f"Missing causal static timing features: {sorted(missing)}")
    hist = np.asarray(a["history"])[ids]
    mask = np.asarray(a["history_mask"])[ids].astype(bool)
    age_index = history_names.index("actual_age_days")
    if np.any(hist[..., age_index][mask] < -1e-6):
        raise ValueError("Future history points are not causal timing inputs")
    if "history_time" in a and np.any(np.asarray(a["history_time"])[ids][mask] > np.broadcast_to(a["origin"][ids, 2:3], mask.shape)[mask]):
        raise ValueError("Future history timestamps are not causal timing inputs")
    dates = pd.to_datetime(a["origin"][ids, 2], unit="s", utc=True)
    doy = dates.dayofyear.to_numpy() + (dates.hour.to_numpy() * 3600 + dates.minute.to_numpy() * 60 + dates.second.to_numpy()) / 86400.
    parts = [doy[:, None] / 366., a["static"][ids][:, [static_names.index(v) for v in STATIC]]]
    names = ["issue_day_of_year_fraction", *STATIC]
    lags = np.asarray(manifest["history_lags_days"], float)
    east, north = history_names.index("east_1000km"), history_names.index("north_1000km")
    for lag in LAGS:
        matched = np.flatnonzero(lags == lag)
        if len(matched) != 1:
            raise ValueError("Frozen timing history lag is missing or duplicated")
        j = int(matched[0])
        values = hist[:, j, [east, north, age_index]].copy()
        values[:, 2] /= 365.
        values[~mask[:, j]] = 0
        parts.extend([values, mask[:, j, None].astype(float)])
        names.extend([f"past_{lag:g}d_east_1000km", f"past_{lag:g}d_north_1000km", f"past_{lag:g}d_actual_age_years", f"past_{lag:g}d_available"])
    matrix = np.column_stack(parts).astype(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite observed timing features")
    return matrix, names


def case_weights(issues):
    """Equal physical animal weight, then its years, then available issue dates."""
    n_issues = issues.groupby(["individual", "year"]).case_id.transform("size").to_numpy(float)
    years = issues.groupby("individual").year.transform("nunique").to_numpy(float)
    w = 1 / (n_issues * years)
    return w / w.sum()


def fit_ridge(x, y, weights, alpha=10.):
    """Weighted, train-only standardization and ridge coefficients for both dates."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    if alpha <= 0 or x.ndim != 2 or y.shape != (len(x), 2) or w.shape != (len(x),) or np.any(w <= 0):
        raise ValueError("Invalid timing ridge fit inputs")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all():
        raise ValueError("Nonfinite timing training inputs")
    w = w / w.sum()
    centre = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - centre) ** 2, axis=0, weights=w))
    scale[scale < 1e-8] = 1.
    z = (x - centre) / scale
    intercept = np.average(y, axis=0, weights=w)
    effective = w * len(x)
    lhs = z.T @ (effective[:, None] * z) + alpha * np.eye(z.shape[1])
    beta = np.linalg.solve(lhs, z.T @ (effective[:, None] * (y - intercept)))
    return {"centre": centre, "scale": scale, "intercept": intercept, "beta": beta, "alpha": float(alpha)}


def ridge_mean(model, x):
    return model["intercept"] + ((np.asarray(x) - model["centre"]) / model["scale"]) @ model["beta"]


def fit_models(x, y, issues, alpha=10., minimum_std_days=3.):
    """Training only; held-out animals cannot tune normalizers or forecast spread."""
    if not set(issues.split) == {"train"}:
        raise ValueError("Only training rows may fit timing models")
    if minimum_std_days <= 0:
        raise ValueError("Timing scale floor must be positive")
    groups = np.asarray(issues.individual)
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three training physical animals are needed for leave-one-animal-out spread")
    w = case_weights(issues)
    prior_mean = np.average(y, axis=0, weights=w)
    prior_std = np.maximum(minimum_std_days, np.sqrt(np.average((y - prior_mean) ** 2, axis=0, weights=w)))
    fitted = fit_ridge(x, y, w, alpha)
    predictions = np.zeros_like(y, dtype=float)
    for animal in unique:
        train, held = groups != animal, groups == animal
        fold = fit_ridge(x[train], y[train], case_weights(issues.loc[train]), alpha)
        predictions[held] = ridge_mean(fold, x[held])
    fitted["std"] = np.maximum(minimum_std_days, np.sqrt(np.average((y - predictions) ** 2, axis=0, weights=w)))
    return {"calendar": {"mean": prior_mean, "std": prior_std}, "ridge": fitted,
            "training_animals": len(unique), "training_rows": len(x),
            "uncertainty_rule": "Calendar spread is training weighted variance; ridge spread is leave-one-training-animal-out RMSE, with fold-local training normalizers."}


def sampled_crps(samples, truth):
    """Unbiased Monte Carlo CRPS via sorted sample-pair distances, in days."""
    samples = np.asarray(samples, float)
    n = samples.shape[-1]
    if n < 2:
        raise ValueError("At least two samples are required for CRPS")
    coefficients = 2 * np.arange(1, n + 1) - n - 1
    pair_term = (np.sort(samples, axis=-1) * coefficients).sum(-1) / (n * (n - 1))
    return np.abs(samples - np.asarray(truth)[..., None]).mean(-1) - pair_term


def _binary_log_score(probability, truth):
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(truth * np.log(p) + (1 - truth) * np.log1p(-p))


def score_distribution(mean, std, truth, issue_doy, standard_draws):
    mean, std = np.broadcast_arrays(mean, std)
    if np.any(std <= 0):
        raise ValueError("Timing forecast standard deviations must be positive")
    crps = sampled_crps(mean[..., None] + std[..., None] * standard_draws, truth)
    error = np.abs(mean - truth)
    occurred = truth <= issue_doy
    p_occurred = ndtr((issue_doy - mean) / std)
    metrics = {"crps_days": crps, "absolute_error_days": error,
               "date_nll": .5 * ((truth - mean) / std) ** 2 + np.log(std) + .5 * np.log(2 * np.pi),
               "occurred_by_issue": occurred, "probability_occurred_by_issue": p_occurred,
               "occurred_brier": (p_occurred - occurred) ** 2,
               "occurred_log_score": _binary_log_score(p_occurred, occurred),
               "within_7d_point_error": error <= 7, "within_30d_point_error": error <= 30,
               "date_90pct_coverage": np.abs(truth - mean) <= 1.6448536269514722 * std}
    for horizon in (7, 30):
        probability = ndtr((issue_doy + horizon - mean) / std) - p_occurred
        outcome = (truth > issue_doy) & (truth <= issue_doy + horizon)
        metrics[f"probability_next_{horizon}d"] = probability
        metrics[f"next_{horizon}d_brier"] = (probability - outcome) ** 2
        metrics[f"next_{horizon}d_log_score"] = _binary_log_score(probability, outcome)
    return metrics


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(values)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def summarize(frame):
    rows = []
    scores = [c for c in frame if c.endswith(("brier", "log_score")) or c in
              {"crps_days", "date_nll", "absolute_error_days", "within_7d_point_error", "within_30d_point_error", "date_90pct_coverage"}]
    for (model, split, event), group in frame.groupby(["model", "split", "event"], sort=True):
        for stratum, subset in [("all_fixed_issues", group),
                                ("before_event_diagnostic", group.loc[~group.occurred_by_issue]),
                                ("already_occurred_diagnostic", group.loc[group.occurred_by_issue])]:
            if subset.empty:
                continue
            w = subset.weight.to_numpy(float)
            rows.append({"model": model, "split": split, "event": event, "stratum": stratum,
                         "issues": len(subset), "physical_animals": subset.individual.nunique(),
                         **{key: float(np.average(subset[key], weights=w)) for key in scores},
                         "median_absolute_error_days": _weighted_median(subset.absolute_error_days.to_numpy(), w)})
            rows[-1]["mean_absolute_error_days"] = rows[-1].pop("absolute_error_days")
    return pd.DataFrame(rows)


def paired_bootstrap(frame, replicates=1000, seed=1701):
    rng, rows = np.random.default_rng(seed), []
    validation = frame.loc[frame.split != "train"]
    for split, split_frame in validation.groupby("split"):
        for event in (*EVENTS, "pooled"):
            selected = split_frame if event == "pooled" else split_frame.loc[split_frame.event == event]
            values = {}
            for model, model_frame in selected.groupby("model"):
                values[model] = {animal: np.average(group.crps_days, weights=group.weight)
                                 for animal, group in model_frame.groupby("individual")}
            animals = sorted(set(values.get("calendar", {})) & set(values.get("ridge", {})))
            delta = np.array([values["ridge"][a] - values["calendar"][a] for a in animals])
            interval = None
            if len(animals) >= 2:
                bootstrap = delta[rng.integers(0, len(delta), size=(replicates, len(delta)))].mean(1)
                interval = np.quantile(bootstrap, [.025, .975]).tolist()
            rows.append({"split": split, "event": event, "physical_animals": len(animals),
                         "ridge_minus_calendar_crps_days": float(delta.mean()) if len(delta) else None,
                         "paired_animal_bootstrap_ci95": interval, "replicates": replicates,
                         "warning": "Very small validation population; resampling these animals does not establish population-wide or frontier performance."})
    return rows


def run(data, output, alpha=10., samples=512, seed=1701, bootstrap=1000):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Timing experiment outputs are immutable; choose a new directory")
    if samples < 2 or bootstrap < 1:
        raise ValueError("At least two forecast samples and a positive bootstrap count are required")
    a, manifest, basehashes = load_cases(data)
    origins = pd.read_parquet(Path(data) / "origins.parquet")
    if sha256(Path(data) / "origins.parquet") != manifest["origins_sha256"]:
        raise ValueError("Timing origin metadata changed")
    labels, label_evidence = load_development_timing(data)
    issues, missing = select_issues(origins, labels)
    if issues.empty or not (issues.split == "train").any() or not (issues.split != "train").any():
        raise ValueError("No eligible training/validation fixed issue origins")
    x, feature_names = feature_matrix(a, manifest, issues.case_id.to_numpy())
    joined = issues.merge(labels[["canonical_animal", "year", "split", "departure_doy", "arrival_doy"]],
                          on=["canonical_animal", "year", "split"], how="left", validate="many_to_one", sort=False)
    if not np.array_equal(joined.case_id.to_numpy(), issues.case_id.to_numpy()) or joined[["departure_doy", "arrival_doy"]].isna().any().any():
        raise ValueError("Invalid timing label/origin join")
    truth = joined[["departure_doy", "arrival_doy"]].to_numpy(float)
    train = issues.split.to_numpy() == "train"
    models = fit_models(x[train], truth[train], issues.loc[train], alpha)
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((len(issues), 2, samples))
    weights = np.zeros(len(issues))
    for split in issues.split.unique():
        selected = issues.split.to_numpy() == split
        weights[selected] = case_weights(issues.loc[selected])
    predictions = []
    for name in ("calendar", "ridge"):
        mean = np.broadcast_to(models["calendar"]["mean"], truth.shape) if name == "calendar" else ridge_mean(models["ridge"], x)
        std = np.broadcast_to(models[name]["std"], truth.shape)
        metrics = score_distribution(mean, std, truth, issues.issue_doy.to_numpy()[:, None], draws)
        for j, event in enumerate(EVENTS):
            frame = issues.copy()
            frame["model"], frame["event"], frame["weight"] = name, event, weights
            frame["forecast_mean_doy"], frame["forecast_std_days"], frame["observed_author_doy"] = mean[:, j], std[:, j], truth[:, j]
            for key, value in metrics.items():
                frame[key] = value[:, j]
            predictions.append(frame)
    scored = pd.concat(predictions, ignore_index=True)
    summary = summarize(scored)
    uncertainty = paired_bootstrap(scored, bootstrap, seed + 1)
    # No reserved outcome arrays, animal IDs as predictors or true event dates
    # participate in issue selection, feature construction or validation fitting.
    report = {"protocol": "wildcast-author-timing-development-v1", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Small Wyoming author-defined NSD spring-timing development experiment; no independent final assessment or frontier claim.",
              "data": str(data), "base_hashes": basehashes, "origins_sha256": manifest["origins_sha256"],
              "annotation_evidence": label_evidence, "feature_names": feature_names,
              "code_sha256": {"migration_timing.py": sha256(__file__), "migration_annotations.py": sha256(Path(__file__).with_name("migration_annotations.py"))},
              "settings": {"issue_dates": ISSUES, "max_issue_delay_days": 7, "alpha": alpha, "minimum_std_days": 3.,
                           "samples": samples, "seed": seed, "bootstrap": bootstrap},
              "eligible_issue_counts": issues.groupby("split").size().to_dict(),
              "eligible_animals": issues.groupby("split").individual.nunique().to_dict(), "missing_calendar_issues": missing,
              "source_annotation_cohort": {s: {"animal_years": len(g), "animals": g.canonical_animal.nunique()}
                                           for s, g in labels.groupby("split")},
              "training_animals": models["training_animals"], "uncertainty_fit": models["uncertainty_rule"],
              "score_rule": "Unbiased Monte Carlo CRPS in days with coupled Gaussian draws; it can be slightly negative. Binary log scores use numerical probability clipping at1e-12.",
              "distribution_rule": "Gaussian distributions over full annual day-of-year dates; no at-risk or future-event-date cohort selection. Gaussian tails can extend beyond calendar-year limits.",
              "weight_rule": "Equal physical animal weight, then observed years, then available fixed issue times.",
              "interpretation": "All-issue scores combine forecasting before events and retrospective date estimation after events. Before/after strata are diagnostics, not verified latent biological states or selection cohorts.",
              "selection_rule": "Compare the two predeclared models on all-fixed-issue validation CRPS, pooling departure/arrival. Diagnostic event strata do not select models or issue cohorts.",
              "limitations": ["Very small source-specific development cohort; the source_annotation_cohort records the animal and season counts before issue availability filtering.",
                              "No validation-year labels after2020 and no independently observed departure/arrival/stopover states.",
                              "The two Gaussian date heads are marginal predictions; independently sampled arrival/departure dates are not constrained to be ordered.",
                              "Frozen v3 origin availability was previously filtered for future GPS target availability and speed QC; timing issue selection adds no event-date-dependent filtering.",
                              "No reserved labels were selected, derived, evaluated or released."],
              "paired_comparisons": uncertainty}
    output.mkdir(parents=True, exist_ok=False)
    scored.to_parquet(output / "development-predictions.parquet", index=False)
    issues.to_parquet(output / "fixed-issues.parquet", index=False)
    summary.to_csv(output / "metrics.csv", index=False)
    np.savez_compressed(output / "model.npz", calendar_mean=models["calendar"]["mean"], calendar_std=models["calendar"]["std"],
                        **{f"ridge_{key}": value for key, value in models["ridge"].items()})
    report["files_sha256"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    dump(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=10.)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    report = run(args.data, args.output, args.alpha, args.samples, args.seed, args.bootstrap)
    print(json.dumps({"eligible_animals": report["eligible_animals"], "paired_comparisons": report["paired_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
