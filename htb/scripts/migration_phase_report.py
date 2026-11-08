"""Exploratory development diagnostics for migration-relevant forecast cases.

Causal mobility strata use observed past GPS and TRAIN-only thresholds.
Author NSD phases and future-displacement proxies are separate retrospective
diagnostics. None of these reports select models or create prediction inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wildcast.migration_annotations import load_development_timing
from wildcast.migration_evaluate import SPLITS, load_cases, macro_scores, sha256
from wildcast.sources import dump

QUANTILE = .8
LAGS = (7., 30.)
PHYSICAL_KM = 100.


def mobility(a, manifest, ids):
    """Read only actual observed past fixes for the supplied development IDs."""
    ids = np.asarray(ids, dtype=int)
    if np.any(~np.isin(a["split"][ids], [0, 1, 2])):
        raise ValueError("Reserved queries cannot enter development mobility strata")
    names = manifest.get("history_names", ["east_1000km", "north_1000km", "actual_age_days"])
    east, north, age = [names.index(name) for name in ["east_1000km", "north_1000km", "actual_age_days"]]
    lags = np.asarray(manifest["history_lags_days"], dtype=float)
    result = {}
    for lag in LAGS:
        matches = np.flatnonzero(lags == lag)
        if len(matches) != 1:
            raise ValueError("Frozen mobility lag missing or duplicated")
        j = int(matches[0])
        valid = a["history_mask"][ids, j].astype(bool)
        value = a["history"][ids, j]
        if not np.isfinite(value[valid][:, [east, north, age]]).all() or np.any(value[valid, age] < lag - 1e-5):
            raise ValueError("Mobility history is non-finite or later than its declared past lag")
        if "history_time" in a and np.any(a["history_time"][ids, j][valid] > a["origin"][ids, 2][valid] - lag * 86400 + .001):
            raise ValueError("Mobility history timestamp is not causal")
        result[f"past_{int(lag)}d_km"] = np.where(valid, np.linalg.norm(value[:, [east, north]], axis=-1) * 1000, np.nan)
        result[f"past_{int(lag)}d_available"] = valid
    return pd.DataFrame({"case": ids, **result})


def fit_thresholds(a, manifest):
    """Fixed80th empirical percentiles; equal physical-animal weight per species."""
    ids = np.flatnonzero(a["split"] == 0)
    past = mobility(a, manifest, ids)
    taxon, animals = a.get("metric_species", a["species"])[ids], a["individual"][ids]
    rows = []
    for species in sorted(np.unique(taxon)):
        for lag in LAGS:
            selected = (taxon == species) & past[f"past_{int(lag)}d_available"].to_numpy()
            value, identity = past[f"past_{int(lag)}d_km"].to_numpy()[selected], animals[selected]
            threshold = None
            if len(value):
                _, inverse, counts = np.unique(identity, return_inverse=True, return_counts=True)
                weight = 1 / counts[inverse]
                order = np.argsort(value, kind="stable")
                crossing = np.searchsorted(np.cumsum(weight[order]), QUANTILE * weight.sum(), side="left")
                threshold = float(value[order[min(crossing, len(value) - 1)]])
            rows.append({"species": int(species), "lag_days": lag, "quantile": QUANTILE,
                         "threshold_km": threshold, "training_origins": int(selected.sum()),
                         "training_independent_animals": int(len(np.unique(identity)))})
    return pd.DataFrame(rows)


def classify(a, manifest, ids, thresholds):
    past = mobility(a, manifest, ids)
    taxon = a.get("metric_species", a["species"])[np.asarray(ids, int)]
    lookup = {(int(row.species), int(row.lag_days)): row.threshold_km for row in thresholds.itertuples(index=False)}
    causal, physical = [], []
    for row, species in zip(past.itertuples(index=False), taxon):
        complete = row.past_7d_available and row.past_30d_available
        seven, month = lookup.get((int(species), 7)), lookup.get((int(species), 30))
        if not complete:
            causal.append("missing_past_history")
        elif seven is None or month is None or not np.isfinite([seven, month]).all():
            causal.append("no_training_species_threshold")
        else:
            causal.append("high_past_mobility" if row.past_7d_km > seven or row.past_30d_km > month else "ordinary_past_mobility")
        physical.append("missing_past_history" if not complete else "past_displacement_ge100km" if max(row.past_7d_km, row.past_30d_km) >= PHYSICAL_KM else "past_displacement_lt100km")
    return past.assign(causal_regime=causal, causal_physical_100km=physical)


def author_phases(origins, annotations):
    """Retrospective spring-date interpretation at UTC calendar-day precision."""
    allowed = {"train", "val_individual", "val_year"}
    if not set(annotations.split) <= allowed or not set(origins.split) <= allowed:
        raise ValueError("Reserved annotation/origin rows cannot enter phase diagnostics")
    keys = ["canonical_animal", "year", "split"]
    if annotations.duplicated(keys).any():
        raise ValueError("Duplicate author animal-year timing annotations")
    keep = annotations[keys + ["departure_doy", "arrival_doy"]]
    result = origins.merge(keep, on=keys, how="left", validate="many_to_one")
    dates = pd.to_datetime(result.origin_time, utc=True)
    day = dates.dt.dayofyear.to_numpy()
    known = result.departure_doy.notna().to_numpy() & result.arrival_doy.notna().to_numpy()
    start, end = np.floor(result.departure_doy.to_numpy()), np.floor(result.arrival_doy.to_numpy())
    labels = np.full(len(result), "unannotated", dtype=object)
    labels[known & (day < start)] = "author_predeparture"
    labels[known & (day >= start) & (day <= end)] = "author_in_transit"
    labels[known & (day > end)] = "author_postarrival"
    return result[["case", "canonical_animal", "year"]].assign(author_phase=labels)


def _support_hash(cases, horizons):
    value = np.empty(len(cases), dtype=[("case", "<i8"), ("horizon", "<f8")])
    value["case"], value["horizon"] = cases, horizons
    return hashlib.sha256(value.tobytes()).hexdigest()


def verified_development(data, evaluation):
    a, manifest, hashes = load_cases(data)
    root = Path(evaluation)
    metrics_path = root / "metrics.json"
    metrics_hash = sha256(metrics_path)
    protocol = json.loads(metrics_path.read_text())
    for key, value in hashes.items():
        if protocol.get(key) != value:
            raise ValueError("Evaluation/cache " + key + " mismatch")
    if (protocol.get("assessment") is not None or not protocol["splits"]
        or len(protocol["splits"]) != len(set(protocol["splits"]))
        or not set(protocol["splits"]) <= {"val_individual", "val_year"}):
        raise ValueError("Only retained development evaluation may enter phase diagnostics")
    table = root / "per-case.parquet"
    before = sha256(table)
    frame = pd.read_parquet(table)
    if sha256(table) != before:
        raise ValueError("Retained development table changed while reading")
    required = {"model", "case", "split", "species", "individual", "study", "horizon_days", "endpoint_energy_km",
                "radial_coverage50", "radial_coverage90", "displacement_proxy"}
    if not required <= set(frame):
        raise ValueError("Retained development score schema is incomplete")
    if frame.duplicated(["model", "case", "horizon_days"]).any():
        raise ValueError("Duplicate development model/origin/horizon support")
    if not np.isfinite(frame.endpoint_energy_km).all() or not set(frame.model) == set(protocol["models"]):
        raise ValueError("Retained scores/models are incomplete")
    if len(protocol["models"]) != len(set(protocol["models"])):
        raise ValueError("Retained model inventory is duplicated")
    for column in ("radial_coverage50", "radial_coverage90"):
        if not np.isin(frame[column].to_numpy(), [False, True]).all():
            raise ValueError("Retained calibration flags must be complete binary observations")
    if not set(frame.split) <= {"val_individual", "val_year"}:
        raise ValueError("Reserved score rows cannot enter development diagnostics")
    if set(frame.split) != set(protocol["splits"]):
        raise ValueError("Retained scored splits differ from the declared protocol")
    if not pd.api.types.is_integer_dtype(frame.case):
        raise ValueError("Retained case IDs must be integers")
    ids = np.sort(frame.case.unique().astype(np.int64))
    if np.any(ids < 0) or np.any(ids >= len(a["split"])) or np.any(~np.isin(a["split"][ids], [1, 2])):
        raise ValueError("Development origin IDs outside the permitted frozen cache")
    identity_hash = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
    if identity_hash != protocol["case_ids_sha256"] or len(ids) != protocol["cases"]:
        raise ValueError("Retained origin support checksum differs")
    codes = {v: k for k, v in SPLITS.items()}
    metric_species = a.get("metric_species", a["species"])
    cases = frame.case.to_numpy(dtype=np.int64)
    if (not np.array_equal(frame.individual, a["individual"][cases])
        or not np.array_equal(frame.species, metric_species[cases])
        or not np.array_equal(frame.study, a["study"][cases])
        or not np.array_equal(frame.split, np.asarray([codes[int(v)] for v in a["split"][cases]]))):
        raise ValueError("Retained biological grouping metadata differs from the cache")
    if "prediction_species" in frame and not np.array_equal(frame.prediction_species, a["species"][cases]):
        raise ValueError("Retained prediction-species grouping differs from the cache")
    ii, hh = np.nonzero(a["target_mask"][ids])
    expected = pd.DataFrame({"case": ids[ii], "horizon_days": a["horizons"][hh]}).sort_values(["case", "horizon_days"])
    support = _support_hash(expected.case, expected.horizon_days)
    for model, chunk in frame.groupby("model", sort=False):
        chunk = chunk.sort_values(["case", "horizon_days"])
        if len(chunk) != len(expected) or _support_hash(chunk.case, chunk.horizon_days) != support:
            raise ValueError("Model development support/target masks differ: " + model)
    if len(expected) != protocol["observed_targets"]:
        raise ValueError("Observed development target count differs")
    if frame.groupby("case").displacement_proxy.nunique().max() != 1:
        raise ValueError("Future displacement proxy differs across model support")
    origin_path = Path(data) / "origins.parquet"
    if sha256(origin_path) != manifest["origins_sha256"]:
        raise ValueError("Origin metadata fingerprint differs")
    origins = pd.read_parquet(origin_path, columns=["case_id", "canonical_animal", "year", "split", "origin_time"])
    if sha256(origin_path) != manifest["origins_sha256"] or sha256(metrics_path) != metrics_hash:
        raise ValueError("Retained origin metadata or evaluation protocol changed while reading")
    origins = origins.set_index("case_id").loc[ids].reset_index().rename(columns={"case_id": "case"})
    origin_times = pd.to_datetime(origins.origin_time, utc=True).astype("int64").to_numpy() / 1e9
    if not np.allclose(origin_times, a["origin"][ids, 2], rtol=0, atol=1e-5):
        raise ValueError("Origin annotation clock differs from its frozen cache row")
    return a, manifest, frame, origins, protocol, {**hashes, "metrics_sha256": metrics_hash,
        "per_case_sha256": before, "case_ids_sha256": identity_hash, "target_support_sha256": support,
        "origins_sha256": manifest["origins_sha256"]}


def frozen_routing(a, ids, hashes, path, expected_sha256=None):
    """Apply an immutable TRAIN-fitted router, without fitting any thresholds."""
    from wildcast.migration_regimes import classify as classify_frozen
    path = Path(path)
    if path.is_dir():
        path = path / "rules.json"
    ids = np.asarray(ids, dtype=np.int64)
    if np.any(~np.isin(a["split"][ids], [1, 2])):
        raise ValueError("Reserved queries cannot enter frozen development routing")
    actual = sha256(path)
    evidence = {"path": str(path), "sha256": actual, "thresholds_refitted": False}
    if expected_sha256 is None:
        filter_path = path.parent / "filter.json"
        if not filter_path.is_file():
            raise ValueError("Frozen routing rules require sibling filter.json or --routing-rules-sha256")
        filter_hash = sha256(filter_path)
        frozen = json.loads(filter_path.read_text())
        if sha256(filter_path) != filter_hash:
            raise ValueError("Frozen routing filter manifest changed while reading")
        for key in ("manifest_sha256", "cases_sha256"):
            if frozen.get(key) != hashes[key]:
                raise ValueError("Frozen routing filter/cache " + key + " mismatch")
        expected_sha256 = frozen.get("rules_sha256")
        evidence.update(filter_manifest=str(filter_path), filter_manifest_sha256=filter_hash)
    if actual != expected_sha256:
        raise ValueError("Frozen routing rules checksum mismatch")
    rules = json.loads(path.read_text())
    if sha256(path) != actual:
        raise ValueError("Frozen routing rules changed while reading")
    for key in ("manifest_sha256", "cases_sha256"):
        if rules.get(key) != hashes[key]:
            raise ValueError("Frozen routing rules/cache " + key + " mismatch")
    provenance = rules.get("provenance", {})
    if (provenance.get("causal") is not True or provenance.get("uses_future_targets") is not False
        or provenance.get("threshold_fit_split") != "train"):
        raise ValueError("Frozen routing rules lack TRAIN-only causal provenance")
    routing = classify_frozen(a, ids, rules)
    evidence.update(protocol=rules["protocol"], provenance=provenance,
                    weighting="Predictor-species TRAIN thresholds; global actual-species/physical-animal balanced fallback for unknown or missing species thresholds")
    return pd.DataFrame({"case": ids, "frozen_router_regime": routing["regime"],
                         "frozen_router_global_fallback": routing["global_fallback"]}), evidence


def paired_intervals(frame, reference="seasonal", replicates=300, seed=2026):
    """Joint animal resampling yields both per-horizon and overall paired CIs."""
    models = sorted(frame.model.unique())
    if reference not in models:
        raise ValueError("Seasonal reference missing from the diagnostic cohort")
    pivot = frame.pivot(index=["split", "species", "individual", "case", "horizon_days"], columns="model", values="endpoint_energy_km")
    if pivot.isna().any().any():
        raise ValueError("Paired diagnostic scores require identical support")
    rng, rows = np.random.default_rng(seed), []
    for split, group in pivot.reset_index().groupby("split", sort=True):
        animal = group.groupby(["species", "individual", "horizon_days"])[models].mean()
        horizons = sorted(group.horizon_days.unique())
        blocks = []
        for _, block in animal.groupby(level="species", sort=True):
            identities = block.index.get_level_values("individual").unique()
            values = block.droplevel("species").reindex(pd.MultiIndex.from_product([identities, horizons], names=["individual", "horizon_days"])).to_numpy().reshape(len(identities), len(horizons), len(models))
            present = np.isfinite(values).all(-1)
            blocks.append((np.nan_to_num(values), present))
        def aggregate(resample):
            sums, counts = np.zeros((len(horizons), len(models))), np.zeros(len(horizons))
            for values, present in blocks:
                weight = rng.multinomial(len(values), np.full(len(values), 1 / len(values))) if resample else np.ones(len(values))
                n = weight @ present
                good = n > 0
                total = np.einsum("a,ahm->hm", weight, values)
                sums[good] += total[good] / n[good, None]
                counts[good] += 1
            means = np.full_like(sums, np.nan)
            good = counts > 0
            means[good] = sums[good] / counts[good, None]
            return np.vstack([np.nanmean(means, axis=0), means])
        point = aggregate(False)
        point = point - point[:, models.index(reference):models.index(reference) + 1]
        draws = np.stack([aggregate(True) for _ in range(replicates)])
        draws = draws - draws[:, :, models.index(reference):models.index(reference) + 1]
        low, high = np.nanquantile(draws, [.025, .975], axis=0)
        for h, horizon in enumerate(["all", *horizons]):
            for j, model in enumerate(models):
                rows.append({"model": model, "split": split, "horizon_days": horizon,
                             "delta_vs_seasonal_km": float(point[h, j]), "ci95_low_km": float(low[h, j]),
                             "ci95_high_km": float(high[h, j]), "bootstrap_effective_replicates": int(np.isfinite(draws[:, h, j]).sum())})
    return pd.DataFrame(rows)


def summarize(frame, group_column, view, replicates=300, seed=2026, intervals=True):
    overall, per_horizon, per_species = [], [], []
    for label, group in frame.groupby(group_column, sort=True):
        macro, species, horizon = macro_scores(group)
        base = group.loc[group.model == "seasonal"]
        if base.empty:
            raise ValueError("Seasonal reference missing from retained score table")
        counts = base.groupby("split").agg(origins=("case", "nunique"), observed_targets=("case", "size"),
                                               independent_animals=("individual", "nunique"), species_count=("species", "nunique")).reset_index()
        hcounts = base.groupby(["split", "horizon_days"]).agg(origins=("case", "nunique"), observed_targets=("case", "size"),
                                               independent_animals=("individual", "nunique"), species_count=("species", "nunique")).reset_index()
        macro = macro.merge(counts, on="split", validate="many_to_one").assign(horizon_days="all")
        horizon = horizon.merge(hcounts, on=["split", "horizon_days"], validate="many_to_one")
        if intervals:
            stable_seed = seed + int.from_bytes(hashlib.sha256(f"{view}:{label}".encode()).digest()[:4], "little")
            paired = paired_intervals(group, replicates=replicates, seed=stable_seed)
            macro = macro.merge(paired.loc[paired.horizon_days == "all"], on=["model", "split", "horizon_days"], validate="one_to_one")
            numeric = paired.loc[paired.horizon_days != "all"].copy()
            numeric["horizon_days"] = numeric.horizon_days.astype(float)
            horizon = horizon.merge(numeric, on=["model", "split", "horizon_days"], validate="one_to_one")
        for table in (macro, horizon, species):
            table["view"], table["stratum"] = view, label
        overall.append(macro); per_horizon.append(horizon); per_species.append(species)
    empty = pd.DataFrame(columns=["model", "split", "horizon_days", "view", "stratum", "endpoint_energy_km"])
    return tuple(pd.concat(tables, ignore_index=True) if tables else empty.copy() for tables in [overall, per_horizon, per_species])


def report(data, evaluation, output, bootstrap_replicates=300, seed=2026,
           routing_rules=None, routing_rules_sha256=None):
    if bootstrap_replicates < 2:
        raise ValueError("At least two paired bootstrap replicates required")
    out = Path(output)
    if out.exists():
        raise FileExistsError("Phase report output must be a new directory")
    if routing_rules_sha256 is not None and routing_rules is None:
        raise ValueError("A routing-rules checksum requires a routing-rules file")
    a, manifest, frame, origins, protocol, hashes = verified_development(data, evaluation)
    thresholds = fit_thresholds(a, manifest)
    causal = classify(a, manifest, origins.case.to_numpy(), thresholds)
    labels, annotation_evidence = load_development_timing(data)
    phases = author_phases(origins, labels)
    descriptions = causal.merge(phases, on="case", validate="one_to_one")
    ids = descriptions.case.to_numpy(dtype=int)
    training_identities = np.unique(a["individual"][a["split"] == 0])
    training_species = np.unique(a["species"][a["split"] == 0])
    descriptions["individual_knowledge"] = np.where(np.isin(a["individual"][ids], training_identities), "seen_training_individual", "new_individual")
    descriptions["species_knowledge"] = np.where((a["species"][ids] != 0) & np.isin(a["species"][ids], training_species), "known_training_species", "unknown_training_species")
    routing_evidence = None
    if routing_rules is not None:
        router, routing_evidence = frozen_routing(a, ids, hashes, routing_rules, routing_rules_sha256)
        descriptions = descriptions.merge(router, on="case", validate="one_to_one")
    joined = frame.merge(descriptions, on="case", validate="many_to_one")
    out.mkdir(parents=True)
    thresholds.to_csv(out / "training-mobility-thresholds.csv", index=False)
    descriptions.to_parquet(out / "development-origin-strata.parquet", index=False)
    tables = {}
    views = [
        ("causal-quantile", "causal_regime", joined, True),
        ("causal-physical", "causal_physical_100km", joined, True),
        ("knowledge-animal", "individual_knowledge", joined, True),
        ("knowledge-species", "species_knowledge", joined, True),
        ("author-retrospective", "author_phase", joined.loc[joined.author_phase != "unannotated"], True),
        ("future-displacement-descriptive", "displacement_proxy", joined, False)]
    if routing_evidence is not None:
        views.append(("frozen-router", "frozen_router_regime", joined, True))
    for name, column, chunk, intervals in views:
        summaries = summarize(chunk, column, name, bootstrap_replicates, seed, intervals)
        for suffix, table in zip(["overall", "per-horizon", "per-species"], summaries):
            table.to_csv(out / (name + "-" + suffix + ".csv"), index=False)
        tables[name] = summaries[0]
    counts = []
    count_columns = ["causal_regime", "causal_physical_100km", "individual_knowledge", "species_knowledge", "author_phase"]
    if routing_evidence is not None:
        count_columns += ["frozen_router_regime", "frozen_router_global_fallback"]
    for column in count_columns:
        cells = joined.loc[joined.model == "seasonal"].groupby(["split", column]).agg(
            origins=("case", "nunique"), independent_animals=("individual", "nunique"), species=("species", "nunique"), observed_targets=("case", "size")).reset_index().rename(columns={column: "stratum"})
        cells["view"] = column
        counts.append(cells)
    cohort_counts = pd.concat(counts, ignore_index=True)
    cohort_counts.to_csv(out / "cohort-counts.csv", index=False)
    annotation_counts = {split: {"animal_years": int(len(group)), "animals": int(group.canonical_animal.nunique())}
                         for split, group in labels.groupby("split")}
    result = {"scope": "Exploratory post-development stratification; not an independent test, official benchmark or frontier claim",
              **hashes, "data": str(data), "evaluation": str(evaluation), "models": protocol["models"],
              "evaluated_origins": len(origins), "observed_targets": protocol["observed_targets"],
              "samples_in_retained_evaluation": protocol["samples"], "bootstrap_replicates": bootstrap_replicates,
              "seed": seed, "model_selection_performed": False, "reserved_queries_or_outcomes_used": False,
              "threshold_rule": {"quantile": QUANTILE, "fit_split": "train only", "lags_days": list(LAGS),
                                  "weighting": "Empirical inverse CDF with equal physical-animal weight within species",
                                  "high_rule": "Both lags observed and either displacement strictly above its species training threshold",
                                  "missing_rule": "Either lag unavailable is separate; species lacking training thresholds is separate",
                                  "physical_view_threshold_km": PHYSICAL_KM},
              "author_annotation_cohort": annotation_counts, "annotation_evidence": annotation_evidence,
              "frozen_routing_rules": routing_evidence,
              "code_sha256": {"migration_phase_report.py": sha256(__file__),
                              "migration_annotations.py": sha256(Path(__file__).resolve().parents[1] / "wildcast/migration_annotations.py")},
              "notes": ["Causal strata use only observed past locations at each origin and fixed training-only thresholds; high relative mobility is not verified migration.",
                        "Fixed100km past displacement is a separate causal descriptive view and does not establish biological intent.",
                        "All model scores within a stratum reuse identical frozen origins and observed-target support; no new forecasts or training are run.",
                        "Individual and species knowledge are separate origin-time views, never a cross-product cohort. Whole-calendar-year validation can include new animals and taxa and is not purely familiar-animal forecasting.",
                        "The scientific causal-quantile view uses actual reporting species and separates taxa without TRAIN thresholds. The optional frozen-router view applies the original predictor-species rules, including their TRAIN-balanced global fallback for unknown token0; it neither refits thresholds nor treats that route assignment as biological migration.",
                        "Energy and radial calibration use the evaluator's physical-animal/species/horizon macro aggregation.",
                        "Intervals resample physical animals within species, retaining all their origins/horizons. They condition on the fitted thresholds and do not account for model selection, annotation error or all sources of uncertainty.",
                        "Per-horizon and multi-model exploration is not corrected for multiple comparisons and must not be interpreted as confirmatory significance.",
                        "Author phases are retrospective NSD-derived spring-date labels joined by exact source animal/year/split; not independently observed biological states or predictor inputs.",
                        "Author departure/arrival calendar days are included in transit; published day-of-year values are floored to UTC calendar days, with no inferred subday biological precision.",
                        "The author validation cohort is very small and source-specific; animal-year and independent-animal counts are reported explicitly.",
                        "Future-displacement strata condition on observed outcomes, are descriptive only, and cannot define a prospective origin-time cohort."]}
    lines = ["# Migration-relevance development diagnostics", "", result["scope"] + ".", "",
             f"Retained evaluation: {len(origins):,} origins, {protocol['observed_targets']:,} observed targets, {protocol['samples']} path samples. No new predictions, model fitting or reserved assessment occurred.", "",
             "Past-mobility cutoffs use training data only, with equal physical-animal weight within species. Both seven-day and thirty-day observations must be available. A high relative percentile can still correspond to small absolute displacement; the separate fixed100km view makes that distinction visible.", "",
             "| Author annotation split | Animal-years | Independent animals |", "|---|---:|---:|"]
    for split, count in annotation_counts.items():
        lines.append(f"| {split} | {count['animal_years']} | {count['animals']} |")
    lines += ["", "The source-specific author dates are retrospective NSD delineations, not independently observed migration states. Their small validation cohort cannot establish broad migration accuracy.", "",
              "See `causal-quantile-*` for high/ordinary/missing-history comparisons, `causal-physical-*` for a fixed100km past-displacement view, `knowledge-animal-*` and `knowledge-species-*` for separate familiar/novel origin-time views, `author-retrospective-*` for author phases, and `future-displacement-descriptive-*` for outcome-conditioned descriptions. Each view has overall, per-horizon and per-species tables. Negative paired differences favour the model over seasonal retrieval.", "",
              "Whole-calendar-year validation can include new animals and unseen taxa; it is not purely a familiar-animal test. Knowledge cohorts are reported separately to avoid an uninformative cross-product of small cells.", "",
              "Radial coverage is about the sampled mean, not highest-density-region coverage. Confidence intervals are exploratory and do not incorporate model-selection or threshold-estimation uncertainty. No model is promoted by this diagnostic report."]
    if routing_evidence is not None:
        result["code_sha256"]["migration_regimes.py"] = sha256(Path(__file__).resolve().parents[1] / "wildcast/migration_regimes.py")
        lines += ["", "`frozen-router-*` additionally reports the original immutable router's assignments. It uses predictor-species thresholds and the recorded global fallback for unknown taxa, while `causal-quantile-*` continues to separate taxa lacking an actual-species training threshold. These are distinct diagnostic views; neither is a verified migration state."]
    dump(out / "report.json", result)
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--routing-rules", help="Frozen rules.json (or its directory); never refits router thresholds")
    parser.add_argument("--routing-rules-sha256", help="Expected frozen rules hash; defaults to verified sibling filter.json")
    print(json.dumps(report(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__": main()
