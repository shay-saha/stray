"""Paired uncertainty for exclusion blocks shared across source taxon labels.

An exclusion block can contain possible aliases with different raw labels.  A
within-species resampling scheme would give that same block different weights in
different strata.  Here a single positive Exp(1) weight follows each global block
through all its observed labels, horizons and origins within a split.

These are conditional Bayesian-bootstrap uncertainty intervals.  They are not
frequentist coverage claims, and do not establish uncertainty for unseen species,
regions, selected models, or a population represented by only one observed block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["split", "species", "individual", "horizon_days", "case"]
METHOD = "paired_global_exclusion_cluster_bayesian_bootstrap_exp1"


def _strata(animal, models, reference):
    """Pack fixed observed strata; origin means have already been computed."""
    table = animal.reset_index()
    identity, identities = pd.factorize(table["individual"], sort=False)
    stratum_keys = pd.MultiIndex.from_frame(table[["species", "horizon_days"]])
    stratum, strata = pd.factorize(stratum_keys, sort=False)
    # factorize() drops MultiIndex level names in supported pandas versions.
    horizon, horizons = pd.factorize(strata.get_level_values(1), sort=False)
    species_per_horizon = np.bincount(horizon, minlength=len(horizons))
    coefficients = 1. / (len(horizons) * species_per_horizon[horizon])
    values = table[models].to_numpy(dtype=float)
    differences = values - values[:, [models.index(reference)]]
    return identity, len(identities), stratum, len(strata), coefficients, differences


def paired_global_cluster_bootstrap(frame, reference, seed=2026, replicates=300):
    """Return equal-identity/species/horizon paired delta uncertainty in km.

    Estimand: average origins within each observed (species, exclusion identity,
    horizon), equal identities within each (species,horizon), equal observed
    species at a horizon, then equal horizons.  Model/reference support must be
    identical; missing support is never silently intersected or imputed.

    Independent positive global-cluster weights are normalized separately within
    each observed species/horizon stratum.  Their shared identity across strata
    preserves dependency.  Positive weights retain singleton and sparse strata.
    """
    if not isinstance(replicates, (int, np.integer)) or isinstance(replicates, (bool, np.bool_)) or replicates <= 0:
        raise ValueError("A positive integer number of Bayesian-bootstrap replicates is required")
    required = ["model", *KEYS, "endpoint_energy_km"]
    if not set(required) <= set(frame.columns) or not len(frame):
        raise ValueError("Nonempty per-case endpoint scores with all identity/stratum keys are required")
    if frame[required].isna().any().any():
        raise ValueError("Paired comparisons require complete keys and finite identical model support")
    try:
        scores = frame["endpoint_energy_km"].to_numpy(dtype=float)
        horizons = frame["horizon_days"].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Scores and horizon strata must be finite numeric values") from error
    if not np.isfinite(scores).all() or not np.isfinite(horizons).all():
        raise ValueError("Scores and horizon strata must be finite")
    if frame.duplicated(["model", *KEYS]).any():
        raise ValueError("Duplicate scored migration model/case/horizon")
    pivot = frame.pivot(index=KEYS, columns="model", values="endpoint_energy_km")
    if reference not in pivot or pivot.isna().any().any():
        raise ValueError("Paired comparisons require all models on identical origins and target masks")
    models = list(pivot.columns)
    rng = np.random.default_rng(seed)
    splits = {}
    for split, group in pivot.reset_index().groupby("split", observed=True, sort=True):
        animal = group.groupby(["species", "individual", "horizon_days"], observed=True)[models].mean()
        if not np.isfinite(animal.to_numpy(dtype=float)).all():
            raise ValueError("Origin aggregation produced nonfinite scores")
        identity, n_identity, stratum, n_strata, coefficients, differences = _strata(animal, models, reference)
        draws = np.empty((replicates, len(models)), dtype=float)
        for draw in range(replicates):
            # The mathematical Exp law is strictly positive. Protect its
            # measure-zero floating-point zero representation without deleting
            # a singleton stratum or resampling any row/label independently.
            weights = np.maximum(rng.exponential(1., size=n_identity), np.finfo(float).tiny)
            row_weights = weights[identity]
            denominator = np.bincount(stratum, weights=row_weights, minlength=n_strata)
            for model in range(len(models)):
                numerator = np.bincount(stratum, weights=row_weights * differences[:, model], minlength=n_strata)
                draws[draw, model] = coefficients @ (numerator / denominator)
        # Identical expression/order to the existing macro delta: changing the
        # uncertainty law must not silently change the reported point estimand.
        observed = animal.groupby(level=["species", "horizon_days"]).mean().groupby(level="horizon_days").mean().mean()
        splits[str(split)] = {model: {"delta_km": float(observed[model] - observed[reference]),
                                     "ci95_km": np.quantile(draws[:, i], [.025, .975]).tolist()}
                              for i, model in enumerate(models)}
    return {"reference": reference, "replicates": int(replicates),
            "unit": "global conservative exclusion identity within split; coupled across labels, horizons and origins",
            "splits": splits, "method": METHOD,
            "interval_type": "95% Bayesian bootstrap uncertainty interval conditional on observed exclusion blocks and strata",
            "limitations": ["No frequentist coverage claim or adjustment for model selection",
                            "Singleton strata have no within-stratum block variance; sparse strata remain present",
                            "Does not measure unseen-species or geographic transfer uncertainty"]}
