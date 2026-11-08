import numpy as np
import pandas as pd
import pytest

from wildcast.migration_uncertainty import paired_global_cluster_bootstrap


def scores(records):
    rows = []
    for case, (species, identity, horizon, delta) in enumerate(records):
        for model, value in [("baseline", 0.), ("candidate", float(delta))]:
            rows.append({"model": model, "split": "val_individual", "species": species,
                         "individual": identity, "horizon_days": horizon, "case": case,
                         "endpoint_energy_km": value})
    return pd.DataFrame(rows)


def cancellation():
    # The shared global weights make species A and B cancel at each horizon.
    # Species C exists only at horizon90 and must never disappear from a draw.
    return scores([("A", 1, 1, 10), ("A", 2, 1, -10),
                   ("B", 1, 1, -10), ("B", 2, 1, 10),
                   ("A", 1, 3, 6), ("B", 1, 3, -6),
                   ("C", 3, 90, 100)])


def test_global_weights_couple_cross_label_blocks_and_keep_singleton_horizons():
    result = paired_global_cluster_bootstrap(cancellation(), "baseline", seed=1, replicates=400)
    value = result["splits"]["val_individual"]["candidate"]
    assert value["delta_km"] == 100 / 3
    np.testing.assert_allclose(value["ci95_km"], [100/3, 100/3], atol=1e-12, rtol=0)
    assert result["splits"]["val_individual"]["baseline"] == {"delta_km": 0., "ci95_km": [0., 0.]}
    assert "Bayesian bootstrap uncertainty" in result["interval_type"]
    assert "exclusion" in result["unit"]


def test_observed_delta_exactly_preserves_existing_equal_identity_species_horizon_estimand():
    from wildcast.migration_evaluate import macro_scores
    frame = scores([("A", 1, 1, 0), ("A", 1, 1, 2), ("A", 2, 1, 3),
                    ("B", 1, 1, 5), ("B", 3, 1, 7), ("A", 1, 10, 20)])
    overall = macro_scores(frame)[0].set_index("model")["endpoint_energy_km"]
    result = paired_global_cluster_bootstrap(frame, "baseline", replicates=100)
    expected = float(overall["candidate"] - overall["baseline"])
    actual = result["splits"]["val_individual"]["candidate"]
    assert actual["delta_km"] == expected == 12.
    assert np.isfinite(actual["ci95_km"]).all()


def test_origin_duplication_and_row_order_do_not_change_cluster_uncertainty():
    frame = scores([("A", 1, 1, 1), ("A", 2, 1, 7), ("B", 1, 1, -3), ("B", 3, 1, 4)])
    expanded = pd.concat([frame.assign(case=frame.case + 10*i) for i in range(50)], ignore_index=True)
    first = paired_global_cluster_bootstrap(frame, "baseline", seed=17, replicates=200)
    duplicated = paired_global_cluster_bootstrap(expanded, "baseline", seed=17, replicates=200)
    reordered = paired_global_cluster_bootstrap(expanded.iloc[::-1], "baseline", seed=17, replicates=200)
    assert first["splits"] == duplicated["splits"] == reordered["splits"]


@pytest.mark.parametrize("change", ["missing_model_row", "changed_identity", "duplicate", "nan", "infinite"])
def test_all_models_require_identical_finite_case_support(change):
    frame = cancellation()
    if change == "missing_model_row": frame = frame.iloc[1:].copy()
    if change == "changed_identity": frame.loc[0, "individual"] = 999
    if change == "duplicate": frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    if change == "nan": frame.loc[0, "endpoint_energy_km"] = np.nan
    if change == "infinite": frame.loc[0, "endpoint_energy_km"] = np.inf
    with pytest.raises(ValueError):
        paired_global_cluster_bootstrap(frame, "baseline", replicates=10)


def test_split_scope_and_seed_are_explicit_and_deterministic():
    frame = cancellation()
    year = frame.assign(split="val_year", endpoint_energy_km=frame.endpoint_energy_km*2)
    combined = pd.concat([frame, year], ignore_index=True)
    first = paired_global_cluster_bootstrap(combined, "baseline", seed=41, replicates=50)
    second = paired_global_cluster_bootstrap(combined, "baseline", seed=41, replicates=50)
    assert first == second
    assert set(first["splits"]) == {"val_individual", "val_year"}
    assert first["splits"]["val_year"]["candidate"]["delta_km"] == 200/3
    with pytest.raises(ValueError, match="positive integer"):
        paired_global_cluster_bootstrap(combined, "baseline", replicates=0)
