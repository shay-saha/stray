import copy

import numpy as np
import pytest

from wildcast import migration_data, multiscale_data
from wildcast.migration_protocol import cross_taxon_exclusion_ids, schema_for


def declared():
    manifest = {"protocol": multiscale_data.PROTOCOL,
                "horizons_days": multiscale_data.HORIZONS.tolist(),
                "assessment_target_matcher": "wildcast.multiscale_data.match_targets",
                "history_names": multiscale_data.HISTORY_NAMES,
                "static_names": multiscale_data.STATIC_NAMES,
                "history_lags_days": multiscale_data.LAGS.tolist(), "unit_metres": 1e6,
                "minimum_history_lags": 2, "minimum_history_prior_fixes": 8,
                "minimum_history_span_days": .25, "minimum_observed_future_targets": 1,
                "train_end_year": 2020, "validation_year": 2021,
                "cutoff_boundaries": ["2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"],
                "animals": {"test": 0}, "subday_clock_excluded_animals": [],
                "subday_clock_modes": {"test": "declared_utc"}}
    arrays = {"horizons": multiscale_data.HORIZONS.copy(), "split": np.zeros(2, "int8"),
              "history": np.zeros((2, 27, 10), "float32"), "history_mask": np.zeros((2, 27), bool),
              "static": np.zeros((2, 18), "float32")}
    return copy.deepcopy(manifest), arrays


def test_exact_multiscale_contract_and_legacy_default():
    manifest, arrays = declared()
    assert schema_for(manifest, arrays) is multiscale_data
    assert schema_for({}, legacy_default=True) is migration_data
    with pytest.raises(ValueError, match="protocol/scope"):
        schema_for({})


@pytest.mark.parametrize("mutation,error", [
    (lambda m, a: a.update(horizons=a["horizons"].astype("float32")), "float64"),
    (lambda m, a: m.pop("history_names"), "history_names"),
    (lambda m, a: m.update(history_lags_days=m["history_lags_days"][::-1]), "history_lags_days"),
    (lambda m, a: m.update(minimum_history_prior_fixes=1), "minimum_history_prior_fixes"),
    (lambda m, a: m.update(unit_metres=1000), "unit_metres"),
    (lambda m, a: a.update(static=np.zeros((2, 14))), "static dimensions"),
    (lambda m, a: m.update(subday_clock_modes={}), "clock admission"),
    (lambda m, a: m.pop("cutoff_boundaries"), "calendar cutoffs"),
])
def test_multiscale_rejects_incompatible_inputs(mutation, error):
    manifest, arrays = declared()
    mutation(manifest, arrays)
    with pytest.raises(ValueError, match=error):
        schema_for(manifest, arrays)


def risk_block():
    manifest,arrays=declared()
    manifest.update(animals={"a":0,"b":0},subday_clock_modes={"a":"declared_utc","b":"declared_utc"},
                    ambiguous_identity_records=["a","b"],metric_species={"genus":0,"species":1})
    manifest["cross_taxon_exclusion_groups"]=[{"individual":0,"canonical_animals":["a","b"],
        "raw_species":["genus","species"],"canonical_taxa":{"a":"genus","b":"species"},
        "kind":"uncertain_identity_block"}]
    arrays.update(individual=np.zeros(2,dtype=int),species=np.zeros(2,dtype=int),metric_species=np.array([0,1]))
    return manifest,arrays


def test_explicit_uncertainty_block_is_permitted_without_relabelling_physical_histories():
    manifest,arrays=risk_block()
    assert cross_taxon_exclusion_ids(manifest) == frozenset({0})
    assert schema_for(manifest,arrays) is multiscale_data
    np.testing.assert_array_equal(arrays["individual"],[0,0])
    np.testing.assert_array_equal(arrays["metric_species"],[0,1])


@pytest.mark.parametrize("mutation,error", [
    (lambda m:m.pop("cross_taxon_exclusion_groups"),"Undeclared"),
    (lambda m:m.update(ambiguous_identity_records=["a"]),"Unverified"),
    (lambda m:m["cross_taxon_exclusion_groups"][0].update(canonical_taxa={"a":"genus","b":"other"}),"Unverified"),
])
def test_uncertainty_exemption_cannot_authorize_undocumented_taxon_conflicts(mutation,error):
    manifest,arrays=risk_block();mutation(manifest)
    with pytest.raises(ValueError,match=error):schema_for(manifest,arrays)
