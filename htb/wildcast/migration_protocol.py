"""Shared protocol validation for raw-history inference and sealed assessment."""
from __future__ import annotations

import importlib
from pathlib import Path
import re

import numpy as np

from . import migration_data

MULTISCALE_MODULES = {"wildcast-multiscale-observed-v1": ".multiscale_data",
                      "wildcast-multiscale-repaired-v2": ".multiscale_repaired"}


def protocol_code_files(manifest):
    """Local source closure for feature/assessment protocol interpretation."""
    protocol = manifest.get("protocol", "wildcast-migration-v1")
    names = ["migration_protocol.py", "migration_data.py", "data.py"]
    if protocol in MULTISCALE_MODULES:
        names.extend(["multiscale_data.py", "migration_clock.py"])
        if protocol == "wildcast-multiscale-repaired-v2":
            names.extend(["multiscale_repaired.py", "migration_physical_repair.py", "migration_repaired_features.py"])
    elif protocol != "wildcast-migration-v1":
        raise ValueError("Unsupported migration cache protocol/scope: " + str(protocol))
    return [Path(__file__).with_name(name) for name in names]


def _repaired_declarations(manifest):
    parent, proof = manifest.get("parent_cache"), manifest.get("physical_repair")
    sha = lambda value: isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None
    if (not isinstance(parent,dict) or parent.get("protocol") != "wildcast-parent-physical-repair-accounted-v1"
            or not isinstance(proof,dict) or not sha(proof.get("sha256"))
            or parent.get("physical_repair") != proof or not sha(parent.get("case_accounting_sha256"))):
        raise ValueError("Repaired protocol requires its physical-repair and parent-accounting declarations")
    count, statuses = parent.get("accounted_parent_cases"), parent.get("case_status_counts")
    if (type(count) is not int or count <= 0 or not isinstance(statuses,dict)
            or set(statuses)-{"retained_unaffected","reassigned_repaired_owner","retired"}
            or any(type(value) is not int or value < 0 for value in statuses.values())
            or sum(statuses.values()) != count):
        raise ValueError("Repaired protocol must account for every parent case exactly once")
    sources, rules = manifest.get("sources"), manifest.get("observation_source_rules")
    if not isinstance(sources,list) or not sources or not isinstance(rules,list):
        raise ValueError("Repaired observations require content-bound source ownership selectors")
    if not all(isinstance(row,dict) for row in sources):
        raise ValueError("Invalid repaired observation source declaration")
    source_hashes = [row.get("sha256") for row in sources]
    if not all(sha(value) for value in source_hashes) or len(set(source_hashes)) != len(source_hashes):
        raise ValueError("Repaired source checksum declarations are incomplete or duplicate")
    rule_hashes = []
    for rule in rules:
        if not isinstance(rule,dict):raise ValueError("Invalid repaired observation ownership rule")
        allowed,excluded = rule.get("allowed_raw_animals"),rule.get("excluded_raw_animals")
        if (not isinstance(allowed,list) or not isinstance(excluded,list)
                or not all(isinstance(name,str) for name in [*allowed,*excluded])
                or len(set(allowed)) != len(allowed) or len(set(excluded)) != len(excluded)
                or set(allowed)&set(excluded) or not isinstance(rule.get("excluded_intervals",[]),list)
                or not isinstance(rule.get("excluded_native_times",[]),list)
                or not isinstance(rule.get("excluded_input_records",[]),list)):
            raise ValueError("Repaired source selectors require disjoint explicit raw identities")
        for group in rule.get("excluded_native_times",[]):
            if not isinstance(group,dict):raise ValueError("Invalid exact-native-time source-QC declaration")
            times=group.get("timestamps_ns")
            if (group.get("input_animal") not in set(allowed)|set(excluded)
                    or group.get("time_axis") != "reported_nominal"
                    or not isinstance(times,list) or not times
                    or not all(type(value) is int and -(2**63) < value < 2**63 for value in times)
                    or times != sorted(set(times)) or not sha(group.get("evidence_sha256"))
                    or not isinstance(group.get("reason"),str) or not group["reason"]):
                raise ValueError("Exact-native-time source-QC keys must be sorted, unique and evidence-bound")
        for group in rule.get("excluded_input_records",[]):
            if not isinstance(group,dict):raise ValueError("Invalid original-record source-QC declaration")
            positions,total=group.get("row_positions"),group.get("input_rows")
            if (type(total) is not int or total <= 0 or not isinstance(positions,list) or not positions
                    or not all(type(value) is int and 0 <= value < total for value in positions)
                    or positions != sorted(set(positions))
                    or group.get("row_number_basis") != "zero_based_full_input_order"
                    or not sha(group.get("evidence_sha256"))
                    or not isinstance(group.get("reason"),str) or not group["reason"]):
                raise ValueError("Original-record source-QC keys require full-input positional provenance")
        rule_hashes.append(rule.get("sha256"))
    if (not all(sha(value) for value in rule_hashes) or len(rule_hashes) != len(set(rule_hashes))
            or set(rule_hashes) != set(source_hashes)):
        raise ValueError("Every repaired source must have exactly one frozen ownership selector")


def cross_taxon_exclusion_ids(manifest):
    """Authorize only recorded uncertainty blocks spanning retained taxon labels.

    Canonical physical histories stay separate. This is never authorization to
    merge coordinates or reinterpret conflicting physical species assignments.
    """
    records = manifest.get("cross_taxon_exclusion_groups", [])
    if not isinstance(records, list):
        raise ValueError("Cross-taxon exclusion declarations must be a list")
    if records and manifest.get("protocol") not in MULTISCALE_MODULES:
        raise ValueError("Cross-taxon exclusions require the explicit multiscale protocol")
    allowed = set()
    animals = manifest.get("animals", {})
    ambiguous = set(manifest.get("ambiguous_identity_records", []))
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid cross-taxon uncertainty block")
        identity, names = record.get("individual"), record.get("canonical_animals")
        taxa, mapping = record.get("raw_species"), record.get("canonical_taxa")
        if (type(identity) is not int or identity < 0 or identity in allowed
                or record.get("kind") != "uncertain_identity_block"
                or not isinstance(names, list) or len(names) < 2
                or not all(isinstance(name, str) for name in names) or len(set(names)) != len(names)
                or set(names) != {name for name, value in animals.items() if value == identity}
                or not set(names) <= ambiguous
                or not isinstance(taxa, list) or len(taxa) < 2
                or not all(isinstance(taxon, str) for taxon in taxa) or len(set(taxa)) != len(taxa)
                or not isinstance(mapping, dict) or set(mapping) != set(names)
                or not all(isinstance(taxon, str) for taxon in mapping.values())
                or set(mapping.values()) != set(taxa)):
            raise ValueError("Unverified cross-taxon uncertainty block; physical conflicts remain forbidden")
        allowed.add(identity)
    return frozenset(allowed)


def schema_for(manifest, arrays=None, *, legacy_default=False):
    protocol = manifest.get("protocol", "wildcast-migration-v1" if legacy_default else None)
    if protocol == "wildcast-migration-v1":
        return migration_data
    if protocol not in MULTISCALE_MODULES:
        raise ValueError("Unsupported migration cache protocol/scope: " + str(protocol))
    multiscale_data = importlib.import_module(MULTISCALE_MODULES[protocol], package="wildcast")
    if multiscale_data.PROTOCOL != protocol:
        raise ValueError("Loaded feature module has a different protocol")
    required = {
        "horizons_days": multiscale_data.HORIZONS.tolist(),
        "assessment_target_matcher": multiscale_data.__name__+".match_targets",
        "history_names": multiscale_data.HISTORY_NAMES,
        "static_names": multiscale_data.STATIC_NAMES,
        "history_lags_days": multiscale_data.LAGS.tolist(),
        "unit_metres": multiscale_data.UNIT,
        "minimum_history_lags": multiscale_data.MINIMUM_HISTORY_LAGS,
        "minimum_history_prior_fixes": multiscale_data.MINIMUM_PRIOR_FIXES,
        "minimum_history_span_days": multiscale_data.MINIMUM_HISTORY_SPAN_DAYS,
        "minimum_observed_future_targets": 1,
    }
    for key, expected in required.items():
        value = manifest.get(key)
        if value != expected or (isinstance(value, bool) and isinstance(expected, (int, float))):
            raise ValueError("Multiscale protocol requires its exact " + key)
    train_year, validation_year = manifest.get("train_end_year"), manifest.get("validation_year")
    if (type(train_year) is not int or type(validation_year) is not int
            or validation_year != train_year + 1
            or manifest.get("cutoff_boundaries") != [f"{train_year+1}-01-01T00:00:00Z",
                                                      f"{validation_year+1}-01-01T00:00:00Z"]):
        raise ValueError("Multiscale protocol requires its pinned calendar cutoffs")
    excluded, modes, animals = (manifest.get(key) for key in
                                ("subday_clock_excluded_animals", "subday_clock_modes", "animals"))
    if (not isinstance(excluded, list) or not all(isinstance(name, str) for name in excluded)
            or not isinstance(modes, dict) or not isinstance(animals, dict)
            or any(mode not in ("bounded_utc_or_gps", "declared_utc") for mode in modes.values())
            or len(set(excluded)) != len(excluded) or set(excluded) & modes.keys()
            or set(excluded) | modes.keys() != animals.keys()):
        raise ValueError("Multiscale protocol requires complete canonical subday clock admission declarations")
    allowed = cross_taxon_exclusion_ids(manifest)
    if protocol == "wildcast-multiscale-repaired-v2":_repaired_declarations(manifest)
    if arrays is not None:
        horizons = np.asarray(arrays.get("horizons"))
        if horizons.dtype != np.dtype("float64") or not np.array_equal(horizons, multiscale_data.HORIZONS):
            raise ValueError("Multiscale cases require the exact canonical float64 horizon grid")
        count = len(arrays["split"])
        for key, shape in {"history": (count, len(multiscale_data.LAGS), len(multiscale_data.HISTORY_NAMES)),
                           "history_mask": (count, len(multiscale_data.LAGS)),
                           "static": (count, len(multiscale_data.STATIC_NAMES))}.items():
            if key not in arrays or arrays[key].shape != shape:
                raise ValueError("Multiscale cases have invalid " + key + " dimensions")
        if "individual" in arrays and "species" in arrays:
            labels = arrays.get("metric_species", arrays["species"])
            pairs = np.unique(np.column_stack([arrays["individual"], labels]), axis=0)
            identities, counts = np.unique(pairs[:, 0], return_counts=True)
            if set(identities[counts > 1]) - allowed:
                raise ValueError("Undeclared cross-taxon exclusion IDs in multiscale cases")
            mapping = manifest.get("metric_species", manifest.get("species", {}))
            for record in manifest.get("cross_taxon_exclusion_groups", []):
                permitted = {mapping[name] for name in record["raw_species"] if name in mapping}
                actual = set(labels[arrays["individual"] == record["individual"]])
                if not actual <= permitted:
                    raise ValueError("Case taxon labels disagree with the declared uncertainty block")
    return multiscale_data
