"""Add reviewed goose identity retraction and eight native GPS preferences.

The base contract and source files stay immutable. This derivative makes no
forecast examples, labels, fits, error comparisons or spatial tolerances.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def partition(names, edges):
    groups = {name: name for name in names}

    def find(name):
        if name not in groups:
            raise ValueError("Unknown physical primitive identity")
        while groups[name] != name:
            groups[name] = groups[groups[name]]
            name = groups[name]
        return name

    for edge in edges:
        a, b = find(edge["a"]), find(edge["b"])
        groups[max(a, b)] = min(a, b)
    return {name: find(name) for name in groups}


def derive(base, goose, buzzard, parent_cache, output):
    base, goose, buzzard, output = map(Path, (base, goose, buzzard, output))
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads(base.read_text())
    bird = json.loads(goose.read_text())
    numeric = json.loads(buzzard.read_text())
    if not bird.get("complete") or not numeric.get("complete") or len(bird.get("retracted_edges", [])) != 1:
        raise ValueError("Reviewed complete identity/native-record evidence required")
    if numeric.get("repair", {}).get("sha256") != digest(base) or bird.get("repair", {}).get("sha256") != digest(base):
        raise ValueError("Source reviews refer to a different base partition")
    if (numeric["summary"]["coordinate_values_modified"] != 0
            or numeric["summary"]["conflicting_nominal_timestamps"] != 8
            or numeric["frozen_parent"]["affected_parent_animals"]):
        raise ValueError("Native source preference needs explicit unchanged-coordinate/new-source-only evidence")
    if partition(previous["physical_mapping"], previous["retained_physical_primitives"]) != previous["physical_mapping"]:
        raise ValueError("Base physical partition is not exactly its retained primitive closure")
    contract = copy.deepcopy(previous)
    edge = bird["retracted_edges"][0]
    pair = frozenset([edge["a"], edge["b"]])
    if len(pair) != 2 or any(name not in contract["physical_mapping"] for name in pair):
        raise ValueError("Unknown reviewed goose raw identities")
    retained, removed = [], []
    for primitive in contract["retained_physical_primitives"]:
        (removed if frozenset([primitive["a"], primitive["b"]]) == pair else retained).append(primitive)
    if len(removed) != 2:
        raise ValueError("Both documented copies of the reviewed goose primitive must be removed")
    if any(frozenset([row["a"], row["b"]]) == pair for row in contract["retract_edges"]):
        raise ValueError("Goose edge is already retracted")
    contract["retained_physical_primitives"] = retained
    physical = partition(previous["physical_mapping"], retained)
    changed = {name for name in physical if physical[name] != previous["physical_mapping"][name]}
    if changed != {max(pair)} or any(physical[name] != name for name in pair):
        raise ValueError("The reviewed goose split would alter another physical identity")
    risk = contract["exclusion_mapping"]
    if len({risk[name] for name in pair}) != 1:
        raise ValueError("The goose uncertainty/exclusion union was not preserved")
    contract["physical_mapping"] = physical
    contract["retract_edges"].append({**edge, "evidence_source": "verified_original_GPS_and_conflicting_reference",
        "review": {"path": str(goose), "sha256": digest(goose)}})
    changed_ownership = []
    for row in contract["raw_coordinate_ownership"]:
        expected = sorted({physical[name] for name in row["original_raw_ids"]})
        if expected != row["repaired_physical_ids"]:
            changed_ownership.append({"input_sha256": row["input_sha256"], "input_animal": row["input_animal"],
                "before": row["repaired_physical_ids"], "after": expected})
            row["repaired_physical_ids"] = expected
    # The goose pair and native buzzard events were absent the frozen parent.
    # Abort rather than guessing ancestry if this ever stops being true.
    for row in contract["parent_memberships"]:
        names = set(row["raw_ids"]) | set(row.get("legacy_collapsed_raw_ids", []))
        expected = sorted({physical[name] for name in names})
        if expected != row["repaired_physical_components"]:
            raise ValueError("New source review unexpectedly changes a frozen parent membership")
    preferences = numeric["proposed_source_record_exclusions"]
    if len(preferences) != 1 or len(preferences[0]["row_positions"]) != 8:
        raise ValueError("Exactly eight reviewed sparse processed records must be excluded")
    selector = {**copy.deepcopy(preferences[0]), "evidence_sha256": digest(buzzard)}
    if selector["input_sha256"] not in {row["input_sha256"] for row in contract["raw_coordinate_ownership"]}:
        raise ValueError("Native record selector input is absent the base ownership census")
    contract.setdefault("source_record_exclusions", []).append(selector)
    contract.setdefault("global_raw_source_record_policies", []).append({
        "kind": "content_bound_preferred_native_fix_records", "source_id": "a1d2d3ab-adcd-4101-8775-e0559445b612",
        "original_source_sha256": numeric["record_preferences"][0]["original_source_sha256"],
        "original_record_number_basis": "one_based_logical_CSV_data_record_header_excluded",
        "preferred_original_records": sorted(row["preferred_original_record"] for row in numeric["record_preferences"]),
        "excluded_original_records": sorted(row["suppressed_original_record"] for row in numeric["record_preferences"]),
        "evidence_sha256": digest(buzzard), "reason": selector["reason"],
        "scope": "Documentary original-event lineage; frozen processed-table execution uses its exact source_record_exclusions. A different input byte/order requires fresh positional certification. No generic rounding or proximity rule."})
    contract["inputs"]["goose_conflicting_reference_review"] = {"path": str(goose), "sha256": digest(goose)}
    contract["inputs"]["buzzard_native_record_review"] = {"path": str(buzzard), "sha256": digest(buzzard)}
    contract["inputs"]["reviewed_source_base_contract"] = {"path": str(base), "sha256": digest(base)}
    summary = contract["summary"]
    summary["physical_components"] = len(set(physical.values()))
    summary["retracted_physical_edges"] = len(contract["retract_edges"])
    summary["source_record_excluded_rows"] = sum(len(row["row_positions"]) for row in contract["source_record_exclusions"])
    contract["reviewed_source_policy_derivation"] = {
        "base": {"path": str(base), "sha256": digest(base)},
        "code": {"path": str(Path(__file__)), "sha256": digest(__file__)},
        "removed_duplicate_goose_physical_primitives": removed,
        "changed_coordinate_ownership_records": changed_ownership,
        "new_source_record_excluded_rows": 8, "preferred_native_coordinate_modifications": 0,
        "parent_memberships_unchanged": True, "exclusion_mapping_unchanged": True,
        "parent_roles_freshness_and_retirements_unchanged": True,
        "scope": "Reviewed biological-alias retraction and exact original-native-record preference only; no training, model selection, future-target/error filtering or final release."}
    for key in ["exclusion_mapping", "parent_memberships", "affected_parent_memberships", "freshness_exclusion_animals",
                "parent_role_conflicts", "parent_freshness_conflicts", "parent_cases_sha256", "parent_manifest_sha256", "parent_origins_sha256"]:
        if contract[key] != previous[key]:
            raise ValueError("Source review altered an inherited protection: " + key)
    from wildcast import migration_physical_repair as repair_module
    from wildcast import multiscale_data
    parent = multiscale_data._load_parent(parent_cache, 2020, 2021)
    repair_module.validate_contract(contract, parent)
    repair_module.recovery_outputs(contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "sha256": digest(output), "summary": summary,
        "new_native_record_selector": selector, "physical_mapping_changes": sorted(changed),
        "changed_coordinate_ownership_records": changed_ownership}, indent=2))
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="reports/migration-parent-physical-retraction-v3.json")
    parser.add_argument("--goose", default="reports/migration-goose-ring-identity-review-v1.json")
    parser.add_argument("--buzzard", default="reports/migration-buzzard-native-record-review-v1.json")
    parser.add_argument("--parent-cache", default="data/migration-expanded-dev-v3")
    parser.add_argument("--output", required=True)
    derive(**vars(parser.parse_args()))
