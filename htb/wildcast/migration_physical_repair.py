"""Content-bound coordinate ownership repair, independent of target outcomes.

An exclusion group is a protection against leakage, not evidence that its
members share one body.  This module never infers a physical link from a count
of coincident locations, a compressed alias star, or a model error.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import migration_data as legacy


SCHEMA = "wildcast-parent-physical-retraction-v1"
STATUSES = {"retained_unaffected", "reassigned_repaired_owner", "retired"}
RETIREMENT_REASONS = {
    "ambiguous_origin_ownership", "missing_origin_owner",
    "invalid_repaired_physical_past", "no_valid_repaired_target_support",
    "source_QC_incompatible",
}


class _Groups:
    def __init__(self, names):
        self.parent = {str(n): str(n) for n in names}

    def find(self, name):
        while self.parent[name] != name:
            self.parent[name] = self.parent[self.parent[name]]
            name = self.parent[name]
        return name

    def join(self, a, b):
        if a not in self.parent or b not in self.parent:
            raise ValueError("Repair primitive names an unknown raw identity")
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def _mapping(value, label):
    if (not isinstance(value, dict) or not value or
            any(not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in value.items())):
        raise ValueError(f"Complete nonempty {label} mapping required")
    return dict(value)


def _memberships(contract):
    return {r["parent_canonical"]: r for r in contract["parent_memberships"]}


def validate_contract(contract, parent, *, verify_inputs=True):
    """Validate a reviewed complete partition; never import inherited stars.

    ``parent`` is the strict, sealed ``migration_data.load_parent_cache`` result.
    Validation reads hashes/identity metadata, not reserved future coordinates.
    """
    if contract.get("schema") != SCHEMA:
        raise ValueError("Unsupported physical repair contract")
    for name in ["manifest", "cases", "origins"]:
        if contract.get("parent_" + name + "_sha256") != parent["hashes"][name + "_sha256"]:
            raise ValueError("Physical repair parent checksum differs")
    physical = _mapping(contract.get("physical_mapping"), "physical")
    risk = _mapping(contract.get("exclusion_mapping"), "exclusion")
    if set(physical) != set(risk):
        raise ValueError("Physical and exclusion mappings must cover identical raw identities")
    if any(c not in physical or physical[c] != c for c in physical.values()):
        raise ValueError("Physical mapping must be an idempotent complete partition")
    old_mapping = parent["manifest"].get("identity_mapping", {})
    if (set(old_mapping) | set(parent["manifest"]["assignments"])) - physical.keys():
        raise ValueError("Repair omits inherited raw identities")
    primitives = contract.get("retained_physical_primitives")
    if not isinstance(primitives, list):
        raise ValueError("Explicit reviewed physical primitives required")
    rebuilt = _Groups(physical)
    for edge in primitives:
        if not isinstance(edge, dict) or not all(edge.get(k) for k in ["a", "b", "evidence_source"]):
            raise ValueError("Physical primitive lacks explicit evidence provenance")
        if edge.get("evidence_source") in {
            "inherited_frozen_parent_identity_mapping", "flat_star", "generic_exact_position"}:
            raise ValueError("A compressed alias star is not a physical primitive")
        if edge.get("type") == "complete_observed_track_duplicate":
            # The auditor binds raw-lineage/whole-track proof.  Partial-window
            # duplicates are deliberately insufficient for whole-animal joins.
            if not edge.get("raw_lineage_sha256"):
                raise ValueError("A whole-track duplicate requires original row-lineage proof")
        rebuilt.join(edge["a"], edge["b"])
    if any(rebuilt.find(n) != physical[n] for n in physical):
        raise ValueError("Physical mapping contains unproved links or shadow alias stars")
    for edge in contract.get("retract_edges", []):
        if edge["a"] not in physical or edge["b"] not in physical:
            raise ValueError("Retracted primitive names an unknown identity")
        if physical[edge["a"]] == physical[edge["b"]]:
            raise ValueError("A retracted physical edge was reintroduced")
        if risk[edge["a"]] != risk[edge["b"]]:
            raise ValueError("Retracted physical links must retain their exclusion protection")
    by_physical = defaultdict(set)
    for n, c in physical.items():
        by_physical[c].add(risk[n])
    if any(len(v) != 1 for v in by_physical.values()):
        raise ValueError("A physical identity crosses conservative exclusion groups")
    rows = contract.get("parent_memberships", [])
    memberships = _memberships(contract)
    if (len(rows) != len(memberships) or
            set(memberships) != set(parent["manifest"]["assignments"])):
        raise ValueError("Repair must account for every parent physical membership exactly once")
    raw_parent = {}
    role_by_risk = defaultdict(set)
    for old, record in memberships.items():
        names = set(record.get("raw_ids", [])) | set(record.get("legacy_collapsed_raw_ids", []))
        expected = {n for n, c in old_mapping.items() if c == old} | {old}
        if not expected <= names or not names <= physical.keys():
            raise ValueError("Repair membership omits original or recovered raw identities")
        for n in names:
            if n in raw_parent and raw_parent[n] != old:
                raise ValueError("A recovered raw identity has conflicting parent ownership")
            raw_parent[n] = old
        if record.get("parent_role") != parent["manifest"]["assignments"][old]:
            raise ValueError("Repair changes a frozen parent assignment role")
        child = sorted({physical[n] for n in names})
        if record.get("repaired_physical_components") != child:
            raise ValueError("Parent repair membership excludes a recovered physical child")
        if set(record.get("child_exclusion_groups", [])) != {risk[n] for n in names}:
            raise ValueError("Parent repair exclusion inheritance is incomplete")
        old_fresh = record.get("parent_origin_freshness_values", [])
        actual = [parent["freshness"][old]] if old in parent["freshness"] else []
        if old_fresh != actual:
            raise ValueError("Repair changes frozen parent freshness evidence")
        for group in record["child_exclusion_groups"]:
            role_by_risk[group].add(record["parent_role"])
    if any(len(v) != 1 for v in role_by_risk.values()):
        raise ValueError("Conflicting parent roles in repaired exclusion group")
    # Every old conservative component survives even though its coordinates may
    # now belong to several bodies.  Canonical spelling is not used as evidence.
    old_components = defaultdict(set)
    for old in parent["manifest"]["assignments"]:
        old_group = parent["manifest"].get("exclusion_groups", {}).get(old, old)
        old_components[old_group].update(memberships[old]["child_exclusion_groups"])
    if any(len(v) != 1 for v in old_components.values()):
        raise ValueError("A frozen parent exclusion component was split")
    if verify_inputs:
        for record in contract.get("inputs", {}).values():
            if legacy.sha256(record["path"]) != record["sha256"]:
                raise ValueError("Physical repair evidence checksum differs")
        combined_records = [v for v in contract.get("inputs", {}).values()
                            if v.get("sha256") == contract.get("combined_identity_sha256")]
        if not combined_records:
            raise ValueError("Repair lacks the bound combined identity audit")
        combined = json.loads(Path(combined_records[0]["path"]).read_text())
        inherited_risk = combined.get("uncertainty_fold_groups", {})
        if any(risk.get(n) != group for n, group in inherited_risk.items()):
            raise ValueError("Repair changes a frozen combined exclusion ID")
    return contract


def load_contract(path, parent, *, verify_inputs=True):
    path = Path(path)
    contract = json.loads(path.read_text())
    validate_contract(contract, parent, verify_inputs=verify_inputs)
    return contract, {"kind": "parent_physical_repair", "path": str(path),
                      "sha256": legacy.sha256(path)}


def recovery_outputs(contract):
    """Return verified original-recovery outputs or refuse an incomplete gate.

    Recovered tables replace only explicitly listed already-collapsed input
    animals.  Frozen input bytes remain ancestry evidence and are never edited.
    """
    required = {(r["input_sha256"], r["input_animal"])
                for r in contract["raw_coordinate_ownership"]
                if r["status"] != "unmixed_original_source_id"}
    recovery = contract.get("original_recovery", {})
    if not required:
        return []
    if recovery.get("complete") is not True:
        raise ValueError("Original-trajectory recovery gate is incomplete")
    if recovery.get("manifest"):
        binding = recovery["manifest"]
        if legacy.sha256(binding["path"]) != binding["sha256"]:
            raise ValueError("Original recovery manifest checksum differs")
    outputs = recovery.get("outputs", [])
    covered = set()
    for record in outputs:
        if legacy.sha256(record["path"]) != record["sha256"]:
            raise ValueError("Recovered original observation checksum differs")
        if not record.get("raw_ids") or not record.get("source_QC_provenance"):
            raise ValueError("Recovered observations lack raw-ID/source-QC provenance")
        for old in record.get("replaces", []):
            key = old["input_sha256"], old["input_animal"]
            if key in covered:
                raise ValueError("Collapsed input ownership has overlapping recovery replacements")
            covered.add(key)
    if covered != required:
        raise ValueError("Original recovery does not replace exactly every collapsed input animal")
    return outputs


def canonicalize_owned(frame, contract):
    """Apply only the reviewed map to source-owned rows; no inferred joins.

    Returns the canonical table and an uncollapsed lineage table.  Exactly
    coincident source observations do not create identity edges.  Stable
    canonical/time/original-ID sorting matches assessment target tie-breaking.
    """
    required = {"timestamp", "lat", "lon", "animal", "species", "study", "source"}
    if not required <= set(frame):
        raise ValueError("Source-owned observation columns are incomplete")
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f.timestamp, utc=True).dt.as_unit("ns")
    if (f.timestamp.isna().any() or not f.lat.between(-90, 90).all() or
            not f.lon.between(-180, 180).all() or f[list(required - {"timestamp"})].isna().any().any()):
        raise ValueError("Source-owned observations contain invalid fields")
    mapping = contract["physical_mapping"]
    if set(f.animal) - mapping.keys():
        raise ValueError("Unreviewed raw identities in repaired observation input")
    f["original_animal"] = f.animal
    f["animal"] = f.animal.map(mapping)
    taxa = f[["animal", "species"]].drop_duplicates().groupby("animal").species.nunique()
    if (taxa != 1).any():
        raise ValueError("A repaired physical identity has conflicting raw species")
    lineage = f.drop_duplicates(["original_animal", "timestamp", "lat", "lon"]).copy()
    # A documented physical identity with simultaneous contradictory fixes is
    # an observation-QC issue, not an invitation to choose a favourable source.
    positions = lineage[["animal", "timestamp", "lat", "lon"]].drop_duplicates()
    if positions.duplicated(["animal", "timestamp"]).any():
        raise ValueError("Reviewed physical identity has contradictory simultaneous coordinates")
    anchor = f.sort_values(["animal", "original_animal"], kind="stable").drop_duplicates("animal").set_index("animal")
    f["study"] = f.animal.map(anchor.study)
    f = f.sort_values(["animal", "timestamp", "original_animal"], kind="stable")
    f = f.drop_duplicates(["animal", "timestamp"]).reset_index(drop=True)
    return f, lineage


def inherited_protections(parent, contract):
    """Pin roles/freshness/risk even for recovered identities absent old maps."""
    physical, risk = contract["physical_mapping"], contract["exclusion_mapping"]
    by_group = defaultdict(list)
    inherited_by_child = defaultdict(set)
    raw_parent = {}
    for record in contract["parent_memberships"]:
        old = record["parent_canonical"]
        names = set(record["raw_ids"]) | set(record.get("legacy_collapsed_raw_ids", []))
        for n in names:
            raw_parent[n] = old
            inherited_by_child[physical[n]].add(old)
        for group in {risk[n] for n in names}:
            by_group[group].append(old)
    roles, freshness = {}, {}
    excluded_raw = set(contract.get("freshness_exclusion_animals", []))
    excluded_risk = {risk[n] for n in excluded_raw if n in risk}
    for group, old_names in by_group.items():
        options = {parent["manifest"]["assignments"][n] for n in old_names}
        if len(options) != 1:
            raise ValueError("Conflicting parent roles in repaired exclusion group")
        roles[group] = options.pop()
        freshness[group] = all(parent["freshness"].get(n, False) for n in old_names) and group not in excluded_risk
    reserved = set()
    for row in parent["origins"].itertuples():
        if row.split == "test_year":
            reserved.update(_memberships(contract)[row.canonical_animal]["child_exclusion_groups"])
    return {"roles": roles, "freshness": freshness, "reserved_temporal": reserved,
            "ancestors": dict(inherited_by_child), "raw_parent": raw_parent}


def _selected_parent_sources(parent, frozen_inputs):
    """Replay the parent's source-row ordering for origin rows only.

    The old alias map is a selection index here, never a physical-history map.
    No old future target, target mask or endpoint is read.
    """
    source_order = {r["sha256"]: i for i,r in enumerate(parent["manifest"]["sources"])}
    parts = []
    wanted = parent["origins"][["case_id", "canonical_animal", "origin_time"]].copy()
    wanted["_ns"] = pd.to_datetime(wanted.origin_time, utc=True).dt.as_unit("ns").astype("int64")
    mapping = parent["manifest"]["identity_mapping"]
    seen = set()
    for binding, frame in frozen_inputs:
        digest = binding["sha256"]
        if digest not in source_order or digest in seen or legacy.sha256(binding["path"]) != digest:
            raise ValueError("Certified origin selection needs exact frozen source hashes once each")
        seen.add(digest)
        f = frame[["animal", "timestamp", "lat", "lon"]].copy()
        f["_oldcanonical"] = f.animal.map(mapping)
        f["_ns"] = pd.to_datetime(f.timestamp, utc=True).dt.as_unit("ns").astype("int64")
        f["_input_sha256"] = digest
        f["_source_order"] = source_order[digest]
        f["_source_row"] = np.arange(len(f))
        # Limit replay to original observed issue times.  Coordinates only
        # verify the selected row; they do not select its owner.
        f = f.merge(wanted[["case_id", "canonical_animal", "_ns"]],
                    left_on=["_oldcanonical", "_ns"], right_on=["canonical_animal", "_ns"],
                    how="inner", validate="many_to_one")
        parts.append(f)
    if seen != source_order.keys():
        raise ValueError("Certified origin selection omits a frozen parent input")
    f = pd.concat(parts, ignore_index=True)
    f = f.sort_values(["case_id", "animal", "_source_order", "_source_row"], kind="stable")
    return f.drop_duplicates("case_id").set_index("case_id")


def _historical_selected_rows(contract):
    binding = contract.get("historical_selected_rows")
    if not binding:
        return None, None
    if legacy.sha256(binding["path"]) != binding["sha256"]:
        raise ValueError("Historical source-row ownership table checksum differs")
    metadata_binding = binding.get("metadata")
    if not metadata_binding or legacy.sha256(metadata_binding["path"]) != metadata_binding["sha256"]:
        raise ValueError("Historical selection replay metadata checksum differs")
    metadata = json.loads(Path(metadata_binding["path"]).read_text())
    if (metadata.get("ownership_table_sha256") != binding["sha256"] or
            metadata.get("all_old_tuples_reproduced_exactly") is not True or
            not metadata.get("source_order") or not metadata.get("replay_code_sha256")):
        raise ValueError("Historical ownership needs code-bound stable input/record ordering proof")
    code = metadata.get("replay_code")
    if (not code or code.get("sha256") != metadata["replay_code_sha256"] or
            legacy.sha256(code["path"]) != code["sha256"]):
        raise ValueError("Historical replay code bytes differ from its certified selector")
    for key in ["parent_metadata", "certification"]:
        record = metadata.get(key)
        if record and legacy.sha256(record["path"]) != record["sha256"]:
            raise ValueError("Historical replay ordering/lineage evidence changed")
    table = pd.read_parquet(binding["path"])
    required = {"old_component", "timestamp", "lat", "lon", "animal", "original_row_id"}
    if (not required <= set(table) or table.original_row_id.isna().any() or
            not table.original_row_id.is_unique):
        raise ValueError("Historical selection table lacks exact row ownership")
    table["_ns"] = pd.to_datetime(table.timestamp, utc=True).dt.as_unit("ns").astype("int64")
    if table.duplicated(["old_component", "_ns"]).any():
        raise ValueError("Conflicting historical selected rows for one frozen input origin")
    table = table.set_index(["old_component", "_ns"])
    return table, {"input_sha256": metadata["old_observation_sha256"],
                   "table_sha256": binding["sha256"], "metadata_sha256": metadata_binding["sha256"]}


def resolve_parent_origins(parent, lineage, contract, *, frozen_inputs=None):
    """Resolve old origins by exact source-owned time/coordinate tuples only.

    Shared origins are ambiguous unless their selected original row is proved
    by the frozen historical selector, or every matching raw row has the same
    independently reviewed motion owner.  Future targets are never consulted;
    no nearest-position owner is guessed.
    """
    protections = inherited_protections(parent, contract)
    old = parent["origins"].copy()
    with np.load(parent["paths"]["cases"], allow_pickle=False) as a:
        origin = a["origin"]
    old["_ns"] = pd.to_datetime(old.origin_time, utc=True).dt.as_unit("ns").astype("int64")
    old["_lat"], old["_lon"] = origin[:, 0], origin[:, 1]
    loc = lineage[["original_animal", "animal", "timestamp", "lat", "lon"]].copy()
    loc["parent_canonical"] = loc.original_animal.map(protections["raw_parent"])
    loc = loc.loc[loc.parent_canonical.notna()]
    loc["_ns"] = pd.to_datetime(loc.timestamp, utc=True).dt.as_unit("ns").astype("int64")
    loc = loc.rename(columns={"lat": "_lat", "lon": "_lon"})
    loc = loc[["parent_canonical", "_ns", "_lat", "_lon", "animal", "original_animal"]].drop_duplicates()
    matched = old[["case_id", "canonical_animal", "_ns", "_lat", "_lon"]].merge(
        loc, how="left", left_on=["canonical_animal", "_ns", "_lat", "_lon"],
        right_on=["parent_canonical", "_ns", "_lat", "_lon"], validate="one_to_many")
    matches = {int(case): g for case, g in matched.groupby("case_id", sort=False)}
    selected = _selected_parent_sources(parent, frozen_inputs) if frozen_inputs is not None else None
    historical, historical_proof = _historical_selected_rows(contract) if selected is not None else (None, None)
    statuses = {(r["input_sha256"], r["input_animal"]): r["status"]
                for r in contract["raw_coordinate_ownership"]}
    physical = contract["physical_mapping"]
    records = []
    for row in old.itertuples():
        found = matches[int(row.case_id)]
        owners = sorted(set(found.animal.dropna()))
        raw_ids = sorted(set(found.original_animal.dropna()))
        reason = "missing_origin_owner" if not owners else "ambiguous_origin_ownership" if len(owners) > 1 else ""
        evidence = "exact original source-owned origin timestamp/latitude/longitude tuple; no future outcomes"
        selected_raw, selected_original_row_id = None, None
        if selected is not None:
            if row.case_id not in selected.index:
                owners = []; reason = "missing_origin_owner"
            else:
                chosen = selected.loc[row.case_id]
                # Pandas itertuples renames underscore-prefixed columns. Use
                # exact archive values for coordinate verification instead.
                exact = chosen.lat == origin[row.case_id,0] and chosen.lon == origin[row.case_id,1]
                selected_key = chosen._input_sha256, chosen.animal
                status = statuses.get(selected_key)
                if not exact or status is None:
                    owners = []; reason = "missing_origin_owner"
                elif status == "unmixed_original_source_id":
                    selected_raw = chosen.animal
                    evidence = "certified frozen parent canonical/time/original-ID stable row selection, verified against exact origin tuple; old aliases used only as selection index"
                else:
                    lookup = chosen.animal, pd.Timestamp(row.origin_time).value
                    if (historical is not None and historical_proof["input_sha256"] == chosen._input_sha256
                            and lookup in historical.index):
                        original = historical.loc[lookup]
                        if original.lat == chosen.lat and original.lon == chosen.lon:
                            selected_raw = original.animal
                            selected_original_row_id = original.original_row_id
                            evidence = "certified historical source-file/record-order selector; original-row ownership table "+historical_proof["table_sha256"]+"; coordinates verify equality only"
                    if selected_raw is None:
                        owners = []; reason = "ambiguous_origin_ownership"
                if selected_raw is not None:
                    if selected_raw not in raw_ids or selected_raw not in physical:
                        owners = []; reason = "missing_origin_owner"
                    else:
                        owners = [physical[selected_raw]]; reason = ""
        records.append({"parent_case_id": int(row.case_id), "parent_canonical_animal": row.canonical_animal,
                        "origin_time": row.origin_time, "parent_split": row.split,
                        "parent_fresh_identity": bool(row.fresh_identity),
                        "child_canonical_animal": owners[0] if len(owners) == 1 else None,
                        "ownership_reason": reason, "ownership_raw_ids": json.dumps(raw_ids),
                        "selected_raw_animal": selected_raw, "selected_original_row_id": selected_original_row_id,
                        "ownership_evidence": evidence})
    return pd.DataFrame(records)


def account_parent_cases(parent, ownership, child_origins, contract, failures=None):
    """Account every old case once, allowing certified source-QC retirement."""
    if (ownership.parent_case_id.tolist() != list(range(len(parent["origins"]))) or
            len(ownership) != len(parent["origins"])):
        raise ValueError("Parent case ownership must cover each old case exactly once")
    failures = failures or {}
    child = {(r.canonical_animal, pd.Timestamp(r.origin_time).value): r for r in child_origins.itertuples()}
    if len(child) != len(child_origins):
        raise ValueError("Duplicate repaired physical origin cases")
    memberships = _memberships(contract)
    records = []
    for record in ownership.to_dict("records"):
        old = memberships[record["parent_canonical_animal"]]
        key = record["child_canonical_animal"], pd.Timestamp(record["origin_time"]).value
        reason = record["ownership_reason"]
        result = child.get(key) if not reason else None
        if result is None and not reason:
            reason = failures.get(key)
            if reason not in RETIREMENT_REASONS:
                raise ValueError("A missing repaired parent case lacks an explicit source-QC reason")
        if reason:
            if reason not in RETIREMENT_REASONS:
                raise ValueError("Unapproved parent retirement reason")
            status = "retired"
        else:
            if result.split != record["parent_split"] or (not record["parent_fresh_identity"] and result.fresh_identity):
                raise ValueError("A repaired parent case changes role or promotes freshness")
            if result.split in {"test_individual", "test_year"} and not result.targets_sealed:
                raise ValueError("A repaired reserved parent case was unsealed")
            changed = old.get("physical_partition_changed") or old.get("requires_original_trajectory_recovery")
            status = "reassigned_repaired_owner" if changed else "retained_unaffected"
        records.append({**record, "status": status, "retirement_reason": reason or None,
                        "child_case_id": int(result.case_id) if result is not None else None,
                        "child_split": result.split if result is not None else None,
                        "child_fresh_identity": bool(result.fresh_identity) if result is not None else None})
    output = pd.DataFrame(records)
    output["child_case_id"] = output.child_case_id.astype("Int64")
    for key, path in parent["paths"].items():
        if legacy.sha256(path) != parent["hashes"][key + "_sha256"]:
            raise ValueError("Frozen parent changed during physical repair")
    return output
