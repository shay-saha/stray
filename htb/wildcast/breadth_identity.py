"""Immutable metadata and repeated-GPS identity extension for a broad GPS core.

Physical identity, taxonomic equivalence and possible exclusion links have
different meanings.  Synonymous names alone never link animals.  Hardware/local
IDs alone never prove a cross-source physical identity.  Parent role conflicts
are reported, never resolved by silently changing the old assessment cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .marine_intake import sha256
from .migration_identity import audit as legacy_audit
from .migration_identity import manufacturer_key, norm, qualified_ring, read_reference, ring_key, serial_key, usable, scoped_animals


class Groups:
    def __init__(self, names=()):
        self.parent = {v: v for v in names}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)

    def mapping(self):
        return {v: self.find(v) for v in sorted(self.parent)}


def load_taxonomy(path=None):
    info = json.loads(Path(path).read_text()) if path else {"aliases": [], "unresolved": []}
    aliases = {}
    for row in info["aliases"]:
        if row.get("relationship") not in {"synonym", "genus_change", "included_synonym"} or not row.get("source_url"):
            raise ValueError("Taxon equivalence needs an explicit supported relationship and authority")
        old, new = norm(row["from"]), norm(row["to"])
        if old in aliases and aliases[old] != new:
            raise ValueError("Conflicting taxonomic equivalence")
        aliases[old] = new
        if row.get("source_path") and sha256(row["source_path"]) != row["source_sha256"]:
            raise ValueError("Taxonomic source snapshot hash mismatch")
    def canonical(value):
        value = norm(value)
        # Collapse explicitly named subspecies for species counting, retaining
        # original strings elsewhere. Unknown/hybrid labels remain unresolved.
        if re.fullmatch(r"[A-Z]+ [A-Z-]+ [A-Z-]+", value):
            value = " ".join(value.split()[:2])
        seen = set()
        while value in aliases:
            if value in seen:
                raise ValueError("Cyclic taxonomic equivalence")
            seen.add(value); value = aliases[value]
        return value
    for value in aliases:
        canonical(value)
    return canonical, info


def _inventory(paths):
    names, taxa, evidence = set(), defaultdict(set), []
    for path in paths:
        path = Path(path); q = pq.ParquetFile(path)
        for batch in q.iter_batches(batch_size=200_000, columns=["animal", "species"]):
            frame = batch.to_pandas().drop_duplicates()
            for name, taxon in frame.itertuples(index=False, name=None):
                names.add(name); taxa[name].add(taxon)
        checksum = sha256(path)
        descriptor = {"path": str(path), "sha256": checksum, "rows": q.metadata.num_rows}
        manifest = path.parent / "manifest.json"
        if manifest.exists():
            info = json.loads(manifest.read_text())
            if info.get("observations_sha256") != checksum:
                raise ValueError("Observation table differs from its declared source/clock manifest")
            descriptor.update(manifest=str(manifest), manifest_sha256=sha256(manifest), manifest_schema=info.get("schema"))
        evidence.append(descriptor)
    return names, taxa, evidence


def _expanded_records(reference):
    original = {(v["reference"], v["line"]): v for v in reference["reference_records"] if v["record_type"] == "reference_table"}
    records = []
    for descriptor in reference["references_checked"]:
        path = Path(descriptor["path"])
        if sha256(path) != descriptor["sha256"]:
            raise ValueError("Public reference changed after hash verification")
        rows, _ = read_reference(path)
        for line, row in enumerate(rows, 2):
            base = original.get((str(path), line))
            if not base or not base["observation_animals"]:
                continue
            flippers = []
            for match in re.finditer(r"\b(?:LHS|RHS|left|right)\s+(?:flipper\s+)?(?:tag\s+)?([A-Z]{2,6}\s*\d{3,8})\b", row.get("animal-comments", ""), re.I):
                flippers.append(re.sub(r"\s+", "", match.group(1)).upper())
            records.append({**base, "explicit_flipper_markers": flippers})
    # Preserve explicit qualified-ring evidence printed in event identifiers.
    records.extend(v for v in reference["reference_records"] if v["record_type"] != "reference_table")
    return records


def derived_reference_records(manifests, observed_names):
    """Admit explicitly derived ID tables with pinned original/code lineage.

    These remain identified as derived tables, rather than pretending they were
    independent publisher bitstreams. They cannot create identity evidence from
    fields absent from the original namespace and documented derivation.
    """
    records, evidence, seen_references = [], [], set()
    for manifest_path in manifests:
        manifest_path = Path(manifest_path)
        info = json.loads(manifest_path.read_text())
        for inventory in info.get("source_inventory", []):
            declaration = inventory["declaration"]
            source_id = declaration["spec"]["source_id"]
            directory = Path(declaration["path"]).parent
            source_path = directory / "source.json"
            source = json.loads(source_path.read_text())
            snapshot = source.get("normalizer_code_snapshot")
            if not snapshot:
                continue
            code_path = directory / snapshot["path"]
            if sha256(code_path) != snapshot["sha256"] or source.get("normalizer_sha256") != snapshot["sha256"]:
                raise ValueError("Derived identity reference normalizer evidence mismatch")
            for reference in inventory.get("references", []):
                path = Path(reference["path"])
                key = (source_id, str(path), reference["sha256"])
                if key in seen_references:
                    continue
                seen_references.add(key)
                if sha256(path) != reference["sha256"]:
                    raise ValueError("Derived reference hash mismatch")
                rows, encoding = read_reference(path)
                evidence.append({"path": str(path), "sha256": reference["sha256"], "rows": len(rows), "encoding": encoding,
                                 "source_id": source_id, "QC_manifest": str(manifest_path), "QC_manifest_sha256": sha256(manifest_path),
                                 "source_manifest": str(source_path), "source_manifest_sha256": sha256(source_path),
                                 "normalizer_snapshot": str(code_path), "normalizer_sha256": snapshot["sha256"],
                                 "kind": "source-derived_identity_reference_not_publisher_bitstream"})
                for line, row in enumerate(rows, 2):
                    local, taxon = row.get("animal-id", ""), row.get("animal-taxon", "")
                    if not usable(local) or not usable(taxon):
                        continue
                    delimiter = ":" + taxon + ":"
                    candidates = [name for name in observed_names if name.startswith(source_id + ":") and delimiter in name
                                  and name.rsplit(delimiter, 1)[-1].strip() == local.strip()]
                    bound = scoped_animals(candidates, source_id, taxon, path.name)
                    records.append({"record_type": "source-derived_identity_reference", "source_id": source_id,
                                    "species": taxon, "local_identifier": local, "ring": row.get("animal-ring-id", ""),
                                    "manufacturer": row.get("tag-manufacturer-name", ""), "tag_serial": row.get("tag-serial-no", ""),
                                    "tag_id": row.get("tag-id", ""), "deployment_id": row.get("deployment-id", ""),
                                    "deploy_on": row.get("deploy-on-date", ""), "deploy_off": row.get("deploy-off-date", ""),
                                    "site": row.get("study-site", ""), "sex": row.get("animal-sex", ""), "group": row.get("animal-group-id", ""),
                                    "reference": str(path), "reference_sha256": reference["sha256"], "line": line,
                                    "observation_animals": bound, "normalizer_sha256": snapshot["sha256"]})
    return records, evidence


def metadata_links(records, canonical_taxon):
    """Return strong and uncertain links with distinct, reviewable evidence."""
    rings, hardware, tags = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in records:
        taxon = canonical_taxon(row["species"])
        markers = ([ring_key(row.get("ring", ""))] if ring_key(row.get("ring", "")) else []) + row.get("explicit_flipper_markers", [])
        for marker in markers:
            rings[marker].append(row)
        vendor = manufacturer_key(row.get("manufacturer", ""))
        if serial_key(row.get("tag_serial", "")) and usable(vendor):
            hardware[(taxon, vendor, serial_key(row["tag_serial"]))].append(row)
        if usable(row.get("tag_id", "")) and usable(vendor):
            tags[(taxon, vendor, norm(row["tag_id"]))].append(row)
    strong, ambiguous, ruled_out = [], [], []
    seen = set()
    def emit(a, b, kind, proof, physical=False):
        for left, right in itertools.product(a["observation_animals"], b["observation_animals"]):
            if left == right:
                continue
            pair = tuple(sorted((left, right)))
            key = (pair, kind)
            if key in seen:
                continue
            seen.add(key)
            record = {"a": pair[0], "b": pair[1], "type": kind,
                      "animals": list(pair), "evidence": proof,
                      "references": [{"path": v["reference"], "line": v.get("line"), "sha256": v.get("reference_sha256")} for v in [a, b]]}
            (strong if physical else ambiguous).append(record)
    def conflict(a, b):
        sex = {norm(v.get("sex", "")) for v in [a, b]} & {"M", "F"}
        return len(sex) > 1
    def facts(a, b):
        same_taxon = canonical_taxon(a["species"]) == canonical_taxon(b["species"])
        vendor = usable(a.get("manufacturer", "")) and manufacturer_key(a["manufacturer"]) == manufacturer_key(b.get("manufacturer", ""))
        start = usable(a.get("deploy_on", "")) and a["deploy_on"].strip() == b.get("deploy_on", "").strip()
        site = usable(a.get("site", "")) and len(norm(a["site"])) >= 6 and norm(a["site"]) == norm(b.get("site", ""))
        return bool(same_taxon), bool(vendor), bool(start), bool(site)
    for marker, rows in rings.items():
        for a, b in itertools.combinations(rows, 2):
            same_taxon, vendor, start, site = facts(a, b)
            if conflict(a, b):
                emit(a, b, "conflicting_sex_on_shared_explicit_ring", {"ring": marker}); continue
            # A globally qualified marker with independent corroboration may
            # establish physical identity despite a source taxon-label conflict.
            physical = (same_taxon and (site or vendor and (start or qualified_ring(marker)))) or (qualified_ring(marker) and vendor and start)
            emit(a, b, "same_explicit_ring_corroborated" if physical else "shared_ring_requires_review",
                 {"ring": marker, "same_taxon_key": same_taxon, "matching_vendor": vendor,
                  "matching_deployment_start": start, "matching_site": site}, physical)
    for key, rows in hardware.items():
        for a, b in itertools.combinations(rows, 2):
            _, _, start, _ = facts(a, b)
            ring_a, ring_b = ring_key(a.get("ring", "")), ring_key(b.get("ring", ""))
            if conflict(a, b) or (ring_a and ring_b and ring_a != ring_b):
                ruled_out.append({"type": "different_animals_on_shared_hardware", "key": list(key),
                                  "references": [[v["reference"], v.get("line")] for v in [a, b]]}); continue
            emit(a, b, "same_full_serial_vendor_taxon_deployment" if start else "possible_hardware_redeployment",
                 {"hardware_key": list(key), "matching_deployment_start": start}, start)
    for key, rows in tags.items():
        for a, b in itertools.combinations(rows, 2):
            if a["source_id"] == b["source_id"]:
                continue
            _, _, start, site = facts(a, b)
            ring_a, ring_b = ring_key(a.get("ring", "")), ring_key(b.get("ring", ""))
            if conflict(a, b) or ring_a and ring_b and ring_a != ring_b:
                continue
            # A local tag ID alone has no global meaning. Even matching start/
            # site is an exclusion candidate, not a physical identity assertion.
            if start or site:
                emit(a, b, "same_local_tag_with_corroborating_context_no_physical_merge",
                     {"key": list(key), "matching_deployment_start": start, "matching_site": site})
    return strong, ambiguous, ruled_out


def exact_links(new_path, old_paths, canonical_taxon, known_groups):
    """Five distinct exact time/GPS fixes, moving and spanning an hour; no fuzzing."""
    new_taxa = {canonical_taxon(v) for v in pq.read_table(new_path, columns=["species"]).column(0).unique().to_pylist()}
    old_frames = []
    for path in old_paths:
        q = pq.ParquetFile(path)
        timecol = "timestamp" if "timestamp" in q.schema.names else "timestamp_nominal"
        for batch in q.iter_batches(batch_size=200_000, columns=["animal", "species", timecol, "lat", "lon"]):
            frame = batch.to_pandas().rename(columns={timecol: "stamp"})
            frame["taxon_key"] = frame.species.map(canonical_taxon)
            frame = frame.loc[frame.taxon_key.isin(new_taxa), ["animal", "taxon_key", "stamp", "lat", "lon"]]
            if len(frame):
                old_frames.append(frame)
    if not old_frames:
        return []
    old = pd.concat(old_frames, ignore_index=True).drop_duplicates()
    old.stamp = pd.to_datetime(old.stamp, utc=True)
    keys, stats = ["taxon_key", "stamp", "lat", "lon"], {}
    q = pq.ParquetFile(new_path); timecol = "timestamp" if "timestamp" in q.schema.names else "timestamp_nominal"
    for batch in q.iter_batches(batch_size=100_000, columns=["animal", "species", timecol, "lat", "lon"]):
        frame = batch.to_pandas().rename(columns={timecol: "stamp"})
        frame["taxon_key"] = frame.species.map(canonical_taxon)
        frame.stamp = pd.to_datetime(frame.stamp, utc=True)
        matches = frame.merge(old, on=keys, suffixes=("_new", "_old"))
        for (a, b), group in matches.groupby(["animal_new", "animal_old"], sort=False):
            if a == b or known_groups.find(a) == known_groups.find(b):
                continue
            pair = tuple(sorted((a, b)))
            entry = stats.setdefault(pair, {"fixes": set(), "first": group.stamp.min(), "last": group.stamp.max(),
                                            "position": (group.lat.iloc[0], group.lon.iloc[0]), "moved": False,
                                            "taxon_key": group.taxon_key.iloc[0]})
            entry["first"], entry["last"] = min(entry["first"], group.stamp.min()), max(entry["last"], group.stamp.max())
            entry["moved"] |= bool(((group.lat != entry["position"][0]) | (group.lon != entry["position"][1])).any())
            if len(entry["fixes"]) < 5:
                for row in group[["stamp", "lat", "lon"]].itertuples(index=False, name=None):
                    entry["fixes"].add(row)
                    if len(entry["fixes"]) >= 5:
                        break
    links = []
    for (a, b), entry in sorted(stats.items()):
        span = (entry["last"] - entry["first"]).total_seconds() / 3600
        if len(entry["fixes"]) >= 5 and entry["moved"] and span >= 1:
            links.append({"a": a, "b": b, "type": "exact_GPS_overlap_with_supported_taxon_equivalence",
                          "shared_distinct_fixes_at_least": 5, "span_hours": span, "taxon_key": entry["taxon_key"]})
    return links


def audit(new_observations, old_observations, reference_roots, parent_cache, output, prior_audits=(), taxonomy=None, derived_reference_manifests=()):
    output = Path(output)
    if output.exists():
        raise ValueError("Identity extension output is immutable")
    canonical_taxon, taxon_info = load_taxonomy(taxonomy)
    new_paths = [Path(new_observations)] if isinstance(new_observations, (str, Path)) else [Path(p) for p in new_observations]
    if len(set(new_paths)) != len(new_paths) or set(new_paths) & set(map(Path, old_observations)):
        raise ValueError("Observation inputs must be unique and new/old roles disjoint")
    paths = [*new_paths, *map(Path, old_observations)]
    names, taxa, observation_evidence = _inventory(paths)
    new_names, _, _ = _inventory(new_paths)
    parent_cache = Path(parent_cache)
    parent_path = parent_cache / "manifest.json"; parent = json.loads(parent_path.read_text())
    if (parent.get("reserved_test_targets_sealed") is not True or
            sha256(parent_cache / "cases.npz") != parent["cases_sha256"]):
        raise ValueError("Parent cache integrity/sealing differs from frozen identity protocol")
    origins_path = parent_cache / "origins.parquet"
    if sha256(origins_path) != parent["origins_sha256"]:
        raise ValueError("Parent origin metadata hash mismatch")
    origins = pd.read_parquet(origins_path, columns=["canonical_animal", "fresh_identity"])
    fresh = {name: sorted(set(group.fresh_identity.tolist())) for name, group in origins.groupby("canonical_animal", sort=False)}
    base_path = output.with_name(output.stem + "-public-references.json")
    if base_path.exists():
        raise ValueError("Public reference extension already exists; choose a new output")
    base = legacy_audit(paths, reference_roots, base_path)
    prior = [json.loads(Path(p).read_text()) for p in prior_audits]
    physical = Groups(names)
    for info in [*prior, base]:
        for a, b in info.get("fold_groups", {}).items(): physical.join(a, b)
    for a, b in parent.get("identity_mapping", {}).items(): physical.join(a, b)
    records = _expanded_records(base)
    derived, derived_evidence = derived_reference_records(derived_reference_manifests, names)
    records.extend(derived)
    for info in prior:
        records.extend(info.get("reference_records", []))
    # Earlier canonicalization may have removed a raw alias from the supplied
    # observation files. Frozen parent record names retain that source namespace;
    # recover metadata binding only when its study/reference scope is unique.
    parent_names = sorted(parent.get("identity_mapping", {}))
    parent_by_source = defaultdict(list)
    for name in parent_names:
        parent_by_source[name.split(":", 1)[0]].append(name)
    records = [dict(row) for row in records]
    for row in records:
        candidates = []
        delimiter = ":" + row["species"] + ":"
        for name in parent_by_source.get(row["source_id"], []):
            if name.startswith(row["source_id"] + ":") and delimiter in name and name.rsplit(delimiter, 1)[-1].strip() == row["local_identifier"].strip():
                candidates.append(name)
        if candidates:
            bound = scoped_animals(candidates, row["source_id"], row["species"], Path(row["reference"]).name)
            row["observation_animals"] = sorted(set(row["observation_animals"]) | set(bound))
    unique_records = {}
    for row in records:
        key = (row["reference"], row.get("line"), row.get("reference_sha256"), row["species"], row["local_identifier"])
        if key in unique_records:
            row["observation_animals"] = sorted(set(row["observation_animals"]) | set(unique_records[key]["observation_animals"]))
        unique_records[key] = row
    records = list(unique_records.values())
    strong, ambiguous, ruled_out = metadata_links(records, canonical_taxon)
    for link in strong: physical.join(link["a"], link["b"])
    exact = []
    for i, new_path in enumerate(new_paths):
        links = exact_links(new_path, [*old_observations, *new_paths[:i]], canonical_taxon, physical)
        exact.extend(links)
        for link in links: physical.join(link["a"], link["b"])
    physical_map = physical.mapping()
    risk = Groups(physical_map)
    for a, b in physical_map.items(): risk.join(a, b)
    all_ambiguities = [*base.get("ambiguous", []), *ambiguous]
    for info in prior: all_ambiguities.extend(info.get("ambiguous", []))
    for a, b in parent.get("exclusion_groups", {}).items(): risk.join(a, b)
    for info in [*prior, base]:
        for a, b in info.get("uncertainty_fold_groups", {}).items(): risk.join(a, b)
    for record in all_ambiguities:
        values = record.get("animals", [])
        for a, b in zip(values, values[1:]): risk.join(a, b)
    risk_map = risk.mapping()
    # Emit multi-member exclusion components in the existing preparation API.
    components = defaultdict(list)
    for name, group in risk_map.items(): components[group].append(name)
    inherited_ambiguities = [{"type": "inherited_or_reviewed_exclusion_component_no_coordinate_merge", "animals": members}
                            for members in components.values() if len(members) > 1 and len({physical.find(n) for n in members}) > 1]
    all_ambiguities.extend(inherited_ambiguities)
    parent_members = defaultdict(list)
    for name, role in parent["assignments"].items():
        parent_members[risk.find(name)].append({"animal": name, "assignment": role, "freshness": fresh.get(name, [])})
    conflicts, bindings = [], []
    for group, members in parent_members.items():
        roles = {v["assignment"] for v in members}
        freshness = {value for v in members for value in v["freshness"]}
        if len(roles) > 1 or len(freshness) > 1:
            conflicts.append({"exclusion_group": group, "parent_members": members,
                              "reason": "mixed_parent_roles_or_freshness_requires_protocol_review_no_override"})
    affected = {n for row in all_ambiguities for n in row.get("animals", [])}
    for info in [*prior, base]: affected.update(info.get("freshness_exclusion_animals", []))
    for name in sorted(new_names):
        members = parent_members.get(risk.find(name), [])
        roles = {m["assignment"] for m in members}
        status = ("existing_sealed_individual_or_risk_group" if "test_individual" in roles else
                  "existing_development_identity_or_risk_group" if members else
                  "unresolved_alias_not_certified_fresh" if name in affected else "candidate_new_not_certified_globally_fresh")
        bindings.append({"animal": name, "physical_group": physical.find(name), "exclusion_group": risk.find(name),
                         "parent_members": members, "status": status})
    taxon_groups = defaultdict(set)
    for name, values in taxa.items():
        for value in values: taxon_groups[physical.find(name)].add(canonical_taxon(value))
    taxon_conflicts = [{"physical_group": group, "canonical_taxon_keys": sorted(values)}
                       for group, values in taxon_groups.items() if len(values) > 1]
    raw_keys = {canonical_taxon(value) for name in new_names for value in taxa[name]}
    headline_taxa = sorted(value for value in raw_keys if re.fullmatch(r"[A-Z]+ [A-Z-]+", value) and value.split()[1] not in {"SP", "SPP", "UNKNOWN", "UNIDENTIFIED"})
    report = {"schema": "wildcast-breadth-identity-v1", "observations": observation_evidence,
              "new_observation_inputs": [str(p) for p in new_paths],
              "old_observation_inputs": [str(p) for p in old_observations],
              "parent_manifest": {"path": str(parent_path), "sha256": sha256(parent_path)},
              "parent_origins_sha256": sha256(origins_path), "parent_cases_sha256": parent["cases_sha256"],
              "public_reference_audit": {"path": str(base_path), "sha256": sha256(base_path)},
              "prior_audits": [{"path": str(p), "sha256": sha256(p)} for p in prior_audits],
              "taxonomy": {"path": str(taxonomy), "sha256": sha256(taxonomy), "rules": taxon_info} if taxonomy else None,
              "fold_groups": physical_map, "ambiguous": all_ambiguities, "uncertainty_fold_groups": risk_map,
              "reference_records": records,
              "derived_references_checked": derived_evidence,
              "freshness_exclusion_animals": sorted(affected | {r["animal"] for r in bindings if r["parent_members"]}),
              "metadata_physical_links": strong, "exact_GPS_links": exact, "ruled_out_hardware_redeployments": ruled_out,
              "new_animal_parent_bindings": bindings, "parent_role_conflicts": conflicts,
              "taxonomic_conflict_groups": taxon_conflicts, "new_source_taxon_equivalence_keys": headline_taxa,
              "unresolved_non_binomial_taxon_keys": sorted(raw_keys - set(headline_taxa)),
              "summary": {"new_source_animal_ids": len(new_names), "new_physical_groups": len({physical.find(n) for n in new_names}),
                          "new_exclusion_groups": len({risk.find(n) for n in new_names}), "source_taxon_equivalence_keys": len(headline_taxa),
                          "additional_metadata_physical_links": len(strong), "exact_GPS_links": len(exact),
                          "new_binding_status": dict(Counter(r["status"] for r in bindings)),
                          "parent_role_conflicts": len(conflicts), "taxonomic_conflict_groups": len(taxon_conflicts)},
              "code_sha256": sha256(__file__), "scope": "Reference and repeated raw GPS identity audit only; no forecast labels, reserved outcomes, training or performance claims",
              "limitations": ["Raw taxon labels and source identifiers remain unchanged; supported nomenclature mappings do not establish animal identity",
                              "Historical species splits and ambiguous taxa are not automatically collapsed",
                              "Unmatched metadata does not certify globally fresh animals",
                              "Hardware/local IDs alone are not physical proof; unresolved context matches affect exclusion grouping only",
                              "No parent role/freshness conflict is overridden; preparation must refuse unresolved conflicts",
                              "Exact matching uses reported/validated timestamps without fuzzy clock or spatial correction"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-observations", nargs="+", required=True); p.add_argument("--old-observations", nargs="+", required=True)
    p.add_argument("--reference-roots", nargs="+", required=True); p.add_argument("--parent-cache", required=True)
    p.add_argument("--output", required=True); p.add_argument("--prior-audits", nargs="*", default=[]); p.add_argument("--taxonomy")
    p.add_argument("--derived-reference-manifests", nargs="*", default=[])
    args = p.parse_args()
    result = audit(args.new_observations, args.old_observations, args.reference_roots, args.parent_cache,
                   args.output, args.prior_audits, args.taxonomy, args.derived_reference_manifests)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
