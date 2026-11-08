"""Conservative identity links from public deployment/ring reference tables.

Hardware IDs alone never establish animal identity: transmitters are redeployed.
The output can be applied before data splits and contains no forecasting errors,
targets, or window definitions. Unresolved matches remain explicitly ambiguous.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .sources import dump


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or "")).strip()).upper()


def usable(value):
    return norm(value) not in {"", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN", "?", "U"}


def local_key(value):
    # Local identifiers are case sensitive, unlike catalogue/taxon labels.
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def ring_key(value):
    key = re.sub(r"\s+", "", norm(value))
    # Free descriptions such as 'spiral ring red-yellow' are not serials.
    if not usable(key) or len(key) < 4 or len(key) > 40 or not any(c.isdigit() for c in key):
        return None
    return key


def serial_key(value):
    key = re.sub(r"\s+", "", norm(value))
    if not usable(key) or len(key) < 5 or len(set(key)) < 3 or not any(c.isdigit() for c in key):
        return None
    return key


def manufacturer_key(value):
    key = norm(value)
    return "EOBS" if re.search(r"E[- ]?OBS", key) else key


def qualified_ring(value):
    key = ring_key(value)
    return bool(key and re.fullmatch(r"(?:DER|DEW)[A-Z0-9]{4,10}", key) and sum(c.isdigit() for c in key) >= 3)


def namespace(value):
    return re.sub(r"[^A-Z0-9]+", " ", norm(value)).strip()


def scoped_animals(candidates, source, species, reference_name):
    """A DOI may bundle studies with independent local animal namespaces."""
    candidates = sorted(candidates)
    if len(candidates) <= 1:
        return candidates
    stem = re.sub(r"[-_ ]reference(?:[-_ ]data)?$", "", Path(reference_name).stem, flags=re.I)
    stem = re.sub(r"(?:[-_ ]part)$", "", stem, flags=re.I)
    ref = namespace(stem)
    exact, prefix = [], []
    for animal in candidates:
        study = animal[len(source) + 1:].rsplit(":" + species + ":", 1)[0]
        study = namespace(study)
        if study == ref:
            exact.append(animal)
        elif ref.startswith(study + " ") and re.fullmatch(r"(?:20\d\d(?: 20\d\d)?|PART)", ref[len(study) + 1:]):
            prefix.append(animal)
    matches = exact or prefix
    return matches if len(matches) == 1 else []


def observation_index(paths):
    index, inventory = defaultdict(set), []
    for path in paths:
        path = Path(path)
        q = pq.ParquetFile(path)
        for batch in q.iter_batches(batch_size=250_000, columns=["animal", "source", "species"]):
            unique = pa.Table.from_batches([batch]).group_by(["animal", "source", "species"]).aggregate([])
            for row in unique.to_pylist():
                delimiter = ":" + row["species"] + ":"
                local_id = row["animal"].rsplit(delimiter, 1)[-1]
                index[(row["source"], norm(row["species"]), local_key(local_id))].add(row["animal"])
        inventory.append({"path": str(path), "sha256": sha256(path), "rows": q.metadata.num_rows})
    return index, inventory


def read_reference(path):
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f)), "utf-8-sig"
    except UnicodeDecodeError:
        with path.open(encoding="cp1252", newline="") as f:
            return list(csv.DictReader(f)), "cp1252 (explicit fallback after UTF-8 failed)"


def audit(observations, roots, output):
    observed, observation_manifest = observation_index(observations)
    records, references, skipped, mapping_ambiguities = [], [], [], []
    seen = set()
    for root in roots:
        for directory in sorted(Path(root).glob("*")):
            manifest_path = directory / "source.json"
            if not manifest_path.exists():
                continue
            source = json.loads(manifest_path.read_text())
            for path in sorted(directory.glob("*reference*.csv")):
                checksum = sha256(path)
                # Deduplicate cached copies of the same public reference release.
                if (directory.name, checksum) in seen:
                    continue
                seen.add((directory.name, checksum))
                entry = next((f for f in source.get("files", []) if f["name"] == path.name), None)
                if not entry or entry.get("sha256") != checksum:
                    skipped.append({"path": str(path), "reason": "reference hash lacks matching source manifest evidence"})
                    continue
                rows, encoding = read_reference(path)
                references.append({"path": str(path), "sha256": checksum, "rows": len(rows),
                                   "source_manifest_sha256": sha256(manifest_path), "encoding": encoding,
                                   "columns": list(rows[0]) if rows else []})
                for line, row in enumerate(rows, 2):
                    local = row.get("animal-id", row.get("individual-local-identifier", ""))
                    species = row.get("animal-taxon", row.get("individual-taxon-canonical-name", ""))
                    if not usable(local) or not usable(species):
                        continue
                    record = {"record_type": "reference_table", "source_id": directory.name, "species": species, "local_identifier": local,
                              "ring": row.get("animal-ring-id", ""), "manufacturer": row.get("tag-manufacturer-name", ""),
                              "tag_serial": row.get("tag-serial-no", ""), "tag_id": row.get("tag-id", ""),
                              "deployment_id": row.get("deployment-id", ""), "deploy_on": row.get("deploy-on-date", ""),
                              "deploy_off": row.get("deploy-off-date", ""), "site": row.get("study-site", ""),
                              "sex": row.get("animal-sex", ""), "group": row.get("animal-group-id", ""),
                              "reference": str(path), "reference_sha256": checksum, "line": line}
                    key = (directory.name, norm(species), local_key(local))
                    candidates = observed.get(key, [])
                    record["observation_animals"] = scoped_animals(candidates, directory.name, species, path.name)
                    if candidates and not record["observation_animals"]:
                        record["unresolved_namespace_candidates"] = sorted(candidates)
                        mapping_ambiguities.append({"type": "reference_row_has_unresolved_event_study_namespace",
                                                    "reference": str(path), "line": line, "animals": sorted(candidates)})
                    records.append(record)

    # Some public stork releases have no reference CSV but explicitly print a
    # qualified ring and vendor in the animal identifier. Preserve the precise
    # evidence origin; these fields are not guessed from bare numeric IDs.
    for (source, species, local), animals in observed.items():
        match = re.search(r"\b(DER|DEW)\s*([A-Z0-9]{4,10})\b", local, re.I)
        vendor = re.search(r"\be-?obs\s+(\d{3,})\b", local, re.I)
        if not match or not vendor:
            continue
        ring = match.group(1).upper() + match.group(2).upper()
        if not qualified_ring(ring):
            continue
        records.append({"record_type": "explicit_ring_and_vendor_printed_in_event_identifier",
                        "source_id": source, "species": species, "local_identifier": local,
                        "ring": ring, "manufacturer": "e-obs", "tag_serial": "", "tag_id": vendor.group(1),
                        "deployment_id": "", "deploy_on": "", "deploy_off": "", "site": "", "sex": "", "group": "",
                        "reference": "supplied observation animal identifier", "reference_sha256": None,
                        "line": None, "observation_animals": sorted(animals),
                        "field_evidence": {"printed_identifier": local,
                                           "ring_substring": match.group(0), "vendor_tag_substring": vendor.group(0)}})

    parent = {animal: animal for animals in observed.values() for animal in animals}
    links, ambiguous = [], mapping_ambiguities
    linked_pairs = set()

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def connect(left, right, evidence):
        for a, b in itertools.product(left["observation_animals"], right["observation_animals"]):
            if a == b or tuple(sorted((a, b))) in linked_pairs:
                continue
            a, b = sorted((a, b))
            linked_pairs.add((a, b))
            root_a, root_b = find(a), find(b)
            parent[max(root_a, root_b)] = min(root_a, root_b)
            links.append({"a": a, "b": b, "evidence": {**evidence,
                          "left_reference": left["reference"], "left_line": left["line"],
                          "right_reference": right["reference"], "right_line": right["line"]}})

    def conflict(a, b):
        sexes = {norm(v) for v in (a["sex"], b["sex"]) if norm(v) in {"M", "F"}}
        rings = {ring_key(v) for v in (a["ring"], b["ring"]) if ring_key(v)}
        return len(sexes) > 1 or len(rings) > 1

    local_groups, ring_groups, deployment_groups, hardware_groups = (defaultdict(list) for _ in range(4))
    for r in records:
        local_groups[(r["source_id"], norm(r["species"]), local_key(r["local_identifier"]))].append(r)
        ring = ring_key(r["ring"])
        if ring:
            ring_groups[(norm(r["species"]), ring)].append(r)
        # Full serial + manufacturer + same documented deployment start is a
        # deployment fingerprint. Tag-id on its own is only a local namespace.
        if serial_key(r["tag_serial"]) and usable(r["manufacturer"]):
            hw = (norm(r["species"]), manufacturer_key(r["manufacturer"]), serial_key(r["tag_serial"]))
            hardware_groups[hw].append(r)
            if re.match(r"\d{4}-\d{2}-\d{2}", r["deploy_on"]):
                deployment_groups[hw + (r["deploy_on"].strip(),)].append(r)

    for key, rows in local_groups.items():
        if any(conflict(a, b) for a, b in itertools.combinations(rows, 2)):
            ambiguous.append({"type": "conflicting_reference_group_no_transitive_bridge", "key": list(key),
                              "references": [[r["reference"], r["line"]] for r in rows],
                              "animals": sorted({a for r in rows for a in r["observation_animals"]})})
            continue
        # Local IDs alone are insufficient across separate studies in a DOI.
        # Ring or full deployment evidence below must establish the connection.

    for key, rows in ring_groups.items():
        if any(conflict(a, b) for a, b in itertools.combinations(rows, 2)):
            ambiguous.append({"type": "conflicting_ring_group_no_transitive_bridge", "key": list(key),
                              "references": [[r["reference"], r["line"]] for r in rows],
                              "animals": sorted({a for r in rows for a in r["observation_animals"]})})
            continue
        for a, b in itertools.combinations(rows, 2):
            site = usable(a["site"]) and norm(a["site"]) == norm(b["site"]) and len(norm(a["site"])) >= 6
            manufacturer = usable(a["manufacturer"]) and manufacturer_key(a["manufacturer"]) == manufacturer_key(b["manufacturer"])
            same_start = usable(a["deploy_on"]) and a["deploy_on"].strip() == b["deploy_on"].strip()
            qualified = qualified_ring(key[1])
            if not conflict(a, b) and (site or (manufacturer and (same_start or qualified))):
                connect(a, b, {"type": "same_ring_species_with_corroborating_metadata", "key": list(key),
                               "matching_manufacturer": a["manufacturer"] if manufacturer else None, "matching_site": a["site"] if site else None,
                               "matching_deploy_on": a["deploy_on"] if same_start else None,
                               "qualified_printed_ring": qualified,
                               "left_printed_identifier_evidence": a.get("field_evidence"),
                               "right_printed_identifier_evidence": b.get("field_evidence")})
            else:
                ambiguous.append({"type": "ring_match_lacks_corroboration_or_conflicts", "key": list(key),
                                  "references": [[a["reference"], a["line"]], [b["reference"], b["line"]]],
                                  "animals": sorted(set(a["observation_animals"] + b["observation_animals"]))})

    for key, rows in deployment_groups.items():
        if any(conflict(a, b) for a, b in itertools.combinations(rows, 2)):
            ambiguous.append({"type": "conflicting_deployment_group", "key": list(key),
                              "animals": sorted({a for r in rows for a in r["observation_animals"]})})
            continue
        for a, b in itertools.combinations(rows, 2):
            if not conflict(a, b):
                connect(a, b, {"type": "same_full_hardware_serial_manufacturer_species_deployment_start", "key": list(key)})

    for key, rows in hardware_groups.items():
        different_starts = sorted({r["deploy_on"] for r in rows if usable(r["deploy_on"])})
        different_animals = sorted({(r["source_id"], r["local_identifier"]) for r in rows})
        if len(different_starts) > 1 and len(different_animals) > 1:
            ambiguous.append({"type": "possible_hardware_redeployment_no_automatic_identity_merge", "key": list(key),
                              "deployment_starts": different_starts, "source_local_animals": different_animals,
                              "animals": sorted({a for r in rows for a in r["observation_animals"]})})

    for key, rows in local_groups.items():
        names = sorted({a for r in rows for a in r["observation_animals"]})
        if len({find(a) for a in names}) > 1:
            ambiguous.append({"type": "same_source_local_id_across_studies_not_asserted_identity", "key": list(key), "animals": names})

    groups = {a: find(a) for a in sorted(parent) if find(a) != a}
    for canonical in set(groups.values()):
        groups[canonical] = canonical
    # Plausible aliases share split/bootstrap blocks without asserting identity
    # or combining observations. Strong physical links and uncertainty links
    # therefore remain separate outputs with different semantics.
    risk_parent = dict(parent)
    def risk_find(a):
        while risk_parent[a] != a:
            risk_parent[a] = risk_parent[risk_parent[a]]
            a = risk_parent[a]
        return a
    affected = set()
    for entry in ambiguous:
        names = [a for a in entry.get("animals", []) if a in risk_parent]
        affected.update(names)
        for a, b in zip(names, names[1:]):
            aa, bb = risk_find(a), risk_find(b)
            risk_parent[max(aa, bb)] = min(aa, bb)
    risk_roots = {risk_find(a) for a in affected}
    uncertainty_groups = defaultdict(list)
    for a in sorted(risk_parent):
        if risk_find(a) in risk_roots:
            uncertainty_groups[risk_find(a)].append(a)
    uncertainty_mapping = {a: root for root, names in uncertainty_groups.items() for a in names}
    report = {"observed_utc": datetime.now(timezone.utc).isoformat(), "scope": "Identity audit only; candidate fresh means not identified in available metadata, never certified globally new.",
              "observations": observation_manifest, "references_checked": references, "references_skipped": skipped,
              "fold_groups": groups, "links": links, "ambiguous": ambiguous, "reference_records": records,
              "uncertainty_fold_groups": uncertainty_mapping,
              "uncertainty_groups": [{"group": key, "animals": value} for key, value in sorted(uncertainty_groups.items())],
              "freshness_exclusion_animals": sorted(affected),
              "uncertainty_rule": "Use uncertainty_fold_groups only for split/bootstrap grouping and freshness exclusion; never merge coordinates or claim physical identity from these links.",
              "summary": {"reference_files": len(references), "reference_rows": sum(r["rows"] for r in references),
                          "explicit_identifier_ring_records": sum(r["record_type"] != "reference_table" for r in records),
                          "rows_joined_to_observations": sum(bool(r["observation_animals"]) for r in records),
                          "observed_animals": len(parent), "links": len(links), "ambiguous": len(ambiguous),
                          "uncertainty_groups": len(uncertainty_groups), "uncertainty_affected_animals": len(uncertainty_mapping),
                          "freshness_exclusion_animals": len(affected)},
              "limitations": ["Absent reference matches do not prove new physical animals.",
                              "No merges use a hardware/tag ID alone; redeployments and ambiguous markers remain unresolved.",
                              "Identity links should be unioned with exact-fix links before splitting.",
                              "Reference names map through supplied observation source/species/local identifiers; aliases already removed from these files cannot be recovered without earlier observation inventories."]}
    dump(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", nargs="+", type=Path, required=True)
    p.add_argument("--roots", nargs="+", type=Path, default=[Path("data/raw"), Path("data/v2/raw"), Path("data/migration-corpus/raw"), Path("data/migration-expanded/raw")])
    p.add_argument("--output", type=Path, default=Path("reports/migration-reference-audit.json"))
    a = p.parse_args()
    print(json.dumps(audit(a.observations, a.roots, a.output)["summary"], indent=2))


if __name__ == "__main__":
    main()
