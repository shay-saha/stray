"""Read-only reference-field census; never exports model-ready metadata features."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path


FIELDS = {
    "life_stage_at_deployment": ["animal-life-stage", "animal-life-stage-exact"],
    "sex": ["animal-sex"],
    "mass": ["animal-mass"],
    "length": ["animal-length"],
    "capture_measurements": ["deploy-on-measurements"],
    "birth_date": ["animal-exact-date-of-birth"],
    "birth_year": ["animal-birth-hatch-year"],
    "latest_birth_bound": ["animal-latest-date-born"],
    "earliest_birth_bound": ["animal-earliest-date-born"],
    "deployment_start": ["deploy-on-date", "deploy-on-timestamp"],
    "capture_time": ["capture-timestamp"],
    "group_id": ["animal-group-id"],
    "study_site": ["study-site"],
    "reproductive_condition_at_deployment": ["animal-reproductive-condition"],
}
FORBIDDEN = ["animal-death-comments", "animal-mortality-date", "animal-mortality-type",
             "animal-status", "animal-status-date", "deployment-end-comments", "deployment-end-type",
             "deploy-off-date", "deploy-off-measurements", "deploy-off-sampling", "tag-retrieval-date",
             "animal-comments", "behavior-according-to", "habitat-according-to"]
UNKNOWN = {"", "u", "unknown", "na", "n/a", "nan", "none", "not known", "not specified", "?"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def usable(value, field=None):
    value = str(value or "").strip()
    if field == "sex":
        return value.lower() in {"m", "f", "male", "female"}
    return value.lower() not in UNKNOWN


def read_rows(path):
    for encoding in ["utf-8-sig", "cp1252"]:
        try:
            with Path(path).open(encoding=encoding, newline="") as stream:
                return list(csv.DictReader(stream)), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unreadable reference: {path}")


def audit(identity, parent, output, reference_mirror=None):
    identity, parent = Path(identity), Path(parent)
    evidence, previous = json.loads(identity.read_text()), json.loads(parent.read_text())
    physical, exclusion = evidence["fold_groups"], evidence["uncertainty_fold_groups"]
    new = {r["animal"] for r in evidence["new_animal_parent_bindings"]}
    old = set(previous["identity_mapping"])
    scopes = {"new_clock_roots": new, "parent_v3": old, "parent_plus_new": new | old}
    references = collections.defaultdict(list)
    for record in evidence["reference_records"]:
        if record["record_type"] in {"reference_table", "source-derived_identity_reference"}:
            references[record["reference"]].append(record)
    covered = {scope: {field: set() for field in FIELDS} for scope in scopes}
    timed = {scope: {field: set() for field in FIELDS} for scope in scopes}
    bound = {scope: set() for scope in scopes}
    values = {field: collections.Counter() for field in FIELDS}
    by_species = collections.defaultdict(lambda: {field: set() for field in FIELDS})
    forbidden = collections.Counter()
    source_files, declarations, seen, sex_values = [], [], set(), collections.defaultdict(set)
    for path, records in sorted(references.items()):
        expected = {r["reference_sha256"] for r in records}
        resolved = Path(path)
        if not resolved.exists() and reference_mirror:
            resolved = Path(reference_mirror) / path
        actual = sha(resolved)
        if expected != {actual}:
            raise ValueError(f"Frozen identity/reference hash mismatch: {path}")
        rows, encoding = read_rows(resolved)
        source_files.append({"path": path, "resolved_path": str(resolved), "sha256": actual, "rows": len(rows), "encoding": encoding})
        for record in records:
            index = int(record["line"]) - 2
            if index < 0 or index >= len(rows):
                raise ValueError("Frozen reference row outside source")
            animals = sorted(set(record["observation_animals"]) & (new | old))
            key = (actual, index, tuple(animals))
            if not animals or key in seen:
                continue
            seen.add(key)
            row = rows[index]
            selected = {field: {column: str(row.get(column) or "").strip() for column in columns
                                if usable(row.get(column), field)} for field, columns in FIELDS.items()}
            start = next(iter(selected["deployment_start"].values()), "")
            valid_start = bool(DATE.fullmatch(start))
            # This date is an event-time anchor, never proof of when a public
            # reference field became available to an operational forecaster.
            declaration = {"reference": path, "reference_sha256": actual, "reference_line": record["line"],
                           "source_id": record["source_id"], "raw_species": record["species"],
                           "source_animals": animals, "physical_ids": sorted({physical[n] for n in animals}),
                           "exclusion_ids": sorted({exclusion[n] for n in animals}),
                           "deployment_start_parseable": valid_start,
                           "fields": {field: item for field, item in selected.items() if item},
                           "issue_time_evidence": "not_recorded_in_reference_snapshot",
                           "record_kind": record["record_type"]}
            declarations.append(declaration)
            for field, item in selected.items():
                values[field].update(item.values())
                for animal in animals:
                    if item:
                        by_species[record["species"]][field].add(physical[animal])
                    if field == "sex":
                        sex_values[physical[animal]].update(v.lower()[0] for v in item.values())
            for field in FORBIDDEN:
                if usable(row.get(field)):
                    forbidden[field] += 1
            for scope, members in scopes.items():
                groups = {physical[n] for n in animals if n in members}
                bound[scope].update(groups)
                for field, item in selected.items():
                    if item:
                        covered[scope][field].update(groups)
                        if valid_start:
                            timed[scope][field].update(groups)
    summaries = {}
    for scope, members in scopes.items():
        groups = {physical[n] for n in members}
        summaries[scope] = {"source_ids": len(members), "physical_ids": len(groups),
                            "physical_ids_with_bound_reference": len(bound[scope]),
                            "field_physical_id_coverage": {
                                f: {"any_reported_value": len(covered[scope][f]),
                                    "with_parseable_deployment_anchor": len(timed[scope][f]),
                                    "fraction": len(covered[scope][f]) / len(groups)} for f in FIELDS}}
    result = {"schema": "wildcast-deployment-metadata-readonly-audit-v1",
              "code_sha256": sha(__file__), "identity_audit": {"path": str(identity), "sha256": sha(identity)},
              "parent_manifest": {"path": str(parent), "sha256": sha(parent)}, "scopes": summaries,
              "reference_files": source_files, "deduplicated_bound_reference_rows": len(declarations),
              "reported_values": {field: dict(counter.most_common()) for field, counter in values.items()},
              "raw_species_physical_id_coverage": {
                  species: {field: len(groups) for field, groups in fields.items()} for species, fields in sorted(by_species.items())},
              "sex_conflicting_physical_ids": {group: sorted(v) for group, v in sex_values.items() if len(v) > 1},
              "forbidden_or_review_only_field_row_counts": dict(forbidden), "reference_rows": declarations,
              "limitations": ["Inventory only: no model-ready feature table, input cache mutation or model fitting",
                              "Reported life stage describes deployment start, not necessarily stage at later forecast issue time",
                              "Reference snapshots lack field-update timestamps: operational as-of knowledge is not proven",
                              "Counts indicate at least one reported deployment value, not per-origin causal coverage",
                              "Group/site fields may be study-wide or time varying; no real-time colony membership is inferred",
                              "Raw age categories and measurement units need source-specific mapping; unknown masks must remain explicit",
                              "No death/fate/end-of-deployment/free-text/derived trip-phase field is exported as a candidate feature"],
              "primary_semantics": ["https://www.movebank.org/cms/movebank-content/movebank-attribute-dictionary",
                                     "https://vocab.nerc.ac.uk/collection/MVB/current/MVB000018/"]}
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "scopes": summaries,
                      "reference_files": len(source_files), "rows": len(declarations),
                      "sex_conflicts": len(result["sex_conflicting_physical_ids"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-mirror")
    args = parser.parse_args()
    audit(args.identity, args.parent, args.output, args.reference_mirror)
