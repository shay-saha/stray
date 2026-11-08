"""Audit source annotations without opening reserved forecast outcomes.

Author-derived migration dates are supervision/evaluation labels, never future
input features. The timing loader returns training/development animal-years only.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .migration_assessment import _load_base, _provenance
from .sources import dump

DEVELOPMENT = {"train", "val_individual", "val_year"}
WYOMING = "0e3a4577-b063-47d9-9d6b-edec949aa5fa"
WORKBOOK = Path("data/migration-annotations/raw/ortega-source-data/41467_2023_37750_MOESM4_ESM.xlsx")
TRAITS = {"sex": "animal-sex", "life_stage_at_deployment": "animal-life-stage",
          "exact_birth_date": "animal-exact-date-of-birth", "group_at_deployment": "animal-group-id",
          "study_site": "study-site", "reproductive_condition_at_deployment": "animal-reproductive-condition"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verify(path, expected):
    if not Path(path).is_file() or sha256(path) != expected:
        raise ValueError(f"Annotation provenance checksum differs: {path}")


def _origins(data):
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    _verify(data / "origins.parquet", manifest["origins_sha256"])
    origins = pd.read_parquet(data / "origins.parquet")
    return origins, manifest


def _workbook_provenance(workbook):
    source_path = Path(workbook).parent / "source.json"
    source = json.loads(source_path.read_text())
    entry = next((v for v in source["files"] if Path(v["path"]).name == Path(workbook).name), None)
    if not entry:
        raise ValueError("Workbook lacks source checksum evidence")
    _verify(workbook, entry["sha256"])
    return {"path": str(workbook), "sha256": entry["sha256"], "source": str(source_path),
            "source_sha256": sha256(source_path), "url": source.get("url"), "license": source.get("license")}


def load_development_timing(data, workbook=WORKBOOK):
    """Direct source-specific ID/year join; reserved rows' outcome fields stay unused.

    Dates are the authors' NSD-derived spring migration delineations at day-of-
    year precision, not independently observed intentions or operational inputs.
    """
    import openpyxl
    origins, manifest = _origins(data)
    provenance = _workbook_provenance(workbook)
    keys = defaultdict(set)
    all_keys = defaultdict(set)
    prefix = WYOMING + ":"
    for row in origins.loc[origins.canonical_animal.str.startswith(prefix)].itertuples(index=False):
        key = (row.canonical_animal.rsplit(":", 1)[-1], int(row.year))
        all_keys[key].add(row.split)
        if row.split in DEVELOPMENT:
            keys[key].add((row.canonical_animal, row.split))
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    sheet = book.worksheets[0]
    iterator = sheet.iter_rows(values_only=True)
    header = list(next(iterator))
    columns = {name: header.index(name) for name in ("id_yr", "year", "DOY_Start", "DOY_End") if name in header}
    if len(columns) != 4:
        raise ValueError("Author timing sheet lacks its documented ID/year/start/end columns")
    rows, counts = [], Counter()
    for values in iterator:
        if values[columns["id_yr"]] is None:
            continue
        counts["source_animal_years"] += 1
        identifier = str(values[columns["id_yr"]])
        year = int(values[columns["year"]])
        local, suffix = identifier.rsplit("_", 1)
        if int(suffix) != year:
            raise ValueError("Author animal-year identifier disagrees with its year")
        key = (local, year)
        if key not in keys:
            counts["reserved_selectors_matched_outcomes_not_inspected" if key in all_keys else "unmatched_animal_years"] += 1
            continue
        # Only now may outcome columns be used: the ID/year is development-only.
        start, end = float(values[columns["DOY_Start"]]), float(values[columns["DOY_End"]])
        limit = 366 if calendar.isleap(year) else 365
        if not np.isfinite([start, end]).all() or not 1 <= start <= end <= limit:
            raise ValueError("Invalid author timing labels on a development animal-year")
        if len({name for name, _ in keys[key]}) != 1:
            counts["ambiguous_source_specific_ID_year"] += 1
            continue
        counts["matched_development_animal_years"] += 1
        for name, split in sorted(keys[key]):
            rows.append({"canonical_animal": name, "year": year, "split": split,
                         "departure_doy": start, "arrival_doy": end,
                         "departure_date": pd.Timestamp(f"{year}-01-01", tz="UTC") + pd.Timedelta(days=start - 1),
                         "arrival_date": pd.Timestamp(f"{year}-01-01", tz="UTC") + pd.Timedelta(days=end - 1),
                         "annotation_method": "author NSD-derived migration delineation",
                         "source_sha256": provenance["sha256"]})
    book.close()
    return pd.DataFrame(rows), {"provenance": provenance, "sheet": sheet.title, "columns": header,
            "counts": dict(counts), "join_rule": "Exact Wyoming-source animal local ID plus calendar year; source README explicitly cites this author dataset.",
            "label_status": "Author-derived spring migration dates based on Net Squared Displacement, not independent behavioural observations.",
            "temporal_resolution": "Published day-of-year fields; no finer biological label precision established.",
            "input_rule": "Use as development supervision/evaluation only. Actual future departure/arrival is never an input feature."}


def _reference_coverage(origins, manifest, reference):
    development = origins.loc[origins.split.isin(DEVELOPMENT)].copy()
    development["origin_time"] = pd.to_datetime(development.origin_time, utc=True)
    by_animal = {name: g for name, g in development.groupby("canonical_animal", sort=False)}
    cached, attributes = {}, defaultdict(list)
    for file in reference["references_checked"]:
        path = Path(file["path"])
        _verify(path, file["sha256"])
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                cached[str(path)] = list(csv.DictReader(stream))
        except UnicodeDecodeError:
            with path.open(encoding="cp1252", newline="") as stream:
                cached[str(path)] = list(csv.DictReader(stream))
    for record in reference["reference_records"]:
        if record.get("record_type", "reference_table") != "reference_table" or not record.get("line"):
            continue
        row = cached[record["reference"]][record["line"] - 2]
        names = {manifest["identity_mapping"].get(n, n) for n in record["observation_animals"]}
        for name in names & by_animal.keys():
            on = pd.to_datetime(row.get("deploy-on-date", ""), utc=True, errors="coerce")
            off = pd.to_datetime(row.get("deploy-off-date", ""), utc=True, errors="coerce")
            for label, column in TRAITS.items():
                value = str(row.get(column, "") or "").strip()
                if value.lower() in {"", "u", "unknown", "na", "n/a", "nan"}:
                    continue
                if label == "sex" and value.lower() not in {"m", "f"}:
                    continue
                attributes[label].append((name, value, on, off))
    result = {}
    for label in TRAITS:
        rows = attributes[label]
        per_animal_values = defaultdict(set)
        for name, value, _, _ in rows:
            per_animal_values[name].add(value.lower())
        conflicting = {n for n, values in per_animal_values.items() if label == "sex" and len(values) > 1}
        matched, anchored = defaultdict(set), defaultdict(set)
        for name, value, on, off in rows:
            group = by_animal[name]
            for split, chunk in group.groupby("split", sort=False):
                matched[split].add(name)
                if name in conflicting or pd.isna(on):
                    continue
                available = chunk.origin_time >= on
                if not pd.isna(off):
                    available &= chunk.origin_time <= off
                if label == "exact_birth_date":
                    birth = pd.to_datetime(value, utc=True, errors="coerce")
                    if pd.isna(birth):
                        continue
                    available &= chunk.origin_time >= birth
                anchored[split].update(chunk.loc[available, "case_id"].astype(int))
        result[label] = {"source_field": TRAITS[label], "matched_development_animals": {s: len(v) for s, v in sorted(matched.items())},
                         "deployment_date_anchored_development_cases": {s: len(v) for s, v in sorted(anchored.items())},
                         "conflicting_development_animals": len(conflicting),
                         "matched_development_animal_years": len({(n, int(y)) for n in per_animal_values for y in by_animal[n].year.unique()})}
    return result


def _auxiliary_annotation_coverage(workbook, development_timing):
    """Inspect outcome values only after a development ID/year selector matches."""
    import openpyxl
    keys = {(r.canonical_animal.rsplit(":", 1)[-1], int(r.year)): (r.canonical_animal, r.split)
            for r in development_timing.itertuples(index=False)}
    book = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    out = {}
    for title, sheet_name, header_row, value_column in [
        ("author_stopover_total_days", "Fig4d,4e;SFig4", 2, "stopover.day"),
        ("author_body_condition", "SFig1", 1, "scaledIFBFat")]:
        if sheet_name not in book.sheetnames:
            continue
        sheet = book[sheet_name]
        iterator = sheet.iter_rows(min_row=header_row, values_only=True)
        header = list(next(iterator))
        id_col, value_col = header.index("id_yr"), header.index(value_column)
        matched = defaultdict(list)
        for row in iterator:
            if row[id_col] is None:
                continue
            local, year = str(row[id_col]).rsplit("_", 1)
            key = (local, int(year))
            if key not in keys:
                continue
            # Do not parse a reserved or unmatched outcome field.
            try:
                value = float(row[value_col])
            except (ValueError, TypeError):
                continue
            if not np.isfinite(value):
                continue
            name, split = keys[key]
            matched[split].append((name, int(year)))
        out[title] = {"sheet": sheet_name, "field": value_column,
                      "development_coverage_by_split": {s: {"animal_years": len(v), "animals": len({n for n, _ in v})} for s, v in sorted(matched.items())}}
    book.close()
    if "author_stopover_total_days" in out:
        out["author_stopover_total_days"]["status"] = "Authors' whole-migration high-use stopover-day totals from Brownian-bridge utilization analysis; no timestamped stopover intervals or independently observed foraging/rest states. Supervision only, not an origin-time input."
    if "author_body_condition" in out:
        out["author_body_condition"]["status"] = "Matched physiological body-condition values exist. Exact capture timestamps and input scaling are not supplied by this sheet; causal availability at each forecast origin is not established. Not promoted to input features."
    return out


def audit(data, output):
    data = Path(data)
    arrays, manifest, origins, hashes = _load_base(data)
    provenance = _provenance(manifest, data.resolve())
    sealed = np.isin(arrays["split"], [3, 4])
    reference = json.loads(Path(manifest["reference_audit"]).read_text())
    traits = _reference_coverage(origins, manifest, reference)
    timing, timing_audit = load_development_timing(data)
    timing_audit["development_coverage_by_split"] = {str(s): {"animal_years": len(g), "animals": g.canonical_animal.nunique()}
                                                      for s, g in timing.groupby("split")}
    timing_audit["development_ID_years"] = timing[["canonical_animal", "year", "split"]].to_dict("records")
    auxiliary = _auxiliary_annotation_coverage(WORKBOOK, timing)
    jae = Path("data/migration-corpus/raw/6c481a15-2f87-4418-9476-8bbd84974adf/JAE_meta_data.csv")
    with jae.open(newline="") as stream:
        reader = csv.DictReader(stream)
        selectors = [(r["id"], r["Year"]) for r in reader]
        jae_columns = reader.fieldnames
    jae_source = json.loads((jae.parent / "source.json").read_text())
    jae_entry = next(v for v in jae_source["files"] if v["name"] == jae.name)
    _verify(jae, jae_entry["sha256"])
    external_sources = []
    for path in sorted(Path("data/migration-annotations/raw").glob("*/source.json")):
        external_sources.append({"manifest": str(path), "sha256": sha256(path), **json.loads(path.read_text())})
    acquired_bytes = sum(p.stat().st_size for p in Path("data/migration-annotations/raw").rglob("*") if p.is_file())
    if acquired_bytes > 64_000_000:
        raise ValueError("External metadata acquisition exceeds its 64 MB budget")
    report = {"observed_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Annotation structure, development-only metadata joins and sealed-cache integrity; no final target derivation, release, scoring or inspection.",
              "cache": {"path": str(data), **hashes, "cases": len(origins), "reserved_cases": int(sealed.sum()),
                        "reserved_masks_all_false": not bool(arrays["target_mask"][sealed].any()),
                        "reserved_target_arrays_all_zero": all(bool(np.all(arrays[k][sealed] == 0)) for k in ("y", "target_latlon", "target_time")),
                        "provenance_verified": provenance,
                        "deployment_note": "Paths currently resolve locally. Assessment sidecar records absolute resolved cache/source/checkpoint paths, which must be available in its runtime."},
              "reference_trait_coverage": traits,
              "trait_limitations": ["Life stage is recorded at deployment, not a verified changing behavioural state or current age.",
                                    "Group/site metadata do not establish simultaneous neighbours, social edges, current colony occupancy or a breeding outcome.",
                                    "Blank deployment dates prevent a documented known-by-origin claim; retrospective publication latency is unverified.",
                                    "No death, reproductive outcome or future movement field is used as a predictor."],
              "author_timing_annotations": timing_audit,
              "author_auxiliary_annotations": auxiliary,
              "jae_metadata": {"path": str(jae), "sha256": sha256(jae), "columns": jae_columns,
                               "animal_years": len(selectors), "IDs": len({v[0] for v in selectors}),
                               "years": sorted({int(v[1]) for v in selectors}),
                               "ID_semantics": "All 36 metadata IDs match deployment-id in the associated reference table, not animal-id or tag-id.",
                               "verified_current_cohort_matches": 0,
                               "reason": "Associated tracking source has no retained observations in the current corpus; no corroborated ring/vendor link into current development IDs was found.",
                               "status": "Sex plus breeding-attempt/fledgling outcomes; no departure/arrival/stopover columns. Outcome values not inspected."},
              "crane_season_annotations": {"status": "Author code expects per-individual manually supplied seasonal cutpoint CSVs; those CSVs are absent from the released repository tree.",
                                            "code": "data/migration-annotations/raw/cranes-niches/seg2anno.r", "reference": "https://github.com/syanco/cranes_niches",
                                            "matched_current_individual_seasons": 0, "limitation": "No seasonal cutpoints are fabricated from the code or calendar."},
              "stork_timing_annotations": {"status": "Pinned author repository supplies processing code and a release/death table; no released departure/arrival/stopover annotation table was found.",
                                            "reference": "https://github.com/IrisDB/Delayed_migration_white_stork",
                                            "matched_current_individual_seasons": 0, "limitation": "Experimental release dates and mortality dates are distinct outcomes; the death table was not acquired or inspected."},
              "independently_observed_behaviour_labels": {"departure": 0, "arrival": 0, "stopover": 0,
                                                        "note": "The matched author-derived NSD migration dates above are a separate annotation category, not independent biological ground truth."},
              "external_metadata_acquisition": {"budget_bytes": 64_000_000, "acquired_bytes": acquired_bytes, "sources": external_sources,
                                                "dryad_download_gap": "Public metadata/license/checksum API accessible; file download API returned401 and public file stream403. Official publisher workbook was obtained instead."}}
    dump(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/migration-expanded-dev-v3")
    p.add_argument("--output", default="reports/migration-annotation-audit.json")
    args = p.parse_args()
    report = audit(args.data, args.output)
    print(json.dumps({"cache_valid": True, "reserved_still_sealed": report["cache"]["reserved_target_arrays_all_zero"],
                      "development_timing": report["author_timing_annotations"]["development_coverage_by_split"],
                      "traits": report["reference_trait_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
