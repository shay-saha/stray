"""Build a separate, provenance-preserving marine GPS observation core.

This is source QC and identity work, not training or evaluation.  No coordinate
interpolation, clock correction, per-class accuracy, or biological precision is
invented.  ``timestamp_nominal`` embeds the source clock on a UTC-labelled axis
for ordering only.  Verified UTC is separate.  The legacy ``timestamp`` column is
deliberately absent so future corpus preparation must choose a clock policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .marine_intake import SourceSpec, _method, _resolve, sha256
from .migration_identity import audit as reference_audit


SCHEMA = "wildcast-marine-gps-clean-v1"


def _reference_rows(directory):
    import csv
    rows, evidence = [], []
    for path in sorted(directory.glob("*reference*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            values = list(csv.DictReader(stream))
        digest = sha256(path)
        evidence.append({"path": str(path), "sha256": digest, "rows": len(values)})
        rows.extend({**r, "_reference_file": str(path), "_reference_sha256": digest} for r in values)
    return rows, evidence


def _nominal(value):
    return pd.to_datetime(value, format="mixed", errors="coerce", utc=True)


def _attachment_metadata(raw, cols, references, required=False):
    """Match source animal/taxon/tag to a documented deployment, never tag alone."""
    by_key = defaultdict(list)
    for r in references:
        by_key[(r.get("animal-id", ""), r.get("animal-taxon", ""), r.get("tag-id", ""))].append(
            (json.dumps(r, sort_keys=True, ensure_ascii=False), _nominal(r.get("deploy-on-date", "")),
             _nominal(r.get("deploy-off-date", ""))))
    chosen, accepted, status = [], [], []
    individuals = raw[cols["individual"]]
    taxa = raw[cols["species"]]
    tags = raw[cols["tag"]] if "tag" in cols else pd.Series("", index=raw.index)
    dates = _nominal(raw[cols["timestamp"]])
    for individual, taxon, tag, stamp in zip(individuals, taxa, tags, dates):
        candidates = by_key.get((individual, taxon, tag), [])
        if not candidates:
            chosen.append(""); accepted.append(not required); status.append("reference_unmatched")
            continue
        valid = []
        for serialized, start, end in candidates:
            if required and (pd.isna(start) or pd.isna(end)):
                continue
            if (pd.isna(start) or stamp >= start) and (pd.isna(end) or stamp <= end):
                valid.append(serialized)
        # Identical source copies are not independent deployments.
        valid = sorted(set(valid))
        if len(valid) == 1:
            chosen.append(valid[0]); accepted.append(True)
            status.append("matched_documented_deployment")
        else:
            chosen.append(""); accepted.append(False)
            status.append("outside_or_ambiguous_deployment")
    return np.asarray(accepted), chosen, status


def clean_chunk(raw, path, declaration, references, source_sha256, source_row_start=1, latest="2026-11-09"):
    """Keep source-declared GPS observations, retaining exact source diagnostics."""
    spec = SourceSpec(**declaration["spec"])
    if spec.processing != "observed" or not declaration.get("evidence"):
        raise ValueError("GPS core requires documented observed-fix declaration")
    cols = _resolve(raw.columns, spec)
    if not {"individual", "species", "study", "timestamp"} <= cols.keys():
        raise ValueError("GPS core needs unambiguous source individual/species/study/time")
    get = lambda key: raw[cols[key]] if key in cols else pd.Series("", index=raw.index)
    stamps = _nominal(get("timestamp"))
    lat, lon = pd.to_numeric(get("latitude"), errors="coerce"), pd.to_numeric(get("longitude"), errors="coerce")
    sensor = get("sensor").map(_method)
    keep = sensor.eq("gps") & stamps.notna() & stamps.lt(pd.Timestamp(latest, tz="UTC"))
    keep &= lat.between(-90, 90) & lon.between(-180, 180) & get("individual").ne("") & get("species").ne("")
    reasons = Counter({"input_rows": len(raw), "non_gps_rows": int(sensor.ne("gps").sum()),
                       "invalid_coordinate_rows": int((~(lat.between(-90, 90) & lon.between(-180, 180))).sum()),
                       "unparseable_time_rows": int(stamps.isna().sum()),
                       "future_clock_rows": int(stamps.ge(pd.Timestamp(latest, tz="UTC")).sum())})
    truth, false = {"true", "1", "yes", "t"}, {"false", "0", "no", "f"}
    for flag in ["modelled", "interpolated", "smoothed", "manual_outlier", "algorithm_outlier"]:
        value = get(flag).str.strip().str.lower()
        bad = value.ne("") & ~value.isin(false)
        keep &= ~bad
        reasons[flag + "_or_unparsed_rows"] += int(bad.sum())
    visible = get("visible").str.strip().str.lower()
    bad_visible = visible.ne("") & ~visible.isin(truth)
    keep &= ~bad_visible
    reasons["invisible_or_unparsed_rows"] += int(bad_visible.sum())
    diagnostics_complete = pd.Series(True, index=raw.index)
    for rule in declaration.get("quality_rules", []):
        if not rule.get("evidence") or rule["column"] not in raw:
            raise ValueError("Documented source quality rule is missing evidence or source column")
        value = pd.to_numeric(raw[rule["column"]], errors="coerce")
        if rule["operator"] == ">":
            ok = value > rule["value"]
        elif rule["operator"] == ">=":
            ok = value >= rule["value"]
        elif rule["operator"] == "<=":
            ok = value <= rule["value"]
        elif rule["operator"] == "<":
            ok = value < rule["value"]
        else:
            raise ValueError("Unsupported documented QC operator")
        missing = value.isna()
        diagnostics_complete &= ~missing
        reasons["missing_quality_diagnostic:" + rule["column"]] += int(missing.sum())
        if rule.get("allow_missing", False):
            ok |= missing
        reasons["quality_rule:" + rule["column"]] += int((~ok).sum())
        keep &= ok
    attachment_ok, attachment, attachment_status = _attachment_metadata(raw, cols, references,
                                                    required=declaration.get("require_deployment_bounds", False))
    keep &= attachment_ok
    reasons["outside_or_unresolved_deployment_rows"] += int((~attachment_ok).sum())
    source, study, species, local = spec.source_id, get("study"), get("species"), get("individual")
    animal = source + ":" + study + ":" + species + ":" + local
    # These strings are required to match the exact historical event namespace.
    verified_time = get("timestamp").str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if spec.time_standard == "UTC":
        verified_time[:] = True
    out = pd.DataFrame({"timestamp_nominal": stamps, "timestamp_utc": stamps.where(verified_time),
                        "time_standard_status": np.where(verified_time, "verified_UTC_or_explicit_offset", spec.time_standard or "unknown"),
                        "lat": lat, "lon": lon, "animal": animal, "species": species,
                        "study": source + ":" + study, "source": source,
                        "original_animal": animal, "individual_id_reported": local,
                        "tag_id_reported": get("tag"), "event_id_reported": get("event"),
                        "timestamp_reported": get("timestamp"), "latitude_reported": get("latitude"),
                        "longitude_reported": get("longitude"), "sensor_reported": get("sensor"),
                        "observation_method": "gps", "processing": "observed",
                        "quality_diagnostics_complete": diagnostics_complete,
                        "source_file": str(path), "source_sha256": source_sha256,
                        "source_record": np.arange(source_row_start, source_row_start + len(raw)),
                        "reference_fields_json": attachment, "attachment_status": attachment_status})
    out = out.loc[keep].copy()
    selected = raw.loc[keep]
    out["reported_fields_json"] = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected.to_dict("records")]
    reasons["selected_rows"] += len(out)
    return out, dict(reasons)


def clean(roots, declarations, output, *, latest="2026-11-09", chunksize=50_000):
    """Write a new core; declarations are exact content-pinned, manually reviewed files."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("New marine core output must be empty")
    if isinstance(declarations, (str, Path)):
        declaration_path = Path(declarations)
        plan = json.loads(declaration_path.read_text())
        plan_provenance = {"path": str(declaration_path), "sha256": sha256(declaration_path)}
    else:
        plan, plan_provenance = declarations, {"embedded": declarations}
    files = {}
    for root in roots:
        for p in Path(root).rglob("*.csv"):
            if "reference" not in p.name.lower() and "originals" not in p.relative_to(root).parts:
                files[str(p)] = p
    selected = plan["files"]
    if len({r["path"] for r in selected}) != len(selected):
        raise ValueError("Source plan repeats an input path")
    for row in selected:
        p = Path(row["path"])
        if str(p) not in files or sha256(p) != row["sha256"]:
            raise ValueError("Source file missing from approved roots or content hash mismatch")
        for ev in row["evidence"]:
            if "path" in ev and sha256(ev["path"]) != ev["sha256"]:
                raise ValueError("Source declaration evidence hash mismatch")
    processed = output / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    staging = output / "selected-rows.parquet"
    writer, summaries, inventory = None, {}, []
    for declaration in selected:
        path = Path(declaration["path"])
        refs, ref_evidence = _reference_rows(path.parent)
        stats = Counter()
        position = 1
        for raw in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunksize):
            frame, reasons = clean_chunk(raw, path, declaration, refs, declaration["sha256"], position, latest)
            position += len(raw)
            stats.update(reasons)
            if not len(frame):
                continue
            for animal, group in frame.groupby("animal", sort=False):
                value = summaries.setdefault(animal, {"rows": 0, "first": group.timestamp_nominal.min(),
                                                     "last": group.timestamp_nominal.max(), "species": group.species.iloc[0]})
                value["rows"] += len(group)
                value["first"] = min(value["first"], group.timestamp_nominal.min())
                value["last"] = max(value["last"], group.timestamp_nominal.max())
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(staging, table.schema, compression="zstd")
            writer.write_table(table)
        inventory.append({"declaration": declaration, "source_qc": dict(stats), "references": ref_evidence})
    if writer is None:
        raise ValueError("No GPS observations survived documented source QC")
    writer.close()
    valid_animals = {name for name, stats in summaries.items() if stats["rows"] >= 2 and stats["last"] > stats["first"]}
    destination, writer = processed / "observations.parquet", None
    rows = 0
    for batch in pq.ParquetFile(staging).iter_batches(batch_size=chunksize):
        table = pa.Table.from_batches([batch])
        keep = pa.array([name in valid_animals for name in table["animal"].to_pylist()])
        table = table.filter(keep)
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
        writer.write_table(table)
        rows += len(table)
    writer.close()
    staging.unlink()
    animal_rows = []
    for animal, stats in summaries.items():
        animal_rows.append({"animal": animal, **stats, "span_days": (stats["last"] - stats["first"]).total_seconds() / 86400,
                            "trajectory_retained": animal in valid_animals})
    animal_table = pd.DataFrame(animal_rows)
    animal_table.to_parquet(processed / "animal-audit.parquet", index=False)
    manifest = {"schema": SCHEMA, "rows": rows, "animals": len(valid_animals),
                "species": sorted(animal_table.loc[animal_table.trajectory_retained, "species"].unique()),
                "roots": [str(r) for r in roots], "source_plan": plan_provenance,
                "source_inventory": inventory, "held_sources": plan.get("held", []),
                "observations_sha256": sha256(destination), "animal_audit_sha256": sha256(processed / "animal-audit.parquet"),
                "code_sha256": sha256(__file__), "latest_exclusive_nominal_date": latest,
                "timestamp_contract": "timestamp_nominal is source clock on an ordering axis; verified timestamp_utc is separate. No legacy timestamp column. V4 must explicitly adopt a tolerance policy for unresolved UTC/GPS times.",
                "quality_contract": "Source flags and documented rules only. All source strings and reference rows preserved. No interpolated targets, assumed metre accuracy, or cross-source coordinate merge.",
                "trajectory_rule": "At least two retained rows spanning positive nominal time; does not establish complete future horizons.",
                "limitations": ["Source declarations are supported by supplied papers/metadata, not universal independent verification",
                                "Reference deployment-off data are used only for attachment QC, never as forecast input",
                                "GPS observations retain device error and irregular, surfacing-dependent sampling",
                                "Identity links and sealed-cohort carryover must be audited before future corpus preparation"]}
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def exact_fix_links(new_path, old_paths):
    """Corroborated identity evidence; no fuzzy proximity or hardware-only links."""
    old_frames, old_manifest = [], []
    new_taxa = set(pq.read_table(new_path, columns=["species"]).column(0).unique().to_pylist())
    for path in old_paths:
        q = pq.ParquetFile(path)
        for batch in q.iter_batches(batch_size=200_000, columns=["species", "timestamp", "lat", "lon", "animal"]):
            frame = batch.to_pandas()
            frame = frame.loc[frame.species.isin(new_taxa)]
            if len(frame):
                old_frames.append(frame)
        old_manifest.append({"path": str(path), "sha256": sha256(path)})
    if not old_frames:
        return [], old_manifest
    old = pd.concat(old_frames, ignore_index=True).drop_duplicates()
    old.timestamp = pd.to_datetime(old.timestamp, utc=True)
    taxa, matches = set(old.species), []
    for batch in pq.ParquetFile(new_path).iter_batches(batch_size=200_000,
                        columns=["species", "timestamp_nominal", "lat", "lon", "animal"]):
        frame = batch.to_pandas()
        frame = frame.loc[frame.species.isin(taxa)].rename(columns={"timestamp_nominal": "timestamp"})
        if len(frame):
            merge = frame.merge(old, on=["species", "timestamp", "lat", "lon"], suffixes=("_new", "_old"))
            merge = merge.loc[merge.animal_new.ne(merge.animal_old)]
            if len(merge):
                matches.append(merge)
    links = []
    if matches:
        joined = pd.concat(matches).drop_duplicates(["animal_new", "animal_old", "timestamp", "lat", "lon"])
        for (a, b), group in joined.groupby(["animal_new", "animal_old"], sort=True):
            span = (group.timestamp.max() - group.timestamp.min()).total_seconds() / 3600
            moved = len(group[["lat", "lon"]].drop_duplicates()) > 1
            if len(group) >= 5 and span >= 1 and moved:
                links.append({"a": a, "b": b, "evidence": "exact_species_nominal_time_and_GPS_overlap",
                              "shared_fixes": len(group), "span_hours": span})
    return links, old_manifest


def identity_audit(core, old_observations, reference_roots, frozen_manifest, prior_audit=None):
    """New immutable extension; preserve frozen split risk groups without scores."""
    core = Path(core)
    destination = core / "identity-audit.json"
    if destination.exists():
        raise ValueError("Identity extension is immutable; output already exists")
    new_path = core / "processed/observations.parquet"
    core_manifest = json.loads((core / "processed/manifest.json").read_text())
    if sha256(new_path) != core_manifest["observations_sha256"]:
        raise ValueError("Core observation hash mismatch")
    extension = reference_audit([new_path, *map(Path, old_observations)], reference_roots,
                                core / "reference-audit.json")
    exact, old_manifest = exact_fix_links(new_path, old_observations)
    frozen = json.loads(Path(frozen_manifest).read_text())
    prior = json.loads(Path(prior_audit).read_text()) if prior_audit else {}
    # All existing physical and uncertainty groups persist.  Unknown source IDs
    # are not declared fresh merely because no exact repeated fix was found.
    old_mapping = frozen.get("identity_mapping", {})
    new_names = set(pq.read_table(new_path, columns=["animal"]).column(0).unique().to_pylist())
    nodes = set(new_names) | set(old_mapping) | set(old_mapping.values())
    physical_pairs = list(extension.get("fold_groups", {}).items()) + list(prior.get("fold_groups", {}).items()) + list(old_mapping.items())
    physical_pairs += [(r["a"], r["b"]) for r in exact]
    for a, b in physical_pairs:
        nodes.update([a, b])
    parent = {name: name for name in nodes}
    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]; name = parent[name]
        return name
    def join(a, b):
        aa, bb = find(a), find(b)
        parent[max(aa, bb)] = min(aa, bb)
    for a, b in physical_pairs:
        join(a, b)
    physical = {name: find(name) for name in sorted(nodes)}
    risk_pairs = list(extension.get("uncertainty_fold_groups", {}).items()) + list(prior.get("uncertainty_fold_groups", {}).items())
    risk_pairs += list(frozen.get("exclusion_groups", {}).items())
    for a, b in risk_pairs:
        if a not in parent: parent[a] = a
        if b not in parent: parent[b] = b
        join(a, b)
    risk = {name: find(name) for name in sorted(parent)}
    old_assignments = frozen.get("assignments", {})
    old_by_group = defaultdict(list)
    for name, assignment in old_assignments.items():
        if name in risk:
            old_by_group[risk[name]].append({"animal": name, "assignment": assignment})
    bindings = []
    for name in sorted(new_names):
        matches = old_by_group.get(risk[name], [])
        statuses = {m["assignment"] for m in matches}
        bindings.append({"animal": name, "physical_group": physical[name], "exclusion_group": risk[name],
                         "existing_cohort_members": matches,
                         "status": "existing_sealed_identity_or_risk_group" if statuses & {"test_individual", "test_year"} else
                                   "existing_development_identity_or_risk_group" if matches else "candidate_new_not_certified_fresh"})
    report = {"schema": "wildcast-marine-identity-extension-v1", "source_core_manifest_sha256": sha256(core / "processed/manifest.json"),
              "frozen_manifest": {"path": str(frozen_manifest), "sha256": sha256(frozen_manifest)},
              "prior_reference_audit": {"path": str(prior_audit), "sha256": sha256(prior_audit)} if prior_audit else None,
              "reference_extension_sha256": sha256(core / "reference-audit.json"), "old_observations": old_manifest,
              "fold_groups": physical, "uncertainty_fold_groups": risk, "exact_fix_links": exact,
              "new_animal_cohort_bindings": bindings, "binding_counts": dict(Counter(r["status"] for r in bindings)),
              "freshness_exclusion_animals": sorted(r["animal"] for r in bindings if r["existing_cohort_members"]),
              "code_sha256": sha256(__file__), "scope": "Metadata and repeated raw fixes only; no forecast case labels, model errors or reserved scores accessed",
              "limitations": ["No source-local tag ID alone establishes cross-source physical identity",
                              "All cached references checked are enumerated in reference-audit.json; absent matches do not certify fresh animals",
                              "Existing sealed cohort matches must not become development animals",
                              "Plausible unresolved aliases share exclusion groups without merging coordinates"]}
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--declarations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latest", default="2026-11-09")
    parser.add_argument("--old-observations", nargs="+")
    parser.add_argument("--reference-roots", nargs="+")
    parser.add_argument("--frozen-manifest")
    parser.add_argument("--prior-audit")
    args = parser.parse_args()
    result = clean(args.roots, args.declarations, args.output, latest=args.latest)
    if args.old_observations:
        if not args.reference_roots or not args.frozen_manifest:
            parser.error("Identity extension needs --reference-roots and --frozen-manifest")
        identity = identity_audit(args.output, args.old_observations, args.reference_roots, args.frozen_manifest, args.prior_audit)
        result["identity_binding_counts"] = identity["binding_counts"]
    print(json.dumps({k: result[k] for k in ["rows", "animals", "species", "identity_binding_counts"] if k in result}, indent=2))


if __name__ == "__main__":
    main()
