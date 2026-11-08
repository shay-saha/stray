"""Explicit, source-bound clock compatibility for nominal GPS observation tables.

This adapter does not infer instrument accuracy.  Bounds describe only the
documented UTC-versus-GPS interpretation, conditional on the source clock being
one of those standards.  The original QC table and all original columns survive
unchanged; the compatibility ``timestamp`` may deliberately remain nominal.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = "wildcast-migration-clock-policy-v1"
MODES = {"row_explicit_utc", "declared_utc", "documented_gps", "bounded_utc_or_gps"}
ISO_SYNTAX = (r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
              r"(?:[ T](?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d{1,9})?)?"
              r"(?:Z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)?)?")
# Effective UTC dates, transcribed from NIST's historical positive leap seconds.
GPS_OFFSETS = [("1980-01-06", 0), ("1981-07-01", 1), ("1982-07-01", 2),
               ("1983-07-01", 3), ("1985-07-01", 4), ("1988-01-01", 5),
               ("1990-01-01", 6), ("1991-01-01", 7), ("1992-07-01", 8),
               ("1993-07-01", 9), ("1994-07-01", 10), ("1996-01-01", 11),
               ("1997-07-01", 12), ("1999-01-01", 13), ("2006-01-01", 14),
               ("2009-01-01", 15), ("2012-07-01", 16), ("2015-07-01", 17),
               ("2017-01-01", 18)]
STANDARDS = {
    "reviewed_on": "2026-11-08", "valid_before": "2026-11-09T00:00:00Z",
    "gps_minus_utc_effective_dates": GPS_OFFSETS,
    "references": [
        {"url": "https://vocab.nerc.ac.uk/collection/MVB/current/MVB000200/",
         "finding": "Movebank timestamp attribute allows UTC or GPS time; repository membership alone does not resolve the standard."},
        {"url": "https://www.movebank.org/cms/movebank-content/import-custom-tabular-data",
         "finding": "Movebank's import mapping converts declared dates and time zones into UTC."},
        {"url": "https://github.com/movebank/movebank-api-doc/blob/master/movebank-api.md",
         "finding": "Official API documentation declares stored/API event dates UTC; applying this requires API/export provenance."},
        {"url": "https://www.nist.gov/pml/time-and-frequency-division/time-realization/leap-seconds",
         "finding": "Historical UTC leap-second table supplies the effective dates."},
        {"url": "https://maia.usno.navy.mil/information/eo-values",
         "finding": "GPS minus UTC is 18 seconds from 2017-01-01."},
    ],
    "meaning": "Time-standard interpretation only; neither device clock accuracy nor spatial error is inferred.",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def _check_binding(binding):
    path = Path(binding["path"])
    if not path.is_file() or sha256(path) != binding["sha256"]:
        raise ValueError("Clock provenance content hash mismatch: " + str(path))


def _qc_dependencies(path, manifest):
    """Pin small QC policy/reference reports as well as the observation bytes."""
    dependencies = []
    if manifest.get("animal_audit_sha256"):
        dependencies.append({"path": str((Path(path).parent / "animal-audit.parquet").resolve()),
                             "sha256": manifest["animal_audit_sha256"]})
    if manifest.get("policy"):
        dependencies.append({"path": str(Path(manifest["policy"]).resolve()), "sha256": manifest["policy_sha256"]})
    if manifest.get("source_plan"):
        dependencies.append({"path": str(Path(manifest["source_plan"]["path"]).resolve()),
                             "sha256": manifest["source_plan"]["sha256"]})
    for source in manifest.get("sources", []):
        if source.get("report"):
            dependencies.append({"path": str(Path(source["report"]).resolve()), "sha256": source["report_sha256"]})
    for binding in dependencies:
        _check_binding(binding)
    return dependencies


def _source_policy(source, record):
    record = json.loads(json.dumps(record))
    if record.get("mode") not in MODES:
        raise ValueError("Unsupported or unbounded source clock policy: " + source)
    clock = record.get("documented_clock")
    required_clock = {"declared_utc": "UTC", "documented_gps": "GPS",
                      "bounded_utc_or_gps": "UTC_or_GPS"}
    if record["mode"] in required_clock and clock != required_clock[record["mode"]]:
        raise ValueError("Source policy contradicts documented clock: " + source)
    evidence = record.get("evidence", [])
    if not evidence or not all(e.get("claim") and e.get("kind") and e.get("path") for e in evidence):
        raise ValueError("Every source clock policy needs content-bound evidence")
    valid_kinds = {
        "declared_utc": {"source_declared_utc", "movebank_api_utc"},
        "documented_gps": {"source_declared_gps"},
        "bounded_utc_or_gps": {"source_utc_or_gps", "movebank_attribute_utc_or_gps"},
        "row_explicit_utc": {"row_explicit_utc", "source_declared_utc", "movebank_api_utc"},
    }
    if not any(e["kind"] in valid_kinds[record["mode"]] for e in evidence):
        raise ValueError("Evidence does not establish the chosen source clock policy")
    for e in evidence:
        bound = _bind(e["path"])
        if e.get("sha256") not in (None, bound["sha256"]):
            raise ValueError("Source clock evidence hash mismatch")
        e.update(bound)
    record.pop("source_policy_sha256", None)
    record["source_policy_sha256"] = fingerprint(record)
    return record


def make_policy(manifests, sources, *, train_end_year=2020, validation_year=2021,
                minimum_horizon_days=1.):
    """Bind reviewed per-source decisions and QC inputs, without exporting rows.

    Sources are explicit decisions, never selected by geographic or target data.
    ``declared_utc`` requires source-specific or API evidence; a generic MDR
    timestamp dictionary supports only ``bounded_utc_or_gps``.
    """
    if isinstance(train_end_year, bool) or not isinstance(train_end_year, int):
        raise ValueError("Train end year must be an integer")
    if validation_year != train_end_year + 1:
        raise ValueError("Clock policy requires consecutive pinned calendar cutoffs")
    if not np.isfinite(minimum_horizon_days) or minimum_horizon_days <= 0:
        raise ValueError("Minimum forecast horizon must be positive")
    inputs = []
    for path in manifests:
        binding = _bind(path)
        manifest = json.loads(Path(path).read_text())
        if manifest.get("schema") not in {"wildcast-breadth-clean-v1", "wildcast-marine-gps-core-v1"}:
            # The marine cleaner's published schema spelling is checked below too.
            if manifest.get("schema") != "wildcast-marine-gps-clean-v1":
                raise ValueError("Unsupported nominal-clock QC manifest")
        observations = Path(path).parent / "observations.parquet"
        ob = _bind(observations)
        if ob["sha256"] != manifest.get("observations_sha256"):
            raise ValueError("QC observations hash mismatch")
        inputs.append({"manifest": binding, "observations": ob,
                       "QC_dependencies": _qc_dependencies(path, manifest)})
    if len({i["observations"]["path"] for i in inputs}) != len(inputs):
        raise ValueError("Duplicate nominal-clock input")
    policy = {"schema": SCHEMA, "implementation_sha256": sha256(__file__),
              "inputs": inputs, "sources": {str(s): _source_policy(str(s), r) for s, r in sources.items()},
              "standards": json.loads(json.dumps(STANDARDS)), "standards_sha256": fingerprint(STANDARDS),
              "train_end_year": train_end_year, "validation_year": validation_year,
              "cutoff_boundaries": [f"{train_end_year+1}-01-01T00:00:00Z",
                                    f"{validation_year+1}-01-01T00:00:00Z"],
              "boundary_rule": "exclude rows when possible UTC interpretations straddle a pinned cutoff",
              "minimum_horizon_days": float(minimum_horizon_days),
              "reported_timestamp_syntax": ISO_SYNTAX,
              "bounded_mode_axis": "unchanged nominal values, explicitly not verified UTC",
              "relative_time_contract": "Nominal within-track elapsed values are unchanged; their physical accuracy and constancy of an unresolved clock choice are not asserted.",
              "instrument_clock_accuracy": "unknown unless independently supplied by the source; no universal accuracy added"}
    policy["policy_sha256"] = fingerprint(policy)
    return policy


def validate_policy(policy, *, check_inputs=True):
    unsigned = {k: v for k, v in policy.items() if k != "policy_sha256"}
    if policy.get("schema") != SCHEMA or fingerprint(unsigned) != policy.get("policy_sha256"):
        raise ValueError("Clock policy content hash mismatch")
    if policy.get("implementation_sha256") != sha256(__file__):
        raise ValueError("Clock adapter implementation changed after policy pinning")
    if (policy.get("standards_sha256") != fingerprint(STANDARDS) or
            fingerprint(policy.get("standards")) != fingerprint(STANDARDS)):
        raise ValueError("Clock standard table changed")
    if policy.get("validation_year") != policy.get("train_end_year", 0) + 1:
        raise ValueError("Clock cutoffs are inconsistent")
    expected = [f"{policy['train_end_year']+1}-01-01T00:00:00Z",
                f"{policy['validation_year']+1}-01-01T00:00:00Z"]
    if policy.get("cutoff_boundaries") != expected:
        raise ValueError("Clock cutoff handling is not pinned")
    if policy.get("reported_timestamp_syntax") != ISO_SYNTAX:
        raise ValueError("Reported timestamp syntax differs from the pinned adapter")
    for source, record in policy["sources"].items():
        rebuilt = _source_policy(source, record)
        if rebuilt != record:
            raise ValueError("Per-source clock policy content hash mismatch")
    if check_inputs:
        for binding in policy["inputs"]:
            _check_binding(binding["manifest"])
            _check_binding(binding["observations"])
            for dependency in binding.get("QC_dependencies", []):
                _check_binding(dependency)


def _gps_interpretation(nominal):
    """Convert documented GPS calendar readings; hold POSIX-unrepresentable leap seconds."""
    ns = nominal.astype("int64").to_numpy()
    utc_edges = pd.to_datetime([d for d, _ in GPS_OFFSETS], utc=True).asi8
    offsets = np.asarray([o for _, o in GPS_OFFSETS], dtype=np.int64) * 10**9
    gps_edges = utc_edges + offsets
    index = np.searchsorted(gps_edges, ns, side="right") - 1
    valid = (index >= 0) & (ns < pd.Timestamp(STANDARDS["valid_before"]).value)
    safe = np.clip(index, 0, len(offsets)-1)
    result = ns - offsets[safe]
    # One GPS second at each positive UTC leap has no pandas/POSIX representation.
    following = np.minimum(safe + 1, len(utc_edges)-1)
    leap = (safe < len(utc_edges)-1) & (result >= utc_edges[following])
    valid &= ~leap
    return pd.Series(pd.to_datetime(result, utc=True), index=nominal.index), valid


def adapt_frame(frame, policy):
    """Return a copy plus exclusion counts; never mutate QC columns or infer precision."""
    validate_policy(policy, check_inputs=False)
    required = {"source", "animal", "timestamp_nominal", "timestamp_utc", "time_standard_status", "timestamp_reported"}
    if not required <= set(frame) or "timestamp" in frame:
        raise ValueError("Expected a nominal-clock QC table without legacy timestamp")
    if not frame.index.is_unique:
        raise ValueError("Clock input row index must be unique")
    out = frame.copy(deep=True)
    nominal = pd.to_datetime(frame.timestamp_nominal, utc=True, errors="raise").dt.as_unit("ns")
    explicit = pd.to_datetime(frame.timestamp_utc, utc=True, errors="raise").dt.as_unit("ns")
    if nominal.isna().any():
        raise ValueError("Null nominal timestamp cannot be adapted")
    if (explicit.notna() & explicit.ne(nominal)).any():
        raise ValueError("Conflicting explicit and nominal timestamps require source review")
    if frame.source.isna().any() or not set(frame.source.astype(str)) <= set(policy["sources"]):
        raise ValueError("Unreviewed source clock policy")
    allowed_status = {"UTC_or_GPS", "GPS", "unknown", "verified_UTC_or_explicit_offset"}
    if not set(frame.time_standard_status) <= allowed_status:
        raise ValueError("Unbounded or unsupported clock status")
    if (explicit.notna() != frame.time_standard_status.eq("verified_UTC_or_explicit_offset")).any():
        raise ValueError("Explicit UTC values and clock status disagree")
    out["timestamp"] = nominal
    out["utc_interpretation_lower"] = nominal
    out["utc_interpretation_upper"] = nominal
    out["clock_axis_status"] = ""
    out["clock_source_policy_sha256"] = ""
    reasons = pd.Series("", index=out.index)
    reported = frame.timestamp_reported.astype(str).str.strip()
    syntax_ok = reported.str.fullmatch(ISO_SYNTAX)
    parsed = pd.to_datetime(reported.where(syntax_ok), format="ISO8601", errors="coerce", utc=True).dt.as_unit("ns")
    bad_reported = ~syntax_ok | parsed.isna() | parsed.ne(nominal)
    reasons.loc[bad_reported] = "malformed_or_conflicting_reported_timestamp"
    for source, positions in frame.groupby("source", sort=False).groups.items():
        record = policy["sources"][str(source)];mode = record["mode"]
        stamps = nominal.loc[positions]
        out.loc[positions, "clock_source_policy_sha256"] = record["source_policy_sha256"]
        if mode == "row_explicit_utc":
            if explicit.loc[positions].isna().any():
                raise ValueError("Source requires explicit UTC rows")
            out.loc[positions, "clock_axis_status"] = "row_declared_UTC_accuracy_unknown"
        elif mode == "declared_utc":
            if frame.loc[positions, "time_standard_status"].eq("GPS").any():
                raise ValueError("GPS row contradicts UTC source declaration")
            out.loc[positions, "clock_axis_status"] = "source_declared_UTC_accuracy_unknown"
        elif mode in {"documented_gps", "bounded_utc_or_gps"}:
            unresolved = positions[explicit.loc[positions].isna().to_numpy()]
            if mode == "documented_gps" and len(unresolved) != len(positions):
                raise ValueError("Explicit UTC row contradicts documented GPS source")
            converted, valid = _gps_interpretation(nominal.loc[unresolved])
            held = unresolved[(~valid) & reasons.loc[unresolved].eq("").to_numpy()]
            reasons.loc[held] = "outside_verified_standard_table_or_UTC_leap_second"
            out.loc[unresolved, "utc_interpretation_lower"] = converted
            if mode == "documented_gps":
                out.loc[unresolved, "timestamp"] = converted
                out.loc[unresolved, "utc_interpretation_upper"] = converted
                out.loc[positions, "clock_axis_status"] = "documented_GPS_converted_to_UTC_accuracy_unknown"
            else:
                if frame.loc[unresolved, "time_standard_status"].eq("unknown").any():
                    # Explicit reviewed source documentation, not the row default,
                    # must bound an otherwise unknown row to UTC-or-GPS.
                    if record["documented_clock"] != "UTC_or_GPS":
                        raise ValueError("Unbounded source clock ambiguity")
                out.loc[positions, "clock_axis_status"] = "nominal_bounded_UTC_or_GPS_not_verified_UTC"
                out.loc[positions[explicit.loc[positions].notna().to_numpy()], "clock_axis_status"] = "row_declared_UTC_accuracy_unknown"
    for boundary in policy["cutoff_boundaries"]:
        edge = pd.Timestamp(boundary)
        crosses = out.utc_interpretation_lower.lt(edge) & out.utc_interpretation_upper.ge(edge)
        reasons.loc[crosses & reasons.eq("")] = "UTC_interpretations_straddle_pinned_cutoff"
    out["time_standard_ambiguity_seconds"] = (out.utc_interpretation_upper-out.utc_interpretation_lower).dt.total_seconds()
    out["clock_policy_sha256"] = policy["policy_sha256"]
    counts = Counter(reasons[reasons.ne("")])
    return out.loc[reasons.eq("")].copy(), {"input_rows": len(frame), "output_rows": int(reasons.eq("").sum()),
                                           "excluded": dict(counts)}


def write_policy(path, policy):
    validate_policy(policy)
    with Path(path).open("x") as stream:
        json.dump(policy, stream, indent=2, sort_keys=True, allow_nan=False);stream.write("\n")


def export(policy_path, output, *, batch_size=50_000):
    """Create a new compatibility dataset after review; refuse overwrite or changed inputs."""
    policy_path, output = Path(policy_path), Path(output)
    policy_file_sha256 = sha256(policy_path)
    policy = json.loads(policy_path.read_text());validate_policy(policy)
    if sha256(policy_path) != policy_file_sha256:
        raise ValueError("Clock policy file changed while being loaded")
    if output.exists():
        raise ValueError("Clock compatibility output is immutable; choose a new directory")
    if not policy["inputs"] or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Clock export requires bound QC inputs and a positive batch size")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name+".pending-", dir=output.parent))
    writer = None
    try:
        processed = stage / "processed";processed.mkdir()
        destination = processed / "observations.parquet"
        counts = Counter();excluded = Counter();axis = Counter();sources_seen = set()
        additions = pa.schema([pa.field("timestamp", pa.timestamp("ns", "UTC")),
                               pa.field("utc_interpretation_lower", pa.timestamp("ns", "UTC")),
                               pa.field("utc_interpretation_upper", pa.timestamp("ns", "UTC")),
                               pa.field("clock_axis_status", pa.string()),
                               pa.field("clock_source_policy_sha256", pa.string()),
                               pa.field("time_standard_ambiguity_seconds", pa.float64()),
                               pa.field("clock_policy_sha256", pa.string())])
        schema = pa.unify_schemas([*(pq.ParquetFile(i["observations"]["path"]).schema_arrow
                                    for i in policy["inputs"]), additions], promote_options="permissive")
        max_ambiguity = 0.
        for binding in policy["inputs"]:
            for batch in pq.ParquetFile(binding["observations"]["path"]).iter_batches(batch_size=batch_size):
                frame, audit = adapt_frame(batch.to_pandas(), policy)
                counts.update({k: audit[k] for k in ("input_rows", "output_rows")});excluded.update(audit["excluded"])
                sources_seen.update(frame.source.astype(str));axis.update(frame.clock_axis_status)
                if frame.empty:
                    continue
                max_ambiguity = max(max_ambiguity, float(frame.time_standard_ambiguity_seconds.max()))
                table = pa.Table.from_pandas(frame, preserve_index=False)
                table = pa.Table.from_arrays([table[field.name].cast(field.type) if field.name in table.column_names
                                              else pa.nulls(len(table), type=field.type) for field in schema], schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(destination, schema, compression="zstd")
                writer.write_table(table)
        if writer is None or not counts["output_rows"]:
            raise ValueError("No rows survive the reviewed clock policy")
        writer.close();writer = None
        # Recheck immutable inputs/evidence after streaming, before publishing.
        validate_policy(policy)
        if sha256(policy_path) != policy_file_sha256:
            raise ValueError("Clock policy file changed during export")
        audit = {"schema": "wildcast-migration-clock-compatibility-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                 "policy": _bind(policy_path), "policy_content_sha256": policy["policy_sha256"],
                 "implementation_sha256": sha256(__file__), "inputs": policy["inputs"],
                 "observations_sha256": sha256(destination), "counts": dict(counts), "excluded": dict(excluded),
                 "clock_axis_rows": dict(axis), "sources": sorted(sources_seen),
                 "maximum_time_standard_ambiguity_seconds": max_ambiguity,
                 "ambiguity_fraction_of_minimum_horizon": max_ambiguity / (policy["minimum_horizon_days"] * 86400),
                 "train_end_year": policy["train_end_year"], "validation_year": policy["validation_year"],
                 "cutoff_boundaries": policy["cutoff_boundaries"], "original_QC_unchanged": True,
                 "timestamp_contract": "Compatibility timestamp follows each hashed source policy; bounded UTC/GPS rows retain nominal values. timestamp_utc and original status are preserved, not upgraded.",
                 "limitations": ["Time-standard bounds do not bound device clock error or spatial precision.",
                                 "Relative elapsed values of ambiguous nominal tracks remain numerically unchanged; absolute UTC is not resolved.",
                                 "No targets, cohort assignments, biological phases or model predictions are selected by this adapter."]}
        (processed / "manifest.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False)+"\n")
        (stage / "clock-policy.json").write_bytes(policy_path.read_bytes())
        # rename refuses a non-empty existing destination; the explicit check
        # also covers an empty output directory that should remain immutable.
        if output.exists():
            raise ValueError("Clock compatibility output appeared during export")
        stage.rename(output)
        return audit
    finally:
        if writer is not None:
            writer.close()
        if stage.exists():
            shutil.rmtree(stage)


def check_compatibility(root, *, train_end_year, validation_year, horizons=None):
    """Preflight for the caller of migration_data.prepare; refuses cutoff drift."""
    root = Path(root)
    manifest_path = root / "processed/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "wildcast-migration-clock-compatibility-v1":
        raise ValueError("Expected a clock compatibility dataset")
    policy_path = root / "clock-policy.json"
    policy = json.loads(policy_path.read_text());validate_policy(policy)
    if (sha256(policy_path) != manifest["policy"]["sha256"] or
            policy["policy_sha256"] != manifest["policy_content_sha256"]):
        raise ValueError("Compatibility clock policy hash mismatch")
    if (train_end_year != policy["train_end_year"] or validation_year != policy["validation_year"] or
            train_end_year != manifest["train_end_year"] or validation_year != manifest["validation_year"]):
        raise ValueError("Preparation cutoffs differ from pinned clock cutoff policy")
    if horizons is not None:
        horizons = np.asarray(horizons, dtype=float)
        if (not horizons.size or not np.isfinite(horizons).all() or
                horizons.min() < policy["minimum_horizon_days"]):
            raise ValueError("Preparation horizons are shorter than the pinned clock policy")
    if sha256(root / "processed/observations.parquet") != manifest["observations_sha256"]:
        raise ValueError("Compatibility observations hash mismatch")
    return {"manifest": _bind(manifest_path), "policy": _bind(policy_path),
            "policy_content_sha256": policy["policy_sha256"],
            "train_end_year": train_end_year, "validation_year": validation_year}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("policy");p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--sources", required=True);p.add_argument("--output", required=True)
    p.add_argument("--train-end-year", type=int, default=2020);p.add_argument("--validation-year", type=int, default=2021)
    p.add_argument("--minimum-horizon-days", type=float, default=1.)
    p = sub.add_parser("export");p.add_argument("--policy", required=True);p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()
    if args.command == "policy":
        policy = make_policy(args.manifest, json.loads(Path(args.sources).read_text()),
                             train_end_year=args.train_end_year, validation_year=args.validation_year,
                             minimum_horizon_days=args.minimum_horizon_days)
        write_policy(args.output, policy)
        print(json.dumps({"policy_sha256": policy["policy_sha256"], "sources": len(policy["sources"])}))
    else:
        audit = export(args.policy, args.output, batch_size=args.batch_size)
        print(json.dumps({"counts": audit["counts"], "observations_sha256": audit["observations_sha256"]}))


if __name__ == "__main__":
    main()
