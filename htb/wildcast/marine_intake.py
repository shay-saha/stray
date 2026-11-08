"""Offline, lossless observation intake; this does not build training targets.

Original CSV fields survive as ``reported_fields_json``.  A source declaration
and row flags describe how a location was obtained, independently of its sensor
name.  Argos classes are categorical diagnostics, never invented Gaussian SDs.
Ellipses are reported geometry, not assumed covariance or confidence contours.

Movebank attribute semantics: https://www.movebank.org/cms/movebank-content/
movebank-attribute-dictionary .  In particular ``modelled`` includes changed,
interpolated and model-output positions.  Declaring a file "observed" therefore
never overrides a positive row-level processing flag.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd


SCHEMA = "wildcast-marine-observations-v1"
DICTIONARY = "https://www.movebank.org/cms/movebank-content/movebank-attribute-dictionary"
PROCESSING = {"unknown", "observed", "modelled", "interpolated", "smoothed", "state_space", "kalman"}
METHODS = {"unknown", "gps", "argos_doppler", "acoustic", "light_geolocation", "radio_telemetry", "other"}
ALIASES = {
    "timestamp": ["timestamp"], "date": [], "time": [],
    "latitude": ["location-lat", "location.lat"],
    "longitude": ["location-long", "location.long"],
    "individual": ["individual-local-identifier", "individual.local.identifier"],
    "tag": ["tag-local-identifier", "tag.local.identifier"],
    "deployment": ["deployment-id", "deployment_id"],
    "species": ["individual-taxon-canonical-name", "individual.taxon.canonical.name"],
    "study": ["study-name", "study.name"], "event": ["event-id"],
    "sensor": ["sensor-type", "sensor.type"],
    "modelled": ["modelled"], "interpolated": ["interpolated"], "smoothed": ["smoothed"],
    "visible": ["visible"],
    "manual_outlier": ["manually-marked-outlier"], "algorithm_outlier": ["algorithm-marked-outlier"],
    "argos_lc": ["argos:lc", "argos.lc"],
    "ellipse_major": ["argos:semi-major", "argos.semi.major"],
    "ellipse_minor": ["argos:semi-minor", "argos.semi.minor"],
    "ellipse_orientation": ["argos:orientation", "argos.orientation"],
    "error_radius": ["argos:error-radius", "argos.error.radius", "argos:location-error-radius", "argos.location.error.radius"],
    "gps_hdop": ["gps:hdop", "gps.hdop"], "gps_vdop": ["gps:vdop", "gps.vdop"],
    "gps_pdop": ["gps:dop", "gps:pdop", "gps.dop"],
    "satellites": ["gps:satellite-count", "gps.satellite.count"],
    "receiver": ["receiver-id", "receiver_id"],
    "argos_algorithm": ["argos:location-processing", "argos:location-algorithm"],
}


@dataclass(frozen=True)
class SourceSpec:
    """Documented file-level facts; absent facts remain explicitly unknown.

    ``columns`` maps canonical fields above to exact source header strings.
    ``units`` may specify ellipse_major/minor/error_radius (m or km) and
    ellipse_orientation (degrees).  Orientation convention is a separate fact.
    Defaults never infer GPS, raw processing, UTC, licence, or measurement SD.
    """
    source_id: str
    format: str = "movebank"
    method: str = "unknown"
    processing: str = "unknown"
    timezone: str | None = None
    time_standard: str | None = None
    columns: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    orientation_convention: str | None = None
    species: str | None = None
    study: str | None = None
    sensor_variant: str | None = None
    argos_algorithm: str | None = None
    doi: str | None = None
    license: str | None = None
    evidence_urls: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.source_id.strip() or self.format not in {"movebank", "declared_csv"}:
            raise ValueError("source_id and supported explicit format are required")
        if self.method not in METHODS or self.processing not in PROCESSING:
            raise ValueError("Unknown declared method or processing category")
        if self.time_standard not in {None, "UTC", "GPS", "UTC_or_GPS"}:
            raise ValueError("Unsupported time standard declaration")
        if set(self.columns) - ALIASES.keys():
            raise ValueError("Unknown canonical column name")
        if set(self.units) - {"ellipse_major", "ellipse_minor", "ellipse_orientation", "error_radius"}:
            raise ValueError("Unsupported uncertainty unit declaration")
        for key, unit in self.units.items():
            if unit not in ({"degrees"} if key == "ellipse_orientation" else {"m", "km"}):
                raise ValueError("Unsupported reported uncertainty units")
        if self.timezone is not None:
            pd.Timestamp("2020-01-01").tz_localize(self.timezone)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flag(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "t"}:
        return True
    if value in {"false", "0", "no", "f"}:
        return False
    return None


def _method(value):
    value = str(value).strip().lower()
    if value in {"gps", "gnss", "fastloc-gps", "fastloc gps"}:
        return "gps"
    if value in {"argos-doppler-shift", "argos doppler", "argos_doppler"}:
        return "argos_doppler"
    if value in {"acoustic", "acoustic-telemetry", "acoustic telemetry"}:
        return "acoustic"
    if value in {"light-level", "light-level-geolocator", "light_geolocation", "solar-geolocator"}:
        return "light_geolocation"
    if value in {"radio-transmitter", "radio-telemetry"}:
        return "radio_telemetry"
    return "other" if value else "unknown"


def _timestamp(value, timezone, time_standard=None):
    if not value.strip():
        return pd.NaT, "missing"
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return pd.NaT, "invalid"
        if stamp.tzinfo is None:
            if time_standard in {"GPS", "UTC_or_GPS"}:
                return pd.NaT, "unresolved_time_standard:" + time_standard
            if timezone is None:
                return pd.NaT, "timezone_unknown"
            stamp = stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
            status = "source_declared_timezone"
        else:
            status = "reported_timezone"
        return stamp.tz_convert("UTC"), status
    except (ValueError, TypeError, OverflowError):
        return pd.NaT, "invalid"


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def _namespace(source, study, individual, tag, deployment):
    # Distinguish unknown individual/tag IDs and preserve delimiters and leading zeros.
    identifier = individual or tag
    if not identifier:
        return None
    kind = "individual" if individual else "tag_only"
    return "/".join(quote(str(v), safe="") for v in [source, study, kind, identifier, deployment])


def _resolve(headers, spec):
    columns = {}
    for key in ALIASES:
        if key in spec.columns:
            if spec.columns[key] not in headers:
                raise ValueError(f"Declared column absent: {spec.columns[key]}")
            columns[key] = spec.columns[key]
        elif spec.format == "movebank":
            found = [name for name in ALIASES[key] if name in headers]
            if len(found) > 1:
                raise ValueError(f"Ambiguous source columns for {key}; declare one explicitly")
            if found:
                columns[key] = found[0]
    if not {"latitude", "longitude"} <= columns.keys() or not ({"timestamp", "date"} & columns.keys()):
        raise ValueError("No declared location/time schema; reference tables are not event tables")
    return columns


def read_observations(path, spec: SourceSpec | dict):
    """Return every source row, including invalid, flagged and estimated positions.

    Eligibility is a conservative intake classification, not biological QC or a
    training decision.  Unknown/invalid flags are retained and block eligibility.
    Processing declarations need documentary support; the reader cannot verify a
    paper's truthfulness.  No movement interpolation or source alias linking occurs.
    """
    spec = spec if isinstance(spec, SourceSpec) else SourceSpec(**spec)
    path = Path(path)
    digest = sha256(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers):
            raise ValueError("Duplicate source headers cannot be represented losslessly")
        columns = _resolve(headers, spec)
        rows = []
        for row_number, raw in enumerate(reader, 1):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Malformed CSV record {row_number}")
            get = lambda key: raw.get(columns.get(key), "")
            reported_time = get("timestamp") or " ".join(v for v in [get("date"), get("time")] if v)
            stamp, time_status = _timestamp(reported_time, spec.timezone, spec.time_standard)
            lat, lon = _number(get("latitude")), _number(get("longitude"))
            coord_valid = bool(np.isfinite([lat, lon]).all() and -90 <= lat <= 90 and -180 <= lon <= 180)
            method = _method(get("sensor")) if get("sensor").strip() else spec.method
            conflict = method != spec.method and spec.method != "unknown" and method != "unknown"
            flags = {key: _flag(get(key)) for key in ["modelled", "interpolated", "smoothed", "visible", "manual_outlier", "algorithm_outlier"]}
            invalid_flags = [key for key in flags if get(key).strip() and flags[key] is None]
            processing = spec.processing
            if flags["modelled"] is True:
                processing = "modelled"
            if flags["smoothed"] is True:
                processing = "smoothed"
            if flags["interpolated"] is True:
                processing = "interpolated"
            if processing == "unknown" and flags["modelled"] is False:
                processing = "observed"
            issues = []
            if not coord_valid:
                issues.append("invalid_coordinates")
            if pd.isna(stamp):
                issues.append(time_status)
            if flags["visible"] is False:
                issues.append("source_invisible")
            issues.extend(key for key in ["manual_outlier", "algorithm_outlier"] if flags[key] is True)
            issues.extend("unparsed_flag:" + key for key in invalid_flags)
            if conflict:
                issues.append("sensor_declaration_conflict")
            lc = get("argos_lc").strip().upper()
            if method == "argos_doppler" and lc == "Z":
                issues.append("argos_failed_location")
            if method == "argos_doppler" and lc == "G":
                # Argos LC G denotes a GPS-origin fix, requiring source-specific
                # reconciliation with a simultaneous Doppler sensor declaration.
                issues.append("argos_gps_class_sensor_conflict")
            if lc and lc not in {"3", "2", "1", "0", "A", "B", "Z", "G"}:
                issues.append("unrecognized_argos_class")
            clean = not issues
            observed = processing == "observed"
            kind = ("receiver_detection" if method == "acoustic" else
                    "gps_fix" if method == "gps" and observed else
                    "uncertain_location" if method == "argos_doppler" and observed else
                    "estimated_location" if processing not in {"observed", "unknown"} else "unresolved_location")
            individual, tag, study = get("individual"), get("tag"), get("study") or spec.study or ""
            record = {
                "schema": SCHEMA, "source_id": spec.source_id, "source_file": str(path),
                "source_sha256": digest, "source_record": row_number,
                "event_id_reported": get("event"), "individual_id_reported": individual,
                "tag_id_reported": tag, "deployment_id_reported": get("deployment"),
                "individual_key": _namespace(spec.source_id, study, individual, tag, get("deployment")),
                "identity_basis": "individual" if individual else "tag_only" if tag else "missing",
                "species": get("species") or spec.species or "", "study_reported": study,
                "reported_timestamp": reported_time, "reported_latitude": get("latitude"),
                "reported_longitude": get("longitude"), "timestamp_utc": stamp,
                "time_status": time_status, "time_standard_declared": spec.time_standard,
                "latitude": lat, "longitude": lon,
                "coordinates_valid": coord_valid, "sensor_reported": get("sensor"),
                "observation_method": method, "sensor_variant_declared": spec.sensor_variant,
                "processing": processing, "record_kind": kind,
                "coordinate_role": "receiver" if method == "acoustic" else "animal_location_estimate",
                "argos_lc_reported": get("argos_lc"),
                "argos_algorithm_reported": get("argos_algorithm") or spec.argos_algorithm,
                "receiver_id_reported": get("receiver"),
                "source_flags_json": json.dumps({key: get(key) for key in flags}, sort_keys=True),
                "quality_issues_json": json.dumps(issues),
                "reported_fields_json": json.dumps(raw, sort_keys=True, ensure_ascii=False),
                "gps_observation_eligible": clean and observed and method == "gps",
                "uncertain_observation_eligible": clean and observed and method == "argos_doppler",
                "is_direct_observation": clean and observed and method in {"gps", "argos_doppler"},
            }
            # Preserve both raw quantities and declared units.  Standardized metres
            # appear ONLY with explicit compatible units, never from LC or DOP.
            for key in ["ellipse_major", "ellipse_minor", "ellipse_orientation", "error_radius"]:
                raw_value, unit = get(key), spec.units.get(key)
                record[key + "_reported"] = raw_value
                record[key + "_units"] = unit
                value = _number(raw_value)
                record[key + "_m" if key != "ellipse_orientation" else key + "_degrees"] = (
                    value * (1000 if unit == "km" else 1) if unit and np.isfinite(value) and value >= 0 else np.nan)
            record["ellipse_orientation_convention"] = spec.orientation_convention
            record["uncertainty_model"] = "not_assumed"
            record["gps_hdop_reported"], record["gps_vdop_reported"] = get("gps_hdop"), get("gps_vdop")
            record["gps_pdop_reported"], record["gps_satellites_reported"] = get("gps_pdop"), get("satellites")
            rows.append(record)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    frame.attrs.update({"source_spec": asdict(spec), "source_sha256": digest, "resolved_columns": columns})
    return frame


def summarize(frame):
    if not len(frame):
        return {"rows": 0, "groups": []}
    groups = []
    for key, group in frame.groupby(["species", "observation_method", "processing", "record_kind"], dropna=False, sort=True):
        groups.append(dict(zip(["species", "method", "processing", "record_kind"], key)) | {
            "rows": len(group), "source_individual_ids": group.individual_key.nunique(),
            "gps_eligible_rows": int(group.gps_observation_eligible.sum()),
            "uncertain_eligible_rows": int(group.uncertain_observation_eligible.sum()),
            "reported_argos_classes": group.argos_lc_reported.value_counts().to_dict(),
            "reported_ellipse_rows": int(group.ellipse_major_reported.ne("").sum()),
            "known_timezone_rows": int(group.timestamp_utc.notna().sum()),
        })
    return {"rows": len(frame), "groups": groups,
            "note": "Source IDs are not cross-source deduplicated animals; eligibility is intake classification, not final QC. No targets or trajectories are generated."}


def ingest(path, spec, output):
    """Write a new immutable intake table and auditable source declaration."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Intake output must be empty; preserve previous provenance")
    frame = read_observations(path, spec)
    output.mkdir(parents=True, exist_ok=True)
    table = output / "observations.parquet"
    frame.to_parquet(table, index=False)
    manifest = {"schema": SCHEMA, **frame.attrs, "table_sha256": sha256(table),
                "code_sha256": sha256(__file__), "quality_census": summarize(frame),
                "limitations": ["No assumed error distribution or LC-to-sigma conversion",
                                "No cross-source identity linkage or training split assignment",
                                "Estimated and receiver locations cannot be GPS observation targets"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def census(root, output, taxa=None, declarations=None, chunksize=100_000):
    """Stream source diagnostics without copying large originals or creating labels.

    Declarations map package ID (or relative file path) to SourceSpec overrides.
    Without a declaration, absent ``modelled`` stays UNKNOWN; the time values are
    used solely as nominal dates for broad spans, never asserted UTC synchronization.
    Exact duplicate CSV content is counted once, with all paths retained.  Counts
    of source IDs and nominal tracking spans do not establish independent animals
    or complete observed forecast horizons.
    """
    from collections import Counter
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Census output already exists")
    declarations = declarations or {}
    groups, identities, files, seen = {}, {}, [], {}
    sources = []
    for meta in sorted(root.rglob("source.json")):
        sources.append({"path": str(meta), "sha256": sha256(meta)})
    for path in sorted(root.rglob("*.csv")):
        if "reference" in path.name.lower():
            continue
        relative = str(path.relative_to(root))
        package = path.relative_to(root).parts[0]
        digest = sha256(path)
        if digest in seen:
            files.append({"path": str(path), "sha256": digest, "duplicate_of": seen[digest]})
            continue
        seen[digest] = str(path)
        entry = {"path": str(path), "sha256": digest, "source_id": package}
        files.append(entry)
        overrides = declarations.get(relative, declarations.get(package, {}))
        source = SourceSpec(**({"source_id": package} | overrides))
        # csv preserves exact strings and validates schema before chunk parsing.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            headers = next(csv.reader(stream), [])
        try:
            columns = _resolve(headers, source)
        except ValueError as error:
            entry["status"] = "unsupported_schema"
            entry["reason"] = str(error)
            continue
        entry["resolved_columns"], entry["source_spec"] = columns, asdict(source)
        entry["rows"] = 0
        for raw in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunksize):
            def get(key, default=""):
                return raw[columns[key]] if key in columns else pd.Series(default, index=raw.index)
            species = get("species", source.species or "")
            if taxa is not None:
                keep = species.isin(taxa)
                raw = raw.loc[keep]
                if not len(raw):
                    continue
                species = species.loc[keep]
            n = len(raw)
            entry["rows"] += n
            method_raw = get("sensor")
            method = method_raw.map(_method)
            method = method.where(method_raw.str.strip().ne(""), source.method)
            processing = pd.Series(source.processing, index=raw.index)
            modelled = get("modelled").str.strip().str.lower()
            processing.loc[(processing == "unknown") & modelled.isin(["false", "0", "no", "f"])] = "observed"
            for flag in ["modelled", "smoothed", "interpolated"]:
                processing.loc[get(flag).str.strip().str.lower().isin(["true", "1", "yes", "t"])] = flag
            stamps = get("timestamp")
            if "timestamp" not in columns:
                stamps = get("date") + " " + get("time")
            # Nominal time only: explicit metadata below disclaims UTC/GPS offset.
            nominal = pd.to_datetime(stamps, format="mixed", errors="coerce", utc=True)
            latitude, longitude = pd.to_numeric(get("latitude"), errors="coerce"), pd.to_numeric(get("longitude"), errors="coerce")
            valid_coord = latitude.between(-90, 90) & longitude.between(-180, 180)
            individual, tag = get("individual"), get("tag")
            identity = individual.where(individual.ne(""), "tag_only:" + tag)
            identity = identity.where(individual.ne("") | tag.ne(""), "")
            study = get("study", source.study or "")
            work = pd.DataFrame({"species": species, "method": method, "processing": processing,
                                 "study": study, "identity": identity, "nominal": nominal,
                                 "valid_coords": valid_coord, "sensor_raw": method_raw,
                                 "lc": get("argos_lc"), "ellipse": get("ellipse_major").str.strip().ne(""),
                                 "radius": get("error_radius").str.strip().ne(""),
                                 "satellites": get("satellites").str.strip().ne(""),
                                 "hdop": get("gps_hdop").str.strip().ne(""),
                                 "outlier": get("manual_outlier").str.lower().isin(["true", "1"]) |
                                            get("algorithm_outlier").str.lower().isin(["true", "1"]),
                                 "invisible": get("visible").str.lower().isin(["false", "0"])})
            for key, part in work.groupby(["species", "method", "processing"], dropna=False, sort=False):
                stats = groups.setdefault(key, {"rows": 0, "valid_coordinate_rows": 0, "nominal_time_rows": 0,
                                                "ellipse_reported_rows": 0, "radius_reported_rows": 0,
                                                "satellite_count_reported_rows": 0, "hdop_reported_rows": 0,
                                                "source_outlier_rows": 0, "source_invisible_rows": 0,
                                                "lc": Counter(), "sensor": Counter(), "sources": set()})
                stats["rows"] += len(part)
                for name, col in [("valid_coordinate_rows", "valid_coords"), ("ellipse_reported_rows", "ellipse"),
                                  ("radius_reported_rows", "radius"), ("satellite_count_reported_rows", "satellites"),
                                  ("hdop_reported_rows", "hdop"), ("source_outlier_rows", "outlier"), ("source_invisible_rows", "invisible")]:
                    stats[name] += int(part[col].sum())
                stats["nominal_time_rows"] += int(part.nominal.notna().sum())
                stats["lc"].update(part.lc[part.lc.ne("")])
                stats["sensor"].update(part.sensor_raw)
                stats["sources"].add(package)
                for ikey, animal in part.loc[part.identity.ne("")].groupby(["study", "identity"], dropna=False, sort=False):
                    full = (*key, package, *ikey)
                    animal_stats = identities.setdefault(full, {"rows": 0, "start": None, "end": None, "years": set()})
                    animal_stats["rows"] += len(animal)
                    observed_dates = animal.nominal.dropna()
                    if len(observed_dates):
                        start, end = observed_dates.min(), observed_dates.max()
                        animal_stats["start"] = min(animal_stats["start"], start) if animal_stats["start"] is not None else start
                        animal_stats["end"] = max(animal_stats["end"], end) if animal_stats["end"] is not None else end
                        animal_stats["years"].update(observed_dates.dt.year.unique().tolist())
        entry["status"] = "counted"
    result_groups = []
    for key, stats in sorted(groups.items()):
        animals = [v for k, v in identities.items() if k[:3] == key]
        dated = [a for a in animals if a["start"] is not None]
        spans = np.asarray([(a["end"] - a["start"]).total_seconds() / 86400 for a in dated])
        result_groups.append(dict(zip(["species", "method", "processing"], key)) | {
            **{k: v for k, v in stats.items() if k not in {"lc", "sensor", "sources"}},
            "source_individual_ids": len(animals), "source_individual_years": sum(len(a["years"]) for a in animals),
            "first_nominal_date": str(min(a["start"] for a in dated)) if dated else None,
            "last_nominal_date": str(max(a["end"] for a in dated)) if dated else None,
            "median_nominal_span_days": float(np.median(spans)) if len(spans) else None,
            "source_ids_with_span_at_least_days": {str(d): int((spans >= d).sum()) for d in [1, 3, 7, 14, 30, 60, 90, 365]},
            "argos_classes_reported": dict(stats["lc"]), "sensor_strings_reported": dict(stats["sensor"]),
            "source_packages": sorted(stats["sources"]),
        })
    report = {"schema": SCHEMA, "scope": "source-observation-quality-census-only", "root": str(root),
              "code_sha256": sha256(__file__), "taxa_filter": sorted(taxa) if taxa is not None else None,
              "source_metadata": sources, "files": files, "groups": result_groups,
              "rows": sum(g["rows"] for g in result_groups),
              "time_rule": "Dates/spans are nominal source times; no UTC/GPS synchronization or leap-second correction inferred",
              "limitations": ["Missing modelled flags leave processing unknown unless a documented declaration is supplied",
                              "Source IDs and years are not deduplicated independent physical animals",
                              "Spans do not prove complete observed forecast horizons or migration behaviour",
                              "No uncertainty distribution inferred from Argos class, ellipse, residual or DOP",
                              "All taxa in source files are reported unless taxa_filter explicitly restricts them"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--census-root")
    parser.add_argument("--spec", help="JSON SourceSpec, or census package/file declaration mapping")
    parser.add_argument("--taxa", nargs="+", help="Exact reported taxa for a census")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.census_root:
        report = census(args.census_root, args.output, taxa=args.taxa,
                        declarations=json.loads(Path(args.spec).read_text()) if args.spec else None)
        print(json.dumps({"rows": report["rows"], "groups": len(report["groups"]), "output": args.output}))
    else:
        if not args.spec:
            parser.error("--spec is required for intake")
        result = ingest(args.input, json.loads(Path(args.spec).read_text()), args.output)
        print(json.dumps(result["quality_census"], indent=2))


if __name__ == "__main__":
    main()
