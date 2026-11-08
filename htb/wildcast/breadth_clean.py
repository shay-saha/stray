"""Streaming, resumable GPS QC and time-availability audits for the breadth tranche.

Uses source processing flags and source documents, never the GPS-only legacy
reader.  This stage creates neither model labels nor cohort assignments.  It
preserves nominal clocks and requires an explicit later policy before training.
"""
from __future__ import annotations

import argparse
import codecs
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import ALIASES as EVENT_ALIASES
from .marine_gps_clean import _reference_rows, clean_chunk
from .marine_intake import ALIASES, sha256


SCHEMA = "wildcast-breadth-clean-v1"
DAY = 86400.
LONG_LAGS = np.array([0, .25, .5, 1, 2, 3, 4, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365])
LONG_HORIZONS = np.array([1, 3, 7, 14, 30, 60, 90.])
SHORT_LAGS = np.array([0, 1/288, 1/96, 1/48, 1/24, 1/12, .25, .5, 1, 3, 7])
SHORT_HORIZONS = np.array([1/24, 1/8, .25, .5, 1, 3, 7.])
ISO_CLOCK = r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$"


def named_species_key(value):
    """Counting key only; preserves raw labels and makes no synonym assertion."""
    value = " ".join(unicodedata.normalize("NFKC", str(value)).split())
    if not re.fullmatch(r"[A-Z][a-z]+ [a-z][a-z-]+(?: [a-z][a-z-]+)?", value):
        return ""
    if value.split()[1] in {"sp", "spp", "unknown", "unidentified"}:
        return ""
    return " ".join(value.split()[:2])


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _csv_encoding(path):
    """Validate the entire byte-exact stream before any rows are published."""
    for encoding in ["utf-8-sig", "cp1252"]:
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(2**20), b""):
                    decoder.decode(block)
                decoder.decode(b"", final=True)
            return encoding
        except UnicodeDecodeError:
            if encoding == "cp1252":
                raise


def _csv_batches(path, chunksize, encoding):
    # Without index_col=False, pandas can silently consume an extra CSV field as
    # a row index.  Truncation warnings are fatal rather than dropping fields.
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        yield from pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunksize,
                               encoding=encoding, index_col=False, on_bad_lines="error")


def _source_documents(directory):
    documents, prose = [], []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text, encoding = path.read_text(encoding="utf-8-sig"), "utf-8-sig"
        except UnicodeDecodeError:
            text, encoding = path.read_text(encoding="cp1252"), "cp1252 after explicit UTF-8 failure"
        documents.append({"path": str(path), "sha256": sha256(path), "encoding": encoding})
        # Attribute dictionaries contain generic modelled/interpolated definitions;
        # they are NOT evidence that this particular dataset is interpolated.
        preamble = re.split(r"Terms of Use|Data Attributes", text, maxsplit=1, flags=re.I)[0]
        specifics = re.findall(r"THIS DATASET:[^\n]*(?:\n[^\n]+)?", text, re.I)
        prose.extend([preamble, *specifics])
    return documents, "\n".join(prose)


def source_risks(title, prose):
    """Conservative review flags; absence of a match is not independent proof."""
    flags = []
    if re.search(r"state[- ]space|kalman|smoothed trajectories|interpolated tracks", title, re.I):
        flags.append("processing_model_in_title_requires_raw_fix_evidence")
    for sentence in re.split(r"(?<=[.!?])\s+|\n", prose):
        if re.search(r"original fixes|original (?:GPS )?(?:locations|positions)", sentence, re.I) and re.search(r"\bnot\b", sentence, re.I):
            continue
        if re.search(r"(?:locations|positions|tracks|data)\s+(?:were|are|have been)\s+(?:linearly\s+)?(?:interpolated|regulari[sz]ed|smoothed|modelled|modeled)|(?:interpolated|smoothed|modelled|modeled)\s+(?:locations|positions|trajectories)", sentence, re.I):
            flags.append("source_specific_processed_positions")
    if re.search(r"anosmia|olfactory nerves|olfactory nerve|experimental (?:displacement|translocation)|displacement experiment", title + "\n" + prose, re.I):
        flags.append("experimental_manipulation_requires_cohort_review")
    return sorted(set(flags))


def _column_map(headers):
    translation = {"timestamp": "timestamp", "lat": "latitude", "lon": "longitude",
                   "animal": "individual", "species": "species", "study": "study"}
    result = {}
    for old, new in translation.items():
        found = [v for v in EVENT_ALIASES[old] if v in headers and not (old == "animal" and v.startswith("tag"))]
        if found:
            result[new] = found[0]
    for key, options in ALIASES.items():
        if key not in result:
            match = next((v for v in [*options, *(v.replace("-", ".") for v in options)] if v in headers), None)
            if match:
                result[key] = match
    return result


def prepare_policy(raw_root, plan_path, output, overrides=None):
    """Pin complete acquired sources and make processing holds reviewable first."""
    raw_root, plan_path = Path(raw_root), Path(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get("format") != "wildcast-breadth-metadata-plan-v1":
        raise ValueError("Unsupported pinned acquisition plan")
    overrides = overrides or {}
    jobs, held = [], []
    for selected in plan["selected_sources"]:
        directory = raw_root / selected["id"]
        source_path = directory / "source.json"
        if not source_path.exists():
            held.append({"source_id": selected["id"], "reason": "raw_source_not_acquired"}); continue
        source = json.loads(source_path.read_text())
        if source.get("acquisition_complete") is False:
            held.append({"source_id": selected["id"], "reason": "source_acquisition_incomplete"}); continue
        if source.get("id", source.get("source_id")) != selected["id"]:
            raise ValueError("Acquired source identity differs from pinned plan")
        rights = source.get("metadata", {}).get("dc.rights.uri", selected.get("rights_uris", []))
        if not rights or not all(any(t in v.lower() for t in ["/publicdomain/zero/1.0", "/licenses/by/4.0", "/licenses/by/3.0"]) for v in rights):
            held.append({"source_id": selected["id"], "reason": "unsupported_source_rights"}); continue
        documents, prose = _source_documents(directory)
        risks = source_risks(selected["title"], prose)
        override = overrides.get(selected["id"], {})
        if risks and not override.get("raw_fix_evidence"):
            held.append({"source_id": selected["id"], "reason": "source_processing_or_manipulation_review", "risks": risks,
                         "documents": documents}); continue
        if override.get("hold"):
            held.append({"source_id": selected["id"], "reason": override["hold"]}); continue
        expected_files = {f["name"]: f for f in selected.get("files", []) if f.get("included_for_intake")}
        ref_hashes = {f.get("sha256") for f in source.get("files", [])
                      if f.get("role") == "identity_reference" and f.get("status", "").startswith("complete")}
        _, reference_evidence = _reference_rows(directory)
        for reference in reference_evidence:
            if reference["sha256"] not in ref_hashes:
                raise ValueError("Reference file lacks completed acquisition hash evidence")
        events = []
        for original in source.get("files", []):
            if not original.get("status", "complete").startswith("complete"):
                continue
            if original.get("role") not in {"events", "archive"}:
                continue
            expected_file = expected_files.get(original["name"])
            if expected_file is None:
                held.append({"source_id": selected["id"], "path": original["name"], "reason": "original_not_in_pinned_intake_plan"}); continue
            if (original.get("bytes") != expected_file.get("bytes") or
                    original.get("published_checksum") != expected_file.get("published_checksum")):
                raise ValueError("Acquired original metadata differs from pinned bitstream plan")
            stored = directory / original["stored_as"]
            if not stored.exists() or sha256(stored) != original["sha256"]:
                raise ValueError("Original acquired file hash mismatch")
            checksum = expected_file.get("published_checksum")
            if checksum:
                h = hashlib.new(checksum["checkSumAlgorithm"].lower())
                with stored.open("rb") as stream:
                    for block in iter(lambda: stream.read(2**20), b""):
                        h.update(block)
                if h.hexdigest().lower() != checksum["value"].lower():
                    raise ValueError("Original fails the repository's published checksum")
            for derived in original.get("derived", []):
                offset = 0
                for chunk in derived.get("chunks", []):
                    path = directory / chunk["name"]
                    if sha256(path) != chunk["sha256"]:
                        raise ValueError("Lossless chunk content hash mismatch")
                    encoding = _csv_encoding(path)
                    with path.open(encoding=encoding, newline="") as stream:
                        headers = next(csv.reader(stream), [])
                    if chunk.get("rows") is None and (len(derived["chunks"]) != 1 or not
                            (derived.get("status", "").startswith("byte_exact") or chunk["sha256"] == original["sha256"])):
                        raise ValueError("Unknown row counts require one complete byte-exact original/member chunk")
                    cols = _column_map(headers)
                    required = {"timestamp", "latitude", "longitude", "individual", "species"}
                    if not required <= cols.keys():
                        held.append({"source_id": selected["id"], "path": str(path), "reason": "unsupported_event_schema"}); continue
                    if len(set(headers)) != len(headers):
                        held.append({"source_id": selected["id"], "path": str(path), "reason": "ambiguous_duplicate_event_headers"}); continue
                    virtual = []
                    if "sensor" not in cols:
                        sensors = [v.lower() for v in selected.get("sensors_metadata", [])]
                        explicit = sensors == ["gps"] or bool(re.search(r"(?:^|[-_ ])gps(?:[._ -]|$)", original.get("original_name", original["name"]), re.I))
                        if not explicit:
                            held.append({"source_id": selected["id"], "path": str(path), "reason": "mixed_or_unknown_sensor_without_row_method"}); continue
                        cols["sensor"] = "__intake_declared_sensor"; virtual.append("sensor")
                    if "study" not in cols:
                        cols["study"] = "__intake_declared_study"; virtual.append("study")
                    spec = {"source_id": selected["id"], "method": "gps", "processing": "observed",
                            "time_standard": override.get("time_standard", "UTC_or_GPS"), "columns": cols}
                    events.append({"path": str(path), "sha256": chunk["sha256"], "spec": spec,
                                   "source_record_offset": offset, "rows": chunk["rows"],
                                   "encoding": encoding,
                                   "encoding_evidence": "strict full-stream UTF-8 validation" if encoding == "utf-8-sig" else "explicit CP1252 fallback after UTF-8 decoding failed; bytes untouched",
                                   "original_path": str(stored), "original_sha256": original["sha256"],
                                   "original_member": derived.get("archive_member"), "virtual_fields": virtual,
                                   "evidence": [{"path": str(source_path), "sha256": sha256(source_path),
                                                 "claim": "Source-declared GPS point data; processing and quality flags remain mandatory"}],
                                   "quality_rules": override.get("quality_rules", []),
                                   "require_deployment_bounds": override.get("require_deployment_bounds", False)})
                    offset += chunk["rows"] if chunk["rows"] is not None else 0
        if events:
            jobs.append({"source_id": selected["id"], "source_manifest": str(source_path),
                         "source_manifest_sha256": sha256(source_path), "documents": documents,
                         "source_risks_reviewed": risks, "override": override, "events": events,
                         "references": reference_evidence})
        else:
            held.append({"source_id": selected["id"], "reason": "no_supported_GPS_event_chunks"})
    policy = {"schema": SCHEMA, "plan": str(plan_path), "plan_sha256": sha256(plan_path),
              "raw_root": str(raw_root), "jobs": jobs, "held": held, "code_sha256": sha256(__file__),
              "shared_QC_sha256": sha256(Path(__file__).with_name("marine_gps_clean.py")),
              "intake_sha256": sha256(Path(__file__).with_name("marine_intake.py")),
              "aliases_sha256": _fingerprint(EVENT_ALIASES),
              "contracts": ["Source-declared GPS, never assumed precise ground truth", "No positive modelled/interpolated/smoothed flags",
                            "Dataset-specific processing/manipulation holds; generic dictionary definitions are not dataset evidence",
                            "Original coordinates, taxonomy, source IDs, quality and nominal clock retained",
                            "Clock and documented manipulation anomalies exclude the whole affected source animal",
                            "No labels, identity split assignments, reserved case reads, or model training"]}
    policy["policy_sha256"] = _fingerprint(policy)
    if Path(output).exists():
        raise ValueError("Policy output already exists; use a new immutable path")
    _save(output, policy)
    return policy


def _source_job(job, parts, latest, chunksize):
    """One independent process per source; partial outputs never become final data."""
    target = Path(parts) / (job["source_id"] + ".parquet")
    info_path = target.with_suffix(".json")
    job_hash = _fingerprint({"job": job, "latest": latest, "code": sha256(__file__),
                             "marine_qc_code": sha256(Path(__file__).with_name("marine_gps_clean.py")),
                             "intake_code": sha256(Path(__file__).with_name("marine_intake.py"))})
    if sha256(job["source_manifest"]) != job["source_manifest_sha256"]:
        raise ValueError("Pinned source acquisition manifest changed")
    for document in job["documents"]:
        if sha256(document["path"]) != document["sha256"]:
            raise ValueError("Pinned source processing documentation changed")
    for reference in job["references"]:
        if sha256(reference["path"]) != reference["sha256"]:
            raise ValueError("Pinned reference/deployment table changed")
    for path, checksum in {r["original_path"]: r["original_sha256"] for r in job["events"]}.items():
        if sha256(path) != checksum:
            raise ValueError("Original source bytes changed")
    for event in job["events"]:
        if sha256(event["path"]) != event["sha256"]:
            raise ValueError("Pinned source chunk changed")
    if info_path.exists():
        saved = json.loads(info_path.read_text())
        if saved["job_sha256"] != job_hash or (saved["rows"] and (not target.exists() or sha256(target) != saved["table_sha256"])):
            raise ValueError("Cached source QC differs from current pinned policy")
        return saved
    refs, reference_evidence = _reference_rows(Path(job["source_manifest"]).parent)
    reference_clock_keys, safe_refs = set(), []
    for reference in refs:
        dates = [str(reference.get(key, "")).strip() for key in ["deploy-on-date", "deploy-off-date"]]
        if any(value and (not re.fullmatch(ISO_CLOCK, value) or pd.isna(pd.to_datetime(value, format="mixed", errors="coerce", utc=True))) for value in dates):
            reference_clock_keys.add((reference.get("animal-taxon", ""), reference.get("animal-id", "")))
        else:
            safe_refs.append(reference)
    refs = safe_refs
    temporary = target.with_suffix(".parquet.partial")
    writer, inventory, animals = None, [], {}
    clock_hold, manipulation_hold = set(), set()
    for declaration in job["events"]:
        path = Path(declaration["path"])
        if sha256(path) != declaration["sha256"]:
            raise ValueError("Source chunk changed after policy pinning")
        file_stats, start = Counter(), 1
        for raw in _csv_batches(path, chunksize, declaration["encoding"]):
            cols = declaration["spec"]["columns"]
            virtual_names = []
            for field in declaration["virtual_fields"]:
                name = cols[field]
                if name in raw:
                    raise ValueError("Internal declared-source column collides with real source data")
                raw[name] = "gps" if field == "sensor" else job["source_id"]
                virtual_names.append(name)
            date_text = raw[cols["timestamp"]].str.strip()
            iso_valid = date_text.str.match(ISO_CLOCK)
            non_iso = date_text.ne("") & ~iso_valid
            nominal = pd.to_datetime(date_text.where(iso_valid, ""), format="mixed", errors="coerce", utc=True)
            gps_rows = raw[cols["sensor"]].str.strip().str.lower().isin(["gps", "gnss", "fastloc-gps", "fastloc gps"])
            clock_bad = gps_rows & (nominal.ge(pd.Timestamp(latest, tz="UTC")) | nominal.lt(pd.Timestamp("1980-01-01", tz="UTC")) | non_iso)
            if reference_clock_keys:
                reference_bad = pd.Series([(taxon, individual) in reference_clock_keys for taxon, individual in
                                            zip(raw[cols["species"]], raw[cols["individual"]])], index=raw.index)
                clock_bad |= gps_rows & reference_bad
            for values in raw.loc[clock_bad, [cols["study"], cols["species"], cols["individual"]]].itertuples(index=False, name=None):
                clock_hold.add(job["source_id"] + ":" + ":".join(values))
            chunk_declaration = declaration
            if non_iso.any():
                validated_column = "__intake_validated_timestamp"
                if validated_column in raw:
                    raise ValueError("Internal clock-validation column collides with source data")
                raw[validated_column] = raw[cols["timestamp"]].where(iso_valid, "")
                virtual_names.append(validated_column)
                chunk_declaration = {**declaration, "spec": {**declaration["spec"], "columns": {**cols, "timestamp": validated_column}}}
            frame, stats = clean_chunk(raw, path, chunk_declaration, refs, declaration["sha256"], start, latest)
            start += len(raw); file_stats.update(stats)
            if not len(frame):
                continue
            if virtual_names:
                frame["reported_fields_json"] = [json.dumps({k: v for k, v in json.loads(value).items() if k not in virtual_names},
                                                                 sort_keys=True, ensure_ascii=False) for value in frame.reported_fields_json]
                if "sensor" in declaration["virtual_fields"]:
                    frame["sensor_reported"] = ""
            frame["method_evidence"] = "single_GPS_source_or_explicit_GPS_file" if "sensor" in declaration["virtual_fields"] else "source_row_sensor"
            frame["taxon_raw"] = frame.species
            frame["taxon_display"] = frame.species.map(lambda x: " ".join(unicodedata.normalize("NFKC", x).split()))
            frame["taxon_named_species_key"] = frame.species.map(named_species_key)
            frame["original_source_file"] = declaration["original_path"]
            frame["original_source_sha256"] = declaration["original_sha256"]
            frame["original_archive_member"] = declaration["original_member"] or ""
            frame["original_source_record"] = frame.source_record + declaration["source_record_offset"]
            # Reference endpoint/fate fields are for QC and identity, never model
            # inputs. Only documented baseline manipulation determines this hold.
            ref_cache = {}
            for value in frame.reference_fields_json.unique():
                r = json.loads(value) if value else {}
                manipulation = str(r.get("manipulation-type", "")).strip().lower()
                ref_cache[value] = manipulation not in {"", "none", "control", "unmanipulated", "not manipulated", "unknown"}
            bad_manipulation = frame.reference_fields_json.map(ref_cache)
            manipulation_hold.update(frame.loc[bad_manipulation, "animal"])
            for animal, group in frame.groupby("animal", sort=False):
                value = animals.setdefault(animal, {"rows": 0, "first": group.timestamp_nominal.min().isoformat(),
                                                   "last": group.timestamp_nominal.max().isoformat(),
                                                   "species": group.species.iloc[0], "taxon_display": group.taxon_display.iloc[0],
                                                   "taxon_named_species_key": group.taxon_named_species_key.iloc[0]})
                value["rows"] += len(group)
                value["first"] = min(value["first"], group.timestamp_nominal.min().isoformat())
                value["last"] = max(value["last"], group.timestamp_nominal.max().isoformat())
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
        inventory.append({"declaration": declaration, "qc_counts": dict(file_stats)})
    if writer is not None:
        writer.close(); temporary.replace(target)
    if sha256(job["source_manifest"]) != job["source_manifest_sha256"]:
        raise ValueError("Source acquisition manifest changed during QC")
    result = {"source_id": job["source_id"], "job_sha256": job_hash, "rows": sum(v["rows"] for v in animals.values()),
              "animals": animals, "clock_hold_animals": sorted(clock_hold), "manipulation_hold_animals": sorted(manipulation_hold),
              "files": inventory, "references": reference_evidence, "table": str(target),
              "table_sha256": sha256(target) if writer is not None else None}
    _save(info_path, result)
    return result


def clean(policy_path, output, workers=4, chunksize=50_000, latest="2026-11-09"):
    policy_path, output = Path(policy_path), Path(output)
    policy = json.loads(policy_path.read_text())
    expected = policy.pop("policy_sha256")
    if (_fingerprint(policy) != expected or policy["code_sha256"] != sha256(__file__) or
            policy["shared_QC_sha256"] != sha256(Path(__file__).with_name("marine_gps_clean.py")) or
            policy["intake_sha256"] != sha256(Path(__file__).with_name("marine_intake.py")) or
            policy["aliases_sha256"] != _fingerprint(EVENT_ALIASES)):
        raise ValueError("QC policy or implementation changed after pinning")
    policy["policy_sha256"] = expected
    final_manifest = output / "processed/manifest.json"
    if final_manifest.exists():
        raise ValueError("Completed breadth core is immutable")
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    results = []
    if workers == 1:
        for job in policy["jobs"]:
            result = _source_job(job, parts, latest, chunksize); results.append(result)
            print(f"QC {result['source_id']}: {result['rows']} provisional GPS rows", flush=True)
    else:
        if not 1 <= workers <= 8:
            raise ValueError("Use 1 to 8 independent source workers")
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(_source_job, job, parts, latest, chunksize) for job in policy["jobs"]]
            for future in as_completed(futures):
                result = future.result(); results.append(result)
                print(f"QC {result['source_id']}: {result['rows']} provisional GPS rows", flush=True)
    animals, holds = {}, {}
    for result in results:
        animals.update(result["animals"])
        holds.update({name: "source_clock_anomaly" for name in result["clock_hold_animals"]})
        holds.update({name: "documented_manipulation" for name in result["manipulation_hold_animals"]})
    for animal, stats in animals.items():
        if stats["rows"] < 2 or stats["first"] >= stats["last"]:
            holds[animal] = "singleton_or_zero_span_not_a_trajectory"
    processed = output / "processed"; processed.mkdir(parents=True, exist_ok=True)
    destination, writer, rows = processed / "observations.parquet", None, 0
    for result in sorted(results, key=lambda v: v["source_id"]):
        if not result["rows"]:
            continue
        for batch in pq.ParquetFile(result["table"]).iter_batches(batch_size=chunksize):
            table = pa.Table.from_batches([batch])
            table = table.filter(pa.array([name not in holds for name in table["animal"].to_pylist()]))
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table); rows += len(table)
    if writer is None:
        raise ValueError("No documented GPS trajectories survived source QA")
    writer.close()
    animal_rows = [{"animal": name, **stats, "hold_reason": holds.get(name, ""),
                    "span_days": (pd.Timestamp(stats["last"]) - pd.Timestamp(stats["first"])).total_seconds() / DAY}
                   for name, stats in sorted(animals.items())]
    pd.DataFrame(animal_rows).to_parquet(processed / "animal-audit.parquet", index=False)
    retained = [r for r in animal_rows if not r["hold_reason"]]
    manifest = {"schema": SCHEMA, "scope": "source GPS QC only; identity and frozen cohort audit required before training",
                "policy": str(policy_path), "policy_sha256": sha256(policy_path), "implementation_sha256": sha256(__file__),
                "shared_QC_sha256": sha256(Path(__file__).with_name("marine_gps_clean.py")),
                "rows": rows, "source_animal_ids": len(retained), "raw_taxon_names": sorted({r["species"] for r in retained}),
                "normalized_display_taxa": sorted({r["taxon_display"] for r in retained}),
                "named_species_keys": sorted({r["taxon_named_species_key"] for r in retained if r["taxon_named_species_key"]}),
                "observations_sha256": sha256(destination), "animal_audit_sha256": sha256(processed / "animal-audit.parquet"),
                "sources": [{"source_id": r["source_id"], "report": str(parts / (r["source_id"] + ".json")),
                             "report_sha256": sha256(parts / (r["source_id"] + ".json"))} for r in results],
                "held_sources": policy["held"], "held_animals": holds,
                "clock_hold_bounds": {"earliest_nominal": "1980-01-01", "latest_exclusive_nominal": latest,
                                      "non_ISO_dates": "quarantine pending source-specific format declaration; no MM/DD versus DD/MM guess"},
                "clock_contract": "timestamp_nominal orders original source clocks; timestamp_utc is nullable and independently marked. No legacy timestamp column or GPS-to-UTC correction is invented.",
                "limitations": ["Declared Latin strings are not an independently reconciled species census",
                                "Named-species counting keys collapse explicit subspecies; synonyms and taxonomic revisions still need independent reconciliation",
                                "Source IDs are not independent physical animals until reference/exact-fix audit",
                                "Retained GPS has device error; no assumed measured metre precision",
                                "Tracking spans and raw row counts are not observed long-horizon labels"]}
    _save(final_manifest, manifest)
    return manifest


def _eligible_times(times, *, short=False, cutoff=np.inf):
    """Availability only, matching actual times and never creating target values."""
    times = np.unique(np.asarray(times, dtype=float))
    times.sort()
    lags, horizons = (SHORT_LAGS, SHORT_HORIZONS) if short else (LONG_LAGS, LONG_HORIZONS)
    min_history, stride = (1/24, .25) if short else (7., 7.)
    counts = np.zeros(len(horizons), dtype=np.int64)
    origins, last = 0, -np.inf
    for i in range(8, len(times)):
        stamp = times[i]
        if stamp >= cutoff or stamp - times[0] < min_history * DAY or stamp - last < stride * DAY:
            continue
        requested = stamp - lags * DAY
        past = np.searchsorted(times, requested, side="right") - 1
        valid = past >= 0
        lag_tol = np.maximum(1/96 if short else .25, .2 * lags) * DAY
        valid &= requested - times[np.maximum(past, 0)] <= lag_tol
        if valid.sum() < 5 or (short and len(np.unique(past[valid])) < 5):
            continue
        wanted = stamp + horizons * DAY
        right = np.clip(np.searchsorted(times, wanted), 0, len(times)-1)
        left = np.maximum(right-1, 0)
        selected = np.where(abs(times[left]-wanted) < abs(times[right]-wanted), left, right)
        tol = np.minimum(.25 if short else 2., np.maximum(1/288 if short else .125, .05 * horizons)) * DAY
        mask = (selected > i) & (abs(times[selected]-wanted) <= tol) & (times[selected] < cutoff)
        if mask.sum() < 2:
            continue
        origins += 1; counts += mask; last = stamp
    return origins, counts


def _endpoint_support(times, horizon_days, cutoff=np.inf, long_history=False):
    """One real future observation suffices for an endpoint availability count."""
    times = np.unique(np.asarray(times, dtype=float))
    times.sort()
    if len(times) < 9:
        return 0
    lags = LONG_LAGS if long_history else SHORT_LAGS
    history_days = 7. if long_history else 1/24
    lag_tolerance = np.maximum(.25 if long_history else 1/96, .2 * lags) * DAY
    target_tolerance = min(.25, max(1/288, .05 * horizon_days)) * DAY
    i = max(8, int(np.searchsorted(times, times[0] + history_days * DAY)))
    count = 0
    while i < len(times) and times[i] < cutoff:
        stamp = times[i]
        wanted = stamp + horizon_days * DAY
        requested = stamp - lags * DAY
        past = np.searchsorted(times, requested, side="right") - 1
        valid = (past >= 0) & (requested - times[np.maximum(past, 0)] <= lag_tolerance)
        history_ok = valid.sum() >= 5 and (long_history or len(np.unique(past[valid])) >= 5)
        right = min(int(np.searchsorted(times, wanted)), len(times)-1)
        left = max(right-1, 0)
        selected = left if abs(times[left]-wanted) < abs(times[right]-wanted) else right
        if history_ok and selected > i and times[selected] < cutoff and abs(times[selected]-wanted) <= target_tolerance:
            count += 1
            # Same six-hour origin cadence for each endpoint; this is a count of
            # usable observed endpoints, never a rescaling of daily model errors.
            i = max(i+1, int(np.searchsorted(times, stamp + .25 * DAY)))
        else:
            i += 1
    return count


def availability_audit(observations, output, train_end_year=2020):
    """Time-only per-source availability; no labels, errors or assignments."""
    paths = [Path(v) for v in observations]
    times, species, sources = defaultdict(list), {}, []
    for path in paths:
        q = pq.ParquetFile(path)
        timecol = "timestamp_nominal" if "timestamp_nominal" in q.schema.names else "timestamp"
        for batch in q.iter_batches(batch_size=200_000, columns=["animal", "species", timecol]):
            frame = batch.to_pandas()
            for animal, group in frame.groupby("animal", sort=False):
                if animal in species and species[animal] != group.species.iloc[0]:
                    raise ValueError("One raw animal namespace spans contradictory taxa")
                species[animal] = group.species.iloc[0]
                times[animal].append(pd.to_datetime(group[timecol], utc=True).astype("int64").to_numpy() / 1e9)
        sources.append({"path": str(path), "sha256": sha256(path)})
    records = []
    cutoff = pd.Timestamp(f"{train_end_year+1}-01-01", tz="UTC").timestamp()
    for animal, arrays in sorted(times.items()):
        stamps = np.concatenate(arrays)
        record = {"animal": animal, "species": species[animal], "observed_unique_times": len(np.unique(stamps))}
        for name, short, boundary in [("long_anytime", False, np.inf), ("long_pre_cutoff", False, cutoff),
                                      ("short_anytime", True, np.inf), ("short_pre_cutoff", True, cutoff)]:
            origins, counts = _eligible_times(stamps, short=short, cutoff=boundary)
            record[name + "_origins"] = origins
            record[name + "_targets_by_horizon"] = counts.tolist()
        for name, horizon, long_history in [("one_hour", 1/24, False), ("twelve_hour", .5, False),
                                            ("one_day", 1., False), ("one_hour_seven_day_history", 1/24, True)]:
            record[name + "_anytime_origins"] = _endpoint_support(stamps, horizon, long_history=long_history)
            record[name + "_pre_cutoff_origins"] = _endpoint_support(stamps, horizon, cutoff, long_history)
        records.append(record)
    frame = pd.DataFrame(records)
    summary = []
    for taxon, group in frame.groupby("species", sort=True):
        row = {"species": taxon, "source_animal_ids": len(group)}
        for name in ["long_anytime", "long_pre_cutoff", "short_anytime", "short_pre_cutoff"]:
            row[name + "_origins"] = int(group[name + "_origins"].sum())
            row[name + "_animals"] = int(group[name + "_origins"].gt(0).sum())
            row[name + "_targets_by_horizon"] = np.asarray(group[name + "_targets_by_horizon"].tolist()).sum(0).tolist()
        for name in ["one_hour", "twelve_hour", "one_day", "one_hour_seven_day_history"]:
            for scope in ["anytime", "pre_cutoff"]:
                column = name + "_" + scope + "_origins"
                row[column] = int(group[column].sum())
                row[name + "_" + scope + "_animals"] = int(group[column].gt(0).sum())
        summary.append(row)
    report = {"schema": "wildcast-breadth-time-availability-v1", "observations": sources,
              "code_sha256": sha256(__file__), "train_end_year": train_end_year,
              "long_lags_days": LONG_LAGS.tolist(), "long_horizons_days": LONG_HORIZONS.tolist(),
              "short_lags_days": SHORT_LAGS.tolist(), "short_horizons_days": SHORT_HORIZONS.tolist(),
              "long_policy": "Original >=8 prior rows, >=7 elapsed days, >=5 available lag masks, >=2 actual targets; 7d stride, original tolerance law",
              "short_policy": ">=8 prior rows, >=1h history, >=5 distinct observed lag fixes, >=2 actual targets; 6h stride, target tolerance=min(6h,max(5min,5%horizon))",
              "endpoint_policy": "One actual future observation, same causal history requirements and 6h origin cadence; horizons1h/12h/1d evaluated separately with tolerance=min(6h,max(5min,5%horizon)); optional7d-history1h counter. No daily-error scaling.",
              "counts": {name + "_taxa": sum(r[name + "_origins"] > 0 for r in summary)
                         for name in ["long_anytime", "long_pre_cutoff", "short_anytime", "short_pre_cutoff"]},
              "taxa": summary, "limitations": ["Time-only per-source availability before speed QC, identity pooling and frozen fold assignment; canonicalization may change counts",
                                                "Pre-cutoff counts require both origin and selected targets before Jan1(train_end_year+1)",
                                                "No target coordinates, reserved case labels, scores or model fitting accessed",
                                                "Short-horizon pretraining support is not evidence of observed 90-day migration targets"]}
    for name in ["long_anytime", "long_pre_cutoff", "short_anytime", "short_pre_cutoff"]:
        report["counts"][name + "_named_species"] = len({named_species_key(r["species"]) for r in summary
                                                    if r[name + "_origins"] and named_species_key(r["species"])})
    for endpoint in ["one_hour", "twelve_hour", "one_day", "one_hour_seven_day_history"]:
        for scope in ["anytime", "pre_cutoff"]:
            name = endpoint + "_" + scope
            report["counts"][name + "_taxa"] = sum(r[name + "_origins"] > 0 for r in summary)
            report["counts"][name + "_named_species"] = len({named_species_key(r["species"]) for r in summary
                                                             if r[name + "_origins"] and named_species_key(r["species"])})
    output = Path(output)
    if output.exists():
        raise ValueError("Availability report is immutable")
    _save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("policy")
    p.add_argument("--raw-root", required=True); p.add_argument("--plan", required=True)
    p.add_argument("--overrides"); p.add_argument("--output", required=True)
    p = sub.add_parser("clean")
    p.add_argument("--policy", required=True); p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--chunksize", type=int, default=50_000)
    p.add_argument("--latest", default="2026-11-09")
    p = sub.add_parser("availability")
    p.add_argument("--observations", nargs="+", required=True); p.add_argument("--output", required=True)
    p.add_argument("--train-end-year", type=int, default=2020)
    args = parser.parse_args()
    if args.command == "policy":
        result = prepare_policy(args.raw_root, args.plan, args.output, json.loads(Path(args.overrides).read_text()) if args.overrides else None)
        print(json.dumps({"source_jobs": len(result["jobs"]), "held": len(result["held"])}))
    elif args.command == "clean":
        result = clean(args.policy, args.output, args.workers, args.chunksize, args.latest)
        print(json.dumps({"rows": result["rows"], "source_animal_ids": result["source_animal_ids"],
                          "taxa": len(result["raw_taxon_names"]), "named_species_keys": len(result["named_species_keys"])}))
    else:
        print(json.dumps(availability_audit(args.observations, args.output, args.train_end_year)["counts"]))


if __name__ == "__main__":
    main()
