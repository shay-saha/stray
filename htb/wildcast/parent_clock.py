"""Content-bound supplemental clock evidence for frozen parent observations.

No parent observation, origin or label is rewritten.  A strict source clock
audit can admit new subday supervision; missing evidence always withholds it.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from . import migration_data as legacy
from .migration_clock import (ISO_SYNTAX, STANDARDS, _check_binding, _source_policy,
                              fingerprint, sha256)


AUDIT_SCHEMA="wildcast-parent-source-clock-audit-v1"
EVIDENCE_SCHEMA="wildcast-parent-subday-clock-evidence-v1"


def audit_event_clocks(files, source_decisions, output, *, provenance=None, chunksize=200_000):
    """Audit GPS timestamp strings/flags only, never coordinates or case targets.

    Files are exact historical positive-input event tables.  Invalid/unbounded
    timestamps conservatively withhold the whole source from new subday tasks.
    Missing source clock documentation is not repaired by parsing its dates.
    """
    output=Path(output)
    if output.exists():raise FileExistsError("Use a new source-clock audit path")
    decisions={s:_source_policy(s,r) for s,r in source_decisions.items()}
    grouped=defaultdict(list)
    for row in files:grouped[row["source_id"]].append(row)
    report={"schema":AUDIT_SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),
            "code_sha256":sha256(__file__),"clock_code_sha256":sha256(Path(__file__).with_name("migration_clock.py")),
            "scope":"Raw event timestamp/sensor/visibility/outlier strings only; no coordinates, case labels or outcomes read.",
            "provenance":provenance or [],"sources":{}}
    for source,records in sorted(grouped.items()):
        if source not in decisions:
            report["sources"][source]={"status":"held_missing_clock_documentation","files":records}
            continue
        totals=Counter();inventory=[]
        for record in records:
            path=Path(record["path"]);digest=sha256(path)
            if digest != record["sha256"]:raise ValueError("Historical GPS source content changed")
            encoding=record.get("encoding","utf-8-sig")
            with path.open(encoding=encoding,newline="") as stream:header=next(csv.reader(stream))
            timecol=record.get("timestamp_column","timestamp")
            if timecol not in header:raise ValueError("Clock audit needs an explicitly documented timestamp column")
            columns=[c for c in [timecol,"sensor-type","visible","manually-marked-outlier","algorithm-marked-outlier"] if c in header]
            counts=Counter()
            for frame in pd.read_csv(path,usecols=columns,dtype=str,keep_default_na=False,encoding=encoding,chunksize=chunksize):
                # Exactly the legacy reader's sensor/visibility/manual/algorithm
                # eligibility, without inventing a stronger historical filter.
                keep=pd.Series(True,index=frame.index)
                if "sensor-type" in frame:keep &= frame["sensor-type"].str.lower().eq("gps")
                if "visible" in frame:keep &= frame.visible.str.lower().isin(["true","1"])
                for flag in ["manually-marked-outlier","algorithm-marked-outlier"]:
                    if flag in frame:keep &= ~frame[flag].str.lower().isin(["true","1"])
                text=frame.loc[keep,timecol].str.strip();syntax=text.str.fullmatch(ISO_SYNTAX)
                parsed=pd.to_datetime(text.where(syntax),format="ISO8601",errors="coerce",utc=True)
                counts["GPS_sensor_filtered_rows"]+=len(text)
                counts["malformed_or_impossible_timestamp_rows"]+=int((~syntax|parsed.isna()).sum())
                counts["outside_reviewed_standard_dates"]+=int((parsed.lt(pd.Timestamp("1980-01-06",tz="UTC"))|
                                                               parsed.ge(pd.Timestamp(STANDARDS["valid_before"]))).sum())
            expected=record.get("accepted_sensor_rows_recorded")
            if expected is not None and counts["GPS_sensor_filtered_rows"] != expected:
                counts["historical_filter_count_mismatch"]+=1
            if sha256(path) != digest:raise ValueError("Historical GPS source changed while clock-auditing")
            inventory.append({**record,"verified_sha256":digest,"counts":dict(counts)})
            totals.update(counts)
        bad=any(totals.get(k,0) for k in ["malformed_or_impossible_timestamp_rows","outside_reviewed_standard_dates","historical_filter_count_mismatch"])
        status="strict_clock_validated" if totals["GPS_sensor_filtered_rows"] and not bad else "held_source_clock_anomaly"
        report["sources"][source]={"status":status,"counts":dict(totals),"files":inventory,
                                    "source_policy_sha256":decisions[source]["source_policy_sha256"]}
        print("CLOCK_AUDIT",source,status,totals["GPS_sensor_filtered_rows"],flush=True)
    for binding in report["provenance"]:_check_binding(binding)
    output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return report


def _admissions(manifest, decisions, raw_audit):
    originals={name:{name} for name in manifest["animals"]}
    for original,canonical in manifest["identity_mapping"].items():
        originals[canonical].add(original)
    admitted={};held={}
    for canonical,names in originals.items():
        sources={name.split(":",1)[0] for name in names}
        failures=[];modes=[]
        for source in sorted(sources):
            record=decisions.get(source)
            audit=raw_audit["sources"].get(source,{})
            if (record is None or record["mode"] not in {"bounded_utc_or_gps","declared_utc"} or
                    audit.get("status") != "strict_clock_validated" or
                    audit.get("source_policy_sha256") != record.get("source_policy_sha256")):
                failures.append(source)
            else:modes.append(record["mode"])
        if failures:held[canonical]={"unresolved_sources":failures,"reason":"Missing or failed source-bound clock audit; inherited day-scale tasks only"}
        else:admitted[canonical]="bounded_utc_or_gps" if "bounded_utc_or_gps" in modes else "declared_utc"
    return admitted,held


def make_evidence(parent_cache, source_decisions, raw_clock_audit, output):
    """Create an immutable supplemental audit, preserving every parent byte."""
    output=Path(output)
    if output.exists():raise FileExistsError("Use a new parent-clock evidence path")
    root=Path(parent_cache);meta=json.loads((root/"manifest.json").read_text())
    parent=legacy.load_parent_cache(root,meta["train_end_year"],meta["validation_year"],meta["horizons_days"],meta["stride_days"])
    decisions={s:_source_policy(s,r) for s,r in source_decisions.items()}
    audit=json.loads(Path(raw_clock_audit).read_text())
    if audit.get("schema") != AUDIT_SCHEMA:raise ValueError("Unsupported source clock audit")
    if audit.get("clock_code_sha256") != sha256(Path(__file__).with_name("migration_clock.py")):
        raise ValueError("Source clock audit used a different timestamp/standard policy")
    admitted,held=_admissions(meta,decisions,audit)
    evidence={"schema":EVIDENCE_SCHEMA,"created_utc":datetime.now(timezone.utc).isoformat(),
              "parent_cache":{"path":str(root.resolve()),**parent["hashes"]},
              "parent_sources":meta["sources"],"source_decisions":decisions,
              "raw_clock_audit":{"path":str(Path(raw_clock_audit).resolve()),"sha256":sha256(raw_clock_audit)},
              "admitted_parent_animals":admitted,"withheld_parent_animals":held,
              "minimum_horizon_days":1/24,"train_end_year":meta["train_end_year"],"validation_year":meta["validation_year"],
              "code_sha256":sha256(__file__),"clock_code_sha256":sha256(Path(__file__).with_name("migration_clock.py")),
              "rule":"An admitted child must have evidence for every frozen parent canonical ancestor. A new strong alias cannot upgrade an undocumented parent clock. Parent data/day-scale targets remain unchanged.",
              "limitations":["UTC/GPS bounds concern time-standard interpretation, not device clock accuracy.",
                              "Unknown/local clocks are not corrected or assumed bounded.",
                              "No reserved target values or short-horizon support counts were inspected."]}
    evidence["evidence_sha256"]=fingerprint(evidence)
    output.write_text(json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return evidence


def validate_evidence(path,parent):
    evidence=json.loads(Path(path).read_text())
    unsigned={k:v for k,v in evidence.items() if k != "evidence_sha256"}
    if evidence.get("schema") != EVIDENCE_SCHEMA or fingerprint(unsigned) != evidence.get("evidence_sha256"):
        raise ValueError("Parent clock evidence content hash mismatch")
    if any(evidence.get("parent_cache",{}).get(k) != v for k,v in parent["hashes"].items()):
        raise ValueError("Supplemental clock evidence is for a different frozen parent")
    if evidence.get("parent_sources") != parent["manifest"]["sources"]:
        raise ValueError("Parent source-clock evidence lacks exact observation ancestry")
    if (evidence.get("code_sha256") != sha256(__file__) or
            evidence.get("clock_code_sha256") != sha256(Path(__file__).with_name("migration_clock.py"))):
        raise ValueError("Supplemental parent clock implementation changed")
    _check_binding(evidence["raw_clock_audit"])
    audit=json.loads(Path(evidence["raw_clock_audit"]["path"]).read_text())
    decisions={s:_source_policy(s,r) for s,r in evidence["source_decisions"].items()}
    if decisions != evidence["source_decisions"]:raise ValueError("Parent source clock evidence changed")
    admitted,held=_admissions(parent["manifest"],decisions,audit)
    if admitted != evidence["admitted_parent_animals"] or held != evidence["withheld_parent_animals"]:
        raise ValueError("Parent clock admission was changed after audit")
    return evidence
