"""Regenerate reviewed clock artifacts on a new host without rewriting originals.

Only documentary paths change for the bird/marine/parent policies.  Raw parent
timestamp validation remains an explicitly retained, hash-bound local audit.
Broad policy decisions require the actual source event timestamp dictionary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import re
from datetime import datetime,timezone
import requests

import pyarrow.parquet as pq

from wildcast.migration_clock import (_source_policy, check_compatibility, export,
                                     fingerprint, make_policy, sha256, write_policy)
from wildcast.parent_clock import make_evidence


def save(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")


def verify_assets():
    manifest=json.loads(Path("reports/migration-clock-server-assets-v1.json").read_text())
    for record in manifest["files"]:
        path=Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Transferred clock asset differs: "+str(path))
    print("ASSETS_VERIFIED",len(manifest["files"]),flush=True)


def relocate(decisions,old_prefix):
    result=copy.deepcopy(decisions)
    assets=json.loads(Path("reports/migration-clock-server-assets-v1.json").read_text())["files"]
    for record in result.values():
        record.pop("source_policy_sha256",None)
        for item in record["evidence"]:
            path=Path(item["path"])
            if path.is_absolute():path=path.relative_to(old_prefix)
            if not path.is_file() or sha256(path) != item["sha256"]:
                # The original policy normalized symlinks on its host.  Find
                # the explicitly transferred copy of the identical whole file.
                matches=[Path(a["path"]) for a in assets if a["sha256"] == item["sha256"]]
                if not matches:raise ValueError("No byte-identical relocated documentary asset")
                path=sorted(matches)[0]
            if sha256(path) != item["sha256"]:raise ValueError("Relocated documentary bytes differ")
            item["path"]=str(path)
    return result


def existing_core(name,old_prefix):
    original=Path(f"reports/migration-{name}-clock-policy-v1.json")
    old=json.loads(original.read_text());decisions=relocate(old["sources"],old_prefix)
    manifest=f"data/migration-{name}-gps-clean-v1/processed/manifest.json"
    policy=make_policy([manifest],decisions,minimum_horizon_days=1/24)
    policy["relocation"]={"original_policy_path":str(original),"original_policy_sha256":sha256(original),
                          "rule":"Document paths relocated after exact byte verification; source clock decisions unchanged."}
    policy["policy_sha256"]=fingerprint({k:v for k,v in policy.items() if k != "policy_sha256"})
    path=Path(f"reports/migration-{name}-clock-policy-server-v1.json");write_policy(path,policy)
    result=export(path,f"data/migration-{name}-clock-v1",batch_size=100_000)
    print("CLOCK_READY",name,result["counts"],result["observations_sha256"],flush=True)


def parent(old_prefix):
    original=Path("reports/migration-parent-source-clock-audit-v1.json")
    decision_path=Path("reports/migration-parent-clock-source-decisions-v1.json")
    local_evidence=Path("reports/migration-parent-subday-clock-evidence-v1.json")
    old_decisions=json.loads(local_evidence.read_text())["source_decisions"]
    old_audit=json.loads(original.read_text());relocated=copy.deepcopy(old_audit)
    decisions=relocate(old_decisions,old_prefix)
    for source,decision in old_decisions.items():
        canonical=copy.deepcopy(decision);canonical.pop("source_policy_sha256",None)
        expected=fingerprint(canonical)
        if old_audit["sources"][source].get("source_policy_sha256") != expected:
            raise ValueError("Original clock audit/decision binding differs")
        current=_source_policy(source,decisions[source])
        relocated["sources"][source]["original_source_policy_sha256"]=expected
        relocated["sources"][source]["source_policy_sha256"]=current["source_policy_sha256"]
    relocated["relocation"]={"original_audit":{"path":str(original),"sha256":sha256(original)},
        "original_source_decisions":{"path":str(decision_path),"sha256":sha256(decision_path)},
        "original_normalized_evidence":{"path":str(local_evidence),"sha256":sha256(local_evidence)},
        "code_path":str(Path(__file__).resolve()),"code_sha256":sha256(__file__),
        "rule":"Only documentary policy paths/hashes relocated after exact byte checks. Raw event timestamp validation was performed on the original host and is retained verbatim; raw trajectories were not re-audited on this host."}
    path=Path("reports/migration-parent-source-clock-audit-server-v1.json");save(path,relocated)
    result=make_evidence("data/migration-expanded-dev-v3",decisions,path,
                         "reports/migration-parent-subday-clock-evidence-server-v1.json")
    print("PARENT_CLOCK_READY",len(result["admitted_parent_animals"]),len(result["withheld_parent_animals"]),flush=True)


def broad(version):
    root=Path("data/migration-breadth-clean-v1")
    manifest=json.loads((root/"processed/manifest.json").read_text())
    policy_path=Path(manifest["policy"])
    if sha256(policy_path) != manifest["policy_sha256"]:raise ValueError("Breadth QC policy changed")
    qc=json.loads(policy_path.read_text());jobs={j["source_id"]:j for j in qc["jobs"]}
    counts=Counter()
    for batch in pq.ParquetFile(root/"processed/observations.parquet").iter_batches(columns=["source"],batch_size=250_000):
        counts.update(batch.column(0).to_pylist())
    decisions={};held={}
    for source in sorted(counts):
        job=jobs[source];matches=[]
        for event in job["events"]:
            if event["spec"]["columns"]["timestamp"] != "timestamp":
                held[source]="Custom timestamp field lacks a separately reviewed clock definition";break
            with Path(event["path"]).open(encoding=event["encoding"],newline="") as stream:
                if "timestamp" not in next(csv.reader(stream)):raise ValueError("QC timestamp schema changed")
        if source in held:continue
        for doc in job["documents"]:
            path=Path(doc["path"])
            if sha256(path) != doc["sha256"]:raise ValueError("Actual source timestamp dictionary changed")
            lines=path.read_text(encoding=doc["encoding"].split()[0]).splitlines()
            for i,line in enumerate(lines):
                if re.match(r"^\s*timestamp\s*:",line,re.I):
                    text="\n".join(lines[i:i+7])
                    if re.search(r"\bUTC\b[^\n]*\bor\b[^\n]*\bGPS\b",text,re.I):matches.append((doc,i+1,text))
        delegated=None
        if not matches:
            for doc in job["documents"]:
                prose=Path(doc["path"]).read_text(encoding=doc["encoding"].split()[0])
                if re.search(r"attributes.*defined.*Movebank Attribute Dictionary",prose,re.I):
                    delegated=doc;break
            if delegated is None:
                held[source]="No actual event timestamp dictionary explicitly bounds UTC versus GPS";continue
            cache=Path("data/migration-clock-primary-v1/MVB000200.html");cache.parent.mkdir(parents=True,exist_ok=True)
            url="https://vocab.nerc.ac.uk/collection/MVB/current/MVB000200/"
            if not cache.exists():
                response=requests.get(url,timeout=60);response.raise_for_status()
                if "UTC or GPS time" not in response.text:raise ValueError("Official timestamp definition changed")
                with cache.open("xb") as stream:stream.write(response.content)
                save(cache.with_suffix(".json"),{"url":url,"retrieved_utc":datetime.now(timezone.utc).isoformat(),"sha256":sha256(cache),"scope":"Official event timestamp definition delegated by this source README; not provider-membership inference."})
            if "UTC or GPS time" not in cache.read_text():raise ValueError("Cached official timestamp definition differs")
            doc,line,text=delegated,None,"Actual source README explicitly delegates its attribute definitions to the Movebank Attribute Dictionary; the timestamp field was verified in every input header. Official MVB000200 permits UTC or GPS time."
        else:doc,line,text=matches[0]
        source_meta=Path(job["source_manifest"])
        if sha256(source_meta) != job["source_manifest_sha256"]:raise ValueError("Source metadata changed")
        decisions[source]={"mode":"bounded_utc_or_gps","documented_clock":"UTC_or_GPS","evidence":[
            {"path":doc["path"],"sha256":doc["sha256"],"kind":"movebank_attribute_utc_or_gps","line":line,
             "claim":"Actual source event timestamp field explicitly permits UTC or GPS time; field/header correspondence verified.","timestamp_section":text},
            {"path":str(policy_path),"sha256":sha256(policy_path),"kind":"strict_source_QC_provenance",
             "claim":"Strict original clock syntax, source identities, processing flags, and time anomalies audited by the frozen breadth QC policy."},
            {"path":str(source_meta),"sha256":job["source_manifest_sha256"],"kind":"source_metadata_provenance",
             "claim":"Source-specific original acquisition metadata and published byte checksums."}]}
        if delegated:
            decisions[source]["evidence"][0]["claim"]="Actual source README delegates the released attributes to the named official dictionary; this is an indirect field-definition binding, not an embedded units declaration."
            decisions[source]["evidence"].append({"path":str(cache),"sha256":sha256(cache),"kind":"movebank_attribute_utc_or_gps", "url":url,
                "claim":"The source-delegated authoritative event timestamp definition explicitly permits UTC or GPS time; no UTC-only or local-clock correction inferred."})
    save(f"reports/migration-breadth-clock-source-decisions-{version}.json",{"sources":decisions,"held_sources":held,"source_rows":dict(counts)})
    print("BROAD_CLOCK_REVIEW",len(decisions),"documented",len(held),"held",sum(counts.values()),"rows",flush=True)
    if held:raise ValueError("Explicit source clock evidence needed before exporting held sources: "+repr(held))
    policy=make_policy([root/"processed/manifest.json"],decisions,minimum_horizon_days=1/24)
    path=Path(f"reports/migration-breadth-clock-policy-{version}.json");write_policy(path,policy)
    result=export(path,"data/migration-breadth-clock-v1",batch_size=100_000)
    print("BROAD_CLOCK_READY",result["counts"],result["observations_sha256"],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("operation",choices=["existing","broad","parent"])
    p.add_argument("--original-prefix",required=True);p.add_argument("--review-version",default="v1");a=p.parse_args()
    if not re.fullmatch(r"v\d+",a.review_version):raise ValueError("Use a simple immutable review version")
    if a.operation == "existing":
        verify_assets();existing_core("marine",a.original_prefix);existing_core("bird",a.original_prefix);parent(a.original_prefix)
    elif a.operation == "parent":verify_assets();parent(a.original_prefix)
    else:broad(a.review_version)


if __name__ == "__main__":main()
