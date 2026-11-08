"""Freeze source/time QA holds for duplicated Manx deployments of unknown owner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def resolve_original(path,digest,source_id):
    if Path(path).exists() and sha(path)==digest:return str(path)
    for root in ["data/migration-breadth-stage1/raw","data/migration-marine-stage1/raw","data/migration-marine-targeted/raw"]:
        directory=Path(root)/source_id
        manifest=directory/"source.json"
        if not manifest.exists():continue
        for entry in json.loads(manifest.read_text()).get("files",[]):
            if entry.get("sha256")!=digest:continue
            candidate=directory/entry["stored_as"]
            if candidate.exists() and sha(candidate)==digest:return str(candidate)
    raise ValueError("A checksum-identical publisher original is required")


def build(identity, review, output):
    identity,review=Path(identity),Path(review)
    c,r=json.loads(identity.read_text()),json.loads(review.read_text())
    pairs=[row for row in r["pair_reviews"] if row.get("historical_scope")=="new_Manx_exact_alias"]
    if len(pairs)!=5:raise ValueError("Expected exactly five independently audited duplicated deployment pairs")
    source=next(row for row in r["inputs"] if row["source_id"]=="12d8637e-80b8-4850-994f-71f6d6734007")
    resolved_original=resolve_original(source["path"],source["sha256"],source["source_id"])
    resolved_reference=resolve_original(source["reference"]["path"],source["reference"]["sha256"],source["source_id"])
    rules=[]
    for pair in pairs:
        windows=[(a,b) for a in pair["a_reference_records"] for b in pair["b_reference_records"]
                 if a["deploy-on-date"]==b["deploy-on-date"] and a["deploy-off-date"]==b["deploy-off-date"]]
        if not windows:raise ValueError("Duplicate stream has no source-defined shared deployment bounds")
        for a,b in windows:
            for animal,reference in [(pair["a"],a),(pair["b"],b)]:
                start=pd.Timestamp(reference["deploy-on-date"],tz="UTC")
                end=pd.Timestamp(reference["deploy-off-date"],tz="UTC")
                if not start<end:raise ValueError("Invalid source deployment interval")
                rules.append({"input_animal":animal,"source_id":source["source_id"],
                              "individual_local_identifier":reference["animal-id"],
                              "source_deployment_id":reference["deployment-id"],"source_tag_id":reference["tag-id"],
                              "start_time":start.timestamp(),"end_time":end.timestamp(),"closed":"both",
                              "start_time_reported":reference["deploy-on-date"],"end_time_reported":reference["deploy-off-date"],
                              "reference_row_number":reference["reference_row_number"],
                              "reference_sha256":source["reference"]["sha256"],"source_original_sha256":source["sha256"],
                              "evidence_sha256":sha(review),"reason":"ambiguous_duplicate_deployment"})
    if len(rules)!=14:raise ValueError("Expected seven duplicated deployment intervals held on both labels")
    names={row["input_animal"] for row in rules}
    occurrences,expanded=[],[]
    for item in c["observations"]:
        path=Path(item["path"])
        if sha(path)!=item["sha256"]:raise ValueError("Frozen input hash differs")
        columns=pq.ParquetFile(path).schema_arrow.names
        requested=[name for name in ["animal","timestamp_nominal","timestamp","source_file","source_sha256",
                                     "original_source_file","original_source_sha256"] if name in columns]
        frame=pq.read_table(path,columns=requested,filters=[("animal","in",sorted(names))],use_threads=False).to_pandas()
        for animal,block in frame.groupby("animal",observed=True,sort=True):
            nominal=pd.to_datetime(block["timestamp_nominal"] if "timestamp_nominal" in block else block["timestamp"],utc=True)
            held=pd.Series(False,index=block.index)
            these=[]
            for rule in rules:
                if rule["input_animal"]!=animal:continue
                selected=(nominal>=pd.to_datetime(rule["start_time"],unit="s",utc=True)) & (nominal<=pd.to_datetime(rule["end_time"],unit="s",utc=True))
                held|=selected
                these.append({**rule,"input_sha256":item["sha256"],"input_path":str(path),"rows_in_this_occurrence":int(selected.sum())})
            expanded.extend(these)
            source_columns=[column for column in requested if column not in ["animal","timestamp_nominal","timestamp"]]
            source_values=block.loc[held,source_columns].drop_duplicates().to_dict("records") if source_columns else []
            occurrences.append({"input_path":str(path),"input_sha256":item["sha256"],"input_animal":animal,
                                "rows":len(block),"quarantined_rows":int(held.sum()),"other_rows_retained":int((~held).sum()),
                                "source_occurrences":source_values,"virtual_identity_after_filter":bool(held.all())})
    if {row["input_animal"] for row in occurrences}!=names:
        raise ValueError("A duplicated raw identity has no verified frozen input occurrence")
    result={"schema":"wildcast-global-source-interval-quarantine-v1","code_sha256":sha(__file__),
            "identity_audit":{"path":str(identity),"sha256":sha(identity)},
            "source_QC_evidence":{"path":str(review),"sha256":sha(review)},
            "primary_source":{"path":source["path"],"sha256":source["sha256"],"url":source["original_url"],
                              "published_checksum":source["published_checksum"],"reference":source["reference"],
                              "checksum_identical_resolved_original":resolved_original,
                              "checksum_identical_resolved_reference":resolved_reference},
            "global_raw_source_interval_rules":rules,"source_interval_exclusions":expanded,"source_occurrences":occurrences,
            "comparison_axis":"Original reported nominal time axis, preserving exact subminute bounds; UTC-like storage does not assert measured UTC synchronization",
            "identity_policy":"Keep raw/physical/exclusion/fold/freshness identities and primitives, including virtual IDs with zero eligible observations. This is observation QA, not a physical merge or an outcome label.",
            "application_policy":["Apply the global raw-ID/source-original-SHA/interval rules to every copied source occurrence before history or target case construction",
                                  "Expanded per-input-SHA rules bind all currently frozen root occurrences; a future root must also apply the global source rule, never treat a new clock-root hash as permission to reintroduce the stream",
                                  "Prefer preserved timestamp_nominal or original timestamp_reported. Never floor time or substitute a future-derived trip boundary",
                                  "Remove only these deployment intervals from both ambiguous owner histories; retain all other observed deployments",
                                  "Apply identical source selection during frozen assessment. Anonymous pretraining use needs a separately justified protocol and cannot inherit bird metadata."],
            "summary":{"duplicate_pairs":len(pairs),"duplicated_deployment_windows":len(rules)//2,
                       "global_raw_id_interval_rules":len(rules),"affected_raw_ids":len(names),"frozen_source_occurrences":len(occurrences),
                       "quarantined_rows_counted_per_frozen_occurrence":sum(o["quarantined_rows"] for o in occurrences),
                       "virtual_raw_ids":sorted({o["input_animal"] for o in occurrences if o["virtual_identity_after_filter"]})},
            "scope":"Content-bound source observation ownership QA only, using publisher deployment intervals and audited duplicated streams; no model outcomes or case/target labels read"}
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(output),"sha256":sha(output),"summary":result["summary"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["identity","review","output"]:parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    build(args.identity,args.review,args.output)
