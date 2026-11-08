"""Bind approved source-native ambiguity holds without changing identity partitions."""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def build(base,review,summary,nest,output):
    base,review,summary,nest,output=map(Path,[base,review,summary,nest,output])
    c,r,s,n=[json.loads(p.read_text()) for p in [base,review,summary,nest]]
    if sha(base)!="88639cba594ed226411f5a898926eccfee744ab6fa50401c56953393a7099e55":
        raise ValueError("Expected reviewed frozen v4 base")
    if s["full_record_review"]["sha256"]!=sha(review):raise ValueError("Full/summary source review binding differs")
    if any(g["multi_tag_native_times"] or g["times_with_other_declared_active_owner"]
           or g["times_with_other_owner_unknown_end"] or g["times_with_matching_owner_deployment"]!=g["conflict_native_times"]
           for g in s["groups"]):raise ValueError("Namespace/deployment ambiguity needs physical review before exact-time holds")
    protected={key:copy.deepcopy(c[key]) for key in ["physical_mapping","exclusion_mapping","parent_memberships",
              "affected_parent_memberships","freshness_exclusion_animals","source_record_exclusions","source_interval_exclusions"]}
    rules=[{**rule,"evidence_sha256":sha(review)} for rule in s["proposed_source_native_timestamp_exclusions"]]
    rules.extend(n["source_native_timestamp_exclusions"])
    if len(rules)!=80:raise ValueError("Expected55 source groups plus independently reviewedNest151 across80occurrences")
    global_by_animal={}
    for rule in rules:
        times=rule["timestamps_ns"]
        if (rule["time_axis"]!="reported_nominal" or times!=sorted(set(times))
                or any(not isinstance(t,int) or isinstance(t,bool) for t in times)):
            raise ValueError("Exact sorted native timestamp ns membership required")
        group=global_by_animal.setdefault(rule["input_animal"],{"input_animal":rule["input_animal"],"time_axis":"reported_nominal",
                    "timestamps_ns":set(),"source_original_sha256":set(),"evidence_sha256":rule["evidence_sha256"],
                    "reason":rule["reason"]})
        group["timestamps_ns"].update(times)
        group["source_original_sha256"].update(rule.get("source_original_sha256",[]))
    for rule in n["global_raw_source_native_timestamp_rules"]:
        global_by_animal[rule["input_animal"]]["source_original_sha256"].update(rule["source_original_sha256"])
    global_rules=[{**g,"timestamps_ns":sorted(g["timestamps_ns"]),"source_original_sha256":sorted(g["source_original_sha256"])}
                  for _,g in sorted(global_by_animal.items())]
    if len(global_rules)!=56:raise ValueError("Expected56 affected raw identities")
    impact=[]
    before_ids,after_ids=collections.defaultdict(set),collections.defaultdict(set)
    expected_held=0
    by_input=collections.defaultdict(list)
    for rule in rules:by_input[(rule["input_path"],rule["input_sha256"])].append(rule)
    for (path,digest),input_rules in sorted(by_input.items()):
        if sha(path)!=digest:raise ValueError("Affected frozen input checksum differs")
        names=sorted({rule["input_animal"] for rule in input_rules})
        frame=pq.read_table(path,columns=["animal","timestamp","species"],filters=[("animal","in",names)],use_threads=False).to_pandas()
        frame["ns"]=pd.to_datetime(frame.timestamp,utc=True).astype("int64")
        for rule in input_rules:
            block=frame.loc[frame.animal==rule["input_animal"]]
            held=block.ns.isin(rule["timestamps_ns"])
            if int(held.sum())!=rule["rows_in_this_occurrence"]:raise ValueError("Native-time row count guard differs")
            before_ids[rule["input_animal"]].update(block.ns)
            after_ids[rule["input_animal"]].update(block.loc[~held,"ns"])
            expected_held+=int(held.sum())
            impact.append({"input_path":path,"input_sha256":digest,"animal":rule["input_animal"],
                           "raw_species":sorted(block.species.unique()),"before_rows":len(block),
                           "excluded_rows":int(held.sum()),"retained_rows":int((~held).sum()),
                           "virtual_in_this_occurrence":bool(held.all())})
    virtual=sorted(n for n in before_ids if not after_ids[n])
    if virtual:raise ValueError("An unexpected whole raw identity is removed; review required")
    c["source_native_timestamp_exclusions"]=[*c.get("source_native_timestamp_exclusions",[]),*rules]
    c["global_raw_source_native_timestamp_rules"]=[*c.get("global_raw_source_native_timestamp_rules",[]),*global_rules]
    bindings={"native_source_full_review":review,"native_source_summary":summary,"tropicbird_native_policy":nest,
              "tropicbird_native_source_review":Path(n["evidence"]["path"]),
              "native_source_extraction":Path(r["extraction"]["path"]),
              "native_source_conflict_records":Path(json.loads(Path(r["extraction"]["path"]).read_text())["conflicting_records"]["path"]),
              "native_source_extraction_code":Path("scripts/migration_within_raw_source_qc.py"),
              "native_source_review_code":Path("scripts/migration_review_native_time_conflicts.py"),
              "native_policy_binding_code":Path("scripts/migration_bind_native_source_policy.py"),
              "goose_review_code":Path("scripts/migration_goose_identity_review.py"),
              "buzzard_numeric_review_code":Path("scripts/migration_buzzard_numeric_review.py"),
              "source_derivative_code":Path("scripts/migration_reviewed_source_derivative.py")}
    for key,path in bindings.items():c["inputs"][key]={"path":str(path),"sha256":sha(path)}
    expected_codes={"goose_review_code":"d6aae82102820ef34e3ac848f74cb90df1c6af4ff33393dd75f7f9543d978796",
                    "buzzard_numeric_review_code":"4efcf700b1b15754774d48a807bafe4a6778cb5682b1a15efe268aeebcc5d81c",
                    "source_derivative_code":"eaa4047d2b30ceb57878edd778642c52f139b9041461cb69251764d692c2103c"}
    if any(c["inputs"][key]["sha256"]!=digest for key,digest in expected_codes.items()):raise ValueError("Reviewed source derivative code changed")
    c["native_source_policy_derivation"]={"parent_contract":{"path":str(base),"sha256":sha(base)},
        "source_review_approval":"Root approved source-only native-time quarantine after original-record and unique-active-owner review; no model outcomes used",
        "scope":"All competing observations at exact native times removed; every other row and identity retained. Manx intervals,goose retraction,buzzard record choices,risk/role/freshness remain unchanged",
        "code_sha256":sha(__file__),"status":"approved_frozen"}
    c["native_source_policy_impact"]={"source_occurrences":len(rules),"affected_raw_ids":len(global_rules),
        "unique_raw_id_native_times":sum(len(g["timestamps_ns"]) for g in global_rules),
        "input_occurrence_native_times":sum(len(rule["timestamps_ns"]) for rule in rules),
        "excluded_rows_counted_per_input_occurrence":expected_held,"new_virtual_raw_ids":virtual,
        "existing_virtual_raw_ids_from_Manx_interval_policy":json.loads(Path(c["inputs"]["quarantine"]["path"]).read_text())["summary"]["virtual_raw_ids"],
        "affected_raw_taxa":sorted({s for row in impact for s in row["raw_species"]}),
        "taxa_removed_by_native_policy":[],"source_occurrence_details":impact}
    for key,original in protected.items():
        if c[key]!=original:raise AssertionError("Source-native policy altered a protected identity/fold/previous-policy partition")
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(c,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(output),"sha256":sha(output),"impact":{k:v for k,v in c["native_source_policy_impact"].items() if k!="source_occurrence_details"}}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["base","review","summary","nest","output"]:parser.add_argument("--"+name,required=True)
    a=parser.parse_args();build(a.base,a.review,a.summary,a.nest,a.output)
