"""Verify native duplicate-time records and inspect deployment ownership metadata."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path,algorithm="sha256"):
    h=hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def review(extraction,output):
    extraction,output=Path(extraction),Path(output)
    if output.exists():raise FileExistsError(output)
    e=json.loads(extraction.read_text());record=e["conflicting_records"]
    if sha(record["path"])!=record["sha256"]:raise ValueError("Conflict extraction hash differs")
    rows=pd.read_parquet(record["path"])
    unique=rows.drop_duplicates(["source_sha256","source_record"])
    expected=collections.defaultdict(dict)
    for row in unique.itertuples(index=False):
        expected[row.source_sha256][int(row.source_record)]=json.loads(row.reported_fields_json)
    checks=[];ref_sources={};source_index={s["source_id"]:s for s in e["original_sources"]}
    for source in e["original_sources"]:
        for file in source["files"]:
            path=Path(file["path"])
            if sha(path)!=file["sha256"]:raise ValueError("Publisher source SHA mismatch")
            if file.get("published_checksum"):
                checksum=file["published_checksum"]
                if sha(path,checksum["checkSumAlgorithm"].lower().replace("-",""))!=checksum["value"].lower():
                    raise ValueError("Published source checksum mismatch")
            if file["role"]=="identity_reference":
                with path.open(encoding="utf-8-sig",newline="") as stream:
                    refs=list(csv.DictReader(stream))
                by_tag=collections.defaultdict(list)
                for index,ref in enumerate(refs,2):
                    by_tag[ref.get("tag-id")].append((index,ref,
                        pd.to_datetime(ref.get("deploy-on-date"),errors="coerce",utc=True),
                        pd.to_datetime(ref.get("deploy-off-date"),errors="coerce",utc=True)))
                ref_sources[source["source_id"]]={"path":str(path),"sha256":file["sha256"],"rows":refs,"by_tag":by_tag}
            wanted=expected.get(file["sha256"])
            if not wanted:continue
            verified=set();count=0
            with path.open(encoding="utf-8-sig",newline="") as stream:
                for number,raw in enumerate(csv.DictReader(stream),1):
                    count=number
                    if number not in wanted:continue
                    if raw!=wanted[number]:raise ValueError("Raw publisher record differs from retained original-field JSON")
                    verified.add(number)
            if verified!=set(wanted):raise ValueError("Some original conflict records are absent")
            checks.append({"path":str(path),"sha256":file["sha256"],"records_scanned":count,
                           "conflict_original_records_verified":len(verified),"all_original_fields_equal":True,
                           "published_checksum":file.get("published_checksum")})
    if set(expected)!={row["sha256"] for row in checks}:raise ValueError("Original source verification incomplete")
    groups=[]
    native_exclusions=[]
    for animal,block in rows.groupby("animal",sort=True,observed=True):
        source_id,local=animal.split(":",1)[0],animal.rsplit(":",1)[-1]
        refs=ref_sources[source_id]
        source_rows=block.drop_duplicates(["source_sha256","source_record"])
        parsed=[json.loads(v) for v in source_rows.reported_fields_json]
        time_groups=source_rows.groupby("timestamp",sort=True,observed=True)
        diagnostics=[]
        owned_refs=[dict(r,reference_row_number=i+2) for i,r in enumerate(refs["rows"]) if r.get("animal-id")==local]
        for stamp,observations in time_groups:
            matching=[];other=[];unknown_end_others=[]
            tags=set(observations.tag_id_reported)
            candidates=[candidate for tag in tags for candidate in refs["by_tag"].get(tag,[])]
            for index,ref,on,off in candidates:
                if pd.isna(on) or on>stamp or (pd.notna(off) and off<stamp):continue
                item={"reference_row_number":index,"animal_id":ref["animal-id"],"tag_id":ref["tag-id"],
                      "deployment_id":ref.get("deployment-id"),"deploy_on":ref.get("deploy-on-date"),"deploy_off":ref.get("deploy-off-date")}
                if ref["animal-id"]==local:matching.append(item)
                elif pd.isna(off):unknown_end_others.append(item)
                else:other.append(item)
            diagnostics.append({"timestamp_ns":int(stamp.value),"timestamp_reported":sorted(set(observations.timestamp_reported)),
                                "tag_ids":sorted(tags),"event_ids":observations.event_id_reported.tolist(),
                                "source_records":observations.source_record.astype(int).tolist(),
                                "matching_owner_deployments":matching,"other_declared_active_owners":other,
                                "other_owner_rows_with_unknown_end":unknown_end_others})
        values=collections.defaultdict(collections.Counter)
        quality_names=["gps:hdop","gps:satellite-count","gps:maximum-signal-strength","gps-time-to-fix","manually-marked-outlier","algorithm-marked-outlier","visible","sensor-type"]
        for raw in parsed:
            for field in quality_names:
                if field in raw:values[field][raw[field]]+=1
        delegated=(source_id=="a1d2d3ab-adcd-4101-8775-e0559445b612" or local=="Nest151" and source_id=="21114ef4-73b2-4aaa-a38a-3db8b0f2c3d4")
        status="delegated_independent_source_review" if delegated else "native_same_clock_measurement_ambiguity_no_valid_winner_rule_found"
        if any(d["other_declared_active_owners"] for d in diagnostics):status="deployment_namespace_overlap_requires_review_before_time_hold"
        groups.append({"animal":animal,"source_id":source_id,"local_id":local,"conflict_native_times":time_groups.ngroups,
                       "original_records":len(source_rows),"all_input_copies_records":len(block),
                       "multi_tag_native_times":sum(len(d["tag_ids"])>1 for d in diagnostics),
                       "multiple_reported_timestamp_strings_for_one_time":sum(len(d["timestamp_reported"])>1 for d in diagnostics),
                       "times_with_matching_owner_deployment":sum(bool(d["matching_owner_deployments"]) for d in diagnostics),
                       "times_with_other_declared_active_owner":sum(bool(d["other_declared_active_owners"]) for d in diagnostics),
                       "times_with_other_owner_unknown_end":sum(bool(d["other_owner_rows_with_unknown_end"]) for d in diagnostics),
                       "reference":{"path":refs["path"],"sha256":refs["sha256"],"owner_rows":owned_refs},
                       "quality_values":{k:dict(v) for k,v in values.items()},"diagnosis":status,"native_times":diagnostics})
        if not delegated and status=="native_same_clock_measurement_ambiguity_no_valid_winner_rule_found":
            for (input_path,input_sha),copy in block.groupby(["input_path","input_sha256"],sort=True,observed=True):
                native_exclusions.append({"input_path":input_path,"input_sha256":input_sha,"input_animal":animal,
                                          "timestamps_ns":sorted(set(copy.timestamp.astype("int64").tolist())),
                                          "time_axis":"reported_nominal","reason":"unresolved_native_duplicate_time_observations",
                                          "rows_in_this_occurrence":len(copy),"source_original_sha256":sorted(set(copy.source_sha256))})
    result={"schema":"wildcast-native-time-source-review-v1","code_sha256":sha(__file__),
            "extraction":{"path":str(extraction),"sha256":sha(extraction)},"original_record_checks":checks,
            "summary":{"raw_id_groups":len(groups),"original_conflict_records_verified":len(unique),
                       "native_conflict_times":sum(g["conflict_native_times"] for g in groups),
                       "groups_with_multiple_tags_at_one_time":sum(g["multi_tag_native_times"]>0 for g in groups),
                       "groups_with_other_active_declared_owner":sum(g["times_with_other_declared_active_owner"]>0 for g in groups),
                       "source_native_time_proposals":len(native_exclusions)},
            "groups":groups,"proposed_source_native_timestamp_exclusions":native_exclusions,
            "policy_status":"PROPOSAL_ONLY_NOT_APPLIED_REQUIRES_SCIENTIFIC_REVIEW",
            "interpretation":["Identical native reported time strings and distinct original event/record IDs prove ambiguity predates clock/QC export",
                              "One tag at each native time plus matching source owner/deployment supports retaining original raw namespace, not a new physical merge",
                              "Missing deployment ends are uncertainty, not proof of overlapping animal ownership",
                              "No guessed subsecond ordering, timestamp shifts, interpolation, coordinate means, HDOP-to-metres conversion or first-row winner",
                              "Exact-time removal of all variants is proposed only where no source-derived valid winner can be established; other observations and identity/fold protections remain"],
            "primary_sources":[{"url":"https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145402",
                                "finding":"Kestrel schedules range1s to10min; paper quality exclusions use satellite/DOP/altitude fields absent from this MDR event header. No public rule resolves repeated native timestamps."},
                               {"url":"https://link.springer.com/article/10.1186/s40462-019-0170-8",
                                "finding":"Bat loggers schedule30/60min fixes; author removes HDOP>=10 and handling fixes. All122 competing records at61times have reported HDOP<10, so this rule does not resolve their ownership/time ambiguity."}],
            "scope":"Original observation/source metadata only; no case targets, predictions, errors or outcomes used"}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(output),"sha256":sha(output),"summary":result["summary"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--extraction",required=True);parser.add_argument("--output",required=True)
    a=parser.parse_args();review(a.extraction,a.output)
