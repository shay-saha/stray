"""Read-only whole-track review and recovery inventory for legacy alias links.

No target arrays, forecasts, scores or cached histories are modified. Original
event row numbers and timestamp strings are kept in memory only for source
lineage. Output distinguishes duplicate observations from biological identity.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from wildcast.data import ALIASES, distance

SOURCE_PATHS = {
    "jaguar-morato-2018": "data/combined/raw/jaguar-morato-2018/events.csv",
    "8fed4a87-14d3-4f9d-b630-e1d406b15c68": "data/combined/raw/8fed4a87-14d3-4f9d-b630-e1d406b15c68/Movement ecology of the jaguar in the largest floodplain of the world the Brazilian Pantanal.csv",
    "aae5d565-fd19-4fd4-8e2c-b6b5ca88ac5a": "data/combined/raw/aae5d565-fd19-4fd4-8e2c-b6b5ca88ac5a/MPIAB Argos white stork tracking (1991-2017)-gps.csv",
    "12d8637e-80b8-4850-994f-71f6d6734007": "data/migration-marine-stage1/raw/12d8637e-80b8-4850-994f-71f6d6734007/Free-ranging foraging trips of Manx shearwaters.csv"}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_events(path, source, wanted):
    """Historical eligible GPS rows, with original values/row lineage intact."""
    header = pd.read_csv(path, nrows=0).columns
    mapping = {key: next(name for name in candidates if name in header) for key, candidates in ALIASES.items()}
    tags = next((name for name in ("tag-local-identifier", "tag.local.identifier") if name in header), None)
    event = next((name for name in ("event-id", "Event_ID") if name in header), None)
    extras = [name for name in (tags, event, "sensor-type", "visible", "manually-marked-outlier", "algorithm-marked-outlier") if name and name in header]
    frames, offset = [], 0
    for block in pd.read_csv(path, dtype=str, usecols=list(set(mapping.values()) | set(extras)), chunksize=200_000):
        out = pd.DataFrame({key: block[column] for key, column in mapping.items()})
        out["raw_animal_id"], out["raw_study"] = out.animal, out.study
        out["animal"] = source + ":" + out.study.astype(str) + ":" + out.species.astype(str) + ":" + out.animal.astype(str)
        chosen = out.animal.isin(wanted)
        if "sensor-type" in block:
            chosen &= block["sensor-type"].str.lower().eq("gps")
        if "visible" in block:
            chosen &= block.visible.str.lower().isin(["true", "1"])
        for flag in ("manually-marked-outlier", "algorithm-marked-outlier"):
            if flag in block:
                chosen &= ~block[flag].str.lower().isin(["true", "1"])
        out["row_number"] = np.arange(offset + 2, offset + len(block) + 2)
        offset += len(block)
        out["timestamp_reported"], out["lat_reported"], out["lon_reported"] = out.timestamp, out.lat, out.lon
        out["tag_id"] = block[tags] if tags else ""
        out["event_id"] = block[event] if event else ""
        out = out.loc[chosen].copy()
        out["timestamp"] = pd.to_datetime(out.timestamp, utc=True, errors="coerce",
            format="%m/%d/%y %H:%M" if source == "jaguar-morato-2018" else "mixed").dt.as_unit("ns")
        out["lat"], out["lon"] = pd.to_numeric(out.lat, errors="coerce"), pd.to_numeric(out.lon, errors="coerce")
        out = out.loc[out.timestamp.notna() & out.lat.between(-90,90) & out.lon.between(-180,180)]
        out["source"], out["input_path"] = source, str(path)
        frames.append(out)
    return pd.concat(frames, ignore_index=True), offset


def pair_metrics(a, b):
    a, b = [f.drop_duplicates(["timestamp", "lat", "lon"]) for f in (a,b)]
    both = a.merge(b, on="timestamp", suffixes=("_a", "_b"))
    exact = (both.lat_a == both.lat_b) & (both.lon_a == both.lon_b)
    kilometres = distance(both.lat_a.to_numpy(), both.lon_a.to_numpy(), both.lat_b.to_numpy(), both.lon_b.to_numpy())/1000
    examples = both.loc[kilometres > 1].copy()
    examples["distance_km"] = kilometres[kilometres > 1]
    nearest = []
    for left, right in ((a,b),(b,a)):
        left = left.sort_values("timestamp").drop_duplicates("timestamp")
        right = right.sort_values("timestamp").drop_duplicates("timestamp").rename(columns={"timestamp":"other_time"})
        matches = pd.merge_asof(left,right,left_on="timestamp",right_on="other_time",direction="nearest",tolerance=pd.Timedelta(seconds=60),suffixes=("_a","_b"))
        present = matches.other_time.notna(); matched = matches.loc[present]
        metres = distance(matched.lat_a.to_numpy(),matched.lon_a.to_numpy(),matched.lat_b.to_numpy(),matched.lon_b.to_numpy())
        nearest.append({"from_rows":len(left),"matched_within60seconds":int(present.sum()),
                        "coordinate_exact":int(((matched.lat_a==matched.lat_b)&(matched.lon_a==matched.lon_b)).sum()),
                        "within_0_1m":int((metres <= .1).sum()),"within1m":int((metres <= 1).sum()),
                        "maximum_separation_metres":float(metres.max()) if len(metres) else None,
                        "maximum_time_offset_seconds":float((matched.timestamp-matched.other_time).dt.total_seconds().abs().max()) if len(matched) else None})
    evidence = []
    for row in examples.sort_values("distance_km",ascending=False).head(3).itertuples():
        evidence.append({"timestamp":row.timestamp.isoformat(),"a_original_row":int(row.row_number_a),"b_original_row":int(row.row_number_b),
                         "a_timestamp_reported":row.timestamp_reported_a,"b_timestamp_reported":row.timestamp_reported_b,"distance_km":row.distance_km})
    return {"a_rows":len(a),"b_rows":len(b),"simultaneous_pairs":len(both),"exact_simultaneous_pairs":int(exact.sum()),
            "all_simultaneous_identical":bool(len(both) and exact.all()),
            "exact_fraction":float(exact.mean()) if len(both) else None,
            "separation_km_quantiles_0_50_90_99_100":np.quantile(kilometres,[0,.5,.9,.99,1]).tolist() if len(kilometres) else [],
            "same_time_pairs_above1km":int((kilometres > 1).sum()),"contradiction_original_row_examples":evidence,
            "a_tuples_subset_b":bool(len(a)==int(exact.sum()) and exact.all()),
            "b_tuples_subset_a":bool(len(b)==int(exact.sum()) and exact.all()),
            "nearest_time_diagnostics_a_to_b_and_b_to_a":nearest}


def audit(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    old_path=Path("data/combined-landscape/processed/quality-report.json")
    combined_path=Path("reports/migration-combined-identity-v3.json")
    old, combined=[json.loads(path.read_text()) for path in (old_path,combined_path)]
    parent_path=Path(combined["parent_manifest"]["path"]);parent=json.loads(parent_path.read_text())
    if digest(parent_path)!=combined["parent_manifest"]["sha256"]:raise ValueError("Parent manifest changed")
    links=[{**r,"historical_scope":"pre_v3_destructive_alias"} for r in old["alias_links"]]+[{**r,"historical_scope":"new_Manx_exact_alias"} for r in combined["exact_GPS_links"]]
    names={r[key] for r in links for key in ["a","b"]}
    frames, inputs, references = [], [], defaultdict(list)
    for source, value in SOURCE_PATHS.items():
        path=Path(value);metadata_path=path.parent/"source.json";metadata=json.loads(metadata_path.read_text())
        record={"path":value,"resolved_path":str(path.resolve()),"sha256":digest(path),"bytes":path.stat().st_size,
                "source_id":source,"metadata_path":str(metadata_path),"metadata_sha256":digest(metadata_path)}
        if source=="jaguar-morato-2018":
            archive=path.parent/"DataS1_v1_0.zip"
            if digest(archive)!=metadata["sha256"]:raise ValueError("Author Jaguar archive SHA differs")
            with zipfile.ZipFile(archive) as z:
                member=next(n for n in z.namelist() if n.endswith("jaguar_movement_data.csv"))
                if hashlib.sha256(z.read(member)).hexdigest()!=record["sha256"]:raise ValueError("Jaguar event CSV differs from original archive member")
                extra=next(n for n in z.namelist() if n.endswith("Jaguar_additional information.csv"))
                ref=pd.read_csv(io.BytesIO(z.read(extra)),dtype=str).fillna("")
                for row in ref.to_dict("records"):
                    references[source+":"+row["ID"]].append({k:row.get(k,"") for k in ["ID","Sex","Estimated Age","Collar Type","Collar Brand","Planned Schedule"]})
                record["archive"]={"path":str(archive),"sha256":digest(archive),"member":member,"reference_member":extra,"reference_member_sha256":hashlib.sha256(z.read(extra)).hexdigest()}
        else:
            declared=next(r for r in metadata["files"] if r["name"]==path.name)
            if declared["sha256"]!=record["sha256"]:raise ValueError("Published event source SHA differs")
            md5=digest(path,"md5")
            if md5!=declared["published_checksum"]["value"]:raise ValueError("Published event MD5 differs")
            record.update(published_checksum=declared["published_checksum"],published_checksum_verified=True,original_url=declared["url"])
            refpath=next(path.parent.glob("*reference-data.csv"));rf=pd.read_csv(refpath,dtype=str).fillna("")
            record["reference"]={"path":str(refpath),"sha256":digest(refpath)}
            for row_number,row in enumerate(rf.to_dict("records"),2):
                references[source+":"+row["animal-id"]].append({"reference_row_number":row_number,**{k:row.get(k,"") for k in
                    ["animal-id","tag-id","animal-ring-id","animal-sex","deploy-on-date","deploy-off-date","deployment-id","tag-serial-no","tag-processing-type","tag-model"]}})
        frame,total=raw_events(path,source,names)
        record.update(raw_file_records=total,selected_valid_GPS_rows=len(frame),parser="Historical nominal axis; raw timestamp text retained, no precision/clock upgrade",
                      timestamp_examples=frame.timestamp_reported.drop_duplicates().head(3).tolist(),
                      raw_tags=sorted(frame.tag_id.dropna().unique().tolist()))
        inputs.append(record);frames.append(frame)
    frame=pd.concat(frames,ignore_index=True);groups={name:g for name,g in frame.groupby("animal",sort=False)}
    rows=[];retained=[];retracted=[];observational=[]
    for link in links:
        a,b=groups[link["a"]],groups[link["b"]]
        metrics=pair_metrics(a,b)
        tags_a=set(a.tag_id.dropna());tags_b=set(b.tag_id.dropna())
        refs_a=references[a.source.iloc[0]+":"+a.raw_animal_id.iloc[0]]
        refs_b=references[b.source.iloc[0]+":"+b.raw_animal_id.iloc[0]]
        row={**link,**metrics,"a_tags":sorted(tags_a),"b_tags":sorted(tags_b),"a_reference_records":refs_a,"b_reference_records":refs_b}
        if metrics["same_time_pairs_above1km"]:
            decision="retract_physical_merge_retain_exclusion"
            retracted.append({"a":link["a"],"b":link["b"],"type":"incompatible_simultaneous_histories","reason":"Raw simultaneous positions differ by >1km; exact-subset matches do not justify whole-animal aliasing"})
        elif link["historical_scope"]=="new_Manx_exact_alias":
            decision="retract_whole_animal_union_pending_duplicate_deployment_ownership"
            retracted.append({"a":link["a"],"b":link["b"],"type":"duplicate_telemetry_biological_owner_unresolved","reason":"Large identical telemetry overlap but different reported animal/tag/deployment IDs; some animals also have other unrelated-year deployments. Retain one uncertainty exclusion block, no whole-animal physical union."})
            observational.append({"a":link["a"],"b":link["b"],"type":"identical_GPS_deployment_stream","evidence":{"exact_simultaneous_pairs":metrics["exact_simultaneous_pairs"],"a_tags":sorted(tags_a),"b_tags":sorted(tags_b)},"biological_identity_certified":False})
        elif metrics["all_simultaneous_identical"] and tags_a & tags_b:
            decision="retain_cross_archive_same_collar_observation_alias"
            retained.append({"a":link["a"],"b":link["b"],"type":"same_collar_cross_archive_GPS_duplicate","evidence":{"shared_tag_ids":sorted(tags_a&tags_b),"exact_simultaneous_pairs":metrics["exact_simultaneous_pairs"],"time_precision":"Author Morato file has minute precision; MDR retains seconds. Same source publication doi:10.1002/ecy.2379.","nearest_time_diagnostics":metrics["nearest_time_diagnostics_a_to_b_and_b_to_a"]}})
        else:raise ValueError("Unclassified identity evidence")
        row["decision"]=decision;rows.append(row)
    parents={name:name for name in {r[k] for r in old["alias_links"] for k in ["a","b"]}}
    def find(name):
        while parents[name]!=name:parents[name]=parents[parents[name]];name=parents[name]
        return name
    for link in old["alias_links"]:parents[find(link["b"])]=find(link["a"])
    components=defaultdict(list)
    for name in parents:components[find(name)].append(name)
    input_path=Path("data/combined-landscape/processed/observations.parquet");input_sha=digest(input_path)
    schema=pq.read_schema(input_path)
    recovery=[]
    for canonical,members in sorted(components.items()):
        inherited={parent["identity_mapping"][n] for n in members if n in parent["identity_mapping"]}
        source_records=[r for r in inputs if any(n.startswith(r["source_id"]+":") for n in members)]
        recovery.append({"input_path":str(input_path),"input_sha256":input_sha,"collapsed_canonical":canonical,
            "raw_ids":sorted(members),"original_sources":[{"path":r["path"],"sha256":r["sha256"],"source_id":r["source_id"],"status":"verified_original_retained_premerge_rows_available"} for r in source_records],
            "raw_id_rows":{n:len(groups[n]) for n in members},"raw_id_tags":{n:sorted(groups[n].tag_id.dropna().unique().tolist()) for n in members},
            "raw_id_sequence_sha256":{n:hashlib.sha256(groups[n][["row_number","timestamp_reported","lat_reported","lon_reported","event_id","tag_id"]].to_csv(index=False).encode()).hexdigest() for n in members},
            "inherited_parent_canonical_members":sorted(inherited),"inherited_roles":{n:parent.get("assignments",{}).get(n) for n in inherited},
            "has_any_inherited_parent_member":bool(inherited),"freshness":"Conservatively nonfresh; original exclusion component must remain intact",
            "recovery":"Reparse originals by original source/study/local-animal identifier BEFORE aliasing, timestamp de-duplication or hourly thinning. Preserve nominal timestamp text, tag/event IDs and logical CSV row numbers. Do not split the already-collapsed table and assume discarded histories are recovered.",
            "old_origin_owner_caveat":"Collapsed observations lack original-animal/row lineage. Exact co-located records can have multiple raw owners; historical stable file/row order can reproduce the old chosen row, but cannot certify biological identity."})
    raw_names=set(parents)
    result={"schema":"wildcast-legacy-identity-source-review-v1","created_utc":datetime.now(timezone.utc).isoformat(),"code_sha256":digest(__file__),
        "scope":"Raw GPS identity/concordance and original-source recovery audit only. No cached targets, predictions, scores, fits or cache edits.",
        "inputs":inputs,"quality_report":{"path":str(old_path),"sha256":digest(old_path)},
        "combined_audit":{"path":str(combined_path),"sha256":digest(combined_path)},"parent_manifest":combined["parent_manifest"],
        "collapsed_observation_schema":str(schema),"summary":{"legacy_edges":len(old["alias_links"]),"new_Manx_edges":len(combined["exact_GPS_links"]),
            "retained_same_collar_cross_archive_edges":len(retained),"retracted_physical_edges":len(retracted),"raw_simultaneous_contradiction_edges":sum(r["same_time_pairs_above1km"]>0 for r in rows),
            "Manx_duplicate_telemetry_owner_unresolved":len(observational),"legacy_raw_ids":len(raw_names),"legacy_components":len(components),
            "legacy_raw_ids_absent_combined_fold_groups":len(raw_names-set(combined["fold_groups"])),
            "legacy_raw_ids_absent_parent_identity_mapping":len(raw_names-set(parent["identity_mapping"]))},
        "retained_edges":retained,"retracted_edges":retracted,"observational_duplicate_links":observational,"pair_reviews":rows,"original_recovery":recovery,
        "legacy_raw_ids_absent_combined_fold_groups":sorted(raw_names-set(combined["fold_groups"])),
        "legacy_raw_ids_absent_parent_identity_mapping":sorted(raw_names-set(parent["identity_mapping"])),
        "limitations":["Coordinates and timestamps are original reported/accepted nominal values, not verified instrument time/accuracy.",
          "Near-time matching is diagnostic of source formatting; it never interpolates observations or creates target labels.",
          "Fully matching telemetry establishes observational duplication, not automatic biological identity across different reported animals.",
          "All legacy exclusion/fold roles must survive physical-history reconstruction, even when original histories are separated."]}
    for record in inputs:
        if digest(record["path"])!=record["sha256"]:raise ValueError("Original bytes changed during review")
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"output":str(output),"sha256":digest(output),"summary":result["summary"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    audit(parser.parse_args().output)
