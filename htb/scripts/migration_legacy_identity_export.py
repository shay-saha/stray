"""Recover original source-ID GPS histories without changing any old cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from migration_legacy_identity_review import digest, raw_events, SOURCE_PATHS
from wildcast.data import ALIASES


def export(review_path, output, maximum_rows=1_000_000, maximum_bytes=100_000_000):
    output=Path(output);review_path=Path(review_path)
    if output.exists():raise FileExistsError("Recovery outputs are immutable")
    review=json.loads(review_path.read_text())
    legacy={r[k] for r in review["pair_reviews"] if r["historical_scope"]=="pre_v3_destructive_alias" for k in ["a","b"]}
    if len(legacy)!=72:raise ValueError("Expected the exact reviewed72raw-ID cohort")
    source_records={r["source_id"]:r for r in review["inputs"]}
    frames,source_qc=[],[]
    for source,path in SOURCE_PATHS.items():
        selected={n for n in legacy if n.startswith(source+":")}
        if not selected:continue
        record=source_records[source];path=Path(path)
        if digest(path)!=record["sha256"]:raise ValueError("Reviewed original bytes changed")
        frame,total=raw_events(path,source,selected)
        # Parse coordinate numbers through the same CSV float parser as the
        # historical reader. Original decimal strings survive separately.
        # Association is by original record number, never by coordinates.
        header=pd.read_csv(path,nrows=0).columns
        lat=next(n for n in ALIASES["lat"] if n in header)
        lon=next(n for n in ALIASES["lon"] if n in header)
        qc_columns=[n for n in ["sensor-type","visible","manually-marked-outlier","algorithm-marked-outlier","modelled"] if n in header]
        original=pd.read_csv(path,usecols=[lat,lon,*qc_columns])
        positions=frame.row_number.to_numpy(dtype=int)-2
        old_lat,old_lon=frame.lat.to_numpy().copy(),frame.lon.to_numpy().copy()
        frame["lat"],frame["lon"]=original[lat].iloc[positions].to_numpy(dtype=float),original[lon].iloc[positions].to_numpy(dtype=float)
        for name in qc_columns:
            frame["raw_"+name.replace("-","_")]=original[name].iloc[positions].astype("string").to_numpy()
        frame["study"]=source+":"+frame.raw_study
        frame["timestamp_nominal"]=frame.timestamp
        frame["timestamp_utc"]=pd.Series(pd.NaT,index=frame.index,dtype="datetime64[ns, UTC]")
        frame["time_standard_status"]="legacy_reported_nominal_axis_not_instrument_verified_UTC"
        frame["original_input_sha256"]=record["sha256"]
        frame["original_row_id"]=record["sha256"]+":"+frame.row_number.astype(str)
        frame["row_qc_status"]="historical_GPS_sensor_visibility_outlier_eligible_not_new_quality_certification"
        frame["source_modelled_flag_present"]="modelled" in header
        frame["source_clock_mode"]="unresolved" if source=="jaguar-morato-2018" else "bounded_utc_or_gps"
        frames.append(frame)
        evidence={"source_id":source,"original_path":str(path),"original_sha256":record["sha256"],
            "original_file_records":total,"recovered_rows":len(frame),"raw_ids":sorted(selected),
            "source_metadata":{"path":record["metadata_path"],"sha256":record["metadata_sha256"]},
            "row_filters":"Historical explicitGPS(ifreported), visible true(ifreported), explicitoutlierflags false, validtimestamp/lat/lon. No aliasing/thinning/interpolation/speedfilter or future-error filtering.",
            "row_qc_columns_retained":qc_columns,"modelled_flag_present":"modelled" in header,
            "modelled_flag_absence_is_not_proof_of_unmodelled_processing":True,
            "coordinate_parser":"Original pandasCSV float parser; raw decimal strings retained; row association solely original record number",
            "decimal_string_parser_binary_ulp_differences":int(np.sum((old_lat!=frame.lat.to_numpy())|(old_lon!=frame.lon.to_numpy()))),
            "clock":{"mode":"unresolved" if source=="jaguar-morato-2018" else "bounded_utc_or_gps",
                "timestamp_format":"%m/%d/%y %H:%M" if source=="jaguar-morato-2018" else "Original ISO strings with source precision retained",
                "source_precision":"Reported minute precision; no seconds invented" if source=="jaguar-morato-2018" else "Exact reported text retained; storage ns is not instrument accuracy",
                "subday_admission":"Exclude unresolved original Jaguar axis; inherit root documentary policy for UTC-or-GPS sources, no newly verifiedUTC claim",
                "absolute_UTC_and_instrument_accuracy":"not established by this recovery"}}
        if "published_checksum" in record:evidence["published_checksum"]=record["published_checksum"]
        if "archive" in record:evidence["archive"]=record["archive"]
        for name in ["README.txt","README.source.md"]:
            p=path.parent/name
            if p.exists():evidence.setdefault("clock_documentation",[]).append({"path":str(p),"sha256":digest(p)})
        source_qc.append(evidence)
    result=pd.concat(frames,ignore_index=True)
    if len(result)>maximum_rows or set(result.animal)!=legacy:raise ValueError("Recovered cohort/finite row bound differs")
    if result.original_row_id.isna().any() or not result.original_row_id.is_unique:raise ValueError("Original row lineage is not unique")
    if not result["animal"].equals(result.source+":"+result.raw_study+":"+result.species+":"+result.raw_animal_id):raise ValueError("Raw source identity changed")
    old_input=Path(review["original_recovery"][0]["input_path"])
    old_sha=review["original_recovery"][0]["input_sha256"]
    if digest(old_input)!=old_sha:raise ValueError("Frozen old observation input changed")
    collapsed=[r["collapsed_canonical"] for r in review["original_recovery"]]
    existing=pq.read_table(old_input,columns=["animal"],filters=[("animal","in",collapsed)]).to_pandas()
    selectors=sorted(existing.animal.unique().tolist())
    if len(selectors)!=23:raise ValueError("Expected exactly23surviving collapsed selectors")
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=output.name+".pending-",dir=output.parent))
    try:
        processed=stage/"processed";processed.mkdir()
        observations=processed/"observations.parquet"
        result.to_parquet(observations,index=False)
        if observations.stat().st_size>maximum_bytes:raise ValueError("Recovery file exceeded declared finite byte bound")
        final_path=output/"processed/observations.parquet"
        sources=[{"path":r["original_path"],"sha256":r["original_sha256"]} for r in source_qc]
        output_record={"path":str(final_path),"sha256":digest(observations),"rows":len(result),"bytes":observations.stat().st_size,
            "columns":list(result.columns),"schema":str(pq.read_schema(observations)),"raw_ids":sorted(legacy),
            "row_id_column":"original_row_id","source_QC_provenance":source_qc,
            "replaces":[{"input_path":str(old_input),"input_sha256":old_sha,"input_animal":n} for n in selectors],
            "clock_evidence":{"rule":"Inherited original nominal reported axes; no verifiedUTC or precision upgrade", "sources":{r["source_id"]:r["clock"] for r in source_qc}},
            "status":"original_source_ID_histories_unmixed; further preparation/clock admission belongs to reviewed repair protocol"}
        manifest={"schema":"wildcast-legacy-source-ID-recovery-v1","complete":True,"created_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"Exact source-ID/history recovery only, no biological re-identification, source-quality promotion, targets, model error inspection or fitting",
            "review":{"path":str(review_path),"sha256":digest(review_path)},"code_sha256":digest(__file__),
            "review_code_sha256":digest(Path(__file__).with_name("migration_legacy_identity_review.py")),
            "row_id_column":"original_row_id","outputs":[output_record],"original_recovery":review["original_recovery"],
            "bindings":{"parent_manifest":review["parent_manifest"],"old_observations":{"path":str(old_input),"sha256":old_sha},"original_sources":sources},
            "summary":{"raw_ids":len(legacy),"recovered_rows":len(result),"previously_lost_raw_ids":49,"replaced_collapsed_selectors":len(selectors),
                       "former_components_without_surviving_selector":sorted(set(collapsed)-set(selectors)),"all_recovered_IDs_conservatively_nonfresh":True},
            "quality_limits":["All original histories are recovered, including those removed by old minpoints. New scientific eligibility is not implied.",
                "Unknown/modelled processing and manipulation must retain source metadata QA; this export adds no observation accuracy.",
                "Identical/co-located old origins can have multiple source owners. Never reassign to a nearest-time row or a different physical animal.",
                "Preserve every inherited old component as an exclusion/fold-risk group; restore raw histories without making them fresh.",
                "NewManx raw histories were already uncollapsed in their QC input and are not replaced here; duplicate deployment ownership remains a separate QA issue."]}
        (processed/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n")
        for r in source_qc:
            if digest(r["original_path"])!=r["original_sha256"]:raise ValueError("Original source changed during export")
        if output.exists():raise FileExistsError("Recovery output appeared during export")
        stage.rename(output)
        print(json.dumps({"output":str(output),"manifest_sha256":digest(output/"processed/manifest.json"),"observations_sha256":output_record["sha256"],"summary":manifest["summary"]},indent=2))
        return manifest
    finally:
        if stage.exists():shutil.rmtree(stage)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--review",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();export(args.review,args.output)
