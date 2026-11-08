"""Extract exact contradictory-time source records; choose no measurement winner."""
from __future__ import annotations

import argparse
import collections
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


def extract(diagnostic,output):
    diagnostic,output=Path(diagnostic),Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    d=json.loads(diagnostic.read_text())
    names=sorted({r["raw_animals"][0] for r in d["groups"] if len(r["raw_animals"])==1})
    core=[]
    inputs=[]
    for source in d["sources"]:
        path=Path(source["path"])
        if sha(path)!=source["sha256"]:raise ValueError("Diagnostic source hash mismatch")
        columns=pq.ParquetFile(path).schema_arrow.names
        selected=[c for c in ["animal","timestamp","lat","lon"] if c in columns]
        frame=pq.read_table(path,columns=selected,filters=[("animal","in",names)],use_threads=False).to_pandas()
        frame["input_path"],frame["input_sha256"]=str(path),source["sha256"]
        core.append(frame)
        inputs.append({"path":str(path),"sha256":source["sha256"],"candidate_rows":len(frame)})
    allrows=pd.concat(core,ignore_index=True)
    positions=allrows[["animal","timestamp","lat","lon"]].drop_duplicates()
    keys=positions.loc[positions.duplicated(["animal","timestamp"],keep=False),["animal","timestamp"]].drop_duplicates()
    full=[]
    for source in inputs:
        if not source["candidate_rows"]:continue
        path=Path(source["path"])
        columns=pq.ParquetFile(path).schema_arrow.names
        wanted=[c for c in ["animal","timestamp","timestamp_nominal","lat","lon","individual_id_reported","tag_id_reported",
                "event_id_reported","timestamp_reported","latitude_reported","longitude_reported","sensor_reported",
                "processing","quality_diagnostics_complete","source_file","source_sha256","source_record",
                "reference_fields_json","reported_fields_json","original_source_file","original_source_sha256",
                "original_archive_member","original_source_record"] if c in columns]
        frame=pq.read_table(path,columns=wanted,filters=[("animal","in",names)],use_threads=False).to_pandas()
        frame=frame.merge(keys,on=["animal","timestamp"],how="inner",validate="many_to_one")
        frame["input_path"],frame["input_sha256"]=str(path),source["sha256"]
        full.append(frame)
    rows=pd.concat(full,ignore_index=True)
    rows.to_parquet(output/"conflicting-input-records.parquet",index=False)
    sources=[]
    for source_id in sorted({name.split(":",1)[0] for name in names}):
        directory=Path("data/migration-breadth-stage1/raw")/source_id
        manifest=directory/"source.json"
        s=json.loads(manifest.read_text())
        records=[]
        for entry in s["files"]:
            if entry.get("role") not in {"events","identity_reference","source_documentation"}:continue
            file=directory/entry["stored_as"]
            if sha(file)!=entry["sha256"]:raise ValueError("Source ORIGINAL hash mismatch")
            records.append({"path":str(file),"sha256":entry["sha256"],"role":entry["role"],"bytes":file.stat().st_size,
                            "published_checksum":entry.get("published_checksum"),"url":entry.get("url")})
        sources.append({"source_id":source_id,"manifest":str(manifest),"manifest_sha256":sha(manifest),"files":records})
    summary={"schema":"wildcast-within-raw-simultaneous-record-extraction-v1","code_sha256":sha(__file__),
             "diagnostic":{"path":str(diagnostic),"sha256":sha(diagnostic)},"inputs":inputs,"original_sources":sources,
             "conflicting_records":{"path":str(output/"conflicting-input-records.parquet"),"sha256":sha(output/"conflicting-input-records.parquet"),"rows":len(rows)},
             "raw_ids":len(names),"raw_id_timestamp_keys":len(keys),"scope":"Source observation evidence only; no measurement selection, interpolation, time shift, case labels or model outcomes."}
    (output/"extraction.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(output),"raw_ids":len(names),"keys":len(keys),"rows":len(rows)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic",required=True);parser.add_argument("--output",required=True)
    a=parser.parse_args();extract(a.diagnostic,a.output)
