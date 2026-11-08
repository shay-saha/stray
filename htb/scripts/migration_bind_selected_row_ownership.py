"""Bind certified historical source-row replay to an immutable repair partition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(base,proof,metadata,output):
    base,proof,metadata,output=map(Path,[base,proof,metadata,output])
    c,p,m=[json.loads(path.read_text()) for path in [base,proof,metadata]]
    if p.get("complete") is not True or not all(p["checks"].values()):
        raise ValueError("A complete independent source-row replay proof is required")
    record=p["historical_selected_rows"]
    if sha(record["path"])!=record["sha256"] or m["ownership_table_sha256"]!=record["sha256"]:
        raise ValueError("Historical selected-row output differs")
    if m.get("all_old_tuples_reproduced_exactly") is not True or not m.get("source_order"):
        raise ValueError("Historical selection order was not reproduced")
    if m.get("replay_code_sha256")!=p["code_sha256"]:
        raise ValueError("Stable selector metadata lacks the reviewed replay-code binding")
    if m["old_observation_sha256"]!=p["old_observations"]["sha256"]:
        raise ValueError("Frozen original-input hash differs")
    outputs=c["original_recovery"]["outputs"]
    if m["recovered_observation_sha256"] not in {r["sha256"] for r in outputs}:
        raise ValueError("Historical replay uses a different source-owned recovery")
    selected=pd.read_parquet(record["path"])
    if len(selected)!=record["rows"] or len(selected)!=m["rows"] or selected.original_row_id.duplicated().any():
        raise ValueError("Historical selected-row inventory is not unique or complete")
    recovered=pd.concat([pd.read_parquet(row["path"],columns=["original_row_id","animal","timestamp","lat","lon"]) for row in outputs],ignore_index=True)
    joined=selected.merge(recovered,on="original_row_id",how="left",suffixes=("_selected","_recovered"),validate="one_to_one")
    for column in ["animal","timestamp","lat","lon"]:
        if not joined[column+"_selected"].equals(joined[column+"_recovered"]):
            raise ValueError("Selected row is not exactly owned by its declared recovered source record")
    if set(selected.animal)-c["physical_mapping"].keys():raise ValueError("Selected owner is absent from reviewed raw-ID graph")
    original_exclusions=c["exclusion_mapping"].copy()
    original_physical=c["physical_mapping"].copy()
    c["historical_selected_rows"]={"path":record["path"],"sha256":record["sha256"],
                                    "metadata":{"path":str(metadata),"sha256":sha(metadata)},
                                    "proof":{"path":str(proof),"sha256":sha(proof)},
                                    "row_id_column":record["row_id_column"],"rows":record["rows"]}
    c["inputs"]["historical_selected_row_metadata"]={"path":str(metadata),"sha256":sha(metadata)}
    c["inputs"]["historical_selected_row_proof"]={"path":str(proof),"sha256":sha(proof)}
    c["inputs"]["historical_selected_row_table"]={"path":record["path"],"sha256":record["sha256"]}
    c["historical_selection_binding_derivation"]={"parent_contract":{"path":str(base),"sha256":sha(base)},
                                                "code_sha256":sha(__file__),"kind":"Add certified source-row replay only; physical/risk/role/freshness/source-QC partitions unchanged"}
    if c["physical_mapping"]!=original_physical or c["exclusion_mapping"]!=original_exclusions:
        raise AssertionError("Replay binding changed the frozen identity partition")
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(c,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(output),"sha256":sha(output),"historical_rows":len(selected)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["base","proof","metadata","output"]:parser.add_argument("--"+name,required=True)
    args=parser.parse_args();bind(args.base,args.proof,args.metadata,args.output)
