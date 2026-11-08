"""Conservative cross-study identity checks independent of forecast errors."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sources import dump


def case_fingerprint(arrays):
    digest=hashlib.sha256()
    for key in ["x","y","meta","aux"]:
        digest.update(key.encode());digest.update(arrays[key].tobytes())
    return digest.hexdigest()


def verify_audit(info,manifest,arrays):
    if info["observations_sha256"]!=manifest["observations_sha256"]:
        raise ValueError("Identity audit belongs to different observations")
    if info["windows_sha256"]!=manifest["windows_sha256"]:
        if not info.get("case_fingerprint") or case_fingerprint(arrays)!=info["case_fingerprint"]:
            raise ValueError("Identity audit case IDs belong to different forecast windows")


def audit(root,raw_root,output):
    root,raw_root,output=Path(root),Path(raw_root),Path(output)
    manifest=json.loads((root/"processed/windows.json").read_text())
    observations=pd.read_parquet(root/"processed/observations.parquet")
    with np.load(root/"processed/windows.npz") as z:
        meta,aux=z["meta"],z["aux"]
        fingerprint=case_fingerprint(z) if "x" in z.files and "y" in z.files else None
    names=manifest["animals"];parent={name:name for name in names}
    def find(v):
        while parent[v]!=v:parent[v]=parent[parent[v]];v=parent[v]
        return v
    links=[]
    def join(a,b,evidence):
        aa,bb=find(a),find(b)
        if aa!=bb:parent[max(aa,bb)]=min(aa,bb)
        if a!=b:links.append({"a":a,"b":b,"evidence":evidence})
    by_local=defaultdict(list)
    for name,g in observations.groupby("animal",sort=False):
        species,source=g.species.iloc[0],g.source.iloc[0]
        local=name.split(":"+species+":",1)[-1]
        by_local[source,species,local].append(name)
    for key,members in by_local.items():
        for name in members[1:]:join(members[0],name,{"type":"same_source_species_local_identifier","key":list(key)})
    marks=defaultdict(set)
    used=[]
    for directory in sorted((raw_root/"raw").iterdir()):
        for p in directory.glob("*.csv"):
            try:
                header=pd.read_csv(p,nrows=0)
                if "animal-id" not in header or "animal-ring-id" not in header:continue
                cols=[c for c in ["animal-id","animal-taxon","animal-ring-id"] if c in header]
                f=pd.read_csv(p,usecols=cols,dtype=str).fillna("")
            except (ValueError,pd.errors.ParserError,UnicodeDecodeError):continue
            used.append(str(p))
            for _,r in f.iterrows():
                ring=" ".join(r["animal-ring-id"].strip().upper().split())
                if ring in {"","NA","NAN","NONE","UNKNOWN","0","-"}:continue
                species=r.get("animal-taxon","")
                candidates=by_local.get((directory.name,species,r["animal-id"]),[])
                for name in candidates:marks[directory.name,species,ring].add(name)
    for key,members in marks.items():
        members=sorted(members)
        for name in members[1:]:join(members[0],name,{"type":"same_source_species_ring_identifier","key":list(key)})
    clusters=defaultdict(list)
    for name in names:clusters[find(name)].append(name)
    fold_groups={name:find(name) for name in names if len(clusters[find(name)])>1}
    excluded_individual=set()
    for group,members in clusters.items():
        assignments={manifest["animal_assignments"][name] for name in members}
        if "temporal" in assignments:
            excluded_individual.update(names[name] for name in members if manifest["animal_assignments"][name] in ["val_individual","test_individual"])
        # A calibration label of the same physical animal also disqualifies a
        # test individual for the final stacked model.
        if "val_individual" in assignments:
            excluded_individual.update(names[name] for name in members if manifest["animal_assignments"][name]=="test_individual")
    groups_by_id={int(i):find(name) for name,i in names.items()}
    stamps=observations.timestamp.astype("int64").to_numpy()/1e9
    latest={}
    for i in np.flatnonzero(meta[:,0]==0):
        group=groups_by_id[int(meta[i,3])]
        timestamp=max(float(aux[i,2]),float(stamps[int(aux[i,7])]))
        latest[group]=max(latest.get(group,-np.inf),timestamp)
    exclusion=np.zeros(len(meta),bool)
    exclusion|=np.isin(meta[:,0],[3,4])&np.isin(meta[:,3],list(excluded_individual))
    for i in np.flatnonzero(np.isin(meta[:,0],[1,2])):
        if aux[i,2]<=latest.get(groups_by_id[int(meta[i,3])],-np.inf):exclusion[i]=True
    result={"observations_sha256":manifest["observations_sha256"],"windows_sha256":manifest["windows_sha256"],"case_fingerprint":fingerprint,
            "fold_groups":fold_groups,"links":links,"multi_record_clusters":[v for v in clusters.values() if len(v)>1],
            "excluded_individual_ids":sorted(excluded_individual),"excluded_windows":np.flatnonzero(exclusion).tolist(),
            "excluded_by_split":{s:int((exclusion&(meta[:,0]==code)).sum()) for s,code in {"val_temporal":1,"test_temporal":2,"val_individual":3,"test_individual":4}.items()},
            "references_checked":used,"rule":"Source/species/local ID plus source/species/ring ID. Exclude seen identities from individual assessment and temporal origins preceding any same-identity training endpoint.",
            "limitations":"Conservative public-metadata audit, not proof all cross-source aliases are resolved. Tag IDs alone are not used because tags may be redeployed."}
    dump(output,result)
    return result


def audit_router_bank(bank,output,identity_audit):
    import shutil
    bank,output=Path(bank),Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((bank/"bank.json").read_text());audit_info=json.loads(Path(identity_audit).read_text())
    if report["manifest"]["observations_sha256"]!=audit_info["observations_sha256"]:raise ValueError("Identity audit and calibration bank differ")
    a=dict(np.load(bank/"bank.npz"));names={int(i):name for name,i in report["manifest"]["animals"].items()}
    fold={i:int(hashlib.sha256(('router-v1:'+audit_info["fold_groups"].get(name,name)).encode()).hexdigest()[:8],16)%3 for i,name in names.items()}
    a["fold"]=np.asarray([fold[int(i)] for i in a["meta"][:,3]],np.int64)
    a["excluded"]=np.isin(a["ids"],audit_info["excluded_windows"])
    np.savez_compressed(output/"bank.npz",**a)
    shutil.copy2(bank/"experts.pt",output/"experts.pt")
    report.update(fold_groups=audit_info["fold_groups"],identity_audit={"sha256":hashlib.sha256(Path(identity_audit).read_bytes()).hexdigest(),
                   "excluded_calibration_cases":int(a["excluded"].sum()),"rule":audit_info["rule"]})
    report["fold_rule"]="SHA256(router-v1:canonical_physical_identity) mod 3, using source/species/local and ring links. Excluded identities do not fit calibration gates."
    dump(output/"bank.json",report)
    return report["identity_audit"]
