"""Fit small habitat utilities while preserving the frozen movement ensemble."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .data import SPLITS
from .habitat import HabitatForecast
from .model import energy_score
from .raster_context import SatelliteFields
from .sources import dump
from .train import DeviceBatches,load_windows


def train(root,cache,fields,output,epochs=50,patience=10,batch_size=512,lr=.0003,seed=42,device="auto",identity_audit=None):
    torch.set_num_threads(min(8,torch.get_num_threads()));torch.manual_seed(seed)
    device="cuda" if device=="auto" and torch.cuda.is_available() else "cpu" if device=="auto" else device
    a,manifest=load_windows(root);cache,out=Path(cache),Path(output);out.mkdir(parents=True,exist_ok=True)
    cached=json.loads((cache/"manifest.json").read_text())
    if cached["windows_sha256"]!=manifest["windows_sha256"]:raise ValueError("Habitat proposal cache belongs to different windows")
    ids=np.load(cache/"ids.npy");mapping=np.full(len(a["meta"]),-1,np.int64);mapping[ids]=np.arange(len(ids))
    points=np.load(cache/"points.npy",mmap_mode="r")
    bases=torch.load(cache/"bases.pt",map_location="cpu",weights_only=True)
    field=SatelliteFields.from_cache(fields)
    config={"base_configs":bases["configs"],"field_metadata":field.metadata,"n_drivers":a["drivers"].shape[1],"width":48,"inference_candidates":1024}
    model=HabitatForecast(config,field).to(device)
    for base,state in zip(model.bases,bases["states"]):base.load_state_dict(state)
    previous_year=pd.to_datetime(a["aux"][:,2],unit="s",utc=True).year.to_numpy()-1
    eligible=np.zeros(len(a["meta"]),bool)
    for record in field.metadata:
        west,south,east,north=record["bounds"]
        eligible|=(previous_year==record["year"])&(a["aux"][:,0]>=south)&(a["aux"][:,0]<=north)&(a["aux"][:,1]>=west)&(a["aux"][:,1]<=east)
    ti=np.flatnonzero((a["meta"][:,0]==SPLITS["train"])&eligible)
    vi=np.flatnonzero(np.isin(a["meta"][:,0],[SPLITS["val_temporal"],SPLITS["val_individual"]]))
    if identity_audit:
        audit=json.loads(Path(identity_audit).read_text())
        from .identity import verify_audit
        verify_audit(audit,manifest,a)
        vi=vi[~np.isin(vi,audit["excluded_windows"])]
    if np.any(mapping[np.r_[ti,vi]]<0):raise ValueError("Missing train/validation prior proposals")
    _,inv,counts=np.unique(a["meta"][ti][:,[2,4]],axis=0,return_inverse=True,return_counts=True)
    sampler=torch.utils.data.WeightedRandomSampler(torch.as_tensor(1/counts[inv],dtype=torch.double),len(ti),replacement=True,generator=torch.Generator().manual_seed(seed))
    train_all=a["meta"][:,0]==SPLITS["train"];norm={}
    scales=a["aux"][:,8]
    for study,horizon in np.unique(a["meta"][train_all][:,[2,4]],axis=0):
        take=train_all&(a["meta"][:,2]==study)&(a["meta"][:,4]==horizon)
        norm[int(study),int(horizon)]=max(float((np.linalg.norm(a["y"][take],axis=-1)*scales[take]).mean()),1.)

    def dataset(ix):
        weights=np.asarray([scales[i]/norm.get((int(a['meta'][i,2]),int(a['meta'][i,4])),1.) for i in ix],np.float32)
        return TensorDataset(torch.from_numpy(a["x"][ix]),torch.from_numpy(a["meta"][ix,4].astype(np.float32)),
             torch.from_numpy(a["aux"][ix,4:6].astype(np.float32)),torch.from_numpy(a["drivers"][ix]),torch.from_numpy(a["aux"][ix]),
             torch.from_numpy(previous_year[ix].astype(np.int64)),torch.from_numpy(np.asarray(points[mapping[ix]])),
             torch.from_numpy(a["y"][ix]),torch.from_numpy(weights))
    loader=DeviceBatches(dataset(ti),batch_size,device,sampler)
    vloader=DeviceBatches(dataset(vi),batch_size,device)
    _,cells=np.unique(a["meta"][vi][:,[0,2,4]],axis=0,return_inverse=True)
    distance=np.linalg.norm(a["y"][vi],axis=-1)*scales[vi]
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.05)
    training={"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in parameters),
              "epochs":epochs,"lr":lr,"training_cases_with_fields":len(ti),"scope":"Training targets only; global validation selects checkpoint, fresh tests untouched.",
              "proposal_cache":cached,"fields_manifest_sha256":hashlib.sha256((Path(fields)/"manifest.json").read_bytes()).hexdigest(),
              "correction":"Bounded log utility +/-2, zero whenever origin or candidate habitat is missing. Base ensemble frozen."}
    if identity_audit:training["identity_audit_sha256"]=hashlib.sha256(Path(identity_audit).read_bytes()).hexdigest()
    dump(out/"config.json",{**training,"model":config})

    def validate():
        model.eval();values=[]
        gen=torch.Generator(device=device).manual_seed(19000)
        with torch.inference_mode():
            for x,h,v,dr,aux,year,q,truth,w in vloader:
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
                    utility=model.utility(x,h,v,dr,aux,year,q)
                choose=torch.multinomial(utility.softmax(-1),64,replacement=True,generator=gen)
                draws=q.gather(1,choose[...,None].expand(-1,-1,2))
                draws=torch.where((utility.abs().amax(-1)==0)[:,None,None],q[:,:64],draws)
                values.append(energy_score(draws,truth))
        error=torch.cat(values).cpu().numpy()*scales[vi]
        return float(np.mean([error[cells==k].mean()/max(distance[cells==k].mean(),1.) for k in np.unique(cells)]))

    def save(epoch,score):
        torch.save({"model_family":model.family,"config":config,"model":model.state_dict(),"training":training,
                    "manifest":manifest,"epoch":epoch,"val_score":score},out/"best.pt.tmp")
        (out/"best.pt.tmp").replace(out/"best.pt")
    best=validate();save(0,best);print(json.dumps({"epoch":0,"validation_score":best}),flush=True)
    stale=0;history=[]
    for epoch in range(epochs):
        model.train();model.bases.eval();start=time.monotonic();losses=[]
        for x,h,v,dr,aux,year,q,truth,w in loader:
            # Draw a different subset of the frozen prior pool each iteration.
            q=q[:,torch.randperm(q.shape[1],device=device)[:64]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
                utility=model.utility(x,h,v,dr,aux,year,q)
            prob=utility.softmax(-1)
            accuracy=torch.linalg.vector_norm(q-truth[:,None],dim=-1)
            pair=torch.cdist(q,q)
            es=(prob*accuracy).sum(-1)-.5*torch.einsum("bi,bij,bj->b",prob,pair,prob)
            loss=(es*w).mean()+.01*utility.square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("Invalid habitat utility loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();losses.append(loss.detach())
        score=validate();row={"epoch":epoch+1,"train_loss":float(torch.stack(losses).mean().cpu()),"val_macro_normalized_energy":score,"seconds":round(time.monotonic()-start,2)}
        history.append(row);dump(out/"history.json",history);print(json.dumps(row),flush=True)
        if score<best:best,stale=score,0;save(epoch+1,best)
        else:stale+=1
        if stale>=patience:break
    return {"output":str(out),"best_validation_score":best,"epochs":len(history)}
