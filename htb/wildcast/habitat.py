"""Bounded destination-habitat correction to a frozen movement forecast."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import MovementTransformer
from .moveformer import candidate_features
from .raster_context import SatelliteFields


class HabitatForecast(nn.Module):
    family="wildcast_destination_habitat"
    def __init__(self,config,fields=None):
        super().__init__();self.config=config
        self.bases=nn.ModuleList([MovementTransformer(**v) for v in config["base_configs"]])
        for p in self.bases.parameters():p.requires_grad_(False)
        self.fields=fields if fields is not None else SatelliteFields(config["field_metadata"])
        width=config.get("width",48)
        self.query=nn.Sequential(nn.LayerNorm(config["n_drivers"]+23),nn.Linear(config["n_drivers"]+23,width),nn.GELU(),nn.Linear(width,width))
        self.candidate=nn.Sequential(nn.Linear(135,width),nn.GELU(),nn.Linear(width,width))
        nn.init.zeros_(self.candidate[-1].weight);nn.init.zeros_(self.candidate[-1].bias)

    def utility(self,x,h,v,drivers,aux,previous_year,points):
        with torch.no_grad():landscape=self.fields.candidate_features(points,aux,previous_year)
        clock=torch.log1p(h)
        features=torch.cat([drivers,x[:,-1],x.mean(1),torch.stack([clock,torch.sin(clock),torch.cos(clock)],-1)],-1)
        query=self.query(features)
        candidate=self.candidate(torch.cat([landscape,candidate_features(points,v)],-1))
        score=(candidate*query[:,None]).sum(-1)/query.shape[-1]**.5
        # The correction can redistribute mass but cannot create an arbitrary
        # near-delta forecast from weak landscape evidence. No evidence -> zero.
        return 2*torch.tanh(score.float()/2)*landscape[...,-1]

    def predict_samples(self,a,ix,count,device,generator):
        from .evaluate import ensemble_predictions
        points=ensemble_predictions(list(self.bases),a,ix,max(count,self.config.get("inference_candidates",1024)),device,generator)
        x=torch.from_numpy(a["x"][ix]).to(device);dr=torch.from_numpy(a["drivers"][ix]).to(device)
        h=torch.from_numpy(a["meta"][ix,4].astype(np.float32)).to(device)
        v=torch.from_numpy(a["aux"][ix,4:6].astype(np.float32)).to(device)
        aux=torch.from_numpy(a["aux"][ix]).to(device)
        year=torch.as_tensor(pd.to_datetime(a["aux"][ix,2],unit="s",utc=True).year.to_numpy()-1,device=device)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.startswith("cuda")):
            scores=self.utility(x,h,v,dr,aux,year,points)
        selected=torch.multinomial(scores.softmax(-1),count,replacement=True,generator=generator)
        draws=points.gather(1,selected[...,None].expand(-1,-1,2))
        # Unsupported years/areas retain ordinary independent base draws.
        unchanged=scores.abs().amax(-1)==0
        return torch.where(unchanged[:,None,None],points[:,:count],draws)


def cache_proposals(root,base_root,checkpoints,output,count=256,batch_size=256,device="auto"):
    from .train import load_windows
    from .evaluate import load_model,ensemble_predictions
    from .data import SPLITS
    from .sources import dump
    torch.set_num_threads(min(8,torch.get_num_threads()))
    device="cuda" if device=="auto" and torch.cuda.is_available() else "cpu" if device=="auto" else device
    a,manifest=load_windows(root);b,_=load_windows(base_root)
    for key in ["x","y","meta","aux","memory"]:
        if key in a and not np.array_equal(a[key],b[key]):raise ValueError("Context and base windows differ")
    del b
    loaded=[load_model(p,device) for p in checkpoints]
    ids=np.flatnonzero(np.isin(a["meta"][:,0],[SPLITS["train"],SPLITS["val_temporal"],SPLITS["val_individual"]]))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    points=np.lib.format.open_memmap(out/"points.npy",mode="w+",dtype=np.float32,shape=(len(ids),count,2))
    gen=torch.Generator(device=device).manual_seed(7171)
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            ix=ids[start:start+batch_size]
            points[start:start+len(ix)]=ensemble_predictions([m for m,_ in loaded],a,ix,count,device,gen).cpu().numpy()
            if start%(batch_size*40)==0:print(f"Habitat prior proposals {start}/{len(ids)}",flush=True)
    points.flush();np.save(out/"ids.npy",ids)
    torch.save({"configs":[ck["config"] for _,ck in loaded],"states":[m.state_dict() for m,_ in loaded]},out/"bases.pt")
    dump(out/"manifest.json",{"windows_sha256":manifest["windows_sha256"],"source_data":str(root),"base_data":str(base_root),"count":count,
         "cases":len(ids),"points_sha256":hashlib.sha256((out/"points.npy").read_bytes()).hexdigest(),
         "base_checkpoints":[{"path":str(p),"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in checkpoints],
         "scope":"Only train and validation proposals generated; fresh-test forecasts are not generated or inspected."})
    return {"output":str(out),"cases":len(ids),"proposals":count}
