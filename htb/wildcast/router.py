"""Conditional probabilistic stacking with animal-group cross fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from .baselines import AnalogueForecaster
from .model import MovementTransformer

EXPERTS = ["neural", "analogue", "stationary", "recent_home_range", "contracted_neural"]


def export_analogue(analogue):
    return [{"species":int(s),"horizon":int(h),"features":torch.from_numpy(np.asarray(tree.data).copy()),
             "targets":torch.from_numpy(targets.copy()),"centre":torch.from_numpy(loc.copy()),"spread":torch.from_numpy(scale.copy())}
            for (s,h),(tree,targets,loc,scale) in sorted(analogue.indices.items())]


def import_analogue(records):
    analogue=AnalogueForecaster()
    analogue.indices={}
    for r in records:
        analogue.indices[r["species"],r["horizon"]]=(cKDTree(r["features"].cpu().numpy()),r["targets"].cpu().numpy(),
             r["centre"].cpu().numpy(),r["spread"].cpu().numpy())
    return analogue


def routing_features(x, horizon, drivers, indices):
    # Causal, label-free summaries; the feature list is frozen before fitting.
    return torch.cat([drivers[:,indices], x[:,-1], x.mean(1), torch.log1p(horizon)[:,None]],-1)


class ExpertRouter(nn.Module):
    def __init__(self,n_features,conditional=False,width=24):
        super().__init__()
        initial=torch.tensor([.65,.25,.04,.03,.03]).log().repeat(4,1)
        self.logits=nn.Parameter(initial)
        self.conditional=conditional
        if conditional:
            self.network=nn.Sequential(nn.LayerNorm(n_features),nn.Linear(n_features,width),nn.GELU(),nn.Linear(width,len(EXPERTS)))
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,features,horizon):
        hs=torch.tensor([1.,12.,24.,168.],device=features.device)
        code=(horizon[:,None]-hs).abs().argmin(-1)
        logits=self.logits[code]
        if self.conditional:
            logits=logits+2*torch.tanh(self.network(features)/2)
        return logits.softmax(-1)


def expert_draws(bases,analogue,a,ix,count,device,generator):
    from .evaluate import ensemble_predictions
    neural=ensemble_predictions(bases,a,ix,count,device,generator)
    seed=int(torch.randint(2**31,(),device=device,generator=generator).item())
    rng=np.random.default_rng(seed)
    analogue_samples=torch.from_numpy(analogue.sample(a["x"][ix],a["meta"][ix],a["aux"][ix,4:6],count,rng)).to(device)
    x=torch.from_numpy(a["x"][ix]).to(device)
    positions=x[...,:2].sign()*torch.expm1(x[...,:2].abs().clamp(max=15))
    choice=torch.randint(x.shape[1],(len(ix),count),device=device,generator=generator)
    home=positions.gather(1,choice[...,None].expand(-1,-1,2))
    return torch.stack([neural,analogue_samples,torch.zeros_like(neural),home,neural*.25],dim=1)


def mixture_terms(draws,truth):
    """Unbiased ES terms using independent draw indices for ALL expert pairs.

    Cross-expert same-index pairs are excluded too, because neural and
    contracted-neural draws deliberately share randomness at each index.
    """
    b,k,m,_=draws.shape
    accuracy=torch.linalg.vector_norm(draws-truth[:,None,None],dim=-1).mean(-1)
    distance=torch.cdist(draws.flatten(1,2),draws.flatten(1,2)).reshape(b,k,m,k,m)
    total=distance.sum((2,4))
    diag=distance.diagonal(dim1=2,dim2=4).sum(-1)
    pair=(total-diag)/(m*(m-1))
    return accuracy,pair


def weighted_energy(probability,accuracy,pair):
    return (probability*accuracy).sum(-1)-.5*torch.einsum("bi,bij,bj->b",probability,pair,probability)


class RoutedForecast(nn.Module):
    family="wildcast_routed_ensemble"
    def __init__(self,config,analogue_records=None):
        super().__init__();self.config=config
        self.bases=nn.ModuleList([MovementTransformer(**v) for v in config["base_configs"]])
        if config.get("crossfit"):
            self.fold_gates=nn.ModuleList([ExpertRouter(config["n_features"],config["conditional"],config.get("width",24)) for _ in range(3)])
            self.fold_of_animal={int(i):int(hashlib.sha256(('router-v1:'+config.get('fold_groups',{}).get(name,name)).encode()).hexdigest()[:8],16)%3
                                 for name,i in config["animal_names"].items()}
        else:self.gate=ExpertRouter(config["n_features"],config["conditional"],config.get("width",24))
        self.analogue_records=analogue_records
        self.analogue=import_analogue(analogue_records) if analogue_records is not None else None

    def predict_samples(self,a,ix,count,device,generator):
        if self.analogue is None:raise ValueError("Training-only analogue index is missing")
        h=torch.from_numpy(a["meta"][ix,4].astype(np.float32)).to(device)
        x=torch.from_numpy(a["x"][ix]).to(device);dr=torch.from_numpy(a["drivers"][ix]).to(device)
        features=routing_features(x,h,dr,self.config["driver_indices"])
        if self.config.get("crossfit"):
            fold=torch.tensor([self.fold_of_animal.get(int(v),int(v)%3) for v in a["meta"][ix,3]],device=device)
            p=torch.stack([gate(features,h) for gate in self.fold_gates],dim=1)
            probabilities=p[torch.arange(len(ix),device=device),fold]
        else:probabilities=self.gate(features,h)
        draws=expert_draws(list(self.bases),self.analogue,a,ix,count,device,generator).permute(0,2,1,3)
        choice=torch.multinomial(probabilities,count,replacement=True,generator=generator)
        return draws.gather(2,choice[...,None,None].expand(-1,-1,1,2)).squeeze(2)


def build_bank(root,base_root,checkpoints,output,samples=32,batch_size=256,device="auto"):
    from .train import load_windows
    from .evaluate import load_model
    from .data import SPLITS
    from .sources import dump
    torch.set_num_threads(min(8,torch.get_num_threads()))
    device="cuda" if device=="auto" and torch.cuda.is_available() else "cpu" if device=="auto" else device
    a,manifest=load_windows(root);b,_=load_windows(base_root)
    for key in ["x","y","meta","aux","memory"]:
        if key in a and not np.array_equal(a[key],b[key]):raise ValueError("Base and contextual forecast cases differ")
    del b
    loaded=[load_model(p,device) for p in checkpoints];models=[v[0] for v in loaded]
    train=a["meta"][:,0]==SPLITS["train"]
    analogue=AnalogueForecaster().fit(a["x"][train],a["y"][train],a["meta"][train],a["aux"][train,4:6])
    ids=np.flatnonzero(np.isin(a["meta"][:,0],[SPLITS["val_individual"],SPLITS["val_temporal"]]))
    names=manifest["driver_names"]
    prefixes=("memory_","past_","log_memory","log_context","solar_","season_","photoperiod","weather_")
    driver_indices=[i for i,n in enumerate(names) if n.startswith(prefixes)]
    accuracies=[];pairs=[];features=[]
    gen=torch.Generator(device=device).manual_seed(7001)
    with torch.inference_mode():
        for start in range(0,len(ids),batch_size):
            ix=ids[start:start+batch_size]
            draws=expert_draws(models,analogue,a,ix,samples,device,gen)
            accuracy,pair=mixture_terms(draws,torch.from_numpy(a["y"][ix]).to(device))
            accuracies.append(accuracy.cpu().numpy());pairs.append(pair.cpu().numpy())
            h=torch.from_numpy(a["meta"][ix,4].astype(np.float32)).to(device)
            x=torch.from_numpy(a["x"][ix]).to(device);dr=torch.from_numpy(a["drivers"][ix]).to(device)
            features.append(routing_features(x,h,dr,driver_indices).cpu().numpy())
            if start%(batch_size*20)==0:print(f"Router bank {start}/{len(ids)}",flush=True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Study/horizon normalizers come exclusively from the original training set.
    norm={}
    scales=a["aux"][:,8]
    for s,h in np.unique(a["meta"][train][:,[2,4]],axis=0):
        take=train&(a["meta"][:,2]==s)&(a["meta"][:,4]==h)
        norm[int(s),int(h)]=max(float((np.linalg.norm(a["y"][take],axis=-1)*scales[take]).mean()),1.)
    fallback=max(float((np.linalg.norm(a["y"][train],axis=-1)*scales[train]).mean()),1.)
    loss_weights=np.asarray([scales[i]/norm.get((int(a['meta'][i,2]),int(a['meta'][i,4])),fallback) for i in ids],np.float32)
    animal_names={int(v):k for k,v in manifest['animals'].items()}
    folds=np.asarray([int(hashlib.sha256(('router-v1:'+animal_names[int(animal)]).encode()).hexdigest()[:8],16)%3
                      for animal in a["meta"][ids,3]],np.int64)
    np.savez_compressed(out/"bank.npz",ids=ids,accuracy=np.concatenate(accuracies),pair=np.concatenate(pairs),features=np.concatenate(features),
                        meta=a["meta"][ids],aux=a["aux"][ids],fold=folds,loss_weights=loss_weights)
    torch.save({"base_configs":[ck["config"] for _,ck in loaded],"base_states":[m.state_dict() for m in models],
                "analogue":export_analogue(analogue)},out/"experts.pt")
    report={"source_data":str(root),"base_data":str(base_root),"windows_sha256":manifest["windows_sha256"],"driver_indices":driver_indices,
            "samples_per_expert":samples,"experts":EXPERTS,"cases":len(ids),"fold_rule":"SHA256(router-v1:animal_name) mod 3; every case of an animal stays together",
            "base_checkpoints":[{"path":str(p),"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in checkpoints],
            "manifest":manifest,"scope":"Validation-derived calibration development. Out-of-fold predictions exclude each animal from router fitting; base architecture selection previously used validation. Not an untouched test."}
    dump(out/"bank.json",report)
    return {k:v for k,v in report.items() if k not in ["manifest"]}


def fit_router(bank,output,conditional=False,steps=250,lr=.02,seed=42,device="auto"):
    from .sources import dump
    import pandas as pd
    torch.set_num_threads(min(8,torch.get_num_threads()));torch.manual_seed(seed)
    device="cuda" if device=="auto" and torch.cuda.is_available() else "cpu" if device=="auto" else device
    bank,out=Path(bank),Path(output);out.mkdir(parents=True,exist_ok=True)
    a=dict(np.load(bank/"bank.npz"));info=json.loads((bank/"bank.json").read_text());manifest=info["manifest"]
    features=torch.from_numpy(a["features"]).to(device);h=torch.from_numpy(a["meta"][:,4].astype(np.float32)).to(device)
    accuracy=torch.from_numpy(a["accuracy"]).to(device);pair=torch.from_numpy(a["pair"]).to(device)
    _,cell,counts=np.unique(a["meta"][:,[0,2,4]],axis=0,return_inverse=True,return_counts=True)
    weights=torch.from_numpy((a["loss_weights"]/counts[cell]*len(cell)/len(counts)).astype(np.float32)).to(device)
    predictions=np.zeros_like(a["accuracy"]);states=[];trace=[]
    for fold in range(3):
        train=torch.from_numpy((a["fold"]!=fold)&~a.get("excluded",np.zeros(len(a["fold"]),bool))).to(device)
        assessment=torch.from_numpy(a["fold"]==fold).to(device)
        if not train.any() or not assessment.any():raise ValueError("Empty animal cross-fitting partition")
        gate=ExpertRouter(features.shape[-1],conditional).to(device)
        optimizer=torch.optim.AdamW(gate.parameters(),lr=lr,weight_decay=.01)
        initial=gate.logits.detach().clone()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            p=gate(features[train],h[train])
            es=weighted_energy(p,accuracy[train],pair[train])
            loss=(es*weights[train]).mean()+1e-4*(gate.logits-initial).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("Invalid mixture calibration loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(gate.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            if step%50==0:
                row={"fold":fold,"step":step,"fit_loss":float(loss.detach().cpu())};trace.append(row);print(json.dumps(row),flush=True)
        gate.eval()
        with torch.inference_mode():predictions[assessment.cpu().numpy()]=gate(features[assessment],h[assessment]).cpu().numpy()
        states.append({k:v.detach().cpu() for k,v in gate.state_dict().items()})
    with torch.inference_mode():
        value=weighted_energy(torch.from_numpy(predictions).to(device),accuracy,pair).cpu().numpy()
    report=pd.DataFrame({"window":a["ids"],"split":a["meta"][:,0].astype(int),"study":a["meta"][:,2].astype(int),
                         "animal":a["meta"][:,3].astype(int),"horizon_hours":a["meta"][:,4].astype(int),"fold":a["fold"],
                         "router_es_m":value*a["aux"][:,8],
                         "neural_es_m":(a["accuracy"][:,0]-.5*a["pair"][:,0,0])*a["aux"][:,8],
                         "dont_move_es_m":a["accuracy"][:,2]*a["aux"][:,8]})
    for i,name in enumerate(EXPERTS):report["weight_"+name]=predictions[:,i]
    report.to_parquet(out/"crossfit-windows.parquet",index=False)
    cells=report.groupby(["split","study","horizon_hours"])[["router_es_m","neural_es_m","dont_move_es_m"]].mean()
    eligible=~a.get("excluded",np.zeros(len(a["fold"]),bool))
    cells=report[eligible].groupby(["split","study","horizon_hours"])[["router_es_m","neural_es_m","dont_move_es_m"]].mean()
    scores=cells[["router_es_m","neural_es_m"]].div(cells.dont_move_es_m.clip(lower=1),axis=0)
    experts=torch.load(bank/"experts.pt",map_location="cpu",weights_only=True)
    config={"base_configs":experts["base_configs"],"n_features":features.shape[-1],"conditional":conditional,
            "width":24,"driver_indices":info["driver_indices"],"crossfit":True,"animal_names":manifest["animals"],"folds":3,
            "fold_groups":info.get("fold_groups",{})}
    model=RoutedForecast(config,experts["analogue"])
    model.fold_gates=nn.ModuleList([ExpertRouter(features.shape[-1],conditional) for _ in range(3)])
    for gate,state in zip(model.fold_gates,states):gate.load_state_dict(state)
    for base,state in zip(model.bases,experts["base_states"]):base.load_state_dict(state)
    training={"parameters":sum(p.numel() for p in model.parameters()),"seed":seed,"steps":steps,"conditional":conditional,
              "scope":info["scope"],"fold_rule":info["fold_rule"],"bank_sha256":hashlib.sha256((bank/"bank.npz").read_bytes()).hexdigest()}
    if "identity_audit" in info:training["identity_audit"]=info["identity_audit"]
    torch.save({"model_family":model.family,"model":model.state_dict(),"config":config,"analogue":experts["analogue"],
                "training":training,"manifest":manifest,"epoch":steps,"val_score":float(scores.router_es_m.mean())},out/"best.pt")
    result={"crossfit_macro":scores.groupby("split").mean().to_dict(orient="index"),"trace":trace,"training":training,
            "note":"Each animal's weights were predicted by a gate fitted without that animal. All scores here use 32 source draws per expert and analytic mixture integration; confirm with common 64-sample evaluation."}
    dump(out/"crossfit-metrics.json",result);dump(out/"config.json",{**training,"model":config})
    return {"output":str(out),"crossfit_macro":result["crossfit_macro"]}
