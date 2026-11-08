"""Fast environment-conditioned mixtures of frozen joint migration experts."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .migration_ensemble import _batch_costs, _draw, load_member, macro_weights
from .migration_environment_features import Environment, PROTOCOL
from .migration_human_pressure import sha256

KIND="wildcast_migration_environment_gate"
MODES=("actual","availability","masked")


class FeatureStore:
    def __init__(self,root,hashes=None):
        self.root=Path(root);self.metadata=json.loads((self.root/"features.json").read_text())
        if self.metadata.get("protocol")!=PROTOCOL or self.metadata.get("uses_future_targets") is not False:
            raise ValueError("Unrecognized past-only environment cache")
        if any(self.metadata.get(k)!=v for k,v in (hashes or {}).items()):raise ValueError("Environment/GPS cache hashes differ")
        if sha256(self.root/"features.npz")!=self.metadata["features_sha256"]:raise ValueError("Environmental features checksum differs")
        with np.load(self.root/"features.npz",allow_pickle=False) as source:
            self.ids,self.origins,self.values=source["case_ids"],source["origin"],source["features"]
        if (self.values.shape!=(len(self.ids),len(self.metadata["feature_names"])) or np.any(np.diff(self.ids)<=0)
            or hashlib.sha256(self.ids.astype('<i8').tobytes()).hexdigest()!=self.metadata["case_ids_sha256"]):
            raise ValueError("Environment feature alignment differs")
        self.environment,self.human=None,None

    def lookup(self,a,ids):
        ids=np.asarray(ids,int);canonical=np.asarray(a["case_index"])[ids] if "case_index" in a else ids
        rows=np.searchsorted(self.ids,canonical);safe=np.minimum(rows,len(self.ids)-1)
        valid=(rows<len(self.ids))&(self.ids[safe]==canonical)
        valid &= np.all(self.origins[safe]==a["origin"][ids],axis=-1)
        if valid.all():return self.values[rows]
        result=np.empty((len(ids),self.values.shape[1]),np.float32);result[valid]=self.values[rows[valid]]
        if self.environment is None:
            self.environment=Environment(self.metadata["sources"])
            if self.environment.fingerprint!=self.metadata["source_fingerprint"]:raise ValueError("Environmental source changed")
            from .migration_deepssf import GHSLRaster
            hm=self.metadata["human_metadata"]
            self.human=GHSLRaster(hm["sources"],expected=hm["source_fingerprint"])
        from .migration_context_joint import features
        new=ids[~valid]
        result[~valid]=np.concatenate([features(self.human.context,a,new),self.environment.lookup(a["origin"][new])],-1)
        return result

    def close(self):
        if self.environment is not None:self.environment.close()
        if self.human is not None:self.human.close()


def predictors(a,ids,store,mode):
    if mode not in MODES:raise ValueError("Unknown environmental control")
    extra=store.lookup(a,ids).copy()
    if mode=="masked":extra.fill(0)
    elif mode=="availability":extra[:,store.metadata["value_columns"]]=0
    return np.concatenate([a["static"][ids],extra],-1).astype(np.float32)


class Gate(nn.Module):
    def __init__(self,features,species,weights,hidden=64):
        super().__init__()
        weights=torch.as_tensor(weights,dtype=torch.float64)
        if weights.ndim!=1 or torch.any(weights<=0) or not torch.isclose(weights.sum(),torch.tensor(1.,dtype=torch.float64)):
            raise ValueError("Positive simplex of expert weights required")
        self.config=dict(features=int(features),species=int(species),weights=weights.tolist(),hidden=int(hidden))
        self.register_buffer("base_weights",weights/weights.sum())
        self.register_buffer("mean",torch.zeros(features));self.register_buffer("std",torch.ones(features))
        self.species=nn.Embedding(species,8)
        self.network=nn.Sequential(nn.Linear(features+8,hidden),nn.SiLU(),nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,len(weights)))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,x,species):
        x=((x-self.mean)/self.std).clamp(-20,20)
        raw=self.network(torch.cat([x,self.species(species)],-1))
        weights=(self.base_weights.log()+2*raw.double().tanh()).softmax(-1)
        return .1*self.base_weights+.9*weights


class EnvironmentalEnsemble:
    def __init__(self,gate,members,store,mode,device):
        self.gate,self.members,self.store,self.mode,self.device=gate,members,store,mode,str(device)

    @torch.no_grad()
    def predict_cases(self,a,ids,samples,rng=None,baselines=None):
        ids=np.asarray(ids,int);rng=rng or np.random.default_rng(2026)
        x=torch.as_tensor(predictors(a,ids,self.store,self.mode),device=self.device)
        species=torch.as_tensor(a["species"][ids],device=self.device)
        weights=self.gate(x,species).cpu().numpy()
        selected=(rng.random((len(ids),samples,1))>weights.cumsum(-1)[:,None]).sum(-1)
        selected=np.minimum(selected,len(self.members)-1)
        result=np.empty((len(ids),samples,len(a["horizons"]),2),np.float32)
        for expert,member in enumerate(self.members):
            active=np.flatnonzero((selected==expert).any(-1))
            if not len(active):continue
            draws=_draw(member,a,ids[active],samples,np.random.default_rng(int(rng.integers(0,2**63-1))),self.device,baselines)
            for row,source in enumerate(active):
                mask=selected[source]==expert;result[source,mask]=draws[row,mask]
        return result


def load_model(checkpoint,device="cpu"):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if ck.get("model_kind")!=KIND:raise ValueError("Not an environmental ensemble")
    hashes={key:ck[key] for key in ("manifest_sha256","cases_sha256")}
    store=FeatureStore(ck["environment_artifact"],hashes)
    if store.metadata["features_sha256"]!=ck["environment_features_sha256"]:raise ValueError("Gate environmental feature table changed")
    members=[load_member(row["path"],device,row["sha256"],hashes)[0] for row in ck["members"]]
    gate=Gate(**ck["config"]).to(device);gate.load_state_dict(ck["model"]);gate.eval()
    return EnvironmentalEnsemble(gate,members,store,ck["mode"],device),ck


def coefficients(a,ids,members,evidence,samples,device,batch_size,seed):
    endpoint,joint=macro_weights(a,ids)
    A=np.empty((len(ids),len(members)),np.float32);D=np.empty((len(ids),len(members),len(members)),np.float32)
    started=time.monotonic()
    for start in range(0,len(ids),batch_size):
        end=min(start+batch_size,len(ids));batch=ids[start:end]
        draws=[]
        for k,member in enumerate(members):
            rng=np.random.default_rng(seed+start+k*10000019)
            draws.append(_draw(member,a,batch,2*samples,rng,device))
        costs=_batch_costs(np.stack(draws,1),a["origin"][batch],a["target_latlon"][batch],a["target_mask"][batch],device)
        A[start:end]=.5*(np.einsum('nh,nkh->nk',endpoint[start:end],costs["endpoint_first"])+joint[start:end,None]*costs["joint_first"])*len(ids)
        D[start:end]=.5*(np.einsum('nh,nklh->nkl',endpoint[start:end],costs["endpoint_pair"])+joint[start:end,None,None]*costs["joint_pair"])*len(ids)
        if start%(batch_size*64)==0:print(json.dumps({"coefficient_origins":end,"total":len(ids),"seconds":round(time.monotonic()-started)}),flush=True)
    if not np.isfinite(A).all() or not np.isfinite(D).all():raise ValueError("Nonfinite proper-score coefficients")
    return A,D


def prepare_costs(data,ensemble,output,max_train=200000,batch_size=512,device="cuda"):
    from .migration_evaluate import load_cases
    from .migration_train import load_training_data,sampling_probabilities
    from .migration_protocol import cross_taxon_exclusion_ids
    from .migration_deepssf import validation_panel
    torch.set_num_threads(8);torch.set_float32_matmul_precision("high")
    output=Path(output)
    if output.exists():raise FileExistsError("Frozen proper-score caches are immutable")
    _,_,audited=load_training_data(data)
    a,manifest,hashes=load_cases(data)
    if hashes!=audited:raise ValueError("GPS data changed after audit")
    source=torch.load(ensemble,map_location="cpu",weights_only=True)
    if source.get("model_kind")!="wildcast_migration_ensemble" or any(source.get(k)!=v for k,v in hashes.items()):raise ValueError("Incompatible parent ensemble")
    members,evidence=[],[]
    for row in source["members"]:
        model,info=load_member(row["path"],device,row["sha256"],hashes);members.append(model);evidence.append(info)
    train=np.flatnonzero((a["split"]==0)&a["target_mask"].any(-1))
    probability=sampling_probabilities(a,train,"species_sqrt",allowed_cross_taxon_exclusion_ids=cross_taxon_exclusion_ids(manifest))
    rng=np.random.default_rng(1701)
    train=np.sort(rng.choice(train,min(max_train,len(train)),replace=False,p=probability))
    validation=validation_panel(a,4096)
    arrays={"train_ids":train,"validation_ids":validation}
    for name,ids,samples in (("train",train,2),("validation",validation,8)):
        arrays[name+"_A"],arrays[name+"_D"]=coefficients(a,ids,members,evidence,samples,device,batch_size,1701)
    output.mkdir(parents=True,exist_ok=False);np.savez_compressed(output/"costs.npz",**arrays)
    metadata={**hashes,"parent_ensemble":str(Path(ensemble).resolve()),"parent_ensemble_sha256":sha256(ensemble),
        "members":source["members"],"weights":source["weights"].tolist(),"train_origins":len(train),
        "train_taxa":len(np.unique(a["species"][train])),"validation_origins":len(validation),"TRAIN_only_gate_fitting":True,
        "costs_sha256":sha256(output/"costs.npz"),"samples_per_independent_bank":{"train":2,"validation":8},
        "score":"Equal-split endpoint geodesic/joint chordal energy; identical macro identity/species/horizon weights; unbiased independent banks; no reserved truth"}
    (output/"costs.json").write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)
    return metadata


def fit(data,features,costs,output,mode="actual",device="cuda",epochs=40,patience=6,lr=.0003,batch_size=2048):
    from .migration_evaluate import load_cases
    torch.manual_seed(42);torch.set_num_threads(8)
    a,manifest,hashes=load_cases(data);store=FeatureStore(features,hashes);costs=Path(costs);metadata=json.loads((costs/"costs.json").read_text())
    if any(metadata.get(k)!=v for k,v in hashes.items()) or sha256(costs/"costs.npz")!=metadata["costs_sha256"]:raise ValueError("Coefficient cache differs")
    with np.load(costs/"costs.npz",allow_pickle=False) as source:bank={k:source[k] for k in source.files}
    train,validation=bank["train_ids"],bank["validation_ids"]
    if np.any(a["split"][train]!=0) or not np.isin(a["split"][validation],[1,2]).all():raise ValueError("Invalid environmental train/validation split")
    x=predictors(a,train,store,mode);vx=predictors(a,validation,store,mode)
    n_species=max(manifest["species"].values())+1
    gate=Gate(x.shape[1],n_species,metadata["weights"]).to(device)
    gate.mean.copy_(torch.as_tensor(x.mean(0),device=device));gate.std.copy_(torch.as_tensor(np.maximum(x.std(0),1e-3),device=device))
    tx=torch.as_tensor(x,device=device);vs=torch.as_tensor(a["species"][validation],device=device);ts=torch.as_tensor(a["species"][train],device=device);vxt=torch.as_tensor(vx,device=device)
    A=torch.as_tensor(bank["train_A"],device=device);D=torch.as_tensor(bank["train_D"],device=device)
    vA=torch.as_tensor(bank["validation_A"],device=device);vD=torch.as_tensor(bank["validation_D"],device=device)
    optimizer=torch.optim.AdamW(gate.parameters(),lr=lr,weight_decay=.01)
    output=Path(output)
    if output.exists():raise FileExistsError("Gate runs are immutable")
    output.mkdir(parents=True);history=[];best=float('inf');stale=0;started=time.monotonic()
    for epoch in range(epochs+1):
        if epoch:
            gate.train();order=torch.randperm(len(train),device=device)
            for start in range(0,len(order),batch_size):
                ix=order[start:start+batch_size];weights=gate(tx[ix],ts[ix]);energy=(weights*A[ix]).sum(-1)-.5*torch.einsum('ni,nij,nj->n',weights,D[ix],weights)
                kl=(weights*(weights.log()-gate.base_weights.log())).sum(-1)
                loss=energy.mean()+.05*kl.mean()
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(gate.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        gate.eval()
        with torch.no_grad():
            w=gate(vxt,vs);score=float(((w*vA).sum(-1)-.5*torch.einsum('ni,nij,nj->n',w,vD,w)).mean())
        if not np.isfinite(score):raise ValueError("Nonfinite environmental gate selection")
        row={"epoch":epoch,"selection_km":score,"seconds":round(time.monotonic()-started,2)};history.append(row)
        (output/"history.json").write_text(json.dumps(history,indent=2)+'\n')
        ck={"model_kind":KIND,"model":gate.state_dict(),"config":gate.config,"mode":mode,**hashes,
            "members":metadata["members"],"epoch":epoch,"metrics":{"selection":score},"selection_version":"environment-gate-fixed-panel-energy-v1",
            "environment_artifact":str(Path(features).resolve()),"environment_features_sha256":store.metadata["features_sha256"],
            "training":{"TRAIN_only_gate_fitting":True,"train_origins":len(train),"train_taxa":metadata["train_taxa"],"epochs":epochs,"lr":lr,"seed":42,"frozen_experts":True,
                "controls":"All arms same gate/initial mixture; actual includes content, availability removes content, masked removes all environmental fields", "coefficient_cache":str(costs.resolve()),"coefficients":metadata}}
        torch.save(ck,output/"last.pt")
        if score<best-1e-5:best=score;stale=0;torch.save(ck,output/"best.pt")
        elif epoch:stale+=1
        print(json.dumps({"mode":mode,**row,"best_km":best}),flush=True)
        if epoch and stale>=patience:break
    store.close();return {"best_km":best,"epochs":epoch,"mode":mode}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("costs")
    for name in ("data","ensemble","output"):c.add_argument('--'+name,required=True)
    c.add_argument('--max-train',type=int,default=200000);c.add_argument('--batch-size',type=int,default=512);c.add_argument('--device',default="cuda")
    f=sub.add_parser("fit")
    for name in ("data","features","costs","output"):f.add_argument('--'+name,required=True)
    f.add_argument('--mode',choices=MODES,default="actual");f.add_argument('--device',default="cuda")
    f.add_argument('--epochs',type=int,default=40);f.add_argument('--patience',type=int,default=6)
    args=vars(p.parse_args());command=args.pop('command')
    (prepare_costs if command=='costs' else fit)(**args)


if __name__=='__main__':main()
