"""Finite CUDA cost/precision probe; no animal data, weights or accuracy result."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from wildcast.migration_model import MigrationModel, masked_joint_nll, mixture_energy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=3)
    args=p.parse_args()
    if args.steps < 1:raise ValueError("Positive finite timing steps required")
    out=Path(args.output)
    if out.exists():raise FileExistsError("Use a new cost-probe report path")
    torch.set_num_threads(8)
    device=torch.device(args.device)
    report={"scope":"Synthetic cost/precision probe only; no animal data or retained trained weights",
            "torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(device),
            "device":str(device),"steps_per_candidate":args.steps,
            "model_sha256":hashlib.sha256((PROJECT/"wildcast/migration_model.py").read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"dtype":"float32",
            "matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,"candidates":[],"status":"running"}
    out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=out.with_suffix(".tmp");temp.write_text(json.dumps(report,indent=2)+"\n");temp.replace(out)
    save()
    for width,layers,batch in [(128,2,256),(128,2,512),(512,6,256),(512,6,512),(768,8,256),(768,8,512)]:
        torch.manual_seed(42);torch.cuda.reset_peak_memory_stats(device)
        model=optimizer=inputs=target=mask=parameters=loss=energy=None
        row={"width":width,"layers":layers,"batch_size":batch,"components":8}
        try:
            model=MigrationModel(10,18,140,[1/24,.5,1,3,7,14,30,60,90],history_length=27,
                width=width,layers=layers,components=8,variant="history_direct",dropout=.1,
                min_scale=5e-6,covariance_jitter=1e-12).to(device)
            row["parameters"]=sum(x.numel() for x in model.parameters())
            inputs=(torch.randn(batch,27,10,device=device),torch.ones(batch,27,dtype=torch.bool,device=device),
                    torch.randn(batch,18,device=device),torch.randint(0,140,(batch,),device=device))
            target=torch.randn(batch,9,2,device=device)*.01
            mask=torch.rand(batch,9,device=device)>.25;mask[:,0]=True;target[~mask]=torch.nan
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
            def step():
                nonlocal parameters,loss,energy
                optimizer.zero_grad(set_to_none=True)
                parameters=model(*inputs)
                nll=masked_joint_nll(parameters,target,mask)/(2*mask.sum(-1).clamp_min(1))
                energy=mixture_energy(parameters,target,mask,repeats=2)
                loss=(.1*nll+.5*(energy["endpoint_energy"]+energy["path_energy"])).mean()
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite synthetic loss")
                loss.backward()
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),5.)):
                    raise FloatingPointError("Nonfinite synthetic gradients")
                optimizer.step()
            step();torch.cuda.synchronize(device)
            began=time.perf_counter()
            for _ in range(args.steps):step()
            torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-began
            row.update(status="passed",seconds_per_training_step=elapsed/args.steps,
                synthetic_origins_per_second=batch*args.steps/elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        except torch.cuda.OutOfMemoryError:
            row.update(status="out_of_memory",peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        finally:
            model=optimizer=inputs=target=mask=parameters=loss=energy=None
            gc.collect();torch.cuda.empty_cache()
        report["candidates"].append(row);save()
        print(json.dumps(row),flush=True)
    report["status"]="complete";save()


if __name__ == "__main__":main()
