#!/usr/bin/env python3
"""One-off server export of a recursive model scenario; no application mode."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


class YieldGpu(Exception):
    pass


def check_gpu():
    wanted=os.environ.get('ATLAS_EXPORT_GPU_UUID')
    if not wanted:return
    own={os.getpid()}
    own.update(int(x) for x in os.environ.get('ATLAS_EXPORT_ALLOWED_PIDS','').split(',') if x)
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('NSpid:'):own.update(map(int,line.split()[1:]))
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    for line in text.splitlines():
        pid,gpu=map(str.strip,line.split(','))
        if gpu==wanted and int(pid) not in own:raise YieldGpu('Another process is using the export GPU; yielding')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    temp=path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value,allow_nan=False,separators=(',',':')))
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--years',type=int,default=50)
    p.add_argument('--start',default='2026-11-08T00:00:00Z')
    p.add_argument('--samples',type=int,default=16)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    bundle=args.bundle.resolve();output=args.output.resolve()
    sys.path.insert(0,str(bundle))
    os.chdir(bundle)
    import torch
    try:check_gpu();device='cuda'
    except YieldGpu:device='cpu'
    torch.set_num_threads(64 if device=='cpu' else 2)
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.10)
    atomic_json(output/'progress.json',dict(state='running',phase='Loading selected ensemble on '+device,
        device=device,completed_days=0,total_years=args.years))
    print('Export device:',device,flush=True)
    torch.set_float32_matmul_precision('high')
    from verify_package import verify
    verify(bundle)
    # The published selected ensemble consists of four frozen joint members.
    # Use its exact member selection and distribution sampling implementation.
    import wildcast.migration_model as joint
    from wildcast.migration_ensemble import load_model as load_ensemble
    load_joint=joint.load_model
    rng=np.random.default_rng(2026)
    def load_selected(path,device):
        if Path(path).resolve()!=(bundle/'model/selected.pt').resolve():
            return load_joint(path,device)
        model,metadata=load_ensemble(path,device)
        model.config=model.members[0].config
        model.eval=lambda:model
        def sample(history,mask,static,species,samples,generator):
            if device=='cuda':check_gpu()
            data={key:value.detach().cpu().numpy() for key,value in zip(
                ('history','history_mask','static','species'),(history,mask,static,species))}
            data['horizons']=np.asarray(model.config['horizons'])
            draws=model.predict_cases(data,np.arange(len(history)),samples,rng=rng)
            if device=='cuda':time.sleep(.025)
            return torch.from_numpy(draws)
        model.sample_paths=sample
        return model,metadata
    joint.load_model=load_selected
    from cohort_forecast import CohortEngine,Observations
    from scenario import Scenarios,positions_at
    # The one-off exported scenario uses all trained horizons per 90-day cycle.
    # The application's separate local inference keeps its existing cadence.
    Scenarios.step_days=90
    engine=CohortEngine(bundle,device)
    dataset=engine.dataset(args.observations.resolve())
    scenarios=Scenarios(engine)
    started=time.monotonic()
    start=pd.Timestamp(args.start).tz_convert('UTC')
    end=start+pd.DateOffset(years=args.years)
    total_days=(end-start).total_seconds()/86400
    completed=0.;cycle=0;devices={device}
    def progress(phase,done,total):
        if done==total or done%512==0:
            atomic_json(output/'progress.json',dict(state='running',phase=phase,completed=done,total=total,
                completed_days=completed,total_days=total_days,cycles=cycle,device=device,elapsed_seconds=time.monotonic()-started))
    checkpoint=output/'resume.json'
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text())
        devices.update(saved.get('devices_used',[]))
        if saved['checkpoint_sha256']!=engine.checkpoint_hash or saved['observations_sha256']!=dataset.sha256:
            raise ValueError('Refusing to mix resumed output with another model or observation source')
        rows=pd.read_parquet(output/'resume.parquet')
        state=dict(id=saved['id'],dataset=dataset,issue=pd.Timestamp(saved['issue']),start=start,
              generation=saved['generation'],selected=list(rows.groupby('animal',sort=True)),
              sources=saved['sources'],samples=args.samples,seed=2026,excluded=saved['excluded'],result=saved['result'])
        scenarios.states[state['id']]=state
        rng.bit_generator.state=saved['rng']
        completed=float(saved['completed_days']);cycle=int(saved['cycle'])
        result=None
    else:
        result=scenarios.start(dataset,start.isoformat(),samples=args.samples,progress=progress)
    animals=None
    while completed<total_days-1e-9:
        tick=time.monotonic()
        if result is None:
            result=scenarios.advance(state['id'],state['generation'],progress=progress)
        state=scenarios.states[result['scenario']['id']]
        if animals is None:
            animals=[{k:t[k] for k in ('id','species','name','cohort_id','cohort_label','study',
                 'seed_observation','unknown_species_fallback','history_support_matches_training')} for t in result['tracks']]
            if not (output/'animals.json').exists():atomic_json(output/'animals.json',animals)
        offset=float(result['scenario']['elapsed_days'])
        relative=np.asarray(result['tracks'][0]['horizons_days'])
        relative=relative[relative<=min(90,total_days-offset)+1e-9]
        if total_days-offset<90 and (not len(relative) or relative[-1]<total_days-offset-1e-9):
            relative=np.r_[relative,total_days-offset]
        if cycle==0:relative=np.r_[0.,relative]
        coords=np.asarray([positions_at(t,relative) for t in result['tracks']],dtype='<f4').transpose(1,0,2)
        if not np.isfinite(coords).all() or np.any(np.abs(coords[...,0])>90) or np.any(np.abs(coords[...,1])>180):
            raise ValueError('Invalid predicted Earth coordinates')
        chunk=output/f'chunk-{cycle:04d}.npz'
        with chunk.open('wb') as f:np.savez(f,days=offset+relative,coordinates=coords)
        completed=float(offset+relative[-1]);cycle+=1
        # Resume state contains projected history in a separate file, never observations.
        combined=pd.concat([frame for _,frame in state['selected']],ignore_index=True)
        combined.to_parquet(output/'resume.partial.parquet',index=False,compression='zstd')
        (output/'resume.partial.parquet').replace(output/'resume.parquet')
        compact=[dict(id=t['id'],origin=t['origin'],horizons_days=t['horizons_days'],representative_sample=0,
                      sampled_latlon_paths=[t['sampled_latlon_paths'][t['representative_sample']]]) for t in result['tracks']]
        atomic_json(checkpoint,dict(id=state['id'],issue=state['issue'].isoformat(),generation=state['generation'],
            sources=state['sources'],excluded=state['excluded'],result={'tracks':compact},rng=rng.bit_generator.state,
            checkpoint_sha256=engine.checkpoint_hash,observations_sha256=dataset.sha256,completed_days=completed,cycle=cycle,devices_used=sorted(devices)))
        elapsed=time.monotonic()-started
        eta=elapsed/max(completed,1)*(total_days-completed)
        status=dict(state='running',completed_days=completed,total_days=total_days,cycles=cycle,
                    elapsed_seconds=elapsed,cycle_seconds=time.monotonic()-tick,eta_seconds=eta,device=device)
        atomic_json(output/'progress.json',status);print(json.dumps(status),flush=True)
        result=None
        if device=='cpu' and completed<total_days:
            try:check_gpu()
            except YieldGpu:pass
            else:raise YieldGpu('Switching the checkpointed export to an idle GPU')
    # One importable manifest plus a compact time-major float32 coordinate file.
    days=[];binary=output/'positions.f32'
    with binary.open('wb') as f:
        for part in sorted(output.glob('chunk-*.npz')):
            with np.load(part,allow_pickle=False) as chunk:
                days.extend(chunk['days'].tolist());f.write(chunk['coordinates'].astype('<f4').tobytes())
    if np.any(np.diff(days)<=0) or abs(days[-1]-total_days)>1e-8:
        raise ValueError('Export time coverage is incomplete or unordered')
    animals=json.loads((output/'animals.json').read_text())
    manifest=dict(schema='wildlife-atlas-trajectory-archive-v1',kind='forecast',issue_time=start.isoformat(),
        forecast_end_days=total_days,years=args.years,animals=animals,times_days=days,
        positions=dict(file=binary.name,dtype='float32-le',shape=[len(days),len(animals),2],sha256=digest(binary)),
        checkpoint_sha256=engine.checkpoint_hash,model_kind=engine.kind,trained_taxa=len(engine.manifest['species']),
        package_manifest_sha256=digest(bundle/'package-manifest.json'),
        export_code_sha256=digest(Path(__file__)),samples_per_animal=args.samples,seed=2026,
        dataset={**dataset.info(),'observations':''},runtime=dict(device=device,devices_used=sorted(devices),gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
            elapsed_seconds=time.monotonic()-started),
        scope='50-year recursive model scenario for demo playback; not a validated ecological forecast or live locations',
        interpolation='Spherical interpolation between model horizons; predicted states become subsequent model inputs',
        cohort_definition='Study and species; not verified biological colonies')
    atomic_json(output/'forecast.json',manifest)
    atomic_json(output/'progress.json',dict(state='complete',completed_days=completed,total_days=total_days,
        cycles=cycle,elapsed_seconds=time.monotonic()-started,eta_seconds=0,manifest='forecast.json'))


if __name__=='__main__':
    try:main()
    except YieldGpu as error:
        print(str(error),flush=True);sys.exit(75)
