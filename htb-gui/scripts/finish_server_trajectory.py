#!/usr/bin/env python3
"""Finish a server export using array histories and the frozen ensemble sampler."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    temp=path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value,allow_nan=False,separators=(',',':')));temp.replace(path)


def encode(rows,schema):
    from wildcast.data import project
    from wildcast.migration_data import daylength
    lags=schema.LAGS
    ll=[];ht=[];masks=[]
    for times,coords in rows:
        ix,mask=schema.past_indices(times,len(times)-1)
        ll.append(coords[ix]);ht.append(times[ix]);masks.append(mask)
    ll=np.asarray(ll);ht=np.asarray(ht);mask=np.asarray(masks)
    origin=np.asarray([coords[-1] for _,coords in rows])
    now=np.asarray([times[-1] for times,_ in rows])
    xy=project(ll[...,0],ll[...,1],origin[:,0,None],origin[:,1,None])/1_000_000
    age=(now[:,None]-ht)/86400
    days=pd.to_datetime(ht.ravel(),unit='s',utc=True).dayofyear.to_numpy().reshape(ht.shape)
    season=2*np.pi*days/365.25
    history=np.stack([xy[...,0],xy[...,1],age,np.sin(season),np.cos(season),ll[...,0]/90,
        np.sin(np.deg2rad(ll[...,1])),np.cos(np.deg2rad(ll[...,1])),
        np.broadcast_to(lags,ht.shape),mask],axis=-1)
    history[~mask]=0
    a,b=np.deg2rad(origin).T
    season0=2*np.pi*pd.to_datetime(now,unit='s',utc=True).dayofyear.to_numpy()/365.25
    solar=2*np.pi*((now/3600+origin[:,1]/15)%24)/24
    def velocity(lag):
        j=int(np.flatnonzero(lags==lag)[0]);elapsed=age[:,j]
        if np.any(mask[:,j] & (elapsed<=0)):raise ValueError('Invalid lag velocity')
        return np.divide(-xy[:,j],elapsed[:,None],out=np.zeros((len(rows),2)),where=mask[:,j,None]&(elapsed[:,None]>0))
    static=np.column_stack([np.cos(a)*np.cos(b),np.cos(a)*np.sin(b),np.sin(a),
        np.sin(season0),np.cos(season0),np.sin(solar),np.cos(solar),velocity(1),velocity(7),
        np.log1p((now-np.asarray([t[0] for t,_ in rows]))/86400),mask[:,-1],
        daylength(origin[:,0],pd.to_datetime(now,unit='s',utc=True).dayofyear.to_numpy()),
        velocity(1/24),velocity(3/24),velocity(5/1440),velocity(15/1440)])
    return history.astype('float32'),mask,static.astype('float32'),origin


def advance(rows,followed,horizons,issue,lags):
    days=np.unique(np.r_[np.arange(.25,90.01,.25),90-lags[lags<90],horizons])
    days=days[(days>0)&(days<=90)]
    knots=np.r_[0.,horizons]
    ix=np.clip(np.searchsorted(knots,days,side='right'),1,len(knots)-1)
    t=np.clip((days-knots[ix-1])/(knots[ix]-knots[ix-1]),0,1)
    origin=np.asarray([c[-1] for _,c in rows])
    coordinates=np.concatenate([origin[:,None,:],followed],axis=1)
    a,b=np.deg2rad(coordinates).transpose(2,0,1)
    vectors=np.stack([np.cos(a)*np.cos(b),np.sin(a),np.cos(a)*np.sin(b)],axis=-1)
    v=vectors[:,ix-1]*(1-t[None,:,None])+vectors[:,ix]*t[None,:,None]
    length=np.linalg.norm(v,axis=-1,keepdims=True)
    v=np.where(length<1e-10,vectors[:,ix-1],v)
    v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
    coords=np.rad2deg(np.stack([np.arctan2(v[...,1],np.hypot(v[...,0],v[...,2])),np.arctan2(v[...,2],v[...,0])],axis=-1))
    extra=issue+days*86400;result=[]
    cutoff=issue+90*86400-400*86400
    for i,(times,old) in enumerate(rows):
        keep=times>=cutoff
        result.append((np.r_[times[keep],extra],np.concatenate([old[keep],coords[i]],axis=0)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--years',type=int,default=50);p.add_argument('--batch',type=int,default=512)
    args=p.parse_args();root=args.output.resolve();bundle=args.bundle.resolve()
    sys.path.insert(0,str(bundle));os.chdir(bundle)
    import torch
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(.25)
    from verify_package import verify
    verify(bundle)
    from wildcast.migration_ensemble import load_model
    from wildcast.migration_protocol import schema_for
    from wildcast.data import unproject
    manifest=json.loads((bundle/'data/reference/manifest.json').read_text())
    schema=schema_for(manifest,legacy_default=True)
    model,metadata=load_model(bundle/'model/selected.pt','cuda')
    checkpoint_hash=sha(bundle/'model/selected.pt')
    animals=json.loads((root/'animals.json').read_text());names=[a['id'] for a in animals]
    species=np.asarray([manifest['species'].get(a['species'],0) for a in animals],dtype=np.int64)
    horizons=np.asarray(manifest['horizons_days'],dtype=float)
    rng=np.random.default_rng(2026)
    if (root/'fast-resume.json').exists():
        saved=json.loads((root/'fast-resume.json').read_text())
        with np.load(root/'fast-resume.npz',allow_pickle=False) as a:
            offsets=a['offsets'];times=a['times'];coords=a['coordinates'];followed=a['followed']
            rows=[(times[offsets[i]:offsets[i+1]],coords[offsets[i]:offsets[i+1]]) for i in range(len(names))]
    else:
        saved=json.loads((root/'resume.json').read_text())
        frames=pd.read_parquet(root/'resume.parquet')
        grouped=list(frames.groupby('animal',sort=True))
        if [str(k) for k,_ in grouped]!=names:raise ValueError('Resume identity order differs')
        rows=[(pd.DatetimeIndex(f.timestamp).as_unit('ns').asi8/1e9,f[['lat','lon']].to_numpy(dtype=float)) for _,f in grouped]
        previous={t['id']:t for t in saved['result']['tracks']}
        followed=np.asarray([previous[name]['sampled_latlon_paths'][0] for name in names])
    if saved['checkpoint_sha256']!=checkpoint_hash:raise ValueError('Resume/model mismatch')
    start=pd.Timestamp(saved.get('start','2026-11-08T00:00:00Z'))
    issue=pd.Timestamp(saved['issue']).timestamp();cycle=int(saved['cycle']);completed=float(saved['completed_days'])
    if any(abs(t[-1]-issue)>1e-4 for t,_ in rows):raise ValueError('Resume history/clock mismatch')
    rng.bit_generator.state=saved['rng']
    total_days=((start+pd.DateOffset(years=args.years))-start).total_seconds()/86400
    start_seconds=start.timestamp();base_completed=completed;started=time.monotonic();checks=0
    while completed<total_days-1e-9:
        tick=time.monotonic()
        rows=advance(rows,followed,horizons,issue,schema.LAGS);issue+=90*86400
        history,mask,static,origin=encode(rows,schema)
        encoded=time.monotonic()
        if cycle<12 or cycle%25==0:
            for index in np.linspace(0,len(rows)-1,min(128,len(rows)),dtype=int):
                times,coords=rows[index];dates=pd.to_datetime(times,unit='s',utc=True)
                exact=schema.query_features(coords[:,0],coords[:,1],times,dates,len(times)-1)
                np.testing.assert_array_equal(mask[index],exact[1])
                np.testing.assert_allclose(history[index],exact[0],rtol=2e-6,atol=2e-6)
                np.testing.assert_allclose(static[index],exact[2],rtol=2e-6,atol=2e-6)
                checks+=1
        paths=[]
        data=dict(history=history,history_mask=mask,static=static,species=species,horizons=horizons)
        for first in range(0,len(names),args.batch):
            last=min(first+args.batch,len(names))
            draws=model.predict_cases(data,np.arange(first,last),16,rng=rng)
            coords=unproject(draws*1_000_000,origin[first:last,0,None,None],origin[first:last,1,None,None])
            a,b=np.deg2rad(coords).transpose(3,0,1,2)
            xyz=np.stack([np.cos(a)*np.cos(b),np.sin(a),np.cos(a)*np.sin(b)],axis=-1)
            center=xyz.mean(axis=1);center/=np.maximum(np.linalg.norm(center,axis=-1,keepdims=True),1e-12)
            chosen=np.argmin((1-(xyz*center[:,None]).sum(axis=-1)).sum(axis=-1),axis=1)
            paths.append(coords[np.arange(last-first),chosen])
        followed=np.concatenate(paths,axis=0)
        if not np.isfinite(followed).all():raise ValueError('Non-finite model route')
        predicted=time.monotonic()
        offset=(issue-start_seconds)/86400;relative=horizons[horizons<=total_days-offset+1e-9]
        export=followed[:,:len(relative)]
        if total_days-offset<90 and (not len(relative) or relative[-1]<total_days-offset-1e-9):
            remaining=total_days-offset;knots=np.r_[0.,horizons]
            ix=np.clip(np.searchsorted(knots,remaining,side='right'),1,len(knots)-1)
            f=(remaining-knots[ix-1])/(knots[ix]-knots[ix-1])
            pts=np.concatenate([origin[:,None],followed],axis=1)
            a,b=np.deg2rad(pts[:,[ix-1,ix]]).transpose(2,0,1)
            v=np.stack([np.cos(a)*np.cos(b),np.sin(a),np.cos(a)*np.sin(b)],axis=-1)
            v=v[:,0]*(1-f)+v[:,1]*f;v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
            point=np.rad2deg(np.column_stack([np.arctan2(v[:,1],np.hypot(v[:,0],v[:,2])),np.arctan2(v[:,2],v[:,0])]))
            export=np.concatenate([export,point[:,None]],axis=1);relative=np.r_[relative,remaining]
        with (root/f'chunk-{cycle:04d}.npz').open('wb') as f:
            np.savez(f,days=offset+relative,coordinates=export.transpose(1,0,2).astype('<f4'))
        completed=offset+float(relative[-1]);cycle+=1
        if cycle%4==0 or completed>=total_days:
            offsets=np.r_[0,np.cumsum([len(t) for t,_ in rows])]
            with (root/'fast-resume.partial.npz').open('wb') as f:
                np.savez(f,offsets=offsets,times=np.concatenate([t for t,_ in rows]),
                         coordinates=np.concatenate([c for _,c in rows]),followed=followed)
            (root/'fast-resume.partial.npz').replace(root/'fast-resume.npz')
            write_json(root/'fast-resume.json',dict(checkpoint_sha256=checkpoint_hash,issue=pd.Timestamp(issue,unit='s',tz='UTC').isoformat(),
                start=start.isoformat(),cycle=cycle,completed_days=completed,rng=rng.bit_generator.state,
                observations_sha256=saved['observations_sha256']))
        elapsed=time.monotonic()-started
        eta=elapsed/max(completed-base_completed,1)*(total_days-completed)
        progress=dict(state='running',completed_days=completed,total_days=total_days,cycles=cycle,device='cuda',
            elapsed_seconds=elapsed,cycle_seconds=time.monotonic()-tick,encoding_seconds=encoded-tick,
            sampling_seconds=predicted-encoded,eta_seconds=eta,encoder_equivalence_checks=checks)
        write_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
    days=[];binary=root/'positions.f32'
    with binary.open('wb') as f:
        for part in sorted(root.glob('chunk-*.npz')):
            with np.load(part,allow_pickle=False) as chunk:
                days.extend(chunk['days'].tolist());f.write(chunk['coordinates'].astype('<f4').tobytes())
    if np.any(np.diff(days)<=0) or abs(days[-1]-total_days)>1e-8:raise ValueError('Incomplete export')
    taxa=sorted({a['species'] for a in animals})
    result=dict(schema='wildlife-atlas-trajectory-archive-v1',kind='forecast',issue_time=start.isoformat(),
        forecast_end_days=total_days,years=args.years,animals=animals,times_days=days,
        positions=dict(file=binary.name,dtype='float32-le',shape=[len(days),len(animals),2],sha256=sha(binary)),
        checkpoint_sha256=checkpoint_hash,model_kind='wildcast_migration_ensemble',trained_taxa=len(manifest['species']),
        package_manifest_sha256=sha(bundle/'package-manifest.json'),export_code_sha256=sha(Path(__file__)),
        samples_per_animal=16,seed=2026,encoder_equivalence_checks=checks,
        dataset=dict(observations='',recorded_animals=len(animals),recorded_species=len(taxa),species_catalog=taxa,
                     source_sha256=saved['observations_sha256']),runtime=dict(device='cuda',gpu=torch.cuda.get_device_name(0),
            elapsed_seconds=time.monotonic()-started,batch_size=args.batch),
        scope='50-year recursive model scenario for demo playback; not a validated ecological forecast or live locations',
        interpolation='Spherical interpolation between trained model horizons; predicted states feed subsequent forecasts',
        cohort_definition='Study and species; not verified biological colonies')
    write_json(root/'forecast.json',result)
    write_json(root/'progress.json',dict(state='complete',completed_days=completed,total_days=total_days,cycles=cycle,
        elapsed_seconds=time.monotonic()-started,eta_seconds=0,manifest='forecast.json'))


if __name__=='__main__':main()
