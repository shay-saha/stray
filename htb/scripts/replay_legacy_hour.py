"""Replay retained older one-hour validation forecasts in physical kilometres.

This is a separate legacy dataset/model reference, never a current migration-v3
or official MoveBench score. No model training or final-test evaluation occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from wildcast.data import SPLITS, unproject
from wildcast.evaluate import load_model, ensemble_predictions
from wildcast.migration_evaluate import score_paths, macro_scores, paired_animal_bootstrap, great_circle, sha256

KEYS=['window','split','study','animal','horizon_hours']


def audited_cases(comparison):
    comparison=Path(comparison)
    protocol=json.loads((comparison/'comparison.json').read_text())
    table=pd.read_parquet(comparison/'paired-windows.parquet')
    if not set(table.split)<={'val_temporal','val_individual'}:
        raise ValueError('Only audited legacy validation cases may be replayed')
    if table.duplicated(KEYS).any() or table.window.duplicated().any():raise ValueError('Duplicate audited window support')
    actual=hashlib.sha256(table[KEYS].to_csv(index=False).encode()).hexdigest()
    if actual!=protocol['case_keys_sha256']:raise ValueError('Retained audited case fingerprint mismatch')
    cases=table.loc[table.horizon_hours.eq(1)].copy()
    excluded=protocol['identity_audit']['excluded_windows']
    if cases.empty or cases.window.isin(excluded).any():raise ValueError('Missing one-hour support or retained identity exclusions violated')
    return protocol,cases


def inputs(root,expected,cases):
    root=Path(root);npz=root/'processed/windows.npz';manifest_path=root/'processed/windows.json'
    manifest=json.loads(manifest_path.read_text())
    if sha256(npz)!=expected or manifest['windows_sha256']!=expected:raise ValueError('Legacy window/root fingerprint mismatch')
    ids=cases.window.to_numpy(np.int64)
    with np.load(npz,allow_pickle=False) as archive:
        meta=archive['meta'][ids]
        aux=archive['aux'][ids]
        truth=archive['y'][ids]
        view={key:archive[key][ids] for key in ('x','drivers','anchors','memory') if key in archive.files}
    if sha256(npz)!=expected:raise ValueError('Legacy input archive changed while reading')
    if not np.array_equal(meta[:,4],cases.horizon_hours) or not np.array_equal(meta[:,2],cases.study) or not np.array_equal(meta[:,3],cases.animal):
        raise ValueError('Legacy audited study/animal/horizon metadata differs')
    codes=np.asarray([SPLITS[v] for v in cases.split])
    if not np.array_equal(meta[:,0],codes):raise ValueError('Legacy audited split differs from root')
    if aux.shape[1]<=8 or not np.isfinite(aux).all() or np.any(aux[:,8]<=0):raise ValueError('Missing causal legacy metre scales')
    if np.any(np.abs((aux[:,3]-aux[:,2])/3600-1)>.25+1e-8):raise ValueError('One-hour target tolerance differs from source protocol')
    view.update(meta=meta,aux=aux)
    # The predictors receive neither y nor any query future coordinates.
    actual=unproject(truth.astype(float)*aux[:,8:9],aux[:,0],aux[:,1])
    np.testing.assert_allclose(great_circle(actual,aux[:,:2])*1000,cases.dont_move_es_m,atol=.02,rtol=1e-5)
    return view,manifest,actual,{'root':str(root),'windows_sha256':expected,'manifest_sha256':sha256(manifest_path)}


def physical_ids(manifest,audit,animal):
    inverse={int(v):k for k,v in manifest['animals'].items()}
    groups=[audit.get('fold_groups',{}).get(inverse[int(v)],inverse[int(v)]) for v in animal]
    mapping={name:i for i,name in enumerate(sorted(set(groups)))}
    return np.asarray([mapping[v] for v in groups],dtype=np.int64),mapping


def replay(comparison,output,samples=64,seed=2026,batch_size=256,device='cpu'):
    if samples<2 or batch_size<1:raise ValueError('Positive batch size and at least two samples required')
    out=Path(output)
    if out.exists():raise FileExistsError('Replay output must be new')
    protocol,cases=audited_cases(comparison)
    assets=protocol['protocol']['wildcast_comparators']
    matched=protocol['protocol']['comparison_dataset']
    specifications=[('Legacy WildCast matched296',matched,'WildCast 296 inputs'),
                    ('Legacy WildCast ensemble180',assets['best_existing180_ensemble']['data'],'WildCast best ensemble'),
                    ('Retained MoveFormer local adapter',matched,'MoveFormer local adapter')]
    loaded={};metadata={};truths={};manifests={}
    for _,root,label in specifications:
        if root not in loaded:
            a,m,t,evidence=inputs(root,protocol['evidence'][label]['windows_sha256'],cases)
            loaded[root]=a;manifests[root]=m;truths[root]=t;metadata[root]=evidence
    first=loaded[matched]
    for root,a in loaded.items():
        for name in ('meta','aux','x'):
            np.testing.assert_array_equal(a[name],first[name])
        np.testing.assert_array_equal(truths[root],truths[matched])
        if manifests[root]['animals']!=manifests[matched]['animals'] or manifests[root]['species']!=manifests[matched]['species']:
            raise ValueError('Legacy raw case vocabularies differ across feature roots')
    physical,mapping=physical_ids(manifests[matched],protocol['identity_audit'],cases.animal)
    if np.any(first['meta'][:,1]==0):raise ValueError('Unknown legacy prediction species require separate actual-taxon reporting groups')
    device='cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device
    torch.set_num_threads(min(4,torch.get_num_threads()))
    rows=[];checkpoint_records=[];tail_counts={}
    def retain(label,ix,paths):
        a=first;origin=a['aux'][ix,:3]
        paths=np.asarray(paths,dtype=np.float64)
        if paths.shape!=(len(ix),samples,2) or not np.isfinite(paths).all():raise ValueError('Invalid legacy forecast samples')
        # Old units are normalized by a causal per-origin metre scale.
        metres=paths*a['aux'][ix,None,8:9]
        tail_counts[label]=tail_counts.get(label,0)+int((np.linalg.norm(metres,axis=-1)>np.pi*6371008.8).sum())
        result=score_paths(metres[:,:,None]/1_000_000,truths[matched][ix,None],np.ones((len(ix),1),bool),origin)
        for local,case in enumerate(ix):
            rows.append({'model':label,'case':int(cases.iloc[case].window),'split':str(cases.iloc[case].split),
                'study':int(cases.iloc[case].study),'legacy_animal':int(cases.iloc[case].animal),
                'individual':int(physical[case]),'species':int(a['meta'][case,1]),'horizon_days':1/24,
                'actual_elapsed_hours':float((a['aux'][case,3]-a['aux'][case,2])/3600),
                'endpoint_energy_km':float(result['endpoint_energy_km'][local,0]),
                'point_error_km':float(result['point_error_km'][local,0]),
                'representative_error_km':float(result['representative_error_km'][local,0]),
                'radial_coverage50':bool(result['radial_coverage50'][local,0]),'radial_coverage90':bool(result['radial_coverage90'][local,0])})
    for label,root,evidence_label in specifications:
        models=[]
        for record in protocol['evidence'][evidence_label]['checkpoint_info']:
            path=Path(record['path'])
            if sha256(path)!=record['sha256']:raise ValueError('Retained legacy checkpoint SHA mismatch')
            model,ck=load_model(path,device)
            if ck['manifest']['windows_sha256']!=metadata[root]['windows_sha256']:raise ValueError('Checkpoint and correct legacy feature root differ')
            models.append(model);checkpoint_records.append({'replay_label':label,**record})
        generator=torch.Generator(device=device).manual_seed(seed)
        prediction_view={**loaded[root],'aux':loaded[root]['aux'].copy()}
        prediction_view['aux'][:,[3,7]]=0  # Query target timestamp/index never reaches prediction.
        for split in ('val_temporal','val_individual'):
            ids=np.flatnonzero(cases.split.eq(split).to_numpy())
            for start in range(0,len(ids),batch_size):
                ix=ids[start:start+batch_size]
                with torch.inference_mode():draws=ensemble_predictions(models,prediction_view,ix,samples,device,generator).detach().cpu().numpy()
                retain(label,ix,draws)
        for record in protocol['evidence'][evidence_label]['checkpoint_info']:
            if sha256(record['path'])!=record['sha256']:raise ValueError('Legacy checkpoint changed during replay')
        print(label,'replayed',len(cases),'one-hour validation windows',flush=True)
        del models
        if device=='cuda':torch.cuda.empty_cache()
    for start in range(0,len(cases),batch_size):
        ix=np.arange(start,min(start+batch_size,len(cases)));retain('Legacy stationary reference',ix,np.zeros((len(ix),samples,2)))
    frame=pd.DataFrame(rows)
    overall,species,horizon=macro_scores(frame)
    reference='Retained MoveFormer local adapter'
    energy=paired_animal_bootstrap(frame,reference,seed,500)
    point=paired_animal_bootstrap(frame.assign(endpoint_energy_km=frame.point_error_km),reference,seed,500)
    study=frame.groupby(['model','split','study','individual'])[['point_error_km','endpoint_energy_km','radial_coverage90']].mean().groupby(['model','split','study']).mean().groupby(['model','split']).mean().reset_index()
    pooled=frame.groupby(['model','split']).point_error_km.agg(mean='mean',median='median',p90=lambda s:s.quantile(.9)).reset_index()
    counts=frame.loc[frame.model==reference].groupby('split').agg(windows=('case','size'),physical_animals=('individual','nunique'),species=('species','nunique'),studies=('study','nunique')).reset_index()
    overall=overall.merge(counts,on='split',validate='many_to_one')
    out.mkdir(parents=True)
    frame.to_parquet(out/'per-window.parquet',index=False)
    for name,table in [('physical-species-macro',overall),('per-species',species),('legacy-study-balanced',study),('pooled-point-errors',pooled)]:table.to_csv(out/(name+'.csv'),index=False)
    timing=frame.loc[frame.model==reference].groupby('split').actual_elapsed_hours.agg(windows='size',minimum='min',mean='mean',maximum='max').reset_index();timing.to_csv(out/'target-timing.csv',index=False)
    report={'scope':'Standalone replay of older local one-hour models/data. Not current migration-v3, broad-intake training, an independent test, or an official MoveBench result.',
        'comparison':str(comparison),'comparison_sha256':sha256(Path(comparison)/'comparison.json'),'paired_windows_sha256':sha256(Path(comparison)/'paired-windows.parquet'),
        'one_hour_case_ids_sha256':hashlib.sha256(cases.window.to_numpy(dtype='<i8').tobytes()).hexdigest(),
        'source_case_keys_sha256':protocol['case_keys_sha256'],'inputs':metadata,'checkpoints':checkpoint_records,
        'samples':samples,'seed':seed,'batch_size':batch_size,'device':device,'counts':counts.to_dict('records'),
        'physical_species_macro':overall.to_dict('records'),'paired_energy':energy,'paired_mean_location_error':point,
        'physical_identity_rule':'Retained audit canonical fold_groups; unresolved cross-source aliases remain a limitation.',
        'physical_group_names':mapping,'beyond_antipodal_tangent_samples':tail_counts,
        'geometry':'Spherical AEQD inverse with each causal metre scale, then great-circle distances and spherical sample mean. Energy uses off-diagonal U-statistic pairs; radial coverage is not HDR.',
        'draw_scope':f'Only exact audited one-hour cases, each model seed {seed}, temporal then individual split, {batch_size}-case batches. These finite draws differ from the retained all-horizon evaluation stream.',
        'limitations':['All model/member/hyperparameter selection already used these legacy validation observations. Bootstrap is exploratory and conditions on fitted weights and finite draws.',
          'The matched296 comparator and MoveFormer receive the same origin inputs; ensemble180 is an additional different-input older reference.',
          'The legacy corpus/QC predates the new breadth actual/nonmodelled-GPS and clock audit. This replay does not certify legacy source quality.',
          'Primary aggregation gives equal physical-animal weight within actual species, then species; legacy study-balanced and pooled point summaries are separate descriptive outputs.',
          'One-hour targets may be45–75 minutes after origin, as in the frozen source matching rule. No ground-truth interpolation is introduced.',
          'Historical environmental release vintages were not established; this is retrospective replay, not an operational backtest.'],
        'training_performed':False,'final_tests_scored':False,'code_sha256':{Path(__file__).name:sha256(__file__),'evaluate.py':sha256(PROJECT/'wildcast/evaluate.py'),'model.py':sha256(PROJECT/'wildcast/model.py'),'moveformer.py':sha256(PROJECT/'wildcast/moveformer.py'),'physical_scorer':sha256(PROJECT/'wildcast/migration_evaluate.py')}}
    report['query_future_arrays_sent_to_predictors']=False
    (out/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# Older one-hour replay: physical mean-location errors','',report['scope'],'',
        'Mean-location error is the great-circle miss distance of the mean predicted location, in kilometres. Energy measures the whole predictive distribution and is a different quantity. The table uses equal physical-animal weight within species, then equal species weight. These are legacy validation references, not a claim about the current day-to-month model.','',
        '| Older reference | Split | Mean-location error (km) | Energy (km) | 90% radial coverage | Animals |','|---|---|---:|---:|---:|---:|']
    for row in overall.itertuples(index=False):lines.append(f'| {row.model} | {row.split} | {row.point_error_km:.3f} | {row.endpoint_energy_km:.3f} | {100*row.radial_coverage90:.1f}% | {row.physical_animals} |')
    lines+=['','Point-error paired animal bootstrap intervals are in report.json; every case retains its physical error and actual elapsed time. Radial coverage is not highest-density-region coverage.','',*['- '+v for v in report['limitations']]]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',default='reports/moveformer-comparison-audited');p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=64);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--device',default='cpu')
    r=replay(**vars(p.parse_args()));print(json.dumps({'counts':r['counts'],'scores':r['physical_species_macro']},indent=2))

if __name__=='__main__':main()
