import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('legacy_hour',Path(__file__).resolve().parents[1]/'scripts/replay_legacy_hour.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


def fixture(root):
    root.mkdir();p=root/'processed';p.mkdir()
    cases=pd.DataFrame({'window':[0,1],'split':['val_temporal','val_individual'],'study':[0,1],'animal':[10,20],
                        'horizon_hours':[1,1],'dont_move_es_m':[1000.,1000.]})
    aux=np.zeros((2,9));aux[:,1]=179.999;aux[:,2]=1.5e9;aux[:,3]=1.5e9+3600;aux[:,8]=100000
    np.savez_compressed(p/'windows.npz',x=np.zeros((2,2,10),np.float32),y=np.array([[.01,0],[.01,0]],np.float32),
                        meta=np.array([[1,1,0,10,1],[3,2,1,20,1]]),aux=aux)
    digest=replay.sha256(p/'windows.npz');(p/'windows.json').write_text(json.dumps({'windows_sha256':digest}))
    return cases,digest


def test_correct_legacy_metres_and_dateline_inverse_without_query_labels_in_view(tmp_path):
    cases,digest=fixture(tmp_path/'data')
    view,manifest,truth,metadata=replay.inputs(tmp_path/'data',digest,cases)
    assert 'y' not in view and 'target_latlon' not in view
    assert np.all(truth[:,1]<-179.99)
    np.testing.assert_allclose(replay.great_circle(truth,view['aux'][:,:2]),1.,atol=1e-6)
    assert metadata['windows_sha256']==digest


@pytest.mark.parametrize('change',['hash','scale','identity','timing'])
def test_wrong_roots_scales_identity_or_target_tolerance_rejected(tmp_path,change):
    cases,digest=fixture(tmp_path/'data')
    if change=='hash':digest='0'*64
    elif change=='scale':cases['dont_move_es_m']=1.
    elif change=='identity':cases.loc[0,'animal']=99
    else:
        p=tmp_path/'data/processed/windows.npz'
        with np.load(p) as z:a={k:z[k] for k in z.files}
        a['aux'][0,3]+=2000;np.savez_compressed(p,**a);digest=replay.sha256(p)
        (p.parent/'windows.json').write_text(json.dumps({'windows_sha256':digest}))
    with pytest.raises((ValueError,AssertionError)):
        replay.inputs(tmp_path/'data',digest,cases)


def test_audited_case_checksum_and_final_test_guards(tmp_path):
    cases=pd.DataFrame({'window':[0,1],'split':['val_temporal','val_individual'],'study':[0,1],'animal':[10,20],
                        'horizon_hours':[1,1],'dont_move_es_m':[1000.,1000.]})
    cases.to_parquet(tmp_path/'paired-windows.parquet',index=False)
    protocol={'case_keys_sha256':hashlib.sha256(cases[replay.KEYS].to_csv(index=False).encode()).hexdigest(),
              'identity_audit':{'excluded_windows':[]}}
    (tmp_path/'comparison.json').write_text(json.dumps(protocol))
    assert len(replay.audited_cases(tmp_path)[1])==2
    cases.loc[0,'split']='test_temporal';cases.to_parquet(tmp_path/'paired-windows.parquet',index=False)
    with pytest.raises(ValueError,match='validation'):replay.audited_cases(tmp_path)


def test_known_physical_aliases_share_bootstrap_identity():
    manifest={'animals':{'source:year1:animal':10,'source:year2:animal':20,'other':30}}
    audit={'fold_groups':{'source:year2:animal':'source:year1:animal'}}
    ids,names=replay.physical_ids(manifest,audit,[10,20,30])
    assert ids[0]==ids[1] and ids[0]!=ids[2] and len(names)==2
