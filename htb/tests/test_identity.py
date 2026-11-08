import json

import numpy as np
import pandas as pd

from wildcast.identity import audit


def test_cross_year_identity_cannot_appear_as_unseen_or_earlier_than_training(tmp_path):
    root=tmp_path/"data";p=root/"processed";p.mkdir(parents=True)
    raw=tmp_path/"rawdata";(raw/"raw").mkdir(parents=True)
    first="source:2017:Species:X";second="source:2018:Species:X"
    timestamps=[50,60,100,150,110,200]
    ids=[first,first,second,second,first,second]
    f=pd.DataFrame({"animal":ids,"species":"Species","source":"source","study":["source:2017" if v==first else "source:2018" for v in ids],
                    "timestamp":pd.to_datetime(timestamps,unit="s",utc=True),"lat":1.,"lon":2.,"segment":0})
    f.to_parquet(p/"observations.parquet",index=False)
    manifest={"animals":{first:0,second:1},"animal_assignments":{first:"val_individual",second:"temporal"},"observations_sha256":"example","windows_sha256":"windows"}
    (p/"windows.json").write_text(json.dumps(manifest))
    meta=np.zeros((4,5));meta[:,0]=[0,3,1,1];meta[:,3]=[1,0,1,0]
    aux=np.zeros((4,9));aux[:,2]=[100,50,200,110];aux[:,7]=[3,1,5,4]
    np.savez_compressed(p/"windows.npz",meta=meta,aux=aux)
    result=audit(root,raw,tmp_path/"audit.json")
    assert result["excluded_individual_ids"]==[0]
    assert result["excluded_windows"]==[1,3]
    assert result["fold_groups"][first]==result["fold_groups"][second]


def test_identity_mask_cannot_be_reused_after_case_ids_change():
    from wildcast.identity import verify_audit,case_fingerprint
    import pytest
    arrays={k:np.zeros((2,2)) for k in ["x","y","meta","aux"]}
    info={"observations_sha256":"same","windows_sha256":"old-features","case_fingerprint":case_fingerprint(arrays)}
    manifest={"observations_sha256":"same","windows_sha256":"new-features"}
    verify_audit(info,manifest,arrays)
    arrays["meta"][0,0]=1
    with pytest.raises(ValueError,match="forecast windows"):
        verify_audit(info,manifest,arrays)
