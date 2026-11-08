"""Future isolation, missingness controls and a learnable proper-score gate."""
import json
from pathlib import Path

import numpy as np
import pytest
import shapely
import torch

from wildcast.migration_environment_features import Environment,RAW_WEATHER,LOCAL_NAMES
from wildcast.migration_environment_gate import Gate,EnvironmentalEnsemble,predictors


@pytest.fixture(autouse=True)
def small_threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def weather_environment(tmp_path,future=99.):
    import pandas as pd
    root=tmp_path/'weather';root.mkdir(exist_ok=True)
    (root/'manifest.json').write_text(json.dumps({'grid_degrees':1.}))
    dates=pd.date_range('2019-12-01',periods=100,tz='UTC')
    values=np.arange(100,dtype=float);values[dates>=pd.Timestamp('2020-01-14',tz='UTC')]=future
    daily={'time':[d.strftime('%Y-%m-%d') for d in dates]}
    daily.update({name:values.tolist() for name in RAW_WEATHER})
    data={'params':{'latitude':.5,'longitude':.5,'timezone':'GMT','wind_speed_unit':'ms','models':'era5'},'response':{'daily':daily}}
    (root/'0_0.json').write_text(json.dumps(data))
    e=Environment.__new__(Environment);e.root=tmp_path;e.weather={};e.terrain=[];e.terrain_arrays={};e._weather()
    e.months=np.arange(1980*12,2027*12);e.lat=np.asarray([-1.,1.]);e.lon=np.asarray([-1.,1.])
    e.sst=np.ones((len(e.months),2,2),np.float32)*2
    e.land=shapely.box(-2,-2,2,2);shapely.prepare(e.land)
    return e


def test_weather_and_SST_cannot_read_future_days_or_months(tmp_path):
    import pandas as pd
    origin=np.asarray([[.5,.5,pd.Timestamp('2020-01-15T12:00:00Z').timestamp()]])
    first=weather_environment(tmp_path,99.);expected=first.lookup(origin)
    second=weather_environment(tmp_path,999999.)
    # January2020's query selects November2019, excluding later source months.
    second.sst[second.months>=2019*12+11]=10000
    np.testing.assert_array_equal(second.lookup(origin),expected)
    assert expected.shape==(1,len(LOCAL_NAMES))
    # Latest eligible weather day is13January, index43 since1December.
    assert expected[0,0]==43
    assert expected[0,10]==1 and expected[0,64]==1
    np.testing.assert_array_equal(expected[0,71:74],0)
    assert expected[0,74]==expected[0,75]==1


def test_weather_missing_cells_do_not_become_observed_zeros(tmp_path):
    import pandas as pd
    e=weather_environment(tmp_path)
    o=np.asarray([[.5,.5,pd.Timestamp('2020-01-15T00:00:00Z').timestamp()], [5.,5.,pd.Timestamp('2020-01-15T00:00:00Z').timestamp()]])
    result=e.lookup(o)
    assert not result[1,:60].any()
    assert result[0,10:20].all()
    assert not result[1,74] and result[1,75]


class Store:
    metadata={'value_columns':[0,2]}
    def lookup(self,a,ids):return np.asarray([[10.,1.,20.,.5]]*len(ids),np.float32)


def test_content_and_availability_controls_preserve_geometry_and_masks():
    a={'static':np.ones((2,2),np.float32)}
    actual=predictors(a,[0,1],Store(),'actual');availability=predictors(a,[0,1],Store(),'availability');masked=predictors(a,[0,1],Store(),'masked')
    np.testing.assert_array_equal(availability[:,[2,4]],0)
    np.testing.assert_array_equal(availability[:,[3,5]],actual[:,[3,5]])
    np.testing.assert_array_equal(masked[:,:2],actual[:,:2]);assert not masked[:,2:].any()


def test_zero_initialized_gate_preserves_parent_law_and_learns_conditional_proper_score():
    torch.manual_seed(42);g=Gate(1,1,[.5,.5],hidden=8)
    x=torch.tensor([[-1.],[1.]]).repeat(32,1);species=torch.zeros(len(x),dtype=torch.long)
    torch.testing.assert_close(g(x,species),torch.full((len(x),2),.5,dtype=torch.float64),rtol=0,atol=1e-15)
    # Two deterministic experts at0 and10. Truth alternates with the covariate.
    A=torch.tensor([[0.,10.],[10.,0.]]).repeat(32,1)
    D=torch.tensor([[0.,10.],[10.,0.]])
    def score():
        w=g(x,species);return ((w*A).sum(-1)-.5*torch.einsum('ni,ij,nj->n',w,D.double(),w)).mean()
    assert float(score().detach())==pytest.approx(2.5)
    optimizer=torch.optim.Adam(g.parameters(),lr=.01)
    for _ in range(100):
        optimizer.zero_grad();loss=score();loss.backward();optimizer.step()
    assert float(score().detach())<.5
    assert torch.all(g(x,species)>0)
    torch.testing.assert_close(g(x,species).sum(-1),torch.ones(len(x),dtype=torch.float64))


def test_environment_mixture_selects_one_expert_per_route_and_ignores_future_fields():
    class Expert:
        def __init__(self,value):self.value=value
        def predict_cases(self,a,ids,samples,rng,baselines=None):return np.full((len(ids),samples,len(a['horizons']),2),self.value,np.float32)
    a={'static':np.ones((2,2),np.float32),'species':np.zeros(2,np.int64),'horizons':np.asarray([1.,7.,30.]),'y':object(),'target_latlon':object()}
    gate=Gate(6,1,[.5,.5],hidden=8).eval();model=EnvironmentalEnsemble(gate,[Expert(1),Expert(2)],Store(),'actual','cpu')
    paths=model.predict_cases(a,np.arange(2),100,np.random.default_rng(3))
    assert set(np.unique(paths))=={1.,2.}
    np.testing.assert_array_equal(paths[:,:,0],paths[:,:,1]);np.testing.assert_array_equal(paths[:,:,0],paths[:,:,2])
