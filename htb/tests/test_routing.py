import numpy as np
import torch

from wildcast.data import unproject
from wildcast.raster_context import SatelliteFields,unproject_torch
from wildcast.router import mixture_terms,weighted_energy,export_analogue,import_analogue
from wildcast.baselines import AnalogueForecaster


def test_mixture_score_and_gradients_for_two_known_endpoint_atoms():
    draws=torch.zeros(1,2,8,2);draws[:,1,:,0]=2
    accuracy,pair=mixture_terms(draws,torch.zeros(1,2))
    logits=torch.zeros(1,2,requires_grad=True)
    score=weighted_energy(logits.softmax(-1),accuracy,pair)
    torch.testing.assert_close(score,torch.tensor([.5]))
    score.sum().backward()
    assert logits.grad[0,0]<0 and logits.grad[0,1]>0


def test_shared_expert_randomness_excludes_matching_draw_indices():
    a=torch.tensor([[[[0.,0.],[1.,0.],[4.,0.]],[[0.,0.],[.25,0.],[1.,0.]]]])
    _,pair=mixture_terms(a,torch.zeros(1,2))
    expected=np.mean([abs(float(a[0,0,i,0]-a[0,1,j,0])) for i in range(3) for j in range(3) if i!=j])
    assert abs(pair[0,0,1].item()-expected)<1e-6


def test_satellite_fields_mask_future_year_and_recover_valid_overlapping_tile():
    metadata=[{"year":2019,"bounds":[0.,0.,10.,10.]}]*2
    values=torch.full((2,128,128,64),60,dtype=torch.int8);values[0]=-128
    f=SatelliteFields(metadata,values)
    coords=torch.tensor([[[5.,5.],[80.,80.]]])
    out=f.lookup(coords,torch.tensor([2019]))
    assert out[0,0,-1]==1 and out[0,1,-1]==0
    assert torch.count_nonzero(f.lookup(coords,torch.tensor([2018])))==0


def test_device_inverse_projection_matches_reference_near_dateline():
    xy=np.asarray([[[0.,0.],[100000.,200000.]],[[200000.,0.],[-10000.,10000.]]])
    lat=np.asarray([78.,-62.]);lon=np.asarray([179.9,-179.9])
    expected=unproject(xy,lat[:,None],lon[:,None])
    got=unproject_torch(torch.from_numpy(xy),torch.from_numpy(lat),torch.from_numpy(lon)).numpy()
    np.testing.assert_allclose(got,expected,atol=1e-8)


def test_analogue_index_export_preserves_training_only_predictions():
    rng=np.random.default_rng(1);x=rng.normal(size=(24,32,10)).astype(np.float32)
    y=rng.normal(size=(24,2)).astype(np.float32);meta=np.zeros((24,5));meta[:,1]=1;meta[:,4]=24
    v=rng.normal(size=(24,2)).astype(np.float32)
    a=AnalogueForecaster().fit(x,y,meta,v);b=import_analogue(export_analogue(a))
    np.testing.assert_array_equal(a.sample(x[:3],meta[:3],v[:3],16,np.random.default_rng(9)),b.sample(x[:3],meta[:3],v[:3],16,np.random.default_rng(9)))


def test_habitat_correction_starts_neutral_and_masks_unsupported_years():
    from wildcast.habitat import HabitatForecast
    metadata=[{"year":2019,"bounds":[0.,0.,10.,10.]}]
    fields=SatelliteFields(metadata,torch.full((1,128,128,64),60,dtype=torch.int8))
    model=HabitatForecast({"base_configs":[],"field_metadata":metadata,"n_drivers":4,"width":16},fields)
    x=torch.randn(2,8,10);h=torch.tensor([1.,24.]);v=torch.randn(2,2);dr=torch.randn(2,4)
    aux=torch.zeros(2,9,dtype=torch.float64);aux[:,0:2]=5;aux[:,8]=100
    q=torch.randn(2,8,2)
    initial=model.utility(x,h,v,dr,aux,torch.tensor([2019,2019]),q)
    torch.testing.assert_close(initial,torch.zeros_like(initial))
    with torch.no_grad():model.candidate[-1].weight.fill_(.1)
    absent=model.utility(x,h,v,dr,aux,torch.tensor([2018,2018]),q)
    torch.testing.assert_close(absent,torch.zeros_like(absent))


def test_crossfit_router_inference_is_independent_of_future_targets():
    from wildcast.router import RoutedForecast
    from wildcast.model import MovementTransformer
    from wildcast.evaluate import predictions
    rng=np.random.default_rng(22)
    x=rng.normal(size=(12,8,10)).astype(np.float32);y=rng.normal(size=(12,2)).astype(np.float32)
    meta=np.zeros((12,5));meta[:,1]=1;meta[:,4]=24
    v=rng.normal(size=(12,2)).astype(np.float32)
    analogue=AnalogueForecaster().fit(x,y,meta,v)
    base=MovementTransformer(1,width=32,layers=1,n_drivers=4,n_anchors=5,student_df=4)
    config={"base_configs":[base.config],"n_features":23,"conditional":False,"driver_indices":[0,1],
            "crossfit":True,"animal_names":{"one":0},"folds":3}
    model=RoutedForecast(config,export_analogue(analogue)).eval()
    aux=np.zeros((3,9));aux[:,4:6]=v[:3]
    a={"x":x[:3],"y":y[:3].copy(),"meta":meta[:3],"aux":aux,"drivers":np.zeros((3,4),np.float32),"anchors":np.zeros((3,5,2),np.float32)}
    with torch.inference_mode():
        first=predictions(model,a,np.arange(3),16,"cpu",torch.Generator().manual_seed(6))
        a["y"][:]=1e9
        second=predictions(model,a,np.arange(3),16,"cpu",torch.Generator().manual_seed(6))
    torch.testing.assert_close(first,second)
