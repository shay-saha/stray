import numpy as np
import pandas as pd
import pytest

from wildcast import migration_data as legacy
from wildcast import migration_repaired_features as f
from wildcast import multiscale_data as previous
from wildcast.data import project
from wildcast.migration_baselines import _velocity


def track():
    dates=pd.date_range("2019-01-01",periods=3001,freq="5min",tz="UTC")
    t=np.arange(len(dates))
    lat=30.+.0001*t
    lon=(179.5+.0003*t+180)%360-180
    return lat,lon,dates.asi8/1e9,dates


def test_fine_history_retains_every_previous_feature_bitwise_with_real_dateline_fixes():
    args=track();i=len(args[0])-1
    new=f.query_features(*args,i)
    old=previous.query_features(*args,i)
    assert new[0].shape == (29,10) and new[2].shape == (22,)
    for array in (0,1,3,4):np.testing.assert_array_equal(new[array][f.PREVIOUS_LAG_COLUMNS],old[array])
    np.testing.assert_array_equal(new[2][:18],old[2])
    assert f.STATIC_NAMES[:14] == legacy.STATIC_NAMES


def test_actual_five_minute_velocity_has_no_fourteen_minute_floor():
    lat,lon,times,dates=track();i=len(lat)-1
    history,mask,static,_,_=f.query_features(lat,lon,times,dates,i)
    expected=-project(lat[i-1],lon[i-1],lat[i],lon[i])/f.UNIT/((times[i]-times[i-1])/f.DAY)
    np.testing.assert_allclose(static[18:20],expected,rtol=2e-7)
    a={"history":history[None],"history_mask":mask[None]}
    np.testing.assert_allclose(_velocity(a,np.array([0]))[0],expected,rtol=2e-7)


@pytest.mark.parametrize("older_seconds,valid",[(75,True),(75.001,False)])
def test_five_minute_slot_uses_one_sided_quarter_lag_tolerance(older_seconds,valid):
    dates=pd.DatetimeIndex(pd.to_datetime(["2020-01-01T00:00:00Z","2020-01-02T00:00:00Z"]))
    origin=dates[-1].timestamp()
    times=np.array([origin-300-older_seconds,origin])
    index,mask=f.past_indices(times,1)
    slot=np.flatnonzero(f.LAGS == 5/1440)[0]
    assert bool(mask[slot]) is valid
    assert index[slot] == 0


def test_future_poison_is_neither_parsed_nor_used_in_repaired_query_features():
    lat,lon,times,dates=track();i=99
    expected=f.query_features(lat,lon,times,dates,i)
    values=[list(lat[:i+1])+["bad future latitude"],list(lon[:i+1])+["bad future longitude"],
            list(times[:i+1])+["bad future clock"],list(dates[:i+1])+["bad future date"]]
    actual=f.query_features(*values,i)
    for left,right in zip(expected,actual):np.testing.assert_array_equal(left,right)


def test_missing_fine_slots_stay_zero_without_interpolation():
    dates=pd.date_range("2020-01-01",periods=30,freq="h",tz="UTC")
    n=len(dates)
    history,mask,static,ll,ht=f.query_features(np.arange(n)*.001,np.arange(n)*.002,dates.asi8/1e9,dates,n-1)
    for lag in (5/1440,15/1440):
        j=np.flatnonzero(f.LAGS == lag)[0]
        assert not mask[j] and not history[j].any() and not ll[j].any() and ht[j] == 0
    np.testing.assert_array_equal(static[18:],[0,0,0,0])


def test_corrected_identity_policy_never_splices_coincident_herd_members():
    dates=pd.date_range("2020-01-01",periods=24,freq="h",tz="UTC")
    rows=[]
    for animal in ("a","b"):
        for i,stamp in enumerate(dates):
            rows.append({"animal":animal,"timestamp":stamp,"lat":10.+.001*i+(1 if animal == "b" and i >= 5 else 0),
                         "lon":20.+.002*i,"species":"Synthetic deer","study":"source"})
    raw=pd.DataFrame(rows)
    old,old_mapping,_=legacy.canonicalize(raw)
    assert old_mapping["a"] == old_mapping["b"] and len(old) == 24
    repaired,mapping,links=legacy.canonicalize(raw,infer_coordinate_aliases=False)
    assert mapping == {"a":"a","b":"b"} and not links and len(repaired) == 48
    assert repaired.groupby("animal").size().to_dict() == {"a":24,"b":24}
    assert len(raw) == 48


def test_disabling_coordinate_inference_retains_explicit_audited_duplicate_links():
    dates=pd.date_range("2020-01-01",periods=8,freq="h",tz="UTC")
    raw=pd.DataFrame([{"animal":animal,"timestamp":stamp,"lat":10.+i*.001,"lon":20.+i*.002,
                       "species":"Synthetic deer","study":"source"}
                      for animal in ("a","b") for i,stamp in enumerate(dates)])
    _,separate,_=legacy.canonicalize(raw,infer_coordinate_aliases=False)
    assert separate["a"] != separate["b"]
    result,mapping,links=legacy.canonicalize(raw,[("a","b",{"evidence":"independently_reviewed_primitive"})],
                                            infer_coordinate_aliases=False)
    assert len(result) == 8 and mapping["a"] == mapping["b"]
    assert links[0]["evidence"] == "independently_reviewed_primitive"
