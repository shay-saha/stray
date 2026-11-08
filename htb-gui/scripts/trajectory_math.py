"""Batched exact migration features and explicit scenario-state interpolation."""
import numpy as np
import pandas as pd

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


def advance(rows,followed,horizons,issue,lags,step_days=30):
    days=np.unique(np.r_[np.arange(.25,step_days+.01,.25),step_days-lags[lags<step_days],horizons])
    days=days[(days>0)&(days<=step_days)]
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
    cutoff=issue+step_days*86400-400*86400
    for i,(times,old) in enumerate(rows):
        keep=times>=cutoff
        result.append((np.r_[times[keep],extra],np.concatenate([old[keep],coords[i]],axis=0)))
    return result


