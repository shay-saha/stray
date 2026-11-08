"""Vectorized past-weather, ocean-temperature, terrain and land covariates.

All queries use issue positions/times. Reanalysis and static reconstructed
products are explicitly retrospective; no historical forecast vintage claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .migration_human_pressure import sha256

PROTOCOL = "wildcast-migration-environment-v1"
WEATHER = ("temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "precipitation_sum",
           "wind_speed_10m_max", "wind_direction_sin", "wind_direction_cos", "shortwave_radiation_sum",
           "et0_fao_evapotranspiration", "snowfall_sum")
RAW_WEATHER = ("temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "precipitation_sum",
               "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum",
               "et0_fao_evapotranspiration", "snowfall_sum")
WEATHER_NAMES = [f"weather_past_{days}day_{name}{suffix}" for days in (1, 7, 30)
                 for suffix in ("", "_coverage") for name in WEATHER]
SST_NAMES = ["sst_C", "sst_monthly_change_C", "sst_prior10year_anomaly_C", "sst_gradient_C_per_km"]
TERRAIN_NAMES = ["terrain_elevation_m", "terrain_slope_east", "terrain_slope_north"]
LOCAL_NAMES = WEATHER_NAMES + SST_NAMES + [n+"_present" for n in SST_NAMES] + TERRAIN_NAMES + [n+"_present" for n in TERRAIN_NAMES] + ["land_polygon", "land_polygon_available"]
LOCAL_VALUE_COLUMNS = ([i*20+j for i in range(3) for j in range(10)] + list(range(60,64))
                       + list(range(68,71)) + [74])


class Environment:
    def __init__(self, root, verify=True):
        self.root = Path(root)
        self.weather, self.terrain, self.terrain_arrays = {}, [], {}
        manifests = [self.root/"transfer.json", self.root/"terrain-transfer.json"]
        records = []
        for p in manifests:
            if not p.is_file():
                raise FileNotFoundError(p)
            source = json.loads(p.read_text())
            for record in source["files"]:
                asset = self.root / record["path"]
                if not asset.resolve().is_relative_to(self.root.resolve()):
                    raise ValueError("Environmental source escapes its cache")
                if verify and sha256(asset) != record["sha256"]:
                    raise ValueError("Environmental asset checksum differs: " + str(asset))
                records.append({"path":record["path"],"sha256":record["sha256"]})
        self.fingerprint = hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
        self._weather()
        self._sst()
        self._terrain()
        self._land()
        self.provenance = {"root":str(self.root.resolve()), "fingerprint":self.fingerprint, "files":records,
            "weather":"Cached Open-Meteo ERA5 daily values; one-degree query cells; UTC issue day minus2; means of preceding1/7/30 days; missing-day coverage retained",
            "ocean":"NOAA ERSSTv5 monthly2-degree SST; issue calendar month minus2; anomaly only against ten prior same-calendar-month years with>=8 observations",
            "terrain":"Cached Copernicus GLO90 elevation and central-difference slope; explicit missing pixels; static retrospective product",
            "land":"Natural Earth10m land polygons; static geographic context, not habitat type or measured land use",
            "vintage":"Retrospective product reconstructions. No operational historical availability guarantee.",
            "currents":"Not included; a temperature gradient is not an ocean current"}

    def _weather(self):
        groups = {}
        manifest = json.loads((self.root/"weather/manifest.json").read_text())
        self.grid = float(manifest["grid_degrees"])
        for path in sorted((self.root/"weather").glob("*.json")):
            if path.name == "manifest.json":
                continue
            data = json.loads(path.read_text())
            params, response = data.get("params",{}), data.get("response",{})
            if "daily" not in response or any(k not in response["daily"] for k in RAW_WEATHER):
                continue
            if params.get("timezone") not in ("GMT","UTC") or params.get("wind_speed_unit") != "ms":
                raise ValueError("Weather cache must use UTC-compatible days and m/s wind")
            key = tuple(int(v) for v in path.stem.split("_")[:2])
            expected = ((key[0]+.5)*self.grid,(key[1]+.5)*self.grid)
            if not np.allclose([params["latitude"],params["longitude"]],expected,atol=1e-7,rtol=0):
                raise ValueError("Weather point does not match its containing query cell")
            daily = response["daily"]
            dates = pd.to_datetime(daily["time"],utc=True).asi8 // (86400*10**9)
            raw = np.column_stack([daily[name] for name in RAW_WEATHER]).astype(float)
            angle = np.deg2rad(raw[:,5]); values=np.column_stack([raw[:,:5],np.sin(angle),np.cos(angle),raw[:,6:]])
            groups.setdefault(key,[]).append((dates,values))
        for key,parts in groups.items():
            days=np.concatenate([p[0] for p in parts]); values=np.concatenate([p[1] for p in parts])
            start,end=int(days.min()),int(days.max())
            matrix=np.full((end-start+1,10),np.nan)
            assigned=np.zeros_like(matrix,dtype=bool)
            conflicted=np.zeros_like(matrix,dtype=bool)
            for day,value in zip(days,values):
                row=int(day)-start; present=np.isfinite(value)
                conflict=assigned[row]&present&~np.isclose(matrix[row],value,rtol=1e-6,atol=1e-8,equal_nan=True)
                conflicted[row]|=conflict
                fill=~assigned[row]&present;matrix[row,fill]=value[fill];assigned[row,fill]=True
            matrix[conflicted]=np.nan
            good=np.isfinite(matrix)
            total=np.concatenate([np.zeros((1,10)),np.cumsum(np.where(good,matrix,0),axis=0)])
            counts=np.concatenate([np.zeros((1,10)),np.cumsum(good,axis=0)])
            self.weather[key]=(start,end,total,counts)

    def _sst(self):
        from netCDF4 import Dataset,num2date
        with Dataset(self.root/"sst.mnmean.nc") as d:
            dates=num2date(d["time"][:],d["time"].units)
            self.months=np.asarray([v.year*12+v.month-1 for v in dates])
            self.lat=np.asarray(d["lat"][:]);self.lon=np.asarray(d["lon"][:])
            if str(d["sst"].units).lower() not in ("degc","degree_c","degrees_c","degrees c"):
                raise ValueError("SST must be degrees Celsius")
            self.sst=np.asarray(np.ma.filled(d["sst"][:],np.nan),np.float32)

    def _terrain(self):
        import rasterio
        for path in sorted((self.root/"dem").glob("*.tif")):
            d=rasterio.open(path)
            if d.crs != rasterio.crs.CRS.from_epsg(4326):
                d.close();raise ValueError("Terrain input must be WGS84")
            self.terrain.append((path.name,d))

    def _land(self):
        import fiona,shapely
        with fiona.open("zip://"+str((self.root/"ne_10m_land.zip").resolve())) as source:
            self.land=shapely.union_all([shapely.geometry.shape(item["geometry"]) for item in source])
        shapely.prepare(self.land)

    def lookup(self, origins):
        import shapely
        o=np.asarray(origins,float)
        if o.ndim!=2 or o.shape[1]!=3 or not np.isfinite(o).all() or np.any(np.abs(o[:,0])>90) or np.any(np.abs(o[:,1])>180):
            raise ValueError("Expected finite latitude/longitude/issue-time origins")
        result=np.zeros((len(o),len(LOCAL_NAMES)),np.float32)
        keys=np.floor(o[:,:2]/self.grid).astype(int);cutoff=np.floor(o[:,2]/86400).astype(np.int64)-2
        for key in np.unique(keys,axis=0):
            record=self.weather.get(tuple(key))
            if record is None:continue
            ids=np.flatnonzero((keys==key).all(-1));start,end,total,counts=record
            for w,days in enumerate((1,7,30)):
                hi=np.clip(cutoff[ids]-start+1,0,len(total)-1)
                lo=np.clip(cutoff[ids]-days-start+1,0,len(total)-1)
                number=counts[hi]-counts[lo];summed=total[hi]-total[lo]
                mean=np.divide(summed,number,out=np.zeros_like(summed),where=number>0)
                result[ids,w*20:w*20+10]=mean
                result[ids,w*20+10:w*20+20]=number/days
        dates=pd.to_datetime(o[:,2],unit="s",utc=True);selected=dates.year.to_numpy()*12+dates.month.to_numpy()-3
        t=np.searchsorted(self.months,selected);valid=(t>=120)&(t<len(self.months))
        valid &= self.months[np.minimum(t,len(self.months)-1)]==selected
        ids=np.flatnonzero(valid)
        if len(ids):
            ti=t[ids];la=np.abs(self.lat[None]-o[ids,:1]).argmin(1)
            delta=(self.lon[None]-o[ids,1:2]+180)%360-180;lo=np.abs(delta).argmin(1)
            level=self.sst[ti,la,lo];history=np.stack([self.sst[ti-12*y,la,lo] for y in range(1,11)])
            count=np.isfinite(history).sum(0);normal=np.nansum(history,axis=0)/np.maximum(count,1)
            anomaly=np.where(count>=8,level-normal,np.nan);change=level-self.sst[ti-1,la,lo]
            north=np.minimum(la+1,len(self.lat)-1);south=np.maximum(la-1,0)
            dy=np.abs(self.lat[north]-self.lat[south])*111.2
            dx=4*111.2*np.maximum(np.cos(np.deg2rad(self.lat[la])),.01)
            gradient=np.hypot((self.sst[ti,north,lo]-self.sst[ti,south,lo])/np.maximum(dy,1),
                              (self.sst[ti,la,(lo+1)%len(self.lon)]-self.sst[ti,la,(lo-1)%len(self.lon)])/dx)
            values=np.column_stack([level,change,anomaly,gradient]);present=np.isfinite(values)
            result[ids,60:64]=np.where(present,values,0);result[ids,64:68]=present
        terrain=np.full((len(o),3),np.nan)
        for name,d in self.terrain:
            b=d.bounds;ids=np.flatnonzero(np.isnan(terrain[:,0])&(o[:,1]>=b.left)&(o[:,1]<b.right)&(o[:,0]>b.bottom)&(o[:,0]<=b.top))
            if not len(ids):continue
            if name not in self.terrain_arrays:self.terrain_arrays[name]=np.asarray(np.ma.filled(d.read(1,masked=True).astype(float),np.nan))
            a=self.terrain_arrays[name];c=np.floor((o[ids,1]-d.transform.c)/d.transform.a).astype(int);r=np.floor((o[ids,0]-d.transform.f)/d.transform.e).astype(int)
            inside=(r>=0)&(r<d.height)&(c>=0)&(c<d.width);ids,r,c=ids[inside],r[inside],c[inside]
            terrain[ids,0]=a[r,c]
            interior=(r>0)&(r<d.height-1)&(c>0)&(c<d.width-1)
            ii,rr,cc=ids[interior],r[interior],c[interior]
            dx=abs(d.transform.a)*111200*np.maximum(np.cos(np.deg2rad(o[ii,0])),.01)
            dy=abs(d.transform.e)*111200
            terrain[ii,1]=(a[rr,cc+1]-a[rr,cc-1])/(2*dx)
            terrain[ii,2]=(a[rr-1,cc]-a[rr+1,cc])/(2*dy)
        present=np.isfinite(terrain);result[:,68:71]=np.where(present,terrain,0);result[:,71:74]=present
        result[:,74]=shapely.contains_xy(self.land,o[:,1],o[:,0]);result[:,75]=1
        if not np.isfinite(result).all():raise ValueError("Nonfinite environmental covariates")
        return result

    def close(self):
        for _,d in self.terrain:d.close()
        self.terrain.clear();self.terrain_arrays.clear()


def prepare(data,sources,human,output,batch_size=16384):
    data,output,human=Path(data),Path(output),Path(human)
    if output.exists():raise FileExistsError("Environment caches are immutable")
    manifest=json.loads((data/"manifest.json").read_text());hashes={"manifest_sha256":sha256(data/"manifest.json"),"cases_sha256":sha256(data/"cases.npz")}
    if manifest["cases_sha256"]!=hashes["cases_sha256"]:raise ValueError("GPS cache checksum differs")
    with np.load(data/"cases.npz",allow_pickle=False) as a:
        ids=np.flatnonzero(np.isin(a["split"],[0,1,2]));origins=a["origin"][ids];splits=a["split"][ids]
    hm=json.loads((human/"features.json").read_text())
    if any(hm.get(k)!=v for k,v in hashes.items()) or sha256(human/"features.npz")!=hm["features_sha256"]:raise ValueError("Human context/GPS hashes differ")
    with np.load(human/"features.npz",allow_pickle=False) as h:
        if not np.array_equal(h["case_ids"],ids):raise ValueError("Human context origins differ")
        old=h["features"]
    env=Environment(sources)
    try:
        values=np.empty((len(ids),old.shape[1]+len(LOCAL_NAMES)),np.float32);values[:,:old.shape[1]]=old
        for start in range(0,len(ids),batch_size):
            end=min(start+batch_size,len(ids));values[start:end,old.shape[1]:]=env.lookup(origins[start:end])
            if start%(batch_size*8)==0:print(json.dumps({"environment_origins":end,"total":len(ids)}),flush=True)
        metadata={"protocol":PROTOCOL,**hashes,"sources":str(Path(sources).resolve()),"source_fingerprint":env.fingerprint,
            "source_provenance":env.provenance,"human_metadata":hm,"human_artifact":str(human.resolve()),
            "feature_names":hm["feature_names"]+LOCAL_NAMES,
            "value_columns":[0,7,14]+[old.shape[1]+i for i in LOCAL_VALUE_COLUMNS],
            "uses_future_targets":False,"origins":len(ids),"case_ids_sha256":hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
            "coverage_by_split":{str(code):{"weather":float((values[splits==code,old.shape[1]+10:old.shape[1]+20]>0).any(-1).mean()),
                "SST":float((values[splits==code,old.shape[1]+64]>0).mean()),"terrain":float((values[splits==code,old.shape[1]+71]>0).mean())} for code in (0,1,2)},
            "code_sha256":sha256(__file__)}
    finally:env.close()
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/"features.npz",case_ids=ids,origin=origins,features=values)
    metadata["features_sha256"]=sha256(output/"features.npz")
    (output/"features.json").write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({"origins":len(ids),"features":values.shape[1],"coverage":metadata["coverage_by_split"]}),flush=True)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("data","sources","human","output"):p.add_argument('--'+name,required=True)
    prepare(**vars(p.parse_args()))


if __name__=='__main__':main()
