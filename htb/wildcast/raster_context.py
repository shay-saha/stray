"""Shared prior-year satellite fields for scoring possible destinations."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .sources import dump


def prepare_fields(cache_root, output, workers=3):
    """Reuse selected public AEF files, not crops selected using future targets."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.transform import from_bounds
    from affine import Affine
    cache_root, output = Path(cache_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(cache_root / "aef_index.parquet").drop(columns=["geom", "geom_bbox"], errors="ignore")
    entries = []
    for p in sorted(cache_root.glob("*.json")):
        if p.name == "source.json":
            continue
        record = json.loads(p.read_text())
        if "url" not in record:
            continue
        source_path = record["url"].replace("https://storage.googleapis.com/", "gs://", 1)
        match = index[index.path == source_path]
        if len(match) != 1:
            raise ValueError("Satellite source index mismatch")
        tile = match.iloc[0]
        entries.append({"key": p.stem, "url": record["url"], "year": int(tile.year),
                        "bounds": [float(tile.wgs84_west), float(tile.wgs84_south), float(tile.wgs84_east), float(tile.wgs84_north)]})

    def fetch(entry):
        path = output / (entry["key"] + ".npz")
        if not path.exists():
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_MAX_RETRY=1):
                with rasterio.open(entry["url"]) as ds:
                    raw = ds.read(out_shape=(64, ds.height // 128, ds.width // 128), resampling=Resampling.nearest)
                    data = np.full((64,128,128), -128, np.int8)
                    reproject(source=raw, destination=data, src_transform=ds.transform * Affine.scale(128,128),
                              src_crs=ds.crs, src_nodata=-128, dst_transform=from_bounds(*entry["bounds"],128,128),
                              dst_crs="EPSG:4326", dst_nodata=-128, resampling=Resampling.nearest)
            with path.with_suffix(".tmp").open("wb") as f:
                np.savez_compressed(f, values=data.transpose(1,2,0), bounds=entry["bounds"], year=entry["year"])
            path.with_suffix(".tmp").replace(path)
        return {**entry, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(fetch, entries):
            records.append(record)
            print(f"Destination fields {len(records)}/{len(entries)}", flush=True)
    dump(output / "manifest.json", {"records": records, "license": "CC BY 4.0",
         "attribution": "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind.",
         "native_resolution_metres":10, "used_vector_overview_metres":1280,
         "grid": "128x128 WGS84 cells per source tile, nearest reprojection of publisher vector overview; no extra spatial information is created.",
         "temporal_rule": "Previous calendar year only; no future-year fallback.",
         "coverage_selection": "Reuse origin-count-ranked source files from the prior acquisition, not files chosen using target coordinates or errors."})
    return {"tiles":len(records), "bytes":sum(p.stat().st_size for p in output.glob("*.npz"))}


def unproject_torch(xy, latitude, longitude):
    """Spherical AEQD inverse; same geometry as data.unproject, on device."""
    lat0, lon0 = torch.deg2rad(latitude), torch.deg2rad(longitude)
    while lat0.ndim < xy.ndim - 1:
        lat0 = lat0.unsqueeze(-1)
        lon0 = lon0.unsqueeze(-1)
    x,y = xy.unbind(-1)
    rho = torch.linalg.vector_norm(xy,dim=-1)
    c = rho / 6371008.8
    ratio = torch.where(rho > 0, torch.sin(c) / rho.clamp_min(1e-12), torch.full_like(rho,1/6371008.8))
    lat = torch.asin((torch.cos(c)*torch.sin(lat0)+y*ratio*torch.cos(lat0)).clamp(-1,1))
    denominator = rho*torch.cos(lat0)*torch.cos(c)-y*torch.sin(lat0)*torch.sin(c)
    # atan2(0,0) has undefined gradients even behind a torch.where branch.
    denominator = torch.where(rho == 0, torch.ones_like(denominator), denominator)
    lon = lon0 + torch.atan2(x*torch.sin(c), denominator)
    lon = torch.where(rho==0,lon0,lon)
    return torch.stack([torch.rad2deg(lat),(torch.rad2deg(lon)+180)%360-180],-1)


class SatelliteFields(nn.Module):
    def __init__(self, metadata, values=None):
        super().__init__()
        self.metadata = metadata
        self.register_buffer("bounds",torch.tensor([v["bounds"] for v in metadata],dtype=torch.float64))
        self.register_buffer("years",torch.tensor([v["year"] for v in metadata],dtype=torch.long))
        self.register_buffer("values",torch.as_tensor(values,dtype=torch.int8) if values is not None else torch.full((len(metadata),128,128,64),-128,dtype=torch.int8))

    @classmethod
    def from_cache(cls, root):
        root=Path(root);manifest=json.loads((root/"manifest.json").read_text())
        arrays=[]
        for record in manifest["records"]:
            p=root/(record["key"]+".npz")
            if hashlib.sha256(p.read_bytes()).hexdigest()!=record["sha256"]:
                raise ValueError("Destination field checksum mismatch")
            with np.load(p) as a:arrays.append(a["values"])
        return cls(manifest["records"],np.stack(arrays))

    def lookup(self, coords, previous_year):
        """64D source vectors plus mask; coordinates are arbitrary candidates."""
        lat,lon=coords.unbind(-1)
        b=self.bounds.to(coords.dtype)
        match=(lon[...,None]>=b[:,0]) & (lon[...,None]<=b[:,2]) & (lat[...,None]>=b[:,1]) & (lat[...,None]<=b[:,3])
        match &= previous_year[:,None,None] == self.years[None,None]
        result=torch.zeros((*lat.shape,64),device=coords.device)
        present=torch.zeros(lat.shape,device=coords.device,dtype=torch.bool)
        # Adjacent UTM tiles can overlap. Try valid vectors from up to three
        # bounding matches instead of treating nodata as an observed vector.
        for _ in range(3):
            tile=match.to(torch.int8).argmax(-1)
            bb=b[tile]
            row=((bb[...,3]-lat)/(bb[...,3]-bb[...,1])*128).floor().long().clamp(0,127)
            col=((lon-bb[...,0])/(bb[...,2]-bb[...,0])*128).floor().long().clamp(0,127)
            raw=self.values[tile,row,col]
            good=match.any(-1)&(raw!=-128).all(-1)&~present
            vector=raw.float().sign()*(raw.float()/127.5).square()
            result=torch.where(good[...,None],vector,result)
            present |= good
            match.scatter_(-1,tile[...,None],False)
        return torch.cat([result,present.float()[...,None]],-1)

    def candidate_features(self, points, aux, previous_year):
        # Aux contains only origin latitude/longitude/time and causal scale.
        xy=points*aux[:,None,8:9]
        coords=unproject_torch(xy,aux[:,0],aux[:,1])
        vectors=self.lookup(coords,previous_year)
        origin=self.lookup(aux[:,None,:2],previous_year)
        both=vectors[...,-1:]*origin[...,-1:]
        delta=(vectors[...,:64]-origin[...,:64])*both
        return torch.cat([vectors,delta,both],-1)
