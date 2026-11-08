"""Previous-year AlphaEarth context, using the publisher's vector overviews."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import requests

from .sources import dump
from .vegetation import origins

INDEX_URL = "https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/aef_index.parquet"
ATTRIBUTION = "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."


def dequantize(raw):
    """Publisher mapping for signed-int8 vectors; nodata is an absent vector."""
    raw = np.asarray(raw)
    good = (raw != -128).all(axis=-1)
    result = np.sign(raw.astype(np.float32)) * (raw.astype(np.float32) / 127.5) ** 2
    result[~good] = 0
    return result, good


def cache_satellite(root, cache_root, max_files=32, workers=3):
    root, cache_root = Path(root), Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    obs_hash = json.loads((root / "processed/windows.json").read_text())["observations_sha256"]
    if (cache_root / "source.json").exists() and json.loads((cache_root / "source.json").read_text())["observations_sha256"] != obs_hash:
        raise ValueError("Satellite cache row IDs refer to another observations file")
    index = cache_root / "aef_index.parquet"
    if not index.exists():
        with requests.get(INDEX_URL, stream=True, timeout=(15, 90)) as r:
            r.raise_for_status()
            with index.with_suffix(".tmp").open("wb") as out:
                size = 0
                for b in r.iter_content(1024 * 1024):
                    size += len(b)
                    if size > 80_000_000:
                        raise ValueError("AlphaEarth index exceeds expected budget")
                    out.write(b)
        index.with_suffix(".tmp").replace(index)
    f = pd.read_parquet(index).drop(columns=["geom", "geom_bbox"], errors="ignore")
    rows, lat, lon, times = origins(root)
    years = pd.to_datetime(times, unit="s", utc=True).year.to_numpy() - 1
    candidates = []
    for year in np.unique(years):
        if year < 2017:
            continue
        ix = np.flatnonzero(years == year)
        sub = f[(f.year == year) & (f.wgs84_east >= lon[ix].min()) & (f.wgs84_west <= lon[ix].max())
                & (f.wgs84_north >= lat[ix].min()) & (f.wgs84_south <= lat[ix].max())]
        for _, tile in sub.iterrows():
            take = ix[(lon[ix] >= tile.wgs84_west) & (lon[ix] <= tile.wgs84_east)
                      & (lat[ix] >= tile.wgs84_south) & (lat[ix] <= tile.wgs84_north)]
            if len(take):
                candidates.append((tile.to_dict(), take))
    candidates.sort(key=lambda v: (-len(v[1]), v[0]["path"]))

    def fetch(entry):
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform
        tile, take = entry
        url = tile["path"].replace("gs://", "https://storage.googleapis.com/", 1)
        key = hashlib.sha256(url.encode()).hexdigest()[:24]
        path = cache_root / (key + ".npz")
        if path.exists():
            return {"key": key, "status": "cached"}
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_MAX_RETRY=1):
                with rasterio.open(url) as ds:
                    if ds.count != 64 or 128 not in ds.overviews(1):
                        raise ValueError("Expected 64-band AlphaEarth file with 128x vector overview")
                    # The publisher correctly aggregates dequantized vectors and
                    # renormalizes each overview vector. Do not average int8 bands.
                    raw = ds.read(out_shape=(64, ds.height // 128, ds.width // 128), resampling=Resampling.nearest)
                    x, y = transform("EPSG:4326", ds.crs, lon[take].tolist(), lat[take].tolist())
                    rr, cc = rasterio.transform.rowcol(ds.transform, x, y)
                    rr, cc = np.asarray(rr) // 128, np.asarray(cc) // 128
                    good = (rr >= 0) & (cc >= 0) & (rr < raw.shape[1]) & (cc < raw.shape[2])
                    values = np.full((len(take), 64), -128, dtype=np.int8)
                    values[good] = raw[:, rr[good], cc[good]].T
                    vector, present = dequantize(values)
            with path.with_suffix(".tmp").open("wb") as out:
                np.savez_compressed(out, rows=rows[take], values=vector, present=present, year=np.asarray(tile["year"]))
            path.with_suffix(".tmp").replace(path)
            dump(path.with_suffix(".json"), {"url": url, "year": tile["year"], "origins": len(take),
                 "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "native_resolution_metres": 10,
                 "used_overview_factor": 128, "context_resolution_metres": 1280})
            return {"key": key, "status": "cached", "origins": len(take), "valid": int(present.sum())}
        except Exception as e:
            return {"key": key, "status": "error", "error_type": type(e).__name__}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(fetch, candidates[:max_files]):
            records.append(record)
            print(f"Satellite {len(records)}/{min(len(candidates), max_files)}: {record}", flush=True)
    source = {"observations_sha256": obs_hash,
              "attribution": ATTRIBUTION, "license": "CC BY 4.0", "index_url": INDEX_URL,
              "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(), "records": records,
              "temporal_rule": "Previous calendar year only. No nearest-year fallback for pre-2018 tracks.",
              "availability": "Retrospective archive, original publication vintage is not established."}
    dump(cache_root / "source.json", source)
    return {"files": len(records), "cached": sum(v["status"] == "cached" for v in records)}


def add_satellite(source_root, root, cache_root):
    source_root, root, cache_root = Path(source_root), Path(root), Path(cache_root)
    if source_root.resolve() == root.resolve():
        raise ValueError("Use a new data root")
    src, dest = source_root / "processed", root / "processed"
    manifest = json.loads((src / "windows.json").read_text())
    source = json.loads((cache_root / "source.json").read_text())
    if source["observations_sha256"] != manifest["observations_sha256"]:
        raise ValueError("Satellite cache observation fingerprint differs")
    rows, lat, lon, times = origins(source_root)
    years = pd.to_datetime(times, unit="s", utc=True).year.to_numpy() - 1
    values, present = np.zeros((len(rows), 64), np.float32), np.zeros(len(rows), bool)
    # Resolve overlapping UTM coverage deterministically, using valid vectors.
    for p in sorted(cache_root.glob("*.npz")):
        with np.load(p) as c:
            ix = np.searchsorted(rows, c["rows"])
            if np.any(ix >= len(rows)) or not np.array_equal(rows[ix], c["rows"]):
                raise ValueError("Unknown satellite origin IDs")
            keep = ~present[ix] & c["present"] & (years[ix] == int(c["year"]))
            values[ix[keep]] = c["values"][keep]; present[ix[keep]] = True
    a = dict(np.load(src / "windows.npz"))
    inverse = np.searchsorted(rows, a["aux"][:, 6].astype(int))
    raw = np.column_stack([values, present]).astype(np.float32)[inverse]
    train = a["meta"][:, 0] == 0
    centre = np.median(raw[train], axis=0)
    spread = np.maximum(np.quantile(raw[train], .75, axis=0) - np.quantile(raw[train], .25, axis=0), .1)
    a["drivers"] = np.column_stack([a["drivers"], np.clip((raw - centre) / spread, -12, 12).astype(np.float32)])
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "windows.npz", **a)
    for name in ["observations.parquet", "quality-report.json"]:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    report = {"origins": len(rows), "present": int(present.sum()), "fraction": float(present.mean()),
              "features": 65, "source": source}
    manifest.update(parent_windows_sha256=manifest["windows_sha256"],
                    windows_sha256=hashlib.sha256((dest / "windows.npz").read_bytes()).hexdigest(),
                    driver_names=manifest["driver_names"] + [f"satellite_A{i:02}" for i in range(64)] + ["satellite_present"],
                    driver_median=manifest["driver_median"] + centre.tolist(), driver_iqr=manifest["driver_iqr"] + spread.tolist(),
                    satellite=report)
    dump(dest / "windows.json", manifest)
    dump(root / "satellite-report.json", report)
    return {k: v for k, v in report.items() if k != "source"}
