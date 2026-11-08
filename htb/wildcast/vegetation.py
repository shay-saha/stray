"""Past-composite MODIS vegetation at actual GPS origins, with explicit QA.

These are retrospective archive measurements, not historical release vintages.
No target coordinates, future composites, interpolation, or imputed greenness.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import threading
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from .sources import dump

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-13A1-061"
ASSETS = ["500m_16_days_NDVI", "500m_16_days_EVI", "500m_16_days_pixel_reliability"]
FEATURES = [f"{band}_{stat}" for band in ("ndvi", "evi")
            for stat in ("centre", "mean3", "mean11", "mean51", "std51", "grid_x_difference51", "grid_y_difference51")]
FEATURES += ["centre_present", "good_fraction3", "good_fraction11", "good_fraction51", "snow_fraction51", "cloud_fraction51"]
FEATURES += ["grid_x_difference_present", "grid_y_difference_present"]
TOKEN_LOCK = threading.Lock()
TOKENS = {}


def composite_dates(timestamps, lag_days):
    """Nominal Terra MODIS 16-day starts strictly before the forecast origin."""
    if lag_days < 32:
        raise ValueError("Use at least 32 days of lag for completed MODIS composites")
    d = pd.to_datetime(np.asarray(timestamps) - lag_days * 86400, unit="s", utc=True)
    start = pd.to_datetime(d.year.astype(str) + "-01-01", utc=True)
    return start + pd.to_timedelta(16 * ((d.dayofyear - 1) // 16), unit="D")


def tile_indices(lat, lon):
    # MODIS sinusoidal global grid; geographic coordinates are degrees.
    x = 6371007.181 * np.deg2rad(lon) * np.cos(np.deg2rad(lat))
    y = 6371007.181 * np.deg2rad(lat)
    return (np.floor((x + 20015109.354) / 1111950.5196666666).astype(int),
            np.floor((10007554.677 - y) / 1111950.5196666666).astype(int))


def patch_features(ndvi, evi, qa):
    """51x51 native-pixel patches; only reliability=0 and valid VI accepted."""
    arrays = [np.asarray(v, dtype=float) for v in (ndvi, evi, qa)]
    if any(v.shape != (51, 51) for v in arrays):
        raise ValueError("Expected 51x51 vegetation patches")
    ndvi, evi, qa = arrays
    good = (qa == 0) & (ndvi >= -2000) & (ndvi <= 10000) & (evi >= -2000) & (evi <= 10000)
    x_present = good[:, 26:].any() and good[:, :25].any()
    y_present = good[:25].any() and good[26:].any()
    result = []
    for band in [ndvi * .0001, evi * .0001]:
        def mean(rows, cols):
            v = band[rows, cols]; ok = good[rows, cols]
            return float(v[ok].mean()) if ok.any() else 0.
        result.extend([float(band[25, 25]) if good[25, 25] else 0.,
                       mean(slice(24, 27), slice(24, 27)), mean(slice(20, 31), slice(20, 31)),
                       mean(slice(None), slice(None)), float(band[good].std()) if good.any() else 0.,
                       mean(slice(None), slice(26, None)) - mean(slice(None), slice(None, 25)) if x_present else 0.,
                       mean(slice(None, 25), slice(None)) - mean(slice(26, None), slice(None)) if y_present else 0.])
    result.extend([float(good[25, 25]), good[24:27, 24:27].mean(), good[20:31, 20:31].mean(),
                   good.mean(), (qa == 2).mean(), (qa == 3).mean()])
    result.extend([float(x_present), float(y_present)])
    return np.asarray(result, dtype=np.float32)


def origins(root):
    with np.load(Path(root) / "processed/windows.npz") as a:
        rows, first = np.unique(a["aux"][:, 6].astype(np.int64), return_index=True)
        aux = a["aux"][first]
    return rows, aux[:, 0], aux[:, 1], aux[:, 2]


def groups_for_origins(root):
    rows, lat, lon, times = origins(root)
    h, v = tile_indices(lat, lon)
    groups = {}
    for lag in [32, 48]:
        dates = composite_dates(times, lag).strftime("%Y-%m-%d")
        for i, (hh, vv, date) in enumerate(zip(h, v, dates)):
            groups.setdefault((int(hh), int(vv), date), set()).add(i)
    return (rows, lat, lon, times), groups


def _read_points(item, lat, lon):
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import Window
    signed = []
    for name in ASSETS:
        href = item["assets"][name]["href"]
        parts = urlparse(href)
        account, container = parts.hostname.split(".")[0], parts.path.split("/")[1]
        key = (account, container)
        # Microsoft's SDK caches one container token; signing every asset would
        # waste requests and rapidly hit the public authentication quota.
        with TOKEN_LOCK:
            cached = TOKENS.get(key)
            if cached is None or pd.Timestamp(cached["msft:expiry"]).timestamp() < datetime.now(timezone.utc).timestamp() + 60:
                r = requests.get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}", timeout=30)
                r.raise_for_status()
                cached = r.json(); TOKENS[key] = cached
        signed.append(href + "?" + cached["token"])
    result = np.zeros((len(lat), len(FEATURES)), dtype=np.float32)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=25,
                      GDAL_HTTP_MAX_RETRY=1, GDAL_CACHEMAX=128_000_000):
        from contextlib import ExitStack
        with ExitStack() as stack:
            ds = [stack.enter_context(rasterio.open(url)) for url in signed]
            if any(d.shape != ds[0].shape or d.transform != ds[0].transform or d.crs != ds[0].crs for d in ds[1:]):
                raise ValueError("MODIS assets are not aligned")
            x, y = transform("EPSG:4326", ds[0].crs, lon.tolist(), lat.tolist())
            rr, cc = rasterio.transform.rowcol(ds[0].transform, x, y)
            rr, cc = np.asarray(rr), np.asarray(cc)
            bh, bw = ds[0].block_shapes[0]
            keys = np.column_stack([rr // bh, cc // bw])
            for br, bc in np.unique(keys, axis=0):
                ix = np.flatnonzero((keys[:, 0] == br) & (keys[:, 1] == bc))
                # Read each compressed COG block once, with a 25-pixel halo.
                top, left = int(br * bh - 25), int(bc * bw - 25)
                height, width = bh + 50, bw + 50
                layers = [d.read(1, window=Window(left, top, width, height), boundless=True,
                                 fill_value=-3000 if k < 2 else -1) for k, d in enumerate(ds)]
                for i in ix:
                    r0, c0 = int(rr[i] - top - 25), int(cc[i] - left - 25)
                    result[i] = patch_features(*[a[r0:r0 + 51, c0:c0 + 51] for a in layers])
    return result


def cache_vegetation(root, cache_root, max_groups=128, workers=4):
    root, cache_root = Path(root), Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "processed/windows.json").read_text())
    obs_hash = manifest["observations_sha256"]
    meta_path = cache_root / "source.json"
    if meta_path.exists() and json.loads(meta_path.read_text())["observations_sha256"] != obs_hash:
        raise ValueError("Vegetation row IDs refer to a different observations file")
    if meta_path.exists() and json.loads(meta_path.read_text())["features"] != FEATURES:
        # Cached measurements can be regenerated without deleting source data.
        old_features = json.loads(meta_path.read_text())["features"]
        if old_features == FEATURES[:20]:
            for p in cache_root.glob("h*.npz"):
                with np.load(p) as a:
                    values = a["values"]
                    # Existing pilot patches did not record directional masks;
                    # mark gradients unavailable rather than inventing validity.
                    values[:, [5, 6, 12, 13]] = 0
                    values = np.column_stack([values, np.zeros((len(values), 2), np.float32)])
                    rows0, end0 = a["rows"], a["end"]
                np.savez_compressed(p, rows=rows0, values=values, end=end0)
                metadata = json.loads(p.with_suffix(".json").read_text())
                metadata.update(data_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), pilot_gradient_masks_unavailable=True)
                dump(p.with_suffix(".json"), metadata)
        else:
            raise ValueError("Vegetation feature schema differs; use a fresh cache root")
    dump(meta_path, {"observations_sha256": obs_hash, "features": FEATURES, "collection": COLLECTION,
                    "attribution": "NASA LP DAAC MOD13A1.061; Microsoft Planetary Computer",
                    "availability": "Composite ends precede origin. Retrospective archive, not historical publication vintage.",
                    "missing": "No fallback across dates, no fabricated observations. QA=0 only.",
                    "spatial_note": "Native nominal 500m pixels; 3/11/51-pixel patches. Edges outside tile are masked.",
                    "source": "https://doi.org/10.5067/MODIS/MOD13A1.061"})
    (rows, lat, lon, times), groups = groups_for_origins(root)
    ranked = sorted(groups, key=lambda k: (-len(groups[k]), k))[:max_groups]
    rate_limited = threading.Event()

    def fetch(key):
        h, v, date = key
        name = f"h{h:02}v{v:02}_{date}"
        path = cache_root / (name + ".npz")
        if path.exists():
            return {"key": name, "status": "cached"}
        if rate_limited.is_set():
            return {"key": name, "status": "deferred_rate_limit"}
        ix = np.asarray(sorted(groups[key]))
        params = {"collections": COLLECTION, "datetime": date + "T00:00:00Z/" + date + "T23:59:59Z",
                  "query": json.dumps({"modis:horizontal-tile": {"eq": h}, "modis:vertical-tile": {"eq": v}}), "limit": 20}
        try:
            r = requests.get(STAC + "/search", params=params, timeout=45)
            r.raise_for_status()
            items = [j for j in r.json()["features"] if j["id"].startswith("MOD13A1.")
                     and j["properties"].get("start_datetime", "").startswith(date)]
            if not items:
                return {"key": name, "status": "unavailable"}
            item = sorted(items, key=lambda j: j["id"])[-1]
            end = pd.Timestamp(item["properties"]["end_datetime"]).timestamp()
            if not np.all(end < times[ix]):
                raise ValueError("Composite includes future observations")
            values = _read_points(item, lat[ix], lon[ix])
            with path.with_suffix(".tmp").open("wb") as out:
                np.savez_compressed(out, rows=rows[ix], values=values, end=np.asarray(end))
            path.with_suffix(".tmp").replace(path)
            # Deliberately retain only unsigned asset references, never SAS URLs.
            dump(cache_root / (name + ".json"), {"item": item["id"], "start": item["properties"]["start_datetime"],
                 "end": item["properties"]["end_datetime"], "assets": {n: item["assets"][n]["href"] for n in ASSETS},
                 "query": params, "stac_response_sha256": hashlib.sha256(r.content).hexdigest(),
                 "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "origins": len(ix),
                 "retrieved_utc": datetime.now(timezone.utc).isoformat()})
            return {"key": name, "status": "cached", "origins": len(ix), "good_patches": int((values[:, 17] > 0).sum())}
        except Exception as e:
            # GDAL error strings can include signed URLs; never persist them.
            if isinstance(e, requests.HTTPError) and e.response.status_code == 429:
                rate_limited.set()
                return {"key": name, "status": "rate_limited", "retry_after": e.response.headers.get("Retry-After")}
            return {"key": name, "status": "error", "error_type": type(e).__name__}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(fetch, ranked):
            records.append(result)
            print(f"Vegetation {len(records)}/{len(ranked)}: {result}", flush=True)
            dump(cache_root / "manifest.json", {"total_groups": len(groups), "requested_groups": len(ranked), "records": records})
    return {"groups": len(records), "cached": sum(v["status"] == "cached" for v in records), "cache": str(cache_root)}


def add_vegetation(source_root, root, cache_root):
    source_root, root, cache_root = Path(source_root), Path(root), Path(cache_root)
    if source_root.resolve() == root.resolve():
        raise ValueError("Use a new data root")
    src, dest = source_root / "processed", root / "processed"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / "windows.json").read_text())
    source = json.loads((cache_root / "source.json").read_text())
    if source["observations_sha256"] != manifest["observations_sha256"]:
        raise ValueError("Vegetation cache observation fingerprint differs")
    a = dict(np.load(src / "windows.npz"))
    rows, lat, lon, times = origins(source_root)
    h, v = tile_indices(lat, lon)
    columns = []
    available = []
    ends = []
    for lag in [32, 48]:
        vals = np.zeros((len(rows), len(FEATURES)), dtype=np.float32)
        present = np.zeros(len(rows), dtype=bool)
        end_times = np.zeros(len(rows))
        dates = composite_dates(times, lag).strftime("%Y-%m-%d")
        for path in cache_root.glob("h*.npz"):
            prefix, date = path.stem.split("_")
            hh, vv = int(prefix[1:3]), int(prefix[4:6])
            with np.load(path) as c:
                ix = np.searchsorted(rows, c["rows"])
                ok = ix < len(rows)
                ok[ok] &= rows[ix[ok]] == c["rows"][ok]
                ci = np.flatnonzero(ok); ix = ix[ok]
                match = (h[ix] == hh) & (v[ix] == vv) & (dates[ix] == date)
                ix, ci = ix[match], ci[match]
                if np.any(float(c["end"]) >= times[ix]):
                    raise ValueError("Future vegetation composite")
                vals[ix] = c["values"][ci]; present[ix] = True; end_times[ix] = float(c["end"])
        age = np.where(present, (times - end_times) / 86400 / 60, 0)
        columns.append(np.column_stack([vals, present, age]).astype(np.float32))
        available.append(present & (vals[:, 17] > 0)); ends.append(end_times)
    both = available[0] & available[1] & (ends[0] > ends[1])
    days = np.maximum((ends[0] - ends[1]) / 86400, 1)
    trend = np.column_stack([np.where(both, (columns[0][:, i] - columns[1][:, i]) / days * 16, 0) for i in [3, 10]])
    raw = np.column_stack([*columns, trend, both]).astype(np.float32)
    names = [f"vegetation_lag{lag}_{n}" for lag in [32, 48] for n in [*FEATURES, "cached", "age_scaled60d"]]
    names += ["vegetation_ndvi_change16d", "vegetation_evi_change16d", "vegetation_change_present"]
    inverse = np.searchsorted(rows, a["aux"][:, 6].astype(int))
    raw = raw[inverse]
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite vegetation drivers")
    train = a["meta"][:, 0] == 0
    centre = np.median(raw[train], axis=0)
    spread = np.maximum(np.quantile(raw[train], .75, axis=0) - np.quantile(raw[train], .25, axis=0), .1)
    scaled = np.clip((raw - centre) / spread, -12, 12).astype(np.float32)
    a["drivers"] = np.column_stack([a["drivers"], scaled])
    np.savez_compressed(dest / "windows.npz", **a)
    for name in ["observations.parquet", "quality-report.json"]:
        if (src / name).exists():
            shutil.copy2(src / name, dest / name)
    report = {"origins": len(rows), "good_recent_fraction": float(available[0].mean()),
              "good_previous_fraction": float(available[1].mean()), "paired_trend_fraction": float(both.mean()),
              "features": len(names), "cache": str(cache_root), "source": source}
    manifest.update(parent_windows_sha256=manifest["windows_sha256"],
                    windows_sha256=hashlib.sha256((dest / "windows.npz").read_bytes()).hexdigest(),
                    driver_names=manifest["driver_names"] + names,
                    driver_median=manifest["driver_median"] + centre.tolist(),
                    driver_iqr=manifest["driver_iqr"] + spread.tolist(), vegetation=report)
    manifest["unimplemented_drivers"] = [v for v in manifest.get("unimplemented_drivers", []) if v != "vegetation green-up"]
    dump(dest / "windows.json", manifest)
    dump(root / "vegetation-report.json", report)
    return report
