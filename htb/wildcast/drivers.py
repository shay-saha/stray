"""Audited migration cues, causal movement memory, and retrospective weather.

Weather values precede the origin, but this is a present-day ERA5 archive, not
an archive of the forecast/product vintage available on the historical date.
It must not be described as a validated operational weather input.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import project
from .sources import dump, session

WEATHER_VARS = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "precipitation_sum",
                "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum",
                "et0_fao_evapotranspiration", "snowfall_sum"]
WEATHER_SCALES = np.asarray([30., 30., 30., 10., 10., 180., 30., 10., 10.])


def cache_weather(root, max_cells=64, grid_degrees=1., budget_mb=128):
    root = Path(root)
    f = pd.read_parquet(root / "processed/observations.parquet")
    f["cell_lat"] = np.floor(f.lat / grid_degrees).astype(int)
    f["cell_lon"] = np.floor(f.lon / grid_degrees).astype(int)
    groups = f.groupby(["cell_lat", "cell_lon"])
    order = groups.size().sort_values(ascending=False).head(max_cells)
    out = root / "environment/weather"; out.mkdir(parents=True, exist_ok=True)
    records, total = [], 0
    s = requests.Session()
    s.headers["User-Agent"] = "WildCast-HackTheBurgh/0.2 (cached research weather client)"
    for (a, b), n in order.items():
        g = groups.get_group((a, b))
        params = {"latitude": float((a + .5) * grid_degrees), "longitude": float((b + .5) * grid_degrees),
                  "start_date": (g.timestamp.min() - pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
                  "end_date": g.timestamp.max().strftime("%Y-%m-%d"), "daily": ",".join(WEATHER_VARS),
                  "models": "era5", "timezone": "GMT", "cell_selection": "nearest", "elevation": "nan", "wind_speed_unit": "ms"}
        path = out / f"{a}_{b}.json"
        request_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        try:
            if path.exists():
                data = json.loads(path.read_text())
                if data.get("request_hash") != request_hash:
                    raise ValueError("Cached weather query differs; use a fresh cache root")
            else:
                r = s.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=(15, 90))
                if r.status_code == 429:
                    records.append({"cell": [int(a), int(b)], "status": "rate_limited", "retry_after": r.headers.get("Retry-After")})
                    print(f"Weather quota reached; cached data preserved. Retry after {r.headers.get('Retry-After', 'provider reset')} seconds.", flush=True)
                    break
                r.raise_for_status()
                if total + len(r.content) > budget_mb * 1_000_000:
                    break
                j = r.json()
                if "daily" not in j or any(v not in j["daily"] for v in WEATHER_VARS):
                    raise ValueError("Incomplete weather schema")
                data = {"request_hash": request_hash, "params": params, "response": j, "url": r.url,
                        "response_sha256": hashlib.sha256(r.content).hexdigest(),
                        "retrieved_utc": datetime.now(timezone.utc).isoformat()}
                dump(path, data)
                time.sleep(.2)
            total += path.stat().st_size
            records.append({"cell": [int(a), int(b)], "observations": int(n), "path": str(path), "status": "cached"})
            print(f"Weather {len(records)}/{len(order)} cells; {total / 1e6:.1f} MB", flush=True)
        except (requests.RequestException, ValueError) as e:
            records.append({"cell": [int(a), int(b)], "status": "error", "error": str(e)})
            print(f"Weather cell {a},{b}: {e}", flush=True)
    report = {"provider": "Open-Meteo ERA5 archive", "source": "https://open-meteo.com/en/docs/historical-weather-api",
              "attribution": "Open-Meteo; contains modified Copernicus Climate Change Service information. Data CC BY 4.0.",
              "grid_degrees": grid_degrees, "native_grid_degrees": .25, "variables": WEATHER_VARS,
              "note": "Spatially coarsened point queries; daily UTC values. Retrospective reanalysis, not historical forecast vintages.",
              "bytes": total, "records": records}
    dump(out / "manifest.json", report)
    return report


def daylength(latitude, dayofyear):
    """Geometric day length in hours; polar night/day supported, no refraction."""
    gamma = 2 * np.pi * (np.asarray(dayofyear) - 1) / 365.2425
    declination = (.006918 - .399912 * np.cos(gamma) + .070257 * np.sin(gamma)
                   - .006758 * np.cos(2 * gamma) + .000907 * np.sin(2 * gamma)
                   - .002697 * np.cos(3 * gamma) + .00148 * np.sin(3 * gamma))
    angle = -np.tan(np.deg2rad(np.clip(latitude, -89.9999, 89.9999))) * np.tan(declination)
    return 24 / np.pi * np.arccos(np.clip(angle, -1, 1))


def astronomy(lat, lon, timestamp, horizon):
    stamp = pd.Timestamp(timestamp, unit="s", tz="UTC")
    doy = stamp.dayofyear + (timestamp % 86400) / 86400
    theta = 2 * np.pi * doy / 365.2425
    solar = 2 * np.pi * ((timestamp / 3600 + lon / 15) % 24) / 24
    lunar = 2 * np.pi * ((timestamp - 947182440.) / (29.530588853 * 86400))
    daylight = daylength(lat, doy)
    return {"latitude_sin": np.sin(np.deg2rad(lat)), "latitude_cos": np.cos(np.deg2rad(lat)),
            "longitude_sin": np.sin(np.deg2rad(lon)), "longitude_cos": np.cos(np.deg2rad(lon)),
            "solar_sin": np.sin(solar), "solar_cos": np.cos(solar),
            "season_sin": np.sin(theta), "season_cos": np.cos(theta),
            "season2_sin": np.sin(2 * theta), "season2_cos": np.cos(2 * theta),
            "photoperiod": daylight / 24, "photoperiod_change7d": (daylight - daylength(lat, doy - 7)) / 2,
            "future_photoperiod_at_origin": daylength(lat, doy + horizon / 24) / 24,
            "lunar_sin_approx": np.sin(lunar), "lunar_cos_approx": np.cos(lunar)}


class WeatherLookup:
    def __init__(self, root, lag_days=2):
        if lag_days < 1:
            raise ValueError("Weather must exclude the incomplete UTC day containing the origin")
        self.lag_days = lag_days
        self.cells = {}
        self._feature_cache = {}
        path = Path(root) / "environment/weather"
        if not (path / "manifest.json").exists():
            self.grid = 1.
            return
        self.grid = json.loads((path / "manifest.json").read_text())["grid_degrees"]
        for p in path.glob("*.json"):
            if p.name == "manifest.json":
                continue
            j = json.loads(p.read_text())["response"]["daily"]
            dates = pd.to_datetime(j["time"], utc=True).astype("int64").to_numpy() / 1e9
            values = np.column_stack([j[v] for v in WEATHER_VARS]).astype(float)
            key = tuple(int(t) for t in p.stem.split("_")[:2])
            if key in self.cells:
                old_dates, old_values = self.cells[key]
                dates = np.concatenate([old_dates, dates]); values = np.concatenate([old_values, values])
                # Shared point queries may cover different year ranges. Merge
                # observations by valid UTC day without filling absent years.
                order = np.argsort(dates, kind="stable")
                dates, values = dates[order], values[order]
                keep = np.r_[np.diff(dates) != 0, True]
                dates, values = dates[keep], values[keep]
            self.cells[key] = (dates, values)

    def features(self, lat, lon, origin):
        key = (int(np.floor(lat / self.grid)), int(np.floor(lon / self.grid)))
        cutoff = np.floor(origin / 86400) * 86400 - self.lag_days * 86400
        cache_key = (*key, int(cutoff))
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        result = {}
        for days in [1, 7, 30]:
            mean = np.full(len(WEATHER_VARS), np.nan)
            if key in self.cells:
                dates, values = self.cells[key]
                # The selected UTC days END before origin. Never query a target location.
                ix = (dates <= cutoff) & (dates > cutoff - days * 86400)
                if ix.any():
                    v = values[ix]
                    good = np.isfinite(v).sum(0)
                    mean = np.divide(np.nansum(v, axis=0), good, out=mean, where=good > 0)
                    # Dominant directions are circular, not arithmetic averages.
                    angles = np.deg2rad(v[:, 5]); valid = np.isfinite(angles)
                    if valid.any():
                        mean[5] = np.rad2deg(np.arctan2(np.sin(angles[valid]).mean(), np.cos(angles[valid]).mean())) % 360
            for i, name in enumerate(WEATHER_VARS):
                if i == 5:
                    result[f"weather_{days}d_wind_sin"] = float(np.sin(np.deg2rad(mean[i]))) if np.isfinite(mean[i]) else 0.
                    result[f"weather_{days}d_wind_cos"] = float(np.cos(np.deg2rad(mean[i]))) if np.isfinite(mean[i]) else 0.
                else:
                    result[f"weather_{days}d_{name}"] = float(mean[i] / WEATHER_SCALES[i]) if np.isfinite(mean[i]) else 0.
                result[f"weather_{days}d_{name}_present"] = float(np.isfinite(mean[i]))
        self._feature_cache[cache_key] = result
        return result


def history_features(lat, lon, timestamps, origin_index, phase_start, scale):
    """All data are from the same phase/segment and at or before the origin."""
    i = origin_index; now = timestamps[i]
    lo = max(phase_start, int(np.searchsorted(timestamps, now - 365 * 86400)))
    # Bound work for long tracks; preserve temporal coverage via regular subsampling.
    ids = np.arange(lo, i + 1)
    if len(ids) > 2048:
        ids = ids[np.linspace(0, len(ids) - 1, 2048, dtype=int)]
    xy = project(lat[ids], lon[ids], lat[i], lon[i]) / scale
    times = timestamps[ids]
    out = {"log_context_scale_metres": np.log1p(scale), "log_memory_hours": np.log1p((now - times[0]) / 3600)}
    anchors = [np.zeros(2)]
    for hours in [6, 24, 168, 720]:
        ix = times >= now - hours * 3600
        window = xy[ix]; t = times[ix]
        present = len(window) >= 2
        centre = window.mean(0)
        radius = np.sqrt(np.mean(np.sum((window - centre) ** 2, axis=-1)))
        net = np.linalg.norm(window[-1] - window[0])
        step = np.linalg.norm(np.diff(window, axis=0), axis=-1)
        path = step.sum()
        duration = max((t[-1] - t[0]) / 3600, .01)
        out.update({f"memory_{hours}h_centre_e": np.arcsinh(centre[0]), f"memory_{hours}h_centre_n": np.arcsinh(centre[1]),
                    f"memory_{hours}h_radius": np.log1p(radius), f"memory_{hours}h_net": np.log1p(net),
                    f"memory_{hours}h_path": np.log1p(path), f"memory_{hours}h_speed": np.log1p(path / duration),
                    f"memory_{hours}h_straightness": net / max(path, .001),
                    f"memory_{hours}h_rest_fraction": float(np.mean(step * scale < 25)) if len(step) else 1.,
                    f"memory_{hours}h_duration_fraction": min(duration / hours, 1.), f"memory_{hours}h_present": float(present)})
        if hours in [24, 168, 720]:
            anchors.append(centre)
    for days in [1, 7, 30, 365]:
        j = int(np.searchsorted(timestamps, now - days * 86400, side="right") - 1)
        valid = j >= phase_start and (now - timestamps[j]) <= (days + max(1, days * .2)) * 86400
        point = project(lat[j], lon[j], lat[i], lon[i]) / scale if valid else np.zeros(2)
        out.update({f"past_{days}d_e": np.arcsinh(point[0]), f"past_{days}d_n": np.arcsinh(point[1]), f"past_{days}d_present": float(valid)})
    # Anchor 4 is modest short-term momentum; the head also receives horizon.
    anchors.append((xy[-1] - xy[-2]) if len(xy) > 1 else np.zeros(2))
    return out, np.asarray(anchors, dtype=np.float32)


def enrich(source_root, root, weather=True, weather_lag_days=2, traits=False):
    source_root, root = Path(source_root), Path(root)
    src = source_root / "processed"; dest = root / "processed"; dest.mkdir(parents=True, exist_ok=True)
    original = json.loads((src / "windows.json").read_text())
    a = dict(np.load(src / "windows.npz"))
    f = pd.read_parquet(src / "observations.parquet")
    if src.resolve() == dest.resolve():
        raise ValueError("Enrichment requires a new data root to preserve the frozen experiment")
    for name in ["observations.parquet", "quality-report.json"]:
        shutil.copyfile(src / name, dest / name)
    f["phase"] = "train"
    for animal, ids in f.groupby("animal", sort=False).groups.items():
        ids = np.asarray(list(ids)); assignment = original["animal_assignments"][animal]
        if assignment == "temporal":
            f.loc[ids[int(len(ids) * .65):int(len(ids) * .8)], "phase"] = "val_temporal"
            f.loc[ids[int(len(ids) * .8):], "phase"] = "test_temporal"
        else:
            f.loc[ids, "phase"] = assignment
    lookup = WeatherLookup(root, weather_lag_days)
    trait_lookup = None
    if traits:
        from .traits import TraitLookup
        trait_lookup = TraitLookup(source_root, root, f, original["species"])
        dump(dest / "trait-report.json", trait_lookup.report)
    unique_origins = np.unique(a["aux"][:, 6].astype(int))
    origin_windows = {}
    for k, ix in enumerate(a["aux"][:, 6].astype(int)):
        origin_windows.setdefault(int(ix), k)
    features_by_origin, anchors_by_origin = {}, {}
    for _, g in f.groupby(["animal", "segment", "phase"], sort=False):
        ids = g.index.to_numpy(); positions = {int(v): k for k, v in enumerate(ids)}
        selected = [int(v) for v in ids if int(v) in origin_windows]
        lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
        timestamps = g.timestamp.astype("int64").to_numpy() / 1e9
        for row in selected:
            i = positions[row]; scale = a["aux"][origin_windows[row], 8]
            values, anchors = history_features(lat, lon, timestamps, i, 0, scale)
            if weather:
                values.update(lookup.features(lat[i], lon[i], timestamps[i]))
            if trait_lookup is not None:
                values.update(trait_lookup.features(g.animal.iloc[0], g.species.iloc[0], timestamps[i]))
            features_by_origin[row], anchors_by_origin[row] = values, anchors
        print(f"Drivers: {len(features_by_origin)}/{len(unique_origins)} forecast origins", flush=True)
    rows, anchors = [], []
    names = None
    for ix, meta, aux in zip(range(len(a["meta"])), a["meta"], a["aux"]):
        origin = int(aux[6]); values = dict(features_by_origin[origin])
        values.update(astronomy(aux[0], aux[1], aux[2], int(meta[4])))
        if names is None:
            names = list(values)
        rows.append([values[name] for name in names]); anchors.append(anchors_by_origin[origin])
    raw = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite migration drivers")
    train = a["meta"][:, 0] == 0
    centre = np.median(raw[train], axis=0)
    spread = np.maximum(np.quantile(raw[train], .75, axis=0) - np.quantile(raw[train], .25, axis=0), .1)
    a["drivers"] = np.clip((raw - centre) / spread, -12, 12).astype(np.float32)
    a["anchors"] = np.clip(np.asarray(anchors, dtype=np.float32), -10000, 10000)
    np.savez_compressed(dest / "windows.npz", **a)
    manifest = {**original, "parent_windows_sha256": original["windows_sha256"],
                "windows_sha256": hashlib.sha256((dest / "windows.npz").read_bytes()).hexdigest(),
                "driver_names": names, "driver_median": centre.tolist(), "driver_iqr": spread.tolist(),
                "n_anchors": a["anchors"].shape[1], "weather_enabled": weather,
                "traits_enabled": traits,
                "weather_lag_days": weather_lag_days, "weather_cached_cells": len(lookup.cells),
                "protocol": "LOCAL v2: same cases as v1, previously inspected tests remain diagnostic only. Migration cues + causal within-phase memory; retrospective ERA5 if enabled.",
                "availability_note": "Astronomy is deterministic. Weather valid times precede origin, but publication vintages/operational availability are not established.",
                "unimplemented_drivers": ["direct food/prey abundance", "vegetation green-up", "body condition and hormones", "breeding labels", "social cues", "ocean currents", "terrain/barrier rasters"]}
    dump(dest / "windows.json", manifest)
    return {"windows": len(raw), "drivers": len(names), "anchors": manifest["n_anchors"], "weather_cells": len(lookup.cells), "output": str(dest)}
