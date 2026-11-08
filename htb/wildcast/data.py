"""Conservative QC, grouped splits, and causal irregular-time forecast windows."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .sources import dump

R = 6_371_008.8
ALIASES = {
    "timestamp": ["timestamp", "date_time", "datetime"],
    "lat": ["location-lat", "location.lat", "latitude", "lat"],
    "lon": ["location-long", "location.long", "longitude", "lon"],
    "animal": ["individual-local-identifier", "individual.local.identifier (ID)", "individual.local.identifier", "tag-local-identifier", "tag.local.identifier"],
    "species": ["individual-taxon-canonical-name", "individual.taxon.canonical.name", "species"],
    "study": ["study-name", "study.name"],
}


def project(lat, lon, lat0, lon0):
    """Spherical azimuthal equidistant map, in metres east/north at the origin."""
    lat, lon, lat0, lon0 = np.broadcast_arrays(*[np.deg2rad(x) for x in (lat, lon, lat0, lon0)])
    dl = (lon - lon0 + np.pi) % (2 * np.pi) - np.pi
    east = np.cos(lat) * np.sin(dl)
    north = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dl)
    dot = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dl)
    c = np.arctan2(np.hypot(east, north), np.clip(dot, -1, 1))
    factor = R * np.divide(c, np.sin(c), out=np.ones_like(c), where=np.abs(c) > 1e-10)
    return np.stack([factor * east, factor * north], axis=-1)


def unproject(xy, lat0, lon0):
    xy = np.asarray(xy)
    lat0, lon0 = np.deg2rad(lat0), np.deg2rad(lon0)
    x, y = xy[..., 0], xy[..., 1]
    rho = np.hypot(x, y)
    c = rho / R
    ratio = np.divide(np.sin(c), rho, out=np.full_like(rho, 1 / R), where=rho > 0)
    lat = np.arcsin(np.clip(np.cos(c) * np.sin(lat0) + y * ratio * np.cos(lat0), -1, 1))
    lon = lon0 + np.arctan2(x * np.sin(c), rho * np.cos(lat0) * np.cos(c) - y * np.sin(lat0) * np.sin(c))
    lon = np.where(rho == 0, lon0, lon)
    return np.stack([np.rad2deg(lat), (np.rad2deg(lon) + 180) % 360 - 180], axis=-1)


def distance(lat, lon, lat0, lon0):
    return np.linalg.norm(project(lat, lon, lat0, lon0), axis=-1)


def read_events(path, source, preserve_identifiers=False):
    identifier_columns = [*ALIASES["animal"], *ALIASES["species"], *ALIASES["study"]]
    dtype = {name: str for name in identifier_columns} if preserve_identifiers else None
    frame = pd.read_csv(path, low_memory=False, dtype=dtype)
    columns = {}
    for key, aliases in ALIASES.items():
        match = next((a for a in aliases if a in frame), None)
        if match:
            columns[key] = frame[match]
    if not {"timestamp", "lat", "lon", "animal", "species"} <= columns.keys():
        return None
    if "sensor-type" in frame:
        frame_mask = frame["sensor-type"].str.lower().eq("gps")
    else:
        frame_mask = np.ones(len(frame), dtype=bool)
    if "visible" in frame:
        frame_mask &= frame["visible"].astype(str).str.lower().isin(["true", "1"])
    for flag in ["manually-marked-outlier", "algorithm-marked-outlier"]:
        if flag in frame:
            frame_mask &= ~frame[flag].astype(str).str.lower().isin(["true", "1"])
    out = pd.DataFrame(columns).loc[frame_mask].copy()
    out["source"] = source
    if "study" not in out:
        out["study"] = source
    out["study"] = source + ":" + out["study"].astype(str)
    out["animal"] = out["study"].astype(str) + ":" + out["species"].astype(str) + ":" + out["animal"].astype(str)
    fmt = "%m/%d/%y %H:%M" if source == "jaguar-morato-2018" else "mixed"
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce", format=fmt)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    return out


def clean(root, min_points=200, max_gap_hours=48, max_speed_mps=100):
    root = Path(root)
    frames, report = [], {"files": [], "filters": {}}
    dirs = sorted((root / "raw").iterdir(), key=lambda p: (p.name != "jaguar-morato-2018", p.name))
    for directory in dirs:
        if not (directory / "source.json").exists():
            continue
        source_meta = json.loads((directory / "source.json").read_text())
        if "metadata" in source_meta:
            rights = source_meta["metadata"].get("dc.rights.uri", [])
            allowed = ["/publicdomain/zero/1.0", "/licenses/by/4.0", "/licenses/by/3.0"]
            if not rights or not all(any(x in u.lower() for x in allowed) for u in rights):
                report["files"].append({"path": str(directory), "status": "quarantined: unresolved or unsupported rights"})
                continue
        for path in sorted(directory.glob("*.csv")):
            try:
                f = read_events(path, directory.name)
            except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as e:
                report["files"].append({"path": str(path), "error": str(e)})
                continue
            if f is not None:
                frames.append(f)
                report["files"].append({"path": str(path), "accepted_sensor_rows": len(f)})
            else:
                report["files"].append({"path": str(path), "status": "skipped: unsupported schema or reference table"})
    if not frames:
        raise ValueError("No supported GPS event tables. Run fetch/harvest first.")
    f = pd.concat(frames, ignore_index=True)
    report["input_sensor_rows"] = len(f)
    valid = f.timestamp.notna() & f.lat.between(-90, 90) & f.lon.between(-180, 180)
    valid &= f.species.notna() & ~f.animal.str.endswith(":nan")
    report["filters"]["invalid_fields"] = int((~valid).sum())
    f = f.loc[valid].copy()

    # Link aliases conservatively using >=3 exactly shared species/time/GPS fixes.
    # This prevents the same collared animal appearing in different splits via
    # independent papers or re-archived studies with different local identifiers.
    parent = {a: a for a in f.animal.unique()}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    keys = ["species", "timestamp", "lat", "lon"]
    dup = f.loc[f.duplicated(keys, keep=False)]
    counts = Counter()
    for _, g in dup.groupby(keys, sort=False):
        for a, b in itertools.combinations(sorted(g.animal.unique()), 2):
            counts[a, b] += 1
    aliases = []
    for (a, b), n in counts.items():
        if n >= 3 and find(a) != find(b):
            parent[find(b)] = find(a)
            aliases.append({"a": a, "b": b, "shared_fixes": n})
    f["animal"] = f.animal.map(find)
    report["alias_links"] = aliases
    before = len(f)
    # Keep canonical identity/source metadata consistently after merging aliases.
    canonical = f.drop_duplicates("animal").set_index("animal")[["study", "source", "species"]]
    for col in canonical:
        f[col] = f.animal.map(canonical[col])
    f = f.sort_values(["animal", "timestamp"], kind="stable").drop_duplicates(["animal", "timestamp"])
    report["filters"]["duplicate_animal_timestamp"] = before - len(f)
    # Thin to the first OBSERVED fix in each UTC hour; never interpolate a target.
    before = len(f)
    f["hour"] = f.timestamp.dt.floor("h")
    f = f.drop_duplicates(["animal", "hour"]).drop(columns="hour")
    report["filters"]["hourly_thinning"] = before - len(f)
    prev = f.groupby("animal")[["lat", "lon", "timestamp"]].shift()
    dt = (f.timestamp - prev.timestamp).dt.total_seconds().to_numpy()
    speed = distance(f.lat.to_numpy(), f.lon.to_numpy(), prev.lat.to_numpy(), prev.lon.to_numpy()) / dt
    breaks = ~np.isfinite(dt) | (dt > max_gap_hours * 3600) | (speed > max_speed_mps)
    f["segment"] = pd.Series(breaks, index=f.index).groupby(f.animal).cumsum().astype(int)
    report["speed_breaks"] = int((speed > max_speed_mps).sum())
    report["gap_breaks"] = int((dt > max_gap_hours * 3600).sum())
    before = len(f)
    f = f.loc[f.groupby("animal").animal.transform("size") >= min_points].copy()
    report["filters"]["short_individuals"] = before - len(f)
    f = f.reset_index(drop=True)
    report.update({"output_rows": len(f), "animals": f.animal.nunique(), "species": sorted(f.species.unique()),
                   "studies": f.study.nunique(), "config": {"min_points": min_points, "max_gap_hours": max_gap_hours,
                   "max_speed_mps": max_speed_mps},
                   "note": "Speed threshold is a permissive cross-taxon break rule, not proof of biological validity. Manual track review remains required."})
    (root / "processed").mkdir(parents=True, exist_ok=True)
    f.to_parquet(root / "processed" / "observations.parquet", index=False)
    dump(root / "processed" / "quality-report.json", report)
    return report


SPLITS = {"train": 0, "val_temporal": 1, "test_temporal": 2, "val_individual": 3, "test_individual": 4}


def animal_assignments(f, seed):
    """Deterministic per-study/species holdouts, with no training animal IDs as inputs."""
    result = {}
    for _, g in f.groupby(["study", "species"]):
        animals = sorted(g.animal.unique(), key=lambda a: hashlib.sha256(f"{seed}:{a}".encode()).hexdigest())
        n = len(animals)
        held = max(1, round(n * 0.15)) if n >= 3 else 0
        for i, a in enumerate(animals):
            result[a] = "test_individual" if i < held else "val_individual" if i < 2 * held else "temporal"
    return result


def prepare(root, context=32, stride=4, horizons=(1, 12, 24, 168), seed=42):
    if context < 2 or stride < 1 or not horizons or any(h <= 0 for h in horizons):
        raise ValueError("Require context >=2, stride >=1, and positive forecast horizons")
    root = Path(root)
    f = pd.read_parquet(root / "processed" / "observations.parquet")
    assignment = animal_assignments(f, seed)
    # Chronological 65/15/20 split per training individual. Purge whole contexts
    # and targets at the boundaries so validation never overlaps training fixes.
    f["phase"] = "train"
    for animal, ix in f.groupby("animal", sort=False).groups.items():
        ix = np.asarray(list(ix))
        if assignment[animal] == "temporal":
            f.loc[ix[int(len(ix) * .65):int(len(ix) * .8)], "phase"] = "val_temporal"
            f.loc[ix[int(len(ix) * .8):], "phase"] = "test_temporal"
        else:
            f.loc[ix, "phase"] = assignment[animal]
    train = f.loc[f.phase == "train"]
    if train.empty:
        raise ValueError("No training data after grouped split")
    species = {s: i + 1 for i, s in enumerate(sorted(train.species.unique()))}
    studies = {s: i for i, s in enumerate(sorted(f.study.unique()))}
    animals = {s: i for i, s in enumerate(sorted(f.animal.unique()))}
    # Fixed metric units avoid study/test-dependent scale estimates entirely.
    # Scale each forecast using observed context only. This transfers movement
    # dynamics between taxa without fitting any statistics to held-out animals.
    xs, ys, meta, aux = [], [], [], []
    for (_, segment), g in f.groupby(["animal", "segment"], sort=False):
        if len(g) < context + 1:
            continue
        lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
        ts = g.timestamp.astype("int64").to_numpy() / 1e9
        phase = g.phase.to_numpy()
        solar = 2 * np.pi * ((ts / 3600 + lon / 15) % 24) / 24
        season = 2 * np.pi * g.timestamp.dt.dayofyear.to_numpy() / 365.25
        for i in range(context - 1, len(g) - 1, stride):
            start = i - context + 1
            if phase[start] != phase[i]:
                continue
            origins = np.arange(start, i + 1)
            raw_xy = project(lat[origins], lon[origins], lat[i], lon[i])
            raw_steps = np.diff(raw_xy, axis=0)
            gaps = np.diff(ts[origins]) / 3600
            hourly_steps = np.linalg.norm(raw_steps, axis=-1) / np.sqrt(np.maximum(gaps, .01))
            clipped = np.minimum(hourly_steps, np.quantile(hourly_steps, .9))
            scale = float(np.clip(np.sqrt(np.mean(clipped ** 2)), 25., 100_000.))
            xy = raw_xy / scale
            steps = np.diff(xy, axis=0, prepend=xy[:1])
            dt = np.diff(ts[origins], prepend=ts[start]) / 3600
            features = np.column_stack([np.sign(xy) * np.log1p(np.abs(xy)),
                np.sign(steps) * np.log1p(np.abs(steps)), np.log1p(dt), np.log1p((ts[i] - ts[origins]) / 3600),
                np.sin(solar[origins]), np.cos(solar[origins]), np.sin(season[origins]), np.cos(season[origins])])
            last_dt = (ts[i] - ts[i - 1]) / 3600
            velocity = (xy[-1] - xy[-2]) / max(last_dt, 1e-3)
            for h in horizons:
                pos = np.searchsorted(ts, ts[i] + h * 3600)
                choices = [j for j in (pos - 1, pos) if i < j < len(g)]
                if not choices:
                    continue
                j = min(choices, key=lambda j: abs(ts[j] - ts[i] - h * 3600))
                # Match observations only; never fill gaps or interpolate ground truth.
                if abs(ts[j] - ts[i] - h * 3600) > min(3600, h * 3600 * .25):
                    continue
                if phase[j] != phase[i]:
                    continue
                target = project(lat[j], lon[j], lat[i], lon[i]) / scale
                xs.append(features.astype(np.float32)); ys.append(target.astype(np.float32))
                meta.append([SPLITS[phase[i]], species.get(g.species.iloc[0], 0), studies[g.study.iloc[0]], animals[g.animal.iloc[0]], h])
                aux.append([lat[i], lon[i], ts[i], ts[j], *velocity, float(g.index[i]), float(g.index[j]), scale])
    if not xs:
        raise ValueError("No matched forecast windows; reduce context or inspect sampling cadence.")
    output = root / "processed" / "windows.npz"
    np.savez_compressed(output, x=np.stack(xs), y=np.stack(ys), meta=np.asarray(meta, dtype=np.int32), aux=np.asarray(aux))
    report = {"context": context, "stride": stride, "horizons_hours": list(horizons), "seed": seed,
              "scale_mode": "causal context winsorized RMS hourly step, clipped to 25..100000 metres", "features": ["log_relative_east", "log_relative_north", "log_step_east", "log_step_north",
              "log_interval_hours", "log_age_hours", "sin_solar_hour", "cos_solar_hour", "sin_dayofyear", "cos_dayofyear"],
              "species": species, "studies": studies, "animals": animals, "animal_assignments": assignment,
              "splits": {s: int((np.asarray(meta)[:, 0] == v).sum()) for s, v in SPLITS.items()},
              "observations_sha256": hashlib.sha256((root / "processed" / "observations.parquet").read_bytes()).hexdigest(),
              "windows_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "protocol": "LOCAL v1. Not an official MoveBench split or evaluation. No environmental covariates yet."}
    dump(root / "processed" / "windows.json", report)
    return report
