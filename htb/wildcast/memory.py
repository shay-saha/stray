"""A sparse, causal multi-scale history stream from hours to one year."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .data import project
from .sources import dump

LAGS_HOURS = np.asarray([0, 1, 2, 4, 6, 8, 12, 18, 24, 36, 48, 72, 96, 120, 168, 240, 336,
                         504, 720, 1008, 1440, 2160, 2880, 4320, 8760.])
MEMORY_NAMES = ["present", "log_east", "log_north", "log_age_hours", "log_lag_mismatch_hours",
                "sin_season", "cos_season", "sin_solar_hour", "cos_solar_hour", "log_requested_lag_hours"]


def sparse_memory(lat, lon, timestamps, origins, scales):
    """Arrays must contain only one animal/segment/split phase, sorted by time."""
    now = timestamps[origins]
    requested = now[:, None] - LAGS_HOURS[None] * 3600
    indices = np.searchsorted(timestamps, requested, side="right") - 1
    valid = indices >= 0
    indices = indices.clip(0)
    # Only older observations are allowed; never nearest-neighbor into the future.
    age = (now[:, None] - timestamps[indices]) / 3600
    valid &= age <= LAGS_HOURS[None] + np.maximum(24, LAGS_HOURS[None] * .2)
    xy = project(lat[indices], lon[indices], lat[origins, None], lon[origins, None]) / scales[:, None, None]
    dates = pd.to_datetime(timestamps, unit="s", utc=True)
    season = 2 * np.pi * dates.dayofyear.to_numpy() / 365.2425
    solar = 2 * np.pi * ((timestamps / 3600 + lon / 15) % 24) / 24
    m = np.stack([valid, np.sign(xy[..., 0]) * np.log1p(np.abs(xy[..., 0])),
                  np.sign(xy[..., 1]) * np.log1p(np.abs(xy[..., 1])), np.log1p(age),
                  np.log1p(np.abs(age - LAGS_HOURS[None])), np.sin(season[indices]), np.cos(season[indices]),
                  np.sin(solar[indices]), np.cos(solar[indices]),
                  np.broadcast_to(np.log1p(LAGS_HOURS), age.shape)], axis=-1).astype(np.float32)
    m[~valid] = 0
    return m


def add_memory(source_root, root):
    source_root, root = Path(source_root), Path(root)
    if source_root.resolve() == root.resolve():
        raise ValueError("Use a new data root for a new history representation")
    src, dest = source_root / "processed", root / "processed"
    dest.mkdir(parents=True, exist_ok=True)
    a = dict(np.load(src / "windows.npz")); manifest = json.loads((src / "windows.json").read_text())
    f = pd.read_parquet(src / "observations.parquet")
    f["phase"] = "train"
    for animal, ids in f.groupby("animal", sort=False).groups.items():
        ids = np.asarray(list(ids)); assignment = manifest["animal_assignments"][animal]
        if assignment == "temporal":
            f.loc[ids[int(len(ids) * .65):int(len(ids) * .8)], "phase"] = "val_temporal"
            f.loc[ids[int(len(ids) * .8):], "phase"] = "test_temporal"
        else:
            f.loc[ids, "phase"] = assignment
    orig = a["aux"][:, 6].astype(int)
    unique, first, inverse = np.unique(orig, return_index=True, return_inverse=True)
    cache = np.zeros((len(unique), len(LAGS_HOURS), len(MEMORY_NAMES)), dtype=np.float32)
    origin_scales = a["aux"][first, 8]
    for _, g in f.groupby(["animal", "segment", "phase"], sort=False):
        ids = g.index.to_numpy()
        selected = np.flatnonzero(np.isin(ids, unique))
        if not len(selected):
            continue
        rows = np.searchsorted(unique, ids[selected])
        t = g.timestamp.astype("int64").to_numpy() / 1e9
        lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
        for start in range(0, len(selected), 2048):
            chunk = selected[start:start + 2048]; target = rows[start:start + 2048]
            cache[target] = sparse_memory(lat, lon, t, chunk, origin_scales[target])
    assert np.all(cache[:, 0, 0] == 1), "Each memory stream must have an observed origin"
    a["memory"] = cache[inverse]
    np.savez_compressed(dest / "windows.npz", **a)
    for name in ["observations.parquet", "quality-report.json", "trait-report.json"]:
        if (src / name).exists():
            shutil.copyfile(src / name, dest / name)
    result = {**manifest, "parent_windows_sha256": manifest["windows_sha256"],
              "windows_sha256": hashlib.sha256((dest / "windows.npz").read_bytes()).hexdigest(),
              "memory_lags_hours": LAGS_HOURS.tolist(), "memory_names": MEMORY_NAMES,
              "memory_note": "Past-only samples, within the same animal, segment and split phase. Missing history is masked. A year of memory exists only when observed history supports it."}
    dump(dest / "windows.json", result)
    return {"windows": len(orig), "memory_shape": list(a["memory"].shape), "output": str(dest)}
