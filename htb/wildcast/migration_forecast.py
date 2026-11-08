"""Prospective forecasts from observed history; never exports future truth."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import unproject
from .migration_data import features
from .migration_model import load_model
from .sources import dump


def forecast(checkpoint, observations, output, species, animal=None, issue_time=None,
             samples=64, seed=2026, device="cpu"):
    """Issue an individual forecast using a CSV or parquet of actual GPS fixes.

    Input columns are timestamp,lat,lon and optional animal. Observations later
    than issue_time are excluded before constructing any predictive feature.
    """
    if samples < 2: raise ValueError("At least two trajectory samples required")
    checkpoint, observations = Path(checkpoint), Path(observations)
    f = pd.read_parquet(observations) if observations.suffix == ".parquet" else pd.read_csv(observations)
    if not {"timestamp", "lat", "lon"}.issubset(f): raise ValueError("Require timestamp, lat and lon columns")
    if animal is not None:
        if "animal" not in f: raise ValueError("The supplied observations have no animal identity column")
        f = f.loc[f.animal.astype(str) == str(animal)]
    elif "animal" in f and f.animal.nunique() > 1:
        raise ValueError("Select one individual with --animal")
    f = f.copy()
    f["timestamp"] = pd.to_datetime(f.timestamp, utc=True, errors="coerce")
    valid = f.timestamp.notna() & f.lat.between(-90, 90) & f.lon.between(-180, 180)
    f = f.loc[valid].sort_values("timestamp").drop_duplicates("timestamp")
    if issue_time is not None:
        cutoff = pd.Timestamp(issue_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        f = f.loc[f.timestamp <= cutoff]
    if f.empty: raise ValueError("No valid observed history at/before issue time")
    dates = pd.DatetimeIndex(f.timestamp)
    times = dates.asi8 / 1e9
    lat, lon = f.lat.to_numpy(), f.lon.to_numpy()
    history, mask, static, _, _ = features(lat, lon, times, dates, len(f) - 1)
    model, ckpt = load_model(checkpoint, device)
    index = ckpt["manifest"]["species"].get(species, 0)
    tensors = [torch.as_tensor(x, device=device)[None] for x in [history, mask, static]]
    tensors.append(torch.tensor([index], dtype=torch.long, device=device))
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        paths = model.sample_paths(*tensors, samples=samples, generator=generator).cpu().numpy()[0]
    paths_ll = unproject(paths * 1_000_000, lat[-1], lon[-1])
    horizons = ckpt["config"]["horizons"]
    result = {"scope": "Forecast using observed history only; no future observations",
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_manifest_sha256": ckpt["manifest_sha256"], "species": species, "species_index": index,
        "animal": animal, "origin": {"timestamp": dates[-1].isoformat(), "lat": float(lat[-1]), "lon": float(lon[-1])},
        "requested_issue_time": issue_time, "horizons_days": horizons,
        "forecast_timestamps": [(dates[-1] + pd.Timedelta(days=h)).isoformat() for h in horizons],
        "observed_history": [{"timestamp": row.timestamp.isoformat(), "lat": float(row.lat), "lon": float(row.lon)}
                             for row in f.tail(512).itertuples()],
        "sampled_latlon_paths": paths_ll.tolist(), "samples": samples, "seed": seed,
        "notes": ["Each sampled route shares a mixture component and correlated noise across its horizons.",
                  "Horizon times are relative to the last actual observed fix, which may precede requested issue time.",
                  "Intermediate curves are predictions, not observed GPS or verified fine-scale motion.",
                  "This initial model uses geography, season and past movement; it has no future weather, hard barriers or observed social state."]}
    dump(output, result)
    return {"output": str(output), "origin": result["origin"], "samples": samples, "horizons_days": horizons}
