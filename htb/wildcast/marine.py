"""Coarse NOAA ERSST context with strictly past monthly lookups."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def fetch(output="data/ecology-sources", maximum_mb=192):
    """Bounded resumable acquisition; separate from the terrain cache manifest."""
    import requests
    from .sources import dump
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    url = "https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.ersst.v5/sst.mnmean.nc"
    path = destination / "sst.mnmean.nc"
    if not path.exists():
        response = requests.head(url, timeout=30)
        response.raise_for_status()
        size = int(response.headers["Content-Length"])
        if size > maximum_mb * 2**20:
            raise ValueError("Marine source exceeds download budget")
        temporary = path.with_suffix(".nc.chunks")
        start = temporary.stat().st_size if temporary.exists() else 0
        if start > size:
            raise ValueError("Partial marine download is longer than its source")
        while start < size:
            end = min(size - 1, start + 4 * 2**20 - 1)
            chunk = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
            chunk.raise_for_status()
            if chunk.status_code != 206 or len(chunk.content) != end - start + 1:
                raise ValueError("Marine provider did not honor the bounded range request")
            with temporary.open("ab") as stream:
                stream.write(chunk.content)
            start = end + 1
        temporary.replace(path)
    report = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "provider": "NOAA ERSST v5 via PSL", "resolution": "2 degree monthly surface temperature",
              "temporal_rule": "Origin calendar month minus two; anomaly uses ten prior same-calendar-month years"}
    dump(destination / "marine-source.json", report)
    return report


def past_months(seconds):
    dates = pd.to_datetime(seconds, unit="s", utc=True)
    return dates.year.to_numpy() * 12 + dates.month.to_numpy() - 1 - 2


def lookup(sources, aux):
    from netCDF4 import Dataset, num2date
    path = Path(sources) / "sst.mnmean.nc"
    with Dataset(path) as ds:
        dates = num2date(ds["time"][:], ds["time"].units)
        months = np.asarray([d.year * 12 + d.month - 1 for d in dates])
        lat, lon = np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:])
        sea = np.asarray(ds["sst"][:].filled(np.nan), np.float32)
    selected = past_months(aux[:, 2])
    times = np.searchsorted(months, selected)
    if np.any(times >= len(months)) or np.any(times < 120) or not np.array_equal(months[times], selected):
        raise ValueError("SST source does not contain the required past month")
    ilat = np.abs(lat[None] - aux[:, :1]).argmin(1)
    delta_lon = (lon[None] - aux[:, 1:2] + 180) % 360 - 180
    ilon = np.abs(delta_lon).argmin(1)
    level = sea[times, ilat, ilon]
    # All ten reference observations precede the selected, already lagged month.
    history = np.stack([sea[times - 12 * year, ilat, ilon] for year in range(1, 11)])
    valid = np.isfinite(history).sum(0)
    normal = np.nansum(history, axis=0) / np.maximum(valid, 1)
    anomaly = np.where(valid >= 8, level - normal, np.nan)
    change = level - sea[times - 1, ilat, ilon]
    north = sea[times, np.minimum(ilat + 1, len(lat) - 1), ilon]
    south = sea[times, np.maximum(ilat - 1, 0), ilon]
    east, west = sea[times, ilat, (ilon + 1) % len(lon)], sea[times, ilat, (ilon - 1) % len(lon)]
    dy = np.abs(lat[np.minimum(ilat + 1, len(lat) - 1)] - lat[np.maximum(ilat - 1, 0)]) * 111.2
    dx = 4 * 111.2 * np.maximum(np.cos(np.deg2rad(lat[ilat])), .01)
    gradient = np.hypot((north - south) / np.maximum(dy, 1), (east - west) / dx)
    values = {"ecology_sst_C": level, "ecology_sst_change_1mo_C": change,
              "ecology_sst_past10year_anomaly_C": anomaly,
              "ecology_sst_gradient_C_per_km": gradient}
    return values, {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source": "NOAA ERSST v5, monthly 2-degree surface temperature",
                    "time_rule": "Calendar month minus two; prior ten same-month years for anomaly, >=8 valid references",
                    "limitations": "Coarse retrospective surface context, not fine ocean fronts, currents or temperature at diving depth"}
