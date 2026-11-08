"""Observed fine/seasonal history features for the repaired movement protocol.

Five- and fifteen-minute observations are actual past fixes with missing masks.
They are never interpolated. Source clock admission is a separate frozen rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import migration_data as legacy
from . import multiscale_data as previous
from .data import project

DAY, UNIT = legacy.DAY, legacy.UNIT
HORIZONS = previous.HORIZONS.copy()
LAGS = np.sort(np.r_[previous.LAGS, 5/1440, 15/1440])
HISTORY_NAMES = list(previous.HISTORY_NAMES)
STATIC_NAMES = [*previous.STATIC_NAMES, "velocity_5min_east_1000km_per_day",
                "velocity_5min_north_1000km_per_day", "velocity_15min_east_1000km_per_day",
                "velocity_15min_north_1000km_per_day"]
MINIMUM_PRIOR_FIXES = previous.MINIMUM_PRIOR_FIXES
MINIMUM_HISTORY_SPAN_DAYS = previous.MINIMUM_HISTORY_SPAN_DAYS
MINIMUM_HISTORY_LAGS = previous.MINIMUM_HISTORY_LAGS
LEGACY_LAG_COLUMNS = np.array([int(np.flatnonzero(LAGS == lag)[0]) for lag in legacy.LAGS])
PREVIOUS_LAG_COLUMNS = np.array([int(np.flatnonzero(LAGS == lag)[0]) for lag in previous.LAGS])


def past_indices(times, origin):
    wanted = times[origin]-LAGS*DAY
    index = np.searchsorted(times,wanted,side="right")-1
    safe = np.maximum(index,0)
    tolerance = np.maximum(.25,.2*LAGS)
    short = (LAGS > 0)&(LAGS < .25)
    tolerance[short] = np.minimum(1/24,.25*LAGS[short])
    mask = (index >= 0)&((wanted-times[safe])/DAY <= tolerance)
    return safe,mask


def features(lat, lon, times, dates, origin, day_values=None):
    """Return history29x10/mask29/static22/observed-coordinates29x2/times29."""
    ix,mask = past_indices(times,origin)
    xy = project(lat[ix],lon[ix],lat[origin],lon[origin])/UNIT
    day_values = dates.dayofyear.to_numpy() if day_values is None else day_values
    season = 2*np.pi*day_values[ix]/365.25
    history = np.column_stack([xy,(times[origin]-times[ix])/DAY,np.sin(season),np.cos(season),
                               lat[ix]/90,np.sin(np.deg2rad(lon[ix])),np.cos(np.deg2rad(lon[ix])),LAGS,mask])
    history[~mask] = 0
    xyz = [np.cos(np.deg2rad(lat[origin]))*np.cos(np.deg2rad(lon[origin])),
           np.cos(np.deg2rad(lat[origin]))*np.sin(np.deg2rad(lon[origin])),np.sin(np.deg2rad(lat[origin]))]
    season0 = 2*np.pi*day_values[origin]/365.25
    solar = 2*np.pi*((times[origin]/3600+lon[origin]/15)%24)/24
    def velocity(lag):
        j = int(np.flatnonzero(LAGS == lag)[0])
        if not mask[j]:return [0.,0.]
        elapsed = (times[origin]-times[ix[j]])/DAY
        if elapsed <= 0:raise ValueError("Past velocity requires a strictly earlier actual observation")
        return (-xy[j]/elapsed).tolist()
    static = [*xyz,np.sin(season0),np.cos(season0),np.sin(solar),np.cos(solar),
              *velocity(1.),*velocity(7.),np.log1p((times[origin]-times[0])/DAY),
              float(mask[-1]),legacy.daylength(lat[origin],day_values[origin]),
              *velocity(1/24),*velocity(3/24),*velocity(5/1440),*velocity(15/1440)]
    ll = np.column_stack([lat[ix],lon[ix]]);ll[~mask] = 0
    ht = times[ix].copy();ht[~mask] = 0
    return history.astype("float32"),mask,np.asarray(static,"float32"),ll,ht


def query_features(lat, lon, times, dates, origin, day_values=None):
    """Parse only the observed prefix on the explicitly accepted clock axis."""
    if isinstance(origin,(bool,np.bool_)) or not isinstance(origin,(int,np.integer)) or origin < 0:
        raise ValueError("A valid observed origin index is required")
    count = int(origin)+1
    if any(len(value) < count for value in (lat,lon,times,dates)):
        raise ValueError("Query origin exceeds the supplied history")
    lat=np.asarray(lat[:count],dtype=float);lon=np.asarray(lon[:count],dtype=float)
    times=np.asarray(times[:count],dtype=float);dates=pd.DatetimeIndex(dates[:count])
    if dates.tz is None:raise ValueError("Query history requires an explicitly declared clock/timezone axis")
    dates=dates.tz_convert("UTC").as_unit("ns")
    if (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(lat).all() or not np.isfinite(lon).all()
            or not np.all((-90 <= lat)&(lat <= 90)) or not np.all((-180 <= lon)&(lon <= 180))
            or not np.allclose(times,dates.asi8/1e9,rtol=0,atol=1e-6)):
        raise ValueError("Invalid GPS coordinates or inconsistent ordered observation clock")
    days=None if day_values is None else np.asarray(day_values[:count],dtype=float)
    if days is not None and (len(days) != count or not np.isfinite(days).all()):
        raise ValueError("Observed day-of-year values must be complete and finite")
    return features(lat,lon,times,dates,int(origin),days)
