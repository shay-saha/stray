"""Sealed, parent-pinned observed movement cases from one hour to ninety days.

This is an explicit new protocol.  It does not relax migration_data.prepare's
legacy ancestry checks, create interpolated labels, fit normalizers, or expose
reserved truth.  Short-horizon breadth and long-horizon support are separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from functools import lru_cache

import numpy as np
import pandas as pd

from . import migration_data as legacy
from .data import distance, project
from .migration_clock import check_compatibility
from .migration_clock import GPS_OFFSETS, STANDARDS
from .parent_clock import validate_evidence as validate_parent_clock_evidence


PROTOCOL = "wildcast-multiscale-observed-v1"
DAY, UNIT, SPLITS = legacy.DAY, legacy.UNIT, legacy.SPLITS
HORIZONS = np.array([1/24, .5, 1, 3, 7, 14, 30, 60, 90.])
LAGS = np.sort(np.r_[legacy.LAGS, 1/24, 3/24])
MINIMUM_PRIOR_FIXES = 8
MINIMUM_HISTORY_SPAN_DAYS = .25
MINIMUM_HISTORY_LAGS = 2
SHORT_TRACK_SPAN_DAYS = 14.
SHORT_STRIDE_DAYS = .25
LONG_STRIDE_DAYS = 1.
_CLOCK_UTC_EDGES=pd.to_datetime([d for d,_ in GPS_OFFSETS],utc=True).asi8/1e9
_CLOCK_OFFSETS=np.asarray([offset for _,offset in GPS_OFFSETS],dtype=float)
_CLOCK_GPS_EDGES=_CLOCK_UTC_EDGES+_CLOCK_OFFSETS
_CLOCK_VALID_BEFORE=pd.Timestamp(STANDARDS["valid_before"]).timestamp()
HISTORY_NAMES = list(legacy.HISTORY_NAMES)
STATIC_NAMES = [*legacy.STATIC_NAMES, "velocity_1h_east_1000km_per_day",
                "velocity_1h_north_1000km_per_day", "velocity_3h_east_1000km_per_day",
                "velocity_3h_north_1000km_per_day"]
LEGACY_LAG_COLUMNS = np.array([int(np.flatnonzero(LAGS == lag)[0]) for lag in legacy.LAGS])


def target_tolerances(horizons):
    """Public target matching tolerances in DAYS; no timestamps are rounded."""
    horizons=np.asarray(horizons,dtype=float)
    if (horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all() or
            np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0)):
        raise ValueError("Strictly increasing positive finite horizons required")
    return np.where(horizons < 1,np.minimum(1/24,.25*horizons),
                    np.minimum(2.,np.maximum(.125,.05*horizons)))


@lru_cache(maxsize=16)
def _clock_cutoffs(values):
    return tuple(pd.Timestamp(v).timestamp() for v in values)


def target_admission_mask(horizons, canonical_animal, manifest, *, origin_time=None, target_times=None):
    """Pure label-admission rule, using inline frozen declarations only.

    No paths are read.  Without times, return the canonical identity's horizon
    permission.  Unix-second origin/selected-target times additionally enforce
    bounded UTC/GPS interpretation and pinned split cutoffs for subday labels.
    Instrument clock accuracy is not inferred from these standard-only bounds.
    """
    horizons=np.asarray(horizons,dtype=float);target_tolerances(horizons)
    admitted=np.ones(len(horizons),dtype=bool);short=horizons < 1
    mode=manifest.get("subday_clock_modes",{}).get(str(canonical_animal))
    if (str(canonical_animal) in manifest.get("subday_clock_excluded_animals",[]) or
            mode not in {"declared_utc","bounded_utc_or_gps"}):
        admitted[short]=False
        return admitted
    if origin_time is not None:
        if np.asarray(origin_time).ndim != 0:raise ValueError("One observed origin time required")
        if not np.isfinite(origin_time) or origin_time <= 0:admitted[short]=False
    if target_times is not None:
        target_times=np.asarray(target_times,dtype=float)
        if target_times.shape != horizons.shape:raise ValueError("Selected target times must match horizons")
        good=np.isfinite(target_times)&(target_times > 0)
        if origin_time is not None:good &= target_times > origin_time
        admitted[short] &= good[short]
    if mode == "declared_utc":return admitted
    boundaries=_clock_cutoffs(tuple(manifest.get("cutoff_boundaries",[])))
    def bounded_ok(values):
        values=np.atleast_1d(np.asarray(values,dtype=float))
        finite=np.isfinite(values)
        index=np.searchsorted(_CLOCK_GPS_EDGES,values,side="right")-1
        safe=np.clip(index,0,len(_CLOCK_OFFSETS)-1)
        lower=values-_CLOCK_OFFSETS[safe]
        valid=finite&(index >= 0)&(values < _CLOCK_VALID_BEFORE)
        next_index=np.minimum(safe+1,len(_CLOCK_OFFSETS)-1)
        valid &= ~((safe < len(_CLOCK_OFFSETS)-1)&(lower >= _CLOCK_UTC_EDGES[next_index]))
        for edge in boundaries:
            valid &= ~((lower < edge)&(values >= edge))
        return valid
    if origin_time is not None:
        if not bounded_ok(origin_time)[0]:admitted[short]=False
    if target_times is not None:
        admitted[short] &= bounded_ok(target_times)[short]
    return admitted


def match_targets(times, origin, horizons=HORIZONS):
    """Nearest actual future fix, with stricter predeclared subday tolerance."""
    times, horizons = np.asarray(times, dtype=float), np.asarray(horizons, dtype=float)
    if (times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0) or
            not np.isfinite(times).all() or not 0 <= origin < len(times)):
        raise ValueError("Ordered finite actual times and a valid origin required")
    tolerance=target_tolerances(horizons)
    wanted = times[origin] + horizons * DAY
    right = np.clip(np.searchsorted(times, wanted), 0, len(times)-1)
    left = np.maximum(right-1, 0)
    index = np.where(abs(times[left]-wanted) < abs(times[right]-wanted), left, right)
    mask = (index > origin) & (abs(times[index]-wanted) <= tolerance*DAY)
    return index, mask


def past_indices(times, origin):
    wanted = times[origin] - LAGS*DAY
    index = np.searchsorted(times, wanted, side="right")-1
    safe = np.maximum(index, 0)
    # Existing 25 slots retain their original tolerances exactly.  New 1h/3h
    # slots cannot silently substitute a measurement many hours older.
    tolerance = np.maximum(.25, .2*LAGS)
    short = (LAGS > 0) & (LAGS < .25)
    tolerance[short] = np.minimum(1/24, .25*LAGS[short])
    mask = (index >= 0) & ((wanted-times[safe])/DAY <= tolerance)
    return safe, mask


def features(lat, lon, times, dates, origin, day_values=None):
    ix, mask = past_indices(times, origin)
    xy = project(lat[ix], lon[ix], lat[origin], lon[origin])/UNIT
    day_values = dates.dayofyear.to_numpy() if day_values is None else day_values
    season = 2*np.pi*day_values[ix]/365.25
    history = np.column_stack([xy, (times[origin]-times[ix])/DAY, np.sin(season), np.cos(season),
                               lat[ix]/90, np.sin(np.deg2rad(lon[ix])), np.cos(np.deg2rad(lon[ix])), LAGS, mask])
    history[~mask] = 0
    xyz = [np.cos(np.deg2rad(lat[origin]))*np.cos(np.deg2rad(lon[origin])),
           np.cos(np.deg2rad(lat[origin]))*np.sin(np.deg2rad(lon[origin])), np.sin(np.deg2rad(lat[origin]))]
    season0 = 2*np.pi*day_values[origin]/365.25
    solar = 2*np.pi*((times[origin]/3600+lon[origin]/15) % 24)/24
    def velocity(lag):
        j = int(np.flatnonzero(LAGS == lag)[0])
        return (-xy[j]/max((times[origin]-times[ix[j]])/DAY, .01)).tolist() if mask[j] else [0., 0.]
    static = [*xyz, np.sin(season0), np.cos(season0), np.sin(solar), np.cos(solar),
              *velocity(1.), *velocity(7.), np.log1p((times[origin]-times[0])/DAY),
              float(mask[-1]), legacy.daylength(lat[origin], day_values[origin]),
              *velocity(1/24), *velocity(3/24)]
    ll = np.column_stack([lat[ix], lon[ix]]);ll[~mask] = 0
    ht = times[ix].copy();ht[~mask] = 0
    return history.astype("float32"), mask, np.asarray(static, "float32"), ll, ht


def query_features(lat, lon, times, dates, origin, day_values=None):
    """Public inference builder; same five-array result as legacy.features.

    Signature: query_features(lat, lon, times, dates, origin, day_values=None).
    ``times`` are Unix seconds on the explicitly accepted clock axis, ``dates``
    are timezone-aware corresponding instants, and origin is an observed index.
    Returns (history[27,10], mask[27], static[18], history_latlon[27,2],
    history_time[27]).  Only the prefix through origin is parsed or validated;
    appended malformed/future values cannot change the result.  Sparse queries
    can be represented, but validated training support additionally requires the
    published minimum-fix/span/lag constants.
    """
    if not isinstance(origin, (int, np.integer)) or origin < 0:
        raise ValueError("A valid observed origin index is required")
    count = int(origin)+1
    if any(len(value) < count for value in (lat,lon,times,dates)):
        raise ValueError("Query origin exceeds the supplied observation history")
    lat=np.asarray(lat[:count],dtype=float);lon=np.asarray(lon[:count],dtype=float)
    times=np.asarray(times[:count],dtype=float);dates=pd.DatetimeIndex(dates[:count])
    if dates.tz is None:
        raise ValueError("Query dates require an explicitly declared timezone/clock axis")
    dates=dates.tz_convert("UTC").as_unit("ns")
    if (not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
            not np.isfinite(lat).all() or not np.isfinite(lon).all() or
            not np.all((-90 <= lat)&(lat <= 90)) or not np.all((-180 <= lon)&(lon <= 180)) or
            not np.allclose(times,dates.asi8/1e9,rtol=0,atol=1e-6)):
        raise ValueError("Query history has invalid coordinates or inconsistent ordered clock values")
    days=None if day_values is None else np.asarray(day_values[:count],dtype=float)
    return features(lat,lon,times,dates,int(origin),days)


def _load_parent(path, train_end_year, validation_year):
    meta = json.loads((Path(path)/"manifest.json").read_text())
    # Reuse the original strict verifier with the parent's own protocol.  The
    # explicit child protocol, not a relaxed legacy guard, authorizes extension.
    parent = legacy.load_parent_cache(path, train_end_year, validation_year,
                                      meta.get("horizons_days", []), meta.get("stride_days", -1))
    previous = np.asarray(meta["horizons_days"], dtype=float)
    if (not len(previous) or np.any(previous < 1) or
            any(not np.any(HORIZONS == value) for value in previous)):
        raise ValueError("Multiscale ancestry requires preserved parent day-scale horizons")
    if meta.get("unit_metres") != UNIT:
        raise ValueError("Parent units differ from the explicit multiscale protocol")
    return parent


def _input_paths(roots, parent, train_end_year, validation_year):
    parent_hashes = {r["sha256"] for r in parent["manifest"].get("sources", [])}
    if not parent_hashes:
        raise ValueError("Parent lacks source observation checksum evidence")
    paths, evidence, seen = [], [], set()
    for root in roots:
        path = Path(root)
        if path.is_dir():path = path/"processed/observations.parquet"
        digest = legacy.sha256(path)
        if digest in seen:
            raise ValueError("Duplicate source observation content")
        seen.add(digest)
        record = {"path": str(path), "sha256": digest}
        if digest in parent_hashes:
            record["clock_provenance"] = "unchanged exact frozen-parent observation bytes; no upgraded UTC claim"
        else:
            if path.name != "observations.parquet" or path.parent.name != "processed":
                raise ValueError("New inputs must be the actual clock-compatible processed observation table")
            compat = check_compatibility(path.parent.parent, train_end_year=train_end_year,
                                         validation_year=validation_year, horizons=HORIZONS)
            record["clock_compatibility"] = compat
            policy=json.loads((path.parent.parent/"clock-policy.json").read_text())
            record["source_clock_modes"]={s:("bounded_utc_or_gps" if r["mode"] == "bounded_utc_or_gps" else "declared_utc")
                                           for s,r in policy["sources"].items()}
        paths.append(path);evidence.append(record)
    if parent_hashes-seen:
        raise ValueError("Every frozen parent source observation hash must remain in child inputs")
    return paths, evidence


def _read_audits(identity_audit, reference_audit):
    audits = []
    for kind, path in [("identity", identity_audit), ("reference", reference_audit)]:
        if path:
            audits.append((json.loads(Path(path).read_text()),
                           {"kind": kind, "path": str(path), "sha256": legacy.sha256(path)}))
    return audits


def _observations(paths, parent, audits):
    columns = ["timestamp", "lat", "lon", "animal", "species", "study", "source"]
    # Project required columns; full QC tables contain potentially large original
    # JSON diagnostics which must not be materialized for case preparation.
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit("ns")
    valid = frame.timestamp.notna() & frame.lat.between(-90, 90) & frame.lon.between(-180, 180)
    valid &= frame.animal.notna() & frame.species.notna()
    frame = frame.loc[valid].drop_duplicates(["animal", "timestamp", "lat", "lon"])
    known = [(a, b, {"evidence": "inherited_frozen_parent_identity_mapping"}) for a, b in parent["inherited_links"]]
    for audit, provenance in audits:
        known.extend((a, b, {"evidence": provenance["kind"]+"_identity_audit", "audit_sha256": provenance["sha256"]})
                     for a, b in audit.get("fold_groups", {}).items())
    return legacy.canonicalize(frame, known)


def _exclusions(frame, mapping, parent, audits):
    groups = {n:n for n in frame.animal.unique()};ambiguous = set()
    def find(name):
        while groups[name] != name:
            groups[name] = groups[groups[name]];name = groups[name]
        return name
    for audit, _ in audits:
        components = [a.get("animals", []) for a in audit.get("ambiguous", [])]
        risk = defaultdict(set)
        for name, group in audit.get("uncertainty_fold_groups", {}).items():
            risk[group].update([name, group])
        # A full uncertainty mapping can include independent singleton groups;
        # those are not evidence of an ambiguous identity.
        components.extend(mapped for names in risk.values()
                          if len(mapped := ({mapping.get(n,n) for n in names} & groups.keys())) > 1)
        for component in components:
            names = {mapping.get(n,n) for n in component} & groups.keys()
            ambiguous.update(names)
            if len(names) > 1:
                first = min(names)
                for name in names:
                    a,b = find(first),find(name);groups[max(a,b)] = min(a,b)
        ambiguous.update(mapping.get(n,n) for n in audit.get("freshness_exclusion_animals", [])
                         if mapping.get(n,n) in groups)
    bindings = legacy.parent_bindings(parent, mapping, groups, find)
    ambiguous.update(mapping.get(n,n) for n in parent["manifest"].get("ambiguous_identity_records", [])
                     if mapping.get(n,n) in groups)
    exclusion = {n:find(n) for n in groups}
    pins = legacy.parent_role_pins(parent, bindings, exclusion, ambiguous)
    return exclusion, ambiguous, bindings, pins


def _subday_context(frame,parent,bindings,evidence,parent_clock_audit,train_end_year,validation_year):
    supplemental=validate_parent_clock_evidence(parent_clock_audit,parent) if parent_clock_audit else None
    inherited=defaultdict(list)
    for binding in bindings:inherited[binding["child_animal"]].append(binding["parent_animal"])
    source_modes=defaultdict(list)
    for record in evidence:
        for source,mode in record.get("source_clock_modes",{}).items():source_modes[source].append(mode)
    admitted_parent=supplemental["admitted_parent_animals"] if supplemental else {}
    modes={};excluded=[]
    for animal,group in frame.groupby("animal",sort=False):
        ancestors=inherited.get(animal,[])
        # Strong coordinate aliases do not turn an undocumented parent clock
        # into a documented clock merely by adding a newer release.
        if any(name not in admitted_parent for name in ancestors):excluded.append(animal);continue
        options=[admitted_parent[n] for n in ancestors]
        options.extend(m for source in set(group.source) for m in source_modes.get(source,[]))
        if not options:excluded.append(animal);continue
        modes[animal]="bounded_utc_or_gps" if "bounded_utc_or_gps" in options else "declared_utc"
    context={"subday_clock_excluded_animals":sorted(excluded),"subday_clock_modes":modes,
             "cutoff_boundaries":[f"{train_end_year+1}-01-01T00:00:00Z",f"{validation_year+1}-01-01T00:00:00Z"],
             "subday_clock_rule":"All frozen parent canonical ancestors require content-bound supplemental clock evidence; unknown/local parent clocks cannot gain subday labels through strong aliases. Relative history remains nominal; no device accuracy is upgraded."}
    provenance={"kind":"parent_subday_clock","path":str(parent_clock_audit),"sha256":legacy.sha256(parent_clock_audit)} if parent_clock_audit else None
    return context,provenance


def _cross_taxon_declarations(frame,exclusion,ambiguous,animals):
    """Describe uncertain resampling blocks without merging biological labels."""
    physical=frame[["animal","species"]].drop_duplicates()
    if (physical.groupby("animal").species.nunique() != 1).any():
        raise ValueError("A canonical physical identity has conflicting raw species; resolve the physical/taxonomic evidence before preparation")
    physical=physical.assign(exclusion=physical.animal.map(exclusion))
    declarations=[]
    for _,group in physical.groupby("exclusion",sort=True):
        if group.species.nunique() <= 1:continue
        names=set(group.animal)
        if len(names) < 2 or not names <= ambiguous:
            raise ValueError("Cross-taxon exclusion group lacks explicit uncertainty evidence for every canonical member")
        declarations.append({"individual":int(animals[min(names)]),"canonical_animals":sorted(names),
                             "raw_species":sorted(set(group.species)),
                             "canonical_taxa":dict(sorted(zip(group.animal,group.species))),
                             "kind":"uncertain_identity_block"})
    return declarations


def _support_audit(parent, case_map, arrays):
    """Audit old development support; sealed parent support remains undisclosed."""
    old_horizons = np.asarray(parent["manifest"]["horizons_days"])
    child_columns = np.array([int(np.flatnonzero(HORIZONS == h)[0]) for h in old_horizons])
    with np.load(parent["paths"]["cases"], allow_pickle=False) as archive:
        before = archive["target_mask"][case_map.parent_case_id]
        old_times = archive["target_time"][case_map.parent_case_id]
    after = arrays["target_mask"][case_map.child_case_id][:,child_columns]
    new_times = arrays["target_time"][case_map.child_case_id][:,child_columns]
    development = ~case_map.parent_split.isin(["test_individual", "test_year"]).to_numpy()
    lost = before & ~after
    if lost.any():
        raise ValueError("A parent observed development target lost support at a shared day-scale horizon")
    common = before & after & development[:,None]
    return {"horizons_days": old_horizons.tolist(),
            "parent_development_targets": before[development].sum(0).tolist(),
            "preserved_development_targets": common.sum(0).tolist(),
            "gained_development_targets": ((~before)&after&development[:,None]).sum(0).tolist(),
            "changed_actual_target_times": ((old_times != new_times)&common).sum(0).tolist(),
            "lost_development_targets": lost.sum(0).tolist(),
            "reserved_rule": "Every parent reserved origin/split is retained; masks and labels stay zero. Actual reserved support/outcomes are not disclosed."}


def prepare(roots, output, *, parent_cache, identity_audit=None, reference_audit=None,
            existing_manifest=None, train_end_year=2020, validation_year=2021, seed=1701,
            horizons=HORIZONS,parent_clock_audit=None):
    """Prepare the fixed reviewed multiscale protocol; no reserved-release option."""
    if validation_year != train_end_year+1:
        raise ValueError("Validation must immediately follow the pinned training years")
    if not np.array_equal(np.asarray(horizons,dtype=float),HORIZONS):
        raise ValueError("This explicit multiscale protocol freezes its nine horizons")
    parent = _load_parent(parent_cache, train_end_year, validation_year)
    output = Path(output)
    if output.exists():raise FileExistsError("Use a new immutable multiscale output directory")
    paths, evidence = _input_paths(roots, parent, train_end_year, validation_year)
    audits = _read_audits(identity_audit, reference_audit)
    frame, mapping, links = _observations(paths, parent, audits)
    exclusion, ambiguous, bindings, pins = _exclusions(frame, mapping, parent, audits)
    clock_context,clock_provenance=_subday_context(frame,parent,bindings,evidence,parent_clock_audit,train_end_year,validation_year)
    pinned_roles, pinned_freshness, pinned_origins, reserved_temporal, old_origins = pins
    previous = set(json.loads(Path(existing_manifest).read_text())["animals"]) if existing_manifest else set()
    previous_groups = {exclusion.get(mapping.get(n,n), mapping.get(n,n)) for n in previous}
    group_ids = {n:i for i,n in enumerate(sorted(set(exclusion.values())))}
    animals = {n:group_ids[exclusion[n]] for n in sorted(frame.animal.unique())}
    cross_taxon=_cross_taxon_declarations(frame,exclusion,ambiguous,animals)
    species_names = sorted(frame.species.unique());species = {n:i+1 for i,n in enumerate(species_names)}
    studies = {n:i for i,n in enumerate(sorted(frame.study.unique()))}
    cutoff = pd.Timestamp(f"{train_end_year+1}-01-01", tz="UTC").timestamp()
    val_end = pd.Timestamp(f"{validation_year+1}-01-01", tz="UTC").timestamp()
    data = defaultdict(list);rows=[];animal_audit=[];skipped=Counter();assignments={};clock_withheld=defaultdict(lambda:np.zeros(len(HORIZONS),dtype="int64"))
    for animal,g in frame.groupby("animal", sort=True):
        g = g.sort_values("timestamp")
        lat,lon = g.lat.to_numpy(),g.lon.to_numpy();dates = pd.DatetimeIndex(g.timestamp)
        ns = dates.asi8;times=ns/1e9;days=dates.dayofyear.to_numpy()
        span = (times[-1]-times[0])/DAY
        stride = SHORT_STRIDE_DAYS if span < SHORT_TRACK_SPAN_DAYS else LONG_STRIDE_DAYS
        speed = distance(lat[1:],lon[1:],lat[:-1],lon[:-1])/np.maximum(np.diff(times),1.)
        speed_prefix = np.r_[0,np.cumsum(speed > 100)]
        group=exclusion[animal];fresh = group not in previous_groups and animal not in ambiguous
        fresh=pinned_freshness.get(group,fresh)
        bucket=int(hashlib.sha256(f"migration-v1:{seed}:{group}".encode()).hexdigest()[:12],16)%100
        role="test_individual" if fresh and bucket < 15 else "val_individual" if (15 <= bucket < 30 if fresh else bucket < 20) else "temporal"
        role=pinned_roles.get(group,role);assignments[animal]=role
        animal_audit.append({"canonical_animal":animal,"individual":animals[animal],"species":g.species.iloc[0],
                             "study":g.study.iloc[0],"observations":len(g),"first":dates[0],"last":dates[-1],
                             "span_days":span,"observed_years":len(np.unique(dates.year)),"stride_days":stride,
                             "fresh_identity":fresh,"assignment":role})
        last_regular=-np.inf;inherited=pinned_origins.get(animal,set())
        for i in range(MINIMUM_PRIOR_FIXES,len(g)):
            if times[i]-times[0] < MINIMUM_HISTORY_SPAN_DAYS*DAY:continue
            forced=ns[i] in inherited;regular=times[i]-last_regular >= stride*DAY
            if not forced and not regular:continue
            hist,hm,static,hll,ht=features(lat,lon,times,dates,i,days)
            if hm.sum() < MINIMUM_HISTORY_LAGS:continue
            # Regular cadence uses only observed pre-origin history.  Forced
            # parent origins are unioned without resetting that cadence.
            if regular:last_regular=times[i]
            if role == "test_individual":split,boundary=3,np.inf
            elif role == "val_individual":
                if times[i] >= val_end:skipped["development_origins_after_year_boundary"]+=1;continue
                split,boundary=1,val_end
            elif times[i] < cutoff:split,boundary=0,cutoff
            elif times[i] < val_end:split,boundary=2,val_end
            elif fresh or group in reserved_temporal:split,boundary=4,np.inf
            else:skipped["previously_inspected_future_years"]+=1;continue
            ix,mask=match_targets(times,i);mask &= times[ix] < boundary
            admission=target_admission_mask(HORIZONS,animal,clock_context,origin_time=times[i],target_times=times[ix])
            if split < 3:clock_withheld[g.species.iloc[0]] += mask&~admission
            mask &= admission
            if not mask.any():skipped["no_observed_future_target"]+=1;continue
            left=np.searchsorted(times,times[i]-7*DAY);right=int(ix[mask].max())
            if speed_prefix[right] > speed_prefix[left]:skipped["speed_jump_cases"]+=1;continue
            sealed=split >= 3
            legacy_history=(times[i]-times[0] >= 7*DAY and hm[LEGACY_LAG_COLUMNS].sum() >= 5)
            long_history=(times[i]-times[0] >= 30*DAY and hm[LEGACY_LAG_COLUMNS].sum() >= 5
                          and hm[int(np.flatnonzero(LAGS == 7)[0])])
            long_support="sealed" if sealed else "yes" if long_history and mask[HORIZONS >= 30].any() else "no"
            if sealed:
                # Derive no future coordinate values for a sealed case.
                mask[:]=False;y=np.zeros((len(HORIZONS),2),dtype="float32")
                ll=np.zeros((len(HORIZONS),2));target_time=np.zeros(len(HORIZONS));proxy="sealed"
            else:
                y=(project(lat[ix],lon[ix],lat[i],lon[i])/UNIT).astype("float32")
                ll=np.column_stack([lat[ix],lon[ix]]);target_time=times[ix].copy()
                proxy="over_100km" if np.linalg.norm(y[mask],axis=-1).max()*UNIT/1000 > 100 else "under_100km"
                y[~mask]=0;ll[~mask]=0;target_time[~mask]=0
            for key,value in {"history":hist,"history_mask":hm,"static":static,"species":species[g.species.iloc[0]],
                              "y":y,"target_mask":mask,"split":split,"individual":animals[animal],
                              "study":studies[g.study.iloc[0]],"origin":[lat[i],lon[i],times[i]],
                              "target_latlon":ll,"target_time":target_time,"history_latlon":hll,"history_time":ht}.items():
                data[key].append(value)
            rows.append({"case_id":len(rows),"canonical_animal":animal,"individual":animals[animal],
                         "species":g.species.iloc[0],"study":g.study.iloc[0],"origin_time":dates[i],"year":dates.year[i],
                         "animal_year":f"{animal}:{dates.year[i]}","split":next(n for n,v in SPLITS.items() if v == split),
                         "fresh_identity":fresh,"targets_sealed":sealed,"displacement_proxy":proxy,
                         "memory365_available":bool(hm[-1]),"forced_parent_origin":forced,
                         "regular_origin":regular,"legacy_history_support":bool(legacy_history),
                         "long_history_support":bool(long_history),"long_migration_support_proxy":long_support})
    if not rows:raise ValueError("No eligible actual multiscale cases")
    arrays={key:np.asarray(values) for key,values in data.items()};origins=pd.DataFrame(rows)
    train_species={name:i+1 for i,name in enumerate(sorted(set(origins.loc[origins.split == "train","species"])))}
    remap={old:train_species.get(name,0) for name,old in species.items()}
    arrays["species"]=np.asarray([remap[int(v)] for v in arrays["species"]],dtype="int64")
    metric_species={name:i for i,name in enumerate(species_names)}
    arrays["metric_species"]=np.asarray([metric_species[n] for n in origins.species],dtype="int64")
    for key in ["individual","study"]:arrays[key]=arrays[key].astype("int64")
    # Preserve canonical fractional horizons exactly for sidecar reconstruction
    # at tight subday tolerance boundaries.  Existing legacy cache dtype stays
    # unchanged; model/evaluator dimensions remain dynamic.
    arrays["split"]=arrays["split"].astype("int8");arrays["horizons"]=HORIZONS.copy()
    case_map=legacy.parent_case_map(parent,old_origins,origins)
    support=_support_audit(parent,case_map,arrays)
    sealed=arrays["split"] >= 3
    if arrays["target_mask"][sealed].any() or any(arrays[k][sealed].any() for k in ["y","target_latlon","target_time"]):
        raise ValueError("Reserved multiscale truth must remain zero")
    support_rows=[]
    for (split,taxon),group in origins.loc[~origins.targets_sealed].groupby(["split","species"]):
        ids=group.case_id.to_numpy()
        for j,horizon in enumerate(HORIZONS):
            valid=arrays["target_mask"][ids,j];selected=group.loc[valid]
            support_rows.append({"split":split,"species":taxon,"horizon_days":horizon,"cases":int(valid.sum()),
                                 "physical_exclusion_groups":selected.individual.nunique(),
                                 "long_history_cases":int(selected.long_history_support.sum()),
                                 "long_migration_support_proxy_cases":int((selected.long_migration_support_proxy == "yes").sum())})
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=output.name+".pending-",dir=output.parent))
    try:
        np.savez_compressed(stage/"cases.npz",**arrays);origins.to_parquet(stage/"origins.parquet",index=False)
        case_map.to_parquet(stage/"parent-case-map.parquet",index=False)
        pd.DataFrame(animal_audit).to_parquet(stage/"animal-audit.parquet",index=False)
        pd.DataFrame(support_rows).to_csv(stage/"development-support.csv",index=False)
        origins.groupby(["split","species"]).agg(cases=("case_id","size"),animals=("individual","nunique")).reset_index().to_csv(stage/"cohort-counts.csv",index=False)
        manifest={"protocol":PROTOCOL,"preparation_version":PROTOCOL,"horizons_days":HORIZONS.tolist(),"horizons":HORIZONS.tolist(),
                  "assessment_target_matcher":"wildcast.multiscale_data.match_targets",
                  "unit_metres":UNIT,"history_names":HISTORY_NAMES,"static_names":STATIC_NAMES,"history_lags_days":LAGS.tolist(),
                  "minimum_history_lags":MINIMUM_HISTORY_LAGS,"minimum_history_prior_fixes":MINIMUM_PRIOR_FIXES,
                  "minimum_history_span_days":MINIMUM_HISTORY_SPAN_DAYS,
                  "minimum_observed_future_targets":1,
                  "species":train_species,"metric_species":metric_species,"animals":animals,"studies":studies,"splits":SPLITS,
                  "assignments":assignments,"seed":seed,"train_end_year":train_end_year,"validation_year":validation_year,
                  "cases":len(origins),"case_counts":origins.split.value_counts().to_dict(),
                  "fresh_physical_animals":int(sum(r["fresh_identity"] for r in animal_audit)),
                  "target_counts":{n:arrays["target_mask"][arrays["split"] == v].sum(0).tolist() for n,v in SPLITS.items()},
                  "training_taxa_with_any_target":len(train_species),
                  "development_support":"development-support.csv","development_support_sha256":legacy.sha256(stage/"development-support.csv"),
                  "sampling_rule":{"long_track_stride_days":1.,"short_track_stride_days":.25,"short_track_span_below_days":14.,
                                   "minimum_strictly_prior_fixes":8,"minimum_past_span_hours":6,"minimum_valid_history_slots":2,
                                   "minimum_observed_future_targets":1,"parent_origins":"union at exact observed nanosecond timestamps; do not shift regular cadence",
                                   "track_span_scope":"retrospective timestamp-only cadence policy; full track span is never a predictive feature"},
                  "long_support_rule":"Development diagnostic only: >=30d actual past span, >=5 valid legacy lag slots including 7d, and >=1 actual target at 30/60/90d. Observation support proxy, not a verified migration phase.",
                  "target_rule":"Nearest actual future observation. H<1d tolerance=min(1h,25%H); H>=1d tolerance=min(2d,max(3h,5%H)). Actual target time retained; endpoints purged at pinned calendar cutoffs.",
                  "observation_tie_rule":"After raw animal/time/lat/lon dedup, stable sort canonical animal, timestamp, original_animal; retain first canonical animal+timestamp. Input-root and source-row order break remaining exact ties.",
                  "target_tie_rule":"Choose the right/later actual observation when left/right distances are equal, matching the legacy matcher.",
                  "history_rule":"Observed fixes at/before each requested lag only. Added 1h/3h lags tolerate min(1h,25%lag); original 25 slots retain legacy tolerances. No interpolated fixes or train/validation-fitted features.",
                  "reserved_test_targets_sealed":True,"sources":evidence,"identity_mapping":mapping,"identity_links":links,
                  **clock_context,"parent_clock_audit":clock_provenance,
                  "subday_clock_withheld_development_targets":{s:v.tolist() for s,v in clock_withheld.items() if v.any()},
                  "identity_unit":"Strong canonical identities; uncertain aliases share split/retrieval/bootstrap exclusion groups without coordinate merging.",
                  "exclusion_groups":exclusion,"ambiguous_identity_records":sorted(ambiguous),"audits":[p for _,p in audits]+([clock_provenance] if clock_provenance else []),
                  "cross_taxon_exclusion_groups":cross_taxon,
                  "reference_audit":str(reference_audit) if reference_audit else None,"reference_audit_sha256":legacy.sha256(reference_audit) if reference_audit else None,
                  "existing_manifest":str(existing_manifest) if existing_manifest else None,"existing_manifest_sha256":legacy.sha256(existing_manifest) if existing_manifest else None,
                  "cases_sha256":legacy.sha256(stage/"cases.npz"),"origins_sha256":legacy.sha256(stage/"origins.parquet"),"skipped":dict(skipped),
                  "preparation_code_sha256":{name:legacy.sha256(Path(__file__).with_name(name)) for name in ["multiscale_data.py","migration_data.py","migration_clock.py","parent_clock.py","data.py"]},
                  "freshness_status":"Candidate fresh under recorded identity/exclusion rules; parent freshness cannot be promoted. No reserved biological outcomes inspected.",
                  "parent_cache":{"protocol":"wildcast-multiscale-parent-extension-v1","path":parent["root"],**parent["hashes"],
                                  "parent_cases_preserved":len(case_map),"parent_reserved_cases_preserved":int(case_map.parent_split.isin(["test_individual","test_year"]).sum()),
                                  "parent_assignments_pinned":len(bindings),"pinned_roles":dict(Counter(pinned_roles.values())),
                                  "bindings":bindings,"case_map":"parent-case-map.parquet","case_map_sha256":legacy.sha256(stage/"parent-case-map.parquet"),
                                  "parent_clock_audit":clock_provenance,
                                  "shared_horizon_support":support,
                                  "change_authorization":"Explicit new protocol extends horizons/history and changes cadence; the legacy parent loader's default guard is unchanged.",
                                  "reserved_protection":"All parent roles, exclusion groups and reserved temporal years are inherited; conflicting roles abort. Freshness may only be suppressed."},
                  "limitations":["One-hour training breadth is not long-horizon or seasonal migration evidence.",
                                 "Tracking availability and displacement are not verified migration/departure/arrival labels.",
                                 "Clock compatibility is source-specific; nominal bounded UTC/GPS is not instrument-verified UTC.",
                                 "Normalization must be fitted on training cases only by the trainer.",
                                 "Final assessment must dispatch this protocol's subday matcher before any explicit truth release."]}
        # Detect concurrent metadata/source/audit mutations before publishing.
        for path,record in zip(paths,evidence):
            if legacy.sha256(path) != record["sha256"]:raise ValueError("Source observations changed during preparation")
        for _,record in audits:
            if legacy.sha256(record["path"]) != record["sha256"]:raise ValueError("Identity audit changed during preparation")
        if clock_provenance:
            if legacy.sha256(parent_clock_audit) != clock_provenance["sha256"]:raise ValueError("Supplemental parent clock audit changed")
            validate_parent_clock_evidence(parent_clock_audit,parent)
        legacy.parent_case_map(parent,old_origins,origins)
        (stage/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n")
        if output.exists():raise FileExistsError("Multiscale output appeared during preparation")
        stage.rename(output)
        return manifest
    finally:
        if stage.exists():shutil.rmtree(stage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots",nargs="+",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--parent-cache",required=True);parser.add_argument("--identity-audit");parser.add_argument("--reference-audit")
    parser.add_argument("--parent-clock-audit")
    parser.add_argument("--existing-manifest");parser.add_argument("--train-end-year",type=int,default=2020)
    parser.add_argument("--validation-year",type=int,default=2021);parser.add_argument("--seed",type=int,default=1701)
    args=parser.parse_args();manifest=prepare(args.roots,args.output,parent_cache=args.parent_cache,
        identity_audit=args.identity_audit,reference_audit=args.reference_audit,existing_manifest=args.existing_manifest,
        train_end_year=args.train_end_year,validation_year=args.validation_year,seed=args.seed,parent_clock_audit=args.parent_clock_audit)
    print(json.dumps({key:manifest[key] for key in ["cases","case_counts","training_taxa_with_any_target","target_counts"]},indent=2))


if __name__ == "__main__":main()
