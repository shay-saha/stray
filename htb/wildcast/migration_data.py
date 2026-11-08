"""Observed long-horizon cases, physical identities, and sealed final targets.

This protocol concerns migration-relevant movement, not verified behavioural labels.
Observation gaps stay missing; they do not erase an animal's earlier history.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import distance, project
from .sources import dump

DAY = 86400.
UNIT = 1_000_000.
LAGS = np.array([0, .25, .5, 1, 2, 3, 4, 5, 7, 10, 14, 21, 30, 45,
                 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365])
HISTORY_NAMES = ["east_1000km", "north_1000km", "actual_age_days", "season_sin",
    "season_cos", "latitude_90", "longitude_sin", "longitude_cos", "requested_age_days", "available"]
STATIC_NAMES = ["origin_unit_x", "origin_unit_y", "origin_unit_z", "season_sin", "season_cos",
    "solar_sin", "solar_cos", "velocity_1d_east_1000km_per_day", "velocity_1d_north_1000km_per_day",
    "velocity_7d_east_1000km_per_day", "velocity_7d_north_1000km_per_day",
    "log1p_observed_span_days", "memory365_available", "daylength_fraction"]
SPLITS = {"train": 0, "val_individual": 1, "val_year": 2, "test_individual": 3, "test_year": 4}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def canonicalize(frame, known_links=None, *, infer_coordinate_aliases=True):
    """Canonicalize records using explicit links and optional historical inference.

    The old exact-count rule is retained for archival reproduction. It can
    overmerge colocating animals; repaired protocols disable it and supply only
    separately audited primitive links. Local identifiers are not global IDs.
    """
    if not isinstance(infer_coordinate_aliases, (bool, np.bool_)):
        raise ValueError("Coordinate-inference policy must be an explicit boolean")
    f = frame.copy()
    names = sorted(f.animal.unique())
    parent = {n: n for n in names}
    links = []
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]; n = parent[n]
        return n
    def join(a, b, reason):
        if a not in parent or b not in parent: return
        aa, bb = find(a), find(b)
        if aa != bb:
            parent[max(aa, bb)] = min(aa, bb)
            links.append({"a": a, "b": b, **reason})
    # One DOI may bundle several studies with separate local ID namespaces.
    # Repeated local numbers across those studies are not identity evidence.
    edges = known_links.items() if hasattr(known_links, "items") else known_links or []
    for edge in edges:
        a, b = edge[:2]
        join(a, b, edge[2] if len(edge) == 3 else {"evidence": "existing_reference_identity_audit"})
    if infer_coordinate_aliases:
        duplicate = f.loc[f.duplicated(["species", "timestamp", "lat", "lon"], keep=False)]
        overlaps = {}
        for key, g in duplicate.groupby(["species", "timestamp", "lat", "lon"], sort=False):
            for a, b in itertools.combinations(sorted(g.animal.unique()), 2):
                if find(a) == find(b): continue
                stamp, position = key[1], (key[2], key[3])
                if (a, b) not in overlaps: overlaps[a, b] = [0, stamp, stamp, position, False]
                value = overlaps[a, b]
                value[0] += 1; value[1] = min(value[1], stamp); value[2] = max(value[2], stamp)
                value[4] |= value[3] != position
        for (a, b), (count, first, last, _, moved) in overlaps.items():
            duration = (last - first).total_seconds()
            if count >= 5 and moved and duration >= 3600:
                join(a, b, {"evidence": "exact_species_time_GPS_overlap", "shared_fixes": count, "span_hours": duration / 3600})
    mapping = {n: find(n) for n in names}
    f["original_animal"] = f.animal
    f["animal"] = f.animal.map(mapping)
    # A canonical study is stable across aliases, even when records come from
    # multiple papers. Keep original identities separately for provenance.
    anchor = f.sort_values(["animal", "original_animal"]).drop_duplicates("animal").set_index("animal")
    f["study"] = f.animal.map(anchor.study)
    f = f.sort_values(["animal", "timestamp", "original_animal"], kind="stable")
    f = f.drop_duplicates(["animal", "timestamp"]).reset_index(drop=True)
    return f, mapping, links


def load_observations(roots, identity_audit=None, reference_audit=None, inherited_links=None):
    frames, evidence = [], []
    for root in roots:
        path = Path(root)
        if path.is_dir(): path = path / "processed/observations.parquet"
        if not path.exists(): raise FileNotFoundError(path)
        f = pd.read_parquet(path)
        required = {"timestamp", "lat", "lon", "animal", "species", "study", "source"}
        if not required.issubset(f): raise ValueError(f"Missing columns: {required - set(f.columns)}")
        f["timestamp"] = pd.to_datetime(f.timestamp, utc=True)
        frames.append(f[list(required)])
        evidence.append({"path": str(path), "sha256": sha256(path), "rows": len(f)})
    f = pd.concat(frames, ignore_index=True)
    valid = f.timestamp.notna() & f.lat.between(-90, 90) & f.lon.between(-180, 180)
    valid &= f.animal.notna() & f.species.notna()
    f = f.loc[valid].drop_duplicates(["animal", "timestamp", "lat", "lon"])
    known = json.loads(Path(identity_audit).read_text()).get("fold_groups", {}) if identity_audit else {}
    if reference_audit:
        known.update(json.loads(Path(reference_audit).read_text()).get("fold_groups", {}))
    # Keep the legacy dict/override behaviour unless ancestry is requested.
    # Parent edges are additional strong evidence, not keys overwritten by a
    # later audit that happens to choose a different canonical spelling.
    known = known if inherited_links is None else [
        *((a, b, {"evidence": "inherited_frozen_parent_identity_mapping"}) for a, b in inherited_links), *known.items()]
    f, mapping, links = canonicalize(f, known)
    return f, mapping, links, evidence


def load_parent_cache(path, train_end_year, validation_year, horizons, stride_days, release_test=False):
    """Verify frozen ancestry without deriving or revealing any reserved label."""
    root = Path(path)
    paths = {"manifest": root / "manifest.json", "cases": root / "cases.npz", "origins": root / "origins.parquet"}
    hashes = {key + "_sha256": sha256(value) for key, value in paths.items()}
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("cases_sha256") != hashes["cases_sha256"] or manifest.get("origins_sha256") != hashes["origins_sha256"]:
        raise ValueError("Parent cache cases/origins checksum differs from its frozen manifest")
    if manifest.get("reserved_test_targets_sealed") is not True or release_test:
        raise ValueError("Parent-pinned preparation requires sealed reserved targets and cannot release them")
    if manifest.get("eligible_for_final_assessment", True) is not True:
        raise ValueError("A development-only derived cache cannot define a new final-cohort parent")
    if manifest.get("train_end_year") != train_end_year or manifest.get("validation_year") != validation_year:
        raise ValueError("Parent cache freezes calendar cutoffs; changed years require an explicit new protocol")
    if (not np.array_equal(np.asarray(manifest.get("horizons_days", [])), np.asarray(horizons)) or
            float(manifest.get("stride_days", -1)) != float(stride_days) or
            manifest.get("history_names") != HISTORY_NAMES or not np.array_equal(manifest.get("history_lags_days", []), LAGS)):
        raise ValueError("Parent origin/horizon/history protocol differs; use an explicit new protocol")
    with np.load(paths["cases"], allow_pickle=False) as archive:
        required = {"split", "individual", "origin", "target_mask", "y", "target_latlon", "target_time"}
        if not required.issubset(archive.files):
            raise ValueError("Parent cases lack required cohort/label-integrity arrays")
        a = {key: archive[key] for key in required}
    origins = pd.read_parquet(paths["origins"])
    n = len(a["split"])
    required_columns = {"case_id", "canonical_animal", "individual", "origin_time", "split", "fresh_identity", "targets_sealed"}
    if (not required_columns.issubset(origins) or len(origins) != n or manifest.get("cases") != n or
            origins.case_id.tolist() != list(range(n)) or a["origin"].shape != (n, 3)):
        raise ValueError("Parent origins/case identities or dimensions differ")
    if (not np.array_equal(origins["split"].map(SPLITS).to_numpy(), a["split"]) or
            not np.array_equal(origins.individual.to_numpy(), a["individual"]) or
            not np.allclose(pd.to_datetime(origins.origin_time, utc=True).astype("int64") / 1e9,
                            a["origin"][:, 2], rtol=0, atol=1e-6)):
        raise ValueError("Parent origin split/group/time evidence is misaligned")
    sealed = a["split"] >= 3
    if (a["target_mask"][sealed].any() or not origins.loc[sealed, "targets_sealed"].all() or
            any(np.any(a[key][sealed] != 0) for key in ("y", "target_latlon", "target_time"))):
        raise ValueError("Parent reserved targets are not fully sealed")
    assignments = manifest.get("assignments", {})
    allowed = {"temporal", "val_individual", "test_individual"}
    if set(assignments.values()) - allowed or set(manifest.get("animals", {})) != set(assignments):
        raise ValueError("Parent physical-animal assignment metadata is incomplete")
    freshness = {}
    for animal, group in origins.groupby("canonical_animal", sort=False):
        if animal not in assignments or not group.fresh_identity.isin([True, False]).all() or group.fresh_identity.nunique() != 1:
            raise ValueError("Parent physical-animal freshness evidence is inconsistent")
        if any(int(v) != int(manifest["animals"][animal]) for v in group.individual):
            raise ValueError("Parent canonical/physical identity metadata differs")
        expected_splits = {"temporal": {"train", "val_year", "test_year"},
                           "val_individual": {"val_individual"}, "test_individual": {"test_individual"}}[assignments[animal]]
        if set(group["split"]) - expected_splits:
            raise ValueError("Parent assignment role conflicts with recorded origin splits")
        freshness[animal] = bool(group.fresh_identity.iloc[0])
    if sha256(paths["manifest"]) != hashes["manifest_sha256"]:
        raise ValueError("Parent manifest changed during verification")
    return {"root": str(root.resolve()), "paths": paths, "hashes": hashes, "manifest": manifest,
            "origins": origins, "freshness": freshness,
            "inherited_links": list(manifest.get("identity_mapping", {}).items())}


def parent_bindings(parent, mapping, exclusion_parent, exclusion_find):
    """Resolve old record IDs into child physical identities and exclusion sets."""
    meta = parent["manifest"]
    originals = {name: {name} for name in meta["assignments"]}
    for original, canonical in meta.get("identity_mapping", {}).items():
        if canonical not in originals:
            raise ValueError("Parent record mapping names an unknown physical identity")
        originals[canonical].add(original)
    bindings, missing = [], []
    observed_parent = set(parent["origins"].canonical_animal)
    for canonical, old_names in originals.items():
        targets = {mapping.get(name, name) for name in old_names} & exclusion_parent.keys()
        if not targets:
            if canonical in observed_parent:
                missing.append(canonical)
            continue
        if len(targets) != 1:
            raise ValueError("Parent strong physical identity split into multiple child animals")
        bindings.append({"parent_animal": canonical, "parent_role": meta["assignments"][canonical],
                         "parent_fresh_identity": parent["freshness"].get(canonical, False),
                         "parent_freshness_known": canonical in parent["freshness"],
                         "parent_exclusion_group": meta.get("exclusion_groups", {}).get(canonical, canonical),
                         "child_animal": next(iter(targets)), "original_record_ids": sorted(old_names)})
    if missing:
        raise ValueError(f"Parent observed cohort is absent from child inputs ({len(missing)} animals); preserve ancestry or define a new protocol")
    inherited_components = defaultdict(set)
    for record in bindings:
        inherited_components[record["parent_exclusion_group"]].add(record["child_animal"])
    for names in inherited_components.values():
        first = min(names)
        for name in names:
            a, b = exclusion_find(first), exclusion_find(name)
            exclusion_parent[max(a, b)] = min(a, b)
    return bindings


def parent_role_pins(parent, bindings, exclusion_groups, ambiguous_names):
    by_group = defaultdict(list)
    for record in bindings:
        group = exclusion_groups[record["child_animal"]]
        record["child_exclusion_group"] = group
        by_group[group].append(record)
    roles, freshness = {}, {}
    for group, records in by_group.items():
        options = {r["parent_role"] for r in records}
        if len(options) != 1:
            raise ValueError("Conflicting parent roles in a child physical/exclusion component; resolve identity before preparation: " + ",".join(sorted(options)))
        roles[group] = options.pop()
        freshness[group] = all(r["parent_fresh_identity"] for r in records)
        # New uncertainty can conservatively suppress freshness, never promote
        # an old known/ambiguous identity into fresh final eligibility.
        if any(exclusion_groups[name] == group for name in ambiguous_names):
            freshness[group] = False
    targets = {r["parent_animal"]: r["child_animal"] for r in bindings}
    old = parent["origins"].copy()
    old["child_animal"] = old.canonical_animal.map(targets)
    origins = defaultdict(set)
    for record in old.itertuples():
        origins[record.child_animal].add(pd.Timestamp(record.origin_time).value)
    reserved_temporal = {exclusion_groups[r.child_animal] for r in old.itertuples() if r.split == "test_year"}
    return roles, freshness, origins, reserved_temporal, old


def parent_case_map(parent, old, child_origins):
    """Every frozen parent origin keeps its split; numbers may change in child."""
    child = {(r.canonical_animal, pd.Timestamp(r.origin_time).value): r for r in child_origins.itertuples()}
    records = []
    for record in old.itertuples():
        key = (record.child_animal, pd.Timestamp(record.origin_time).value)
        if key not in child:
            raise ValueError("A pinned parent origin lost observed case support; review changed identity/QC/source inputs before a new protocol")
        result = child[key]
        if result.split != record.split or (not record.fresh_identity and result.fresh_identity):
            raise ValueError("Parent origin role/freshness was promoted or otherwise changed")
        if record.split in {"test_individual", "test_year"} and not result.targets_sealed:
            raise ValueError("A parent reserved origin was unsealed")
        records.append({"parent_case_id": record.case_id, "child_case_id": result.case_id,
                        "parent_canonical_animal": record.canonical_animal, "child_canonical_animal": record.child_animal,
                        "origin_time": record.origin_time, "parent_split": record.split, "child_split": result.split,
                        "parent_fresh_identity": bool(record.fresh_identity), "child_fresh_identity": bool(result.fresh_identity)})
    for key, path in parent["paths"].items():
        if sha256(path) != parent["hashes"][key + "_sha256"]:
            raise ValueError("Frozen parent cache changed during child preparation")
    return pd.DataFrame(records)


def past_indices(times, origin, lags=LAGS):
    wanted = times[origin] - lags * DAY
    index = np.searchsorted(times, wanted, side="right") - 1
    safe = np.maximum(index, 0)
    age_error = (wanted - times[safe]) / DAY
    mask = (index >= 0) & (age_error <= np.maximum(.25, .2 * lags))
    return safe, mask


def daylength(lat, day):
    declination = np.deg2rad(23.44) * np.sin(2 * np.pi * (day - 80) / 365.25)
    return np.arccos(np.clip(-np.tan(np.deg2rad(lat)) * np.tan(declination), -1, 1)) / np.pi


def features(lat, lon, times, dates, i, day_values=None):
    ix, mask = past_indices(times, i)
    xy = project(lat[ix], lon[ix], lat[i], lon[i]) / UNIT
    day_values = dates.dayofyear.to_numpy() if day_values is None else day_values
    season = 2 * np.pi * day_values[ix] / 365.25
    hist = np.column_stack([xy, (times[i] - times[ix]) / DAY, np.sin(season), np.cos(season),
        lat[ix] / 90, np.sin(np.deg2rad(lon[ix])), np.cos(np.deg2rad(lon[ix])), LAGS, mask])
    hist[~mask] = 0
    xyz = [np.cos(np.deg2rad(lat[i])) * np.cos(np.deg2rad(lon[i])),
           np.cos(np.deg2rad(lat[i])) * np.sin(np.deg2rad(lon[i])), np.sin(np.deg2rad(lat[i]))]
    season0 = 2 * np.pi * day_values[i] / 365.25
    solar = 2 * np.pi * ((times[i] / 3600 + lon[i] / 15) % 24) / 24
    velocities = []
    for lag in [1., 7.]:
        j = int(np.where(LAGS == lag)[0][0])
        velocities.extend((-xy[j] / max((times[i] - times[ix[j]]) / DAY, .01)).tolist() if mask[j] else [0., 0.])
    static = [*xyz, np.sin(season0), np.cos(season0), np.sin(solar), np.cos(solar), *velocities,
              np.log1p((times[i] - times[0]) / DAY), float(mask[-1]), daylength(lat[i], day_values[i])]
    ll = np.column_stack([lat[ix], lon[ix]]); ll[~mask] = 0
    ht = times[ix].copy(); ht[~mask] = 0
    return hist.astype("float32"), mask, np.asarray(static, "float32"), ll, ht


def match_targets(times, origin, horizons):
    wanted = times[origin] + np.asarray(horizons) * DAY
    right = np.clip(np.searchsorted(times, wanted), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    index = np.where(abs(times[left] - wanted) < abs(times[right] - wanted), left, right)
    # Score at actual observed timestamps, preserving their deviation from the
    # nominal horizon. No matching at/before origin and no interpolated labels.
    tolerance_days = np.minimum(2., np.maximum(.125, np.asarray(horizons) * .05))
    mask = (index > origin) & (abs(times[index] - wanted) <= tolerance_days * DAY)
    return index, mask


def prepare(roots, output, identity_audit=None, existing_manifest="data/combined-landscape/processed/windows.json",
            horizons=(1, 3, 7, 14, 30, 60, 90), stride_days=7., train_end_year=2020,
            validation_year=2021, seed=1701, species_filter=None, release_test=False, reference_audit=None, parent_cache=None):
    if stride_days <= 0 or any(h <= 0 for h in horizons) or np.any(np.diff(horizons) <= 0):
        raise ValueError("Positive stride and strictly increasing positive horizons required")
    if validation_year != train_end_year + 1: raise ValueError("Validation must follow the training end year")
    parent = load_parent_cache(parent_cache, train_end_year, validation_year, horizons, stride_days, release_test) if parent_cache else None
    output = Path(output)
    if (output / "manifest.json").exists(): raise FileExistsError("Use a new output path to preserve frozen migration caches")
    output.mkdir(parents=True, exist_ok=True)
    f, mapping, links, evidence = load_observations(roots, identity_audit, reference_audit,
        parent["inherited_links"] if parent else None)
    if species_filter: f = f.loc[f.species.isin(species_filter)].reset_index(drop=True)
    previous = set()
    if existing_manifest:
        previous = set(json.loads(Path(existing_manifest).read_text())["animals"])
    previously_used = {mapping.get(n, n) for n in previous}
    exclusion_parent = {n: n for n in f.animal.unique()}
    ambiguous_names = set()
    def exclusion_find(n):
        while exclusion_parent[n] != n:
            exclusion_parent[n] = exclusion_parent[exclusion_parent[n]]; n = exclusion_parent[n]
        return n
    if reference_audit:
        reference = json.loads(Path(reference_audit).read_text())
        for ambiguity in reference.get("ambiguous", []):
            names = {mapping.get(n, n) for n in ambiguity.get("animals", [])}
            names &= exclusion_parent.keys()
            ambiguous_names.update(names)
            if len(names) < 2: continue
            first = min(names)
            for n in names:
                a, b = exclusion_find(first), exclusion_find(n)
                exclusion_parent[max(a, b)] = min(a, b)
        ambiguous_names.update(mapping.get(n, n) for n in reference.get("freshness_exclusion_animals", [])
                               if mapping.get(n, n) in exclusion_parent)
    bindings = parent_bindings(parent, mapping, exclusion_parent, exclusion_find) if parent else []
    if parent:
        ambiguous_names.update(mapping.get(n, n) for n in parent["manifest"].get("ambiguous_identity_records", [])
                               if mapping.get(n, n) in exclusion_parent)
    exclusion_groups = {n: exclusion_find(n) for n in exclusion_parent}
    pinned_roles, pinned_freshness, pinned_origins, reserved_temporal, old_parent_origins = ({}, {}, {}, set(), None)
    if parent:
        pinned_roles, pinned_freshness, pinned_origins, reserved_temporal, old_parent_origins = parent_role_pins(
            parent, bindings, exclusion_groups, ambiguous_names)
    previously_used_groups = {exclusion_groups.get(n, n) for n in previously_used}
    group_ids = {n: i for i, n in enumerate(sorted(set(exclusion_groups.values())))}
    species_names = sorted(f.species.unique())
    species = {n: i + 1 for i, n in enumerate(species_names)}
    animals = {n: group_ids[exclusion_groups[n]] for n in sorted(f.animal.unique())}
    studies = {n: i for i, n in enumerate(sorted(f.study.unique()))}
    cutoff = pd.Timestamp(f"{train_end_year + 1}-01-01", tz="UTC").timestamp()
    val_end = pd.Timestamp(f"{validation_year + 1}-01-01", tz="UTC").timestamp()
    data = defaultdict(list); rows, audit_rows = [], []
    assignments = {}; skipped = Counter()
    for animal, g in f.groupby("animal", sort=True):
        g = g.sort_values("timestamp")
        lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
        dates = pd.DatetimeIndex(g.timestamp)
        day_values = dates.dayofyear.to_numpy()
        times = dates.asi8 / 1e9
        raw_speed = distance(lat[1:], lon[1:], lat[:-1], lon[:-1]) / np.maximum(np.diff(times), 1.)
        speed_prefix = np.r_[0, np.cumsum(raw_speed > 100)]
        group = exclusion_groups[animal]
        fresh = group not in previously_used_groups and animal not in ambiguous_names
        if group in pinned_freshness:
            fresh = pinned_freshness[group]
        bucket = int(hashlib.sha256(f"migration-v1:{seed}:{group}".encode()).hexdigest()[:12], 16) % 100
        assignment = "test_individual" if fresh and bucket < 15 else "val_individual" if (15 <= bucket < 30 if fresh else bucket < 20) else "temporal"
        assignment = pinned_roles.get(group, assignment)
        assignments[animal] = assignment
        audit_rows.append({"canonical_animal": animal, "species": g.species.iloc[0], "study": g.study.iloc[0],
            "observations": len(g), "first": dates[0], "last": dates[-1], "span_days": (times[-1] - times[0]) / DAY,
            "observed_years": len(np.unique(dates.year)), "fresh_identity": fresh, "assignment": assignment,
            "gap_over_48h": int((np.diff(times) > 2 * DAY).sum())})
        # Observed origins sampled by elapsed time, not collar sampling density.
        last_origin = -np.inf
        inherited_origins = pinned_origins.get(animal, set())
        for i in range(len(g)):
            if i < 8 or times[i] - times[0] < 7 * DAY: continue
            if dates.asi8[i] not in inherited_origins and times[i] - last_origin < stride_days * DAY: continue
            if assignment == "test_individual": split = 3; boundary = np.inf
            elif assignment == "val_individual":
                if times[i] >= val_end: skipped["development_origins_after_year_boundary"] += 1; continue
                split = 1; boundary = val_end
            elif times[i] < cutoff: split = 0; boundary = cutoff
            elif times[i] < val_end: split = 2; boundary = val_end
            elif fresh or group in reserved_temporal: split = 4; boundary = np.inf
            else: skipped["previously_inspected_future_years"] += 1; continue
            ix, mask = match_targets(times, i, horizons)
            mask &= times[ix] < boundary
            if mask.sum() < 2: continue
            hist, hm, static, hll, ht = features(lat, lon, times, dates, i, day_values)
            if hm.sum() < 5: continue
            # Implausible jumps are excluded as cases, not connected by invented fixes.
            left = np.searchsorted(times, times[i] - 7 * DAY)
            right = int(ix[mask].max())
            if speed_prefix[right] > speed_prefix[left]: skipped["speed_jump_cases"] += 1; continue
            y = project(lat[ix], lon[ix], lat[i], lon[i]) / UNIT
            ll = np.column_stack([lat[ix], lon[ix]])
            target_time = times[ix].copy()
            max_distance = float(np.linalg.norm(y[mask], axis=-1).max() * UNIT / 1000)
            sealed = split >= 3 and not release_test
            if sealed: mask[:] = False
            y[~mask] = 0; ll[~mask] = 0; target_time[~mask] = 0
            for key, value in {"history": hist, "history_mask": hm, "static": static,
                "species": species[g.species.iloc[0]], "y": y.astype("float32"), "target_mask": mask,
                "split": split, "individual": animals[animal], "study": studies[g.study.iloc[0]],
                "origin": [lat[i], lon[i], times[i]], "target_latlon": ll, "target_time": target_time,
                "history_latlon": hll, "history_time": ht}.items(): data[key].append(value)
            # Sealed rows reveal no displacement-based stratum or future summary.
            rows.append({"case_id": len(rows), "canonical_animal": animal, "individual": animals[animal],
                "species": g.species.iloc[0], "study": g.study.iloc[0], "origin_time": dates[i], "year": dates.year[i],
                "animal_year": f"{animal}:{dates.year[i]}", "split": next(n for n, v in SPLITS.items() if v == split),
                "fresh_identity": fresh, "targets_sealed": sealed,
                "displacement_proxy": "sealed" if sealed else "over_100km" if max_distance > 100 else "under_100km",
                "memory365_available": bool(hm[-1])})
            last_origin = times[i]
    if not rows: raise ValueError("No eligible observed long-horizon cases")
    arrays = {key: np.asarray(value) for key, value in data.items()}
    metric_species = {name: i for i, name in enumerate(species_names)}
    arrays["metric_species"] = np.asarray([metric_species[r["species"]] for r in rows], dtype="int64")
    # Species unseen in training use a learned shared fallback, never an
    # untrained reserved-species embedding. Original names stay in origins.
    observed_species = {r["species"] for r in rows if r["split"] == "train"}
    train_species = {name: i + 1 for i, name in enumerate(sorted(observed_species))}
    remap = {old: train_species.get(name, 0) for name, old in species.items()}
    arrays["species"] = np.asarray([remap[int(v)] for v in arrays["species"]], dtype="int64")
    species = train_species
    for key in ["species", "individual", "study"]: arrays[key] = arrays[key].astype("int64")
    arrays["split"] = arrays["split"].astype("int8")
    arrays["horizons"] = np.asarray(horizons, "float32")
    origins = pd.DataFrame(rows)
    parent_map = parent_case_map(parent, old_parent_origins, origins) if parent else None
    np.savez_compressed(output / "cases.npz", **arrays)
    origins.to_parquet(output / "origins.parquet", index=False)
    if parent_map is not None:
        parent_map.to_parquet(output / "parent-case-map.parquet", index=False)
    pd.DataFrame(audit_rows).to_parquet(output / "animal-audit.parquet", index=False)
    # Audit counts expose observation availability, not any reserved future score.
    counts = origins.groupby(["split", "species"]).agg(cases=("case_id", "size"), animals=("individual", "nunique")).reset_index()
    counts.to_csv(output / "cohort-counts.csv", index=False)
    manifest = {"protocol": "wildcast-migration-v1", "preparation_version": "migration-preparation-v3",
        "horizons_days": list(horizons), "horizons": list(horizons),
        "preparation_code_sha256": {"migration_data.py": sha256(__file__), "data.py": sha256(Path(__file__).with_name("data.py"))},
        "unit_metres": UNIT, "history_names": HISTORY_NAMES, "static_names": STATIC_NAMES,
        "history_lags_days": LAGS.tolist(), "species": species, "metric_species": metric_species, "animals": animals, "studies": studies,
        "splits": SPLITS, "assignments": assignments, "seed": seed, "stride_days": stride_days,
        "train_end_year": train_end_year, "validation_year": validation_year, "cases": len(rows),
        "case_counts": origins.split.value_counts().to_dict(), "target_counts": {n: arrays["target_mask"][arrays["split"] == v].sum(0).tolist() for n, v in SPLITS.items()},
        "fresh_physical_animals": int(sum(r["fresh_identity"] for r in audit_rows)),
        "freshness_status": "Candidate unseen identities under the recorded GPS/local/reference rules; unresolved disjoint-deployment aliases still require review before a frontier claim.",
        "identity_unit": "Canonical physical identities where corroborated; plausible unresolved reference aliases share exclusion/resampling groups without merging coordinates.",
        "exclusion_groups": exclusion_groups, "ambiguous_identity_records": sorted(ambiguous_names),
        "reference_audit": str(reference_audit) if reference_audit else None,
        "reference_audit_sha256": sha256(reference_audit) if reference_audit else None,
        "reserved_test_targets_sealed": not release_test, "sources": evidence,
        "identity_links": links, "identity_mapping": mapping,
        "existing_manifest": str(existing_manifest), "existing_manifest_sha256": sha256(existing_manifest) if existing_manifest else None,
        "cases_sha256": sha256(output / "cases.npz"), "origins_sha256": sha256(output / "origins.parquet"),
        "skipped": dict(skipped),
        "target_rule": "Nearest actual future observation; tolerance=min(2d,max(0.125d,5% horizon)); actual timestamp retained; train/validation target endpoints purged at calendar boundaries.",
        "history_rule": "Observed fixes strictly at/before each past requested lag, tolerance=max(0.25d,20% lag), explicit masks; crosses gaps without inventing fixes; no held-out future inputs.",
        "scope": "Migration-relevant long-horizon movement development, not verified migration episodes. Individual split is retrospective generalization; val_year is a whole-calendar-year forecast test against training endpoints before that year.",
        "limitations": ["Conservative identity links cannot prove every cross-source alias is resolved.",
            "No verified departure, arrival, stopover, colony or behavioural labels; displacement strata are proxies.",
            "The initial model has geographic/seasonal/history inputs; candidate-route weather and landscape fields require separate ablations.",
            "Existing animal records have informed earlier WildCast experiments; only new-identity reserved cases qualify as the fresh final cohort."]}
    if parent:
        manifest["parent_cache"] = {"protocol": "wildcast-migration-parent-pinning-v1", "path": parent["root"],
            **parent["hashes"], "parent_cases_preserved": len(parent_map),
            "parent_reserved_cases_preserved": int(parent_map.parent_split.isin(["test_individual", "test_year"]).sum()),
            "unique_child_cases_for_parent": int(parent_map.child_case_id.nunique()),
            "parent_assignments_pinned": len(bindings), "child_exclusion_groups_pinned": len(pinned_roles),
            "pinned_roles": dict(Counter(pinned_roles.values())),
            "parent_freshness_suppressed_conservatively": int((parent_map.parent_fresh_identity & ~parent_map.child_fresh_identity).sum()),
            "parent_animals_without_case_freshness_evidence": sum(not r["parent_freshness_known"] for r in bindings),
            "inherited_strong_record_links": sum(a != b for a, b in parent["inherited_links"]),
            "bindings": bindings, "case_map": "parent-case-map.parquet",
            "case_map_sha256": sha256(output / "parent-case-map.parquet"),
            "rule": "Inherit strong parent record aliases, retain parent exclusion components without coordinate merges, pin physical/exclusion-group roles, reject conflicting roles and changed cutoffs/protocol, preserve every parent origin split. Never promote nonfresh identities or unseal reserved labels.",
            "freshness_rule": "Known parent freshness retained; unknown parent freshness or new ambiguity can only make it more conservative. Previously reserved temporal years remain reserved even if freshness is suppressed."}
    dump(output / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--identity-audit", default="reports/identity-audit.json")
    p.add_argument("--existing-manifest", default="data/combined-landscape/processed/windows.json")
    p.add_argument("--stride-days", type=float, default=7.)
    p.add_argument("--train-end-year", type=int, default=2020)
    p.add_argument("--validation-year", type=int, default=2021)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--species", nargs="+")
    p.add_argument("--reference-audit")
    p.add_argument("--parent-cache")
    a = p.parse_args()
    result = prepare(a.roots, a.output, a.identity_audit, a.existing_manifest,
        stride_days=a.stride_days, train_end_year=a.train_end_year, validation_year=a.validation_year,
        seed=a.seed, species_filter=a.species, reference_audit=a.reference_audit, parent_cache=a.parent_cache)
    print(json.dumps({k: result[k] for k in ["cases", "case_counts", "target_counts", "fresh_physical_animals"]}, indent=2))


if __name__ == "__main__": main()
