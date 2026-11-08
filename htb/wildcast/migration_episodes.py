"""Retrospective observed movement proxies and transparent coverage censuses.

These are candidates for ecological review, not verified migration, departure,
arrival, breeding or behavioural labels. No interpolated locations are created.
Without a split manifest only coordinate-free coverage metadata is audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import distance
from .sources import dump

DAY = 86400.
DEFAULTS = {"shift_distance_km": 100., "sustained_days": 14., "prior_anchor_days": 14.,
            "minimum_anchor_observed_days": 7, "site_radius_km": 25.,
            "minimum_anchor_span_days": 6., "minimum_return_observed_days": 3,
            "endpoint_site_window_days": 7.,
            "minimum_sustained_observed_fraction": .5, "maximum_gap_days": 7.,
            "minimum_far_fraction": .9, "return_sustained_days": 7.,
            "coverage_span_days": 120., "coverage_observed_days": 60, "coverage_months": 4}
EPISODE_COLUMNS = ["episode_id", "canonical_animal", "species", "study", "scope", "proxy_kind",
                   "verified_migration", "retrospective_only", "anchor_time", "anchor_lat", "anchor_lon",
                   "last_prior_site_observation", "first_shift_observation", "sustained_end_observation",
                   "origin_year", "end_year", "sustained_days", "observed_shift_days", "longest_shift_gap_days",
                   "far_observation_fraction", "shift_distance_km", "maximum_displacement_km",
                   "end_lat", "end_lon", "compact_endpoint_site_proxy", "observed_return_cycle_proxy",
                   "first_return_observation", "return_confirmed_observation"]


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def _observed_site(lat, lon):
    """Choose an ACTUAL observed point nearest the spherical sample mean."""
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    unit = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return int(np.argmax(unit @ unit.mean(0)))


def _daily(group):
    result = group.sort_values("timestamp", kind="stable").copy()
    result["calendar_day"] = result.timestamp.dt.floor("D")
    return result.drop_duplicates("calendar_day", keep="first").reset_index(drop=True)


def _anchor_records(daily, rules):
    """Every prior anchor uses observed locations strictly before its query day."""
    times = daily.timestamp.astype("int64").to_numpy() / 1e9
    lat, lon = daily.lat.to_numpy(), daily.lon.to_numpy()
    records = []
    for i in range(len(daily)):
        left = int(np.searchsorted(times, times[i] - rules["prior_anchor_days"] * DAY))
        if i - left < rules["minimum_anchor_observed_days"] or times[i - 1] - times[left] < rules["minimum_anchor_span_days"] * DAY:
            continue
        local = _observed_site(lat[left:i], lon[left:i])
        anchor = left + local
        spread = distance(lat[left:i], lon[left:i], lat[anchor], lon[anchor]) / 1000
        close = spread <= rules["site_radius_km"]
        if close.mean() < rules["minimum_far_fraction"]:
            continue
        last_close = left + np.flatnonzero(close)[-1]
        current_distance = float(distance(lat[i], lon[i], lat[anchor], lon[anchor]) / 1000)
        records.append((i, anchor, last_close, current_distance))
    return times, lat, lon, records


def _return_cycle(times, lat, lon, start, anchor, rules):
    if start >= len(times):
        return None
    near = distance(lat, lon, lat[anchor], lon[anchor]) / 1000 <= rules["site_radius_km"]
    for i in np.flatnonzero(near & (np.arange(len(times)) >= start)):
        end = int(np.searchsorted(times, times[i] + rules["return_sustained_days"] * DAY))
        if end >= len(times):
            break
        if end - i + 1 < rules["minimum_return_observed_days"] or times[end] - times[i] > (rules["return_sustained_days"] + rules["maximum_gap_days"]) * DAY:
            continue
        if np.diff(times[i:end + 1]).max(initial=0) > rules["maximum_gap_days"] * DAY:
            continue
        if near[i:end + 1].mean() >= rules["minimum_far_fraction"] and near[end]:
            return i, end
    return None


def _episodes(daily, animal, scope, rules, anchors=None, details=True):
    if not len(daily):
        return []
    times, lat, lon, records = _anchor_records(daily, rules) if anchors is None else anchors
    output, skip_until = [], -1
    for i, anchor, last_close, first_distance in records:
        if i <= skip_until or first_distance < rules["shift_distance_km"]:
            continue
        end = int(np.searchsorted(times, times[i] + rules["sustained_days"] * DAY))
        if end >= len(times) or times[end] - times[i] > (rules["sustained_days"] + rules["maximum_gap_days"]) * DAY:
            continue
        observed = end - i + 1
        if observed < max(3, int(np.ceil(rules["sustained_days"] * rules["minimum_sustained_observed_fraction"])) + 1):
            continue
        longest_gap = np.diff(times[i:end + 1]).max(initial=0) / DAY
        if longest_gap > rules["maximum_gap_days"]:
            continue
        away = distance(lat[i:end + 1], lon[i:end + 1], lat[anchor], lon[anchor]) / 1000
        far = away >= rules["shift_distance_km"]
        if far.mean() < rules["minimum_far_fraction"] or not far[-1]:
            continue
        skip_until = end
        if not details:
            output.append(True)
            continue
        endpoint_left = max(i, int(np.searchsorted(times, times[end] - rules["endpoint_site_window_days"] * DAY)))
        endpoint = _observed_site(lat[endpoint_left:end + 1], lon[endpoint_left:end + 1]) + endpoint_left
        end_spread = distance(lat[endpoint_left:end + 1], lon[endpoint_left:end + 1], lat[endpoint], lon[endpoint]) / 1000
        returned = _return_cycle(times, lat, lon, end + 1, anchor, rules)
        unique = f"{animal}:{scope}:{times[i]}:{rules['shift_distance_km']}:{rules['sustained_days']}"
        output.append({"episode_id": hashlib.sha256(unique.encode()).hexdigest()[:20],
            "canonical_animal": animal, "species": str(daily.species.iloc[0]), "study": str(daily.study.iloc[0]),
            "scope": scope, "proxy_kind": "sustained_observed_site_shift_needs_validation",
            "verified_migration": False, "retrospective_only": True,
            "anchor_time": daily.timestamp.iloc[anchor], "anchor_lat": float(lat[anchor]), "anchor_lon": float(lon[anchor]),
            "last_prior_site_observation": daily.timestamp.iloc[last_close], "first_shift_observation": daily.timestamp.iloc[i],
            "sustained_end_observation": daily.timestamp.iloc[end], "origin_year": int(daily.timestamp.iloc[i].year),
            "end_year": int(daily.timestamp.iloc[end].year), "sustained_days": float((times[end] - times[i]) / DAY),
            "observed_shift_days": observed, "longest_shift_gap_days": float(longest_gap),
            "far_observation_fraction": float(far.mean()), "shift_distance_km": float(away[-1]),
            "maximum_displacement_km": float(away.max()), "end_lat": float(lat[end]), "end_lon": float(lon[end]),
            "compact_endpoint_site_proxy": bool(np.quantile(end_spread, .9) <= rules["site_radius_km"]),
            "observed_return_cycle_proxy": returned is not None,
            "first_return_observation": daily.timestamp.iloc[returned[0]] if returned else pd.NaT,
            "return_confirmed_observation": daily.timestamp.iloc[returned[1]] if returned else pd.NaT})
    return output


def _year_record(group, coordinates):
    ordered = group.sort_values("timestamp", kind="stable")
    daily = _daily(ordered)
    times = ordered.timestamp.astype("int64").to_numpy() / 1e9
    months = daily.timestamp.dt.month
    span = float((times[-1] - times[0]) / DAY)
    row = {"canonical_animal": str(group.animal.iloc[0]), "species": str(group.species.iloc[0]),
           "study": str(group.study.iloc[0]), "year": int(group.timestamp.dt.year.iloc[0]),
           "scope": str(group.scope.iloc[0]), "first_observation": ordered.timestamp.iloc[0],
           "last_observation": ordered.timestamp.iloc[-1], "observations": len(ordered),
           "observed_calendar_days": len(daily), "observed_span_days": span,
           "longest_observation_gap_days": float(np.diff(times).max(initial=0) / DAY),
           "observed_months": int(months.nunique()), "observed_calendar_quarters": int(((months - 1) // 3).nunique()),
           "calendar_months": ",".join(map(str, sorted(months.unique()))),
           "coverage_heuristic_pass": bool(span >= 120 and len(daily) >= 60 and months.nunique() >= 4),
           "coordinate_audit_available": coordinates}
    row.update({f"month_{month:02d}_observed_days": int((months == month).sum()) for month in range(1, 13)})
    if coordinates:
        lat, lon = daily.lat.to_numpy(), daily.lon.to_numpy()
        start = distance(lat, lon, lat[0], lon[0]) / 1000
        end = distance(lat, lon, lat[-1], lon[-1]) / 1000
        row.update({"first_to_last_observed_distance_km": float(start[-1]),
                    "maximum_observed_nsd_km2": float((start ** 2).max()),
                    "median_observed_nsd_km2": float(np.median(start ** 2)),
                    "last_observed_nsd_km2": float(start[-1] ** 2),
                    "observed_days_near_first_site": int((start <= DEFAULTS["site_radius_km"]).sum()),
                    "observed_days_near_last_site": int((end <= DEFAULTS["site_radius_km"]).sum()),
                    "last_site_first_observed_visit": daily.timestamp.iloc[np.flatnonzero(end <= DEFAULTS["site_radius_km"])[0]],
                    "observed_first_site_return_after_100km": bool(np.any((start <= 25) & (np.maximum.accumulate(start) >= 100)))})
    return row


def audit(observations, output, manifest=None):
    """Audit coverage and allowed development movement; never score forecasts.

    A manifest excludes whole reserved individuals, unknown identities, and all
    observations after the development calendar boundary. Temporal training and
    validation-year episodes are built separately, so training proxy labels do
    not reach into validation. With no manifest, coordinates are not even read.
    """
    source, out = Path(observations), Path(output)
    if source.is_dir():
        source = source / "processed/observations.parquet"
    if (out / "report.json").exists():
        raise FileExistsError("Preserve episode audits: choose a new output directory")
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        metadata = json.loads(manifest_path.read_text())
        manifest_hash = _sha256(manifest_path)
    else:
        metadata = manifest
        manifest_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest() if metadata else None
    coordinates = metadata is not None
    columns = ["animal", "species", "study", "timestamp"] + (["lat", "lon"] if coordinates else [])
    frame = pd.read_parquet(source, columns=columns)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    frame = frame.loc[frame.timestamp.notna() & frame.animal.notna() & frame.species.notna()].copy()
    excluded = {"reserved_individual_rows": 0, "unassigned_identity_rows": 0, "after_development_boundary_rows": 0}
    if coordinates:
        if "assignments" not in metadata or "validation_year" not in metadata or "train_end_year" not in metadata:
            raise ValueError("Episode coordinate auditing requires assignments and development year boundaries")
        frame["animal"] = frame.animal.map(lambda name: metadata.get("identity_mapping", {}).get(name, name))
        assignment = frame.animal.map(metadata["assignments"])
        reserved = assignment.isin(["test_individual", "test_year"])
        unknown = assignment.isna()
        boundary = pd.Timestamp(int(metadata["validation_year"]) + 1, 1, 1, tz="UTC")
        after = frame.timestamp >= boundary
        excluded = {"reserved_individual_rows": int(reserved.sum()), "unassigned_identity_rows": int(unknown.sum()),
                    "after_development_boundary_rows": int((after & ~reserved & ~unknown).sum())}
        selected = ~reserved & ~unknown & ~after
        frame = frame.loc[selected].copy()
        assignment = assignment.loc[selected]
        cutoff = pd.Timestamp(int(metadata["train_end_year"]) + 1, 1, 1, tz="UTC")
        frame["scope"] = np.where(assignment.eq("val_individual"), "validation_individual",
                                  np.where(frame.timestamp < cutoff, "train", "validation_year"))
        valid = frame.lat.between(-90, 90) & frame.lon.between(-180, 180)
        excluded["invalid_allowed_coordinates"] = int((~valid).sum())
        frame = frame.loc[valid].copy()
    else:
        frame["scope"] = "metadata_only_without_split_manifest"
    frame = frame.sort_values(["animal", "timestamp"], kind="stable").drop_duplicates(["animal", "timestamp"])
    rows, candidates = [], []
    sensitivity = {f"{km:g}km_{days:g}days": 0 for km in [50., 100., 200.] for days in [7., 14., 30.]}
    for _, group in frame.groupby(["animal", frame.timestamp.dt.year], sort=True):
        rows.append(_year_record(group, coordinates))
    years = pd.DataFrame(rows)
    if not len(years):
        years = pd.DataFrame(columns=["canonical_animal", "species", "study", "year", "scope",
                                      "observed_calendar_days", "observed_span_days", "observed_months",
                                      "coverage_heuristic_pass", "previous_year_observed", "previous_year_coverage_pass"])
    if len(years):
        previous, next_year = {}, {}
        for animal, group in years.groupby("canonical_animal"):
            lookup = dict(zip(group.year, group.coverage_heuristic_pass))
            for index, row in group.iterrows():
                previous[index] = (row.year - 1 in lookup, bool(lookup.get(row.year - 1, False)))
                next_year[index] = (row.year + 1 in lookup, bool(lookup.get(row.year + 1, False)))
        years["previous_year_observed"] = [previous[i][0] for i in years.index]
        years["previous_year_coverage_pass"] = [previous[i][1] for i in years.index]
        years["next_year_observed_retrospective"] = [next_year[i][0] for i in years.index]
        years["next_year_coverage_pass_retrospective"] = [next_year[i][1] for i in years.index]
    if coordinates:
        for (animal, scope), group in frame.groupby(["animal", "scope"], sort=True):
            daily = _daily(group)
            anchors = _anchor_records(daily, DEFAULTS)
            candidates.extend(_episodes(daily, str(animal), str(scope), DEFAULTS, anchors))
            for km in [50., 100., 200.]:
                for days in [7., 14., 30.]:
                    rules = {**DEFAULTS, "shift_distance_km": km, "sustained_days": days}
                    sensitivity[f"{km:g}km_{days:g}days"] += len(_episodes(daily, str(animal), str(scope), rules, anchors, details=False))
    episodes = pd.DataFrame(candidates, columns=EPISODE_COLUMNS)
    coverage = {}
    for name, span, days, months in [("lenient", 90, 30, 3), ("declared", 120, 60, 4), ("strict", 180, 90, 6)]:
        good = ((years.observed_span_days >= span) & (years.observed_calendar_days >= days) & (years.observed_months >= months)) if len(years) else np.zeros(0, bool)
        coverage[name] = {"minimum_span_days": span, "minimum_observed_days": days, "minimum_months": months,
                          "animal_years": int(good.sum())}
    report = {"scope": "Retrospective movement proxies requiring validation; no verified migration labels or forecast accuracy assessment",
              "observations": str(source), "observations_sha256": _sha256(source), "manifest_sha256": manifest_hash,
              "code_sha256": _sha256(__file__), "model_inputs_generated": False,
              "coordinate_audit_enabled": coordinates, "development_end_exclusive": str(boundary) if coordinates else None,
              "excluded": excluded, "included_rows": len(frame), "included_animals": int(frame.animal.nunique()),
              "animal_years": len(years), "candidate_episodes": len(episodes),
              "observed_return_cycle_proxies": int(episodes.observed_return_cycle_proxy.sum()) if len(episodes) else 0,
              "consecutive_observed_year_pairs": int(years.previous_year_observed.sum()) if len(years) else 0,
              "consecutive_coverage_pass_pairs": int((years.coverage_heuristic_pass & years.previous_year_coverage_pass).sum()) if len(years) else 0,
              "predeclared_rules": DEFAULTS, "coverage_sensitivity": coverage,
              "shift_sensitivity_counts": sensitivity if coordinates else None,
              "notes": ["No forecast, checkpoint, error or model score is used to select an animal or interval.",
                        "Calendar coverage is a transparent heuristic, not proof that an entire biological season was observed.",
                        "Calendar quarters do not encode species-specific breeding or migration seasons.",
                        "Coordinates are excluded for reserved individuals and unassigned identities; future years after the development boundary are excluded.",
                        "Without a split manifest no coordinates are read, and no NSD/site/episode proxies are computed.",
                        "Daily positions are the first ACTUAL observed fix of each UTC day; no interpolation or averaging creates GPS labels.",
                        "Prior anchors are actual observed positions selected using only earlier observations, with a compact prior fortnight requirement.",
                        "Sustained shifts and return confirmations use later allowed observations retrospectively; they are diagnostic labels and must not be prediction inputs.",
                        "Training and validation-year episodes are generated separately so training diagnostic labels do not cross the global calendar cutoff.",
                        "Annual NSD references that year's first observed fix; first/last sites are sampling anchors, not established home/breeding/wintering sites.",
                        "Site radii and displacement/duration thresholds are predeclared physical heuristics; sensitivity counts do not select a winning threshold.",
                        "A compact endpoint flag or an observed return is not proof of arrival, migration or behavioural state.",
                        "Identity exclusion inherits the manifest's conservative alias audit and cannot establish that every biological identity is resolved."]}
    out.mkdir(parents=True, exist_ok=True)
    years.to_csv(out / "animal-year.csv", index=False)
    episodes.to_parquet(out / "candidate-episodes.parquet", index=False)
    dump(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    print(json.dumps(audit(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
