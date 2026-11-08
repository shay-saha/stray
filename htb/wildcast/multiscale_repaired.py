"""Observed 1-hour–90-day cases with certified physical-history repair.

Frozen parent bytes are ancestry evidence.  Unsupported physical joins are
retracted while all old exclusion/role/freshness protections survive.  Every
old case is retained, reassigned to an exact origin owner, or explicitly retired
for source-QC/support reasons.  Reserved outcomes stay sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from . import migration_data as legacy
from . import migration_physical_repair as repair
from . import multiscale_data as previous
from .data import distance, project
from .migration_repaired_features import (
    HORIZONS, LAGS, HISTORY_NAMES, STATIC_NAMES, MINIMUM_PRIOR_FIXES,
    MINIMUM_HISTORY_SPAN_DAYS, MINIMUM_HISTORY_LAGS, LEGACY_LAG_COLUMNS,
    features, query_features, past_indices,
)
from .multiscale_data import target_tolerances, target_admission_mask, match_targets


PROTOCOL = "wildcast-multiscale-repaired-v2"
DAY, UNIT, SPLITS = legacy.DAY, legacy.UNIT, legacy.SPLITS
COLUMNS = ["timestamp", "lat", "lon", "animal", "species", "study", "source"]


def select_source_rows(frame, source_sha256, manifest):
    """Pure source-row ownership selection, shared with assessment derivation.

    This applies frozen raw-ID selectors only, never future coordinates,
    displacements, phases or errors.  Original paths are not required at
    inference.  Retired mixed parent rows stay excluded even when their name is
    also a valid recovered original identity.
    """
    rules = [r for r in manifest.get("observation_source_rules", []) if r["sha256"] == source_sha256]
    if len(rules) != 1 or "animal" not in frame:
        raise ValueError("Exactly one content-bound observation ownership rule required")
    rule = rules[0]
    names = frame.animal
    allowed = set(rule["allowed_raw_animals"])
    excluded = set(rule["excluded_raw_animals"])
    if set(names) - allowed - excluded:
        raise ValueError("Observation source contains raw identities absent its frozen ownership rule")
    keep = names.isin(allowed) & ~names.isin(excluded)
    for interval in rule.get("excluded_intervals", []):
        if (interval.get("closed") != "both" or not interval.get("evidence_sha256") or
                not interval.get("reason") or "timestamp" not in frame):
            raise ValueError("Source-QC interval needs explicit inclusive bounds/evidence")
        numeric = [isinstance(interval[k], (int,float,np.integer,np.floating)) for k in ["start_time","end_time"]]
        if any(numeric):
            if not all(numeric) or interval.get("time_unit") != "unix_seconds_on_nominal_axis":
                raise ValueError("Numeric source-QC bounds require explicit nominal Unix-second units")
            start, end = [pd.Timestamp(interval[k], unit="s", tz="UTC") for k in ["start_time","end_time"]]
        else:
            start, end = pd.Timestamp(interval["start_time"]), pd.Timestamp(interval["end_time"])
        if start.tz is None or end.tz is None or start > end:
            raise ValueError("Source-QC interval needs ordered explicitly labelled clock bounds")
        ns = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit("ns").astype("int64")
        quarantined = names.eq(interval["input_animal"]) & ns.between(start.value, end.value)
        if ("rows_in_this_occurrence" in interval and
                int(quarantined.sum()) != int(interval["rows_in_this_occurrence"])):
            raise ValueError("Frozen source-QC interval row count differs on its exact nominal clock axis")
        keep &= ~quarantined
    native = rule.get("excluded_native_times", [])
    if native:
        if "timestamp" not in frame:
            raise ValueError("Native timestamp source-QC selectors need actual reported times")
        keys, individuals = [], set()
        for item in native:
            values = item.get("timestamps_ns")
            individual = item.get("input_animal")
            if (individual in individuals or individual not in allowed or
                    item.get("time_axis") != "reported_nominal" or not item.get("evidence_sha256") or
                    not item.get("reason") or not isinstance(values,list) or not values or
                    any(type(v) is not int or v <= 0 or v > np.iinfo(np.int64).max for v in values) or
                    any(a >= b for a,b in zip(values,values[1:]))):
                raise ValueError("Native timestamp exclusions need unique ordered integer keys and explicit source-QC provenance")
            individuals.add(individual)
            keys.extend((individual,v) for v in values)
        positions = np.flatnonzero(names.isin(individuals).to_numpy())
        selected_ns = pd.to_datetime(frame.timestamp.iloc[positions],utc=True).dt.as_unit("ns").astype("int64")
        actual = pd.MultiIndex.from_arrays([names.iloc[positions].to_numpy(),selected_ns.to_numpy()])
        marked = actual.isin(pd.MultiIndex.from_tuples(keys))
        counts = names.iloc[positions[marked]].value_counts()
        for item in native:
            if ("rows_in_this_occurrence" in item and
                    int(counts.get(item["input_animal"],0)) != int(item["rows_in_this_occurrence"])):
                raise ValueError("Native timestamp quarantine row count differs from frozen source audit")
        dropped = np.zeros(len(frame),dtype=bool); dropped[positions[marked]]=True
        keep &= ~dropped
    records = rule.get("excluded_input_records", [])
    if records:
        if not frame.index.equals(pd.RangeIndex(len(frame))):
            raise ValueError("Record source-QC selectors require the complete input in original Parquet row order")
        dropped = np.zeros(len(frame),dtype=bool)
        for item in records:
            values = item.get("row_positions")
            if (item.get("row_number_basis") != "zero_based_full_input_order" or
                    type(item.get("input_rows")) is not int or item["input_rows"] != len(frame) or
                    not item.get("evidence_sha256") or not item.get("reason") or
                    not isinstance(values,list) or not values or
                    any(type(v) is not int or not 0 <= v < len(frame) for v in values) or
                    any(a >= b for a,b in zip(values,values[1:]))):
                raise ValueError("Record exclusions need full-input size, ordered integer positions and frozen source-QC proof")
            if dropped[values].any():
                raise ValueError("Overlapping certified input-record exclusions")
            dropped[values] = True
        keep &= ~dropped
    return frame.loc[keep].copy()


def _owned_observations(paths, evidence, contract, *, parent):
    records_by_hash = defaultdict(list)
    for row in contract["raw_coordinate_ownership"]:
        if row["status"] not in {"unmixed_original_source_id", "already_collapsed_needs_original_recovery"}:
            raise ValueError("Unknown original coordinate-ownership status")
        records_by_hash[row["input_sha256"]].append(row)
    outputs = repair.recovery_outputs(contract)
    source_hashes = {r["sha256"] for r in [*evidence,*outputs]}
    fields = {"excluded_intervals":"source_interval_exclusions",
              "excluded_native_times":"source_native_timestamp_exclusions",
              "excluded_input_records":"source_record_exclusions"}
    for field in fields.values():
        if any(row.get("input_sha256") not in source_hashes for row in contract.get(field, [])):
            raise ValueError("Source-QC selectors refer to an absent or changed frozen input hash")
    def extra_rules(digest):
        return {name:[{k:v for k,v in row.items() if k != "input_sha256"}
                      for row in contract.get(field,[]) if row["input_sha256"] == digest]
                for name,field in fields.items()}
    rules, frames, frozen_inputs = [], [], []
    parent_hashes = {r["sha256"] for r in parent["manifest"]["sources"]}
    for path, source in zip(paths, evidence):
        records = records_by_hash[source["sha256"]]
        if not records:
            raise ValueError("Repair contract does not bind an observation input")
        names = [r["input_animal"] for r in records]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate coordinate-ownership input selectors")
        excluded = sorted(r["input_animal"] for r in records if r["status"] != "unmixed_original_source_id")
        allowed = sorted(set(names)-set(excluded))
        rules.append({"sha256": source["sha256"], "allowed_raw_animals": allowed,
                      "excluded_raw_animals": excluded, **extra_rules(source["sha256"]),
                      "kind": "original_QC_input"})
        frame = pd.read_parquet(path, columns=COLUMNS)
        if source["sha256"] in parent_hashes:
            frozen_inputs.append((source, frame))
        if set(frame.animal) != set(names):
            raise ValueError("Actual input identities differ from audited coordinate ownership")
        frames.append(select_source_rows(frame, source["sha256"], {"observation_source_rules": rules}))
    for record in outputs:
        if record["sha256"] in {r["sha256"] for r in rules}:
            raise ValueError("Recovered observations duplicate an existing input hash")
        rule = {"sha256": record["sha256"], "allowed_raw_animals": sorted(record["raw_ids"]),
                "excluded_raw_animals": [], **extra_rules(record["sha256"]),
                "kind": "recovered_original_trajectories"}
        rules.append(rule)
        frame = pd.read_parquet(record["path"], columns=COLUMNS)
        if set(frame.animal) != set(record["raw_ids"]):
            raise ValueError("Recovered output is missing a certified original identity")
        frames.append(select_source_rows(frame, record["sha256"], {"observation_source_rules": rules}))
        evidence.append({"path": record["path"], "sha256": record["sha256"],
                         "kind": "recovered_original_trajectories",
                         "source_QC_provenance": record["source_QC_provenance"],
                         "clock_provenance": record.get("clock_provenance", "Inherited documentary clock proof; no precision upgrade")})
    frame, lineage = repair.canonicalize_owned(pd.concat(frames, ignore_index=True), contract)
    return frame, lineage, rules, frozen_inputs


def _exclusions(frame, contract):
    physical, risk = contract["physical_mapping"], contract["exclusion_mapping"]
    exclusion = {}
    by_risk = defaultdict(set)
    for raw, canonical in physical.items():
        if canonical in exclusion and exclusion[canonical] != risk[raw]:
            raise ValueError("Repaired physical identity crosses exclusion groups")
        exclusion[canonical] = risk[raw]
        by_risk[risk[raw]].add(canonical)
    observed = set(frame.animal)
    exclusion = {n: exclusion[n] for n in observed}
    ambiguous = {n for names in by_risk.values() if len(names) > 1 for n in names} & observed
    ambiguous.update(physical[n] for n in contract.get("freshness_exclusion_animals", []) if physical.get(n) in observed)
    return exclusion, ambiguous


def _clock_context(frame, parent, bindings, evidence, parent_clock_audit, train_end_year, validation_year):
    """Use certified source clocks for recovered IDs with no old case ancestry.

    Existing unknown parent ancestors still block every subday target.  A
    recovered source does not become undocumented simply because an earlier
    cleaner removed all cases for that particular ID.
    """
    documented = {}
    if parent_clock_audit:
        supplemental = previous.validate_parent_clock_evidence(parent_clock_audit, parent)
        raw = json.loads(Path(supplemental["raw_clock_audit"]["path"]).read_text())
        for source, decision in supplemental["source_decisions"].items():
            check = raw.get("sources", {}).get(source, {})
            if (check.get("status") == "strict_clock_validated" and
                    check.get("source_policy_sha256") == decision.get("source_policy_sha256") and
                    decision.get("mode") in {"declared_utc", "bounded_utc_or_gps"}):
                documented[source] = decision["mode"]
    context, provenance = previous._subday_context(
        frame, parent, bindings, [*evidence, {"source_clock_modes":documented}],
        parent_clock_audit, train_end_year, validation_year)
    context["recovered_source_clock_modes"] = documented
    context["subday_clock_rule"] += " Recovered IDs without old canonical ancestors require the same content-bound strict raw-source clock audit and documentary policy; absence of an old case alone does not invalidate documented source clocks."
    return context, provenance


def has_future_timestamp(times, origin, *, boundary=np.inf, canonical_animal, clock_context,
                         horizons=HORIZONS):
    """Scalar timestamp-only eligibility; stop at the first admitted horizon.

    No future coordinate or per-horizon target mask is computed or returned.
    This is the only future-support check used for a reserved case or its
    retirement accounting.  It uses exactly the same nearest/later-tie rule as
    ``match_targets`` without constructing a target vector.
    """
    times = np.asarray(times, dtype=float)
    if not 0 <= origin < len(times):
        raise ValueError("Valid timestamp origin required")
    tolerances = target_tolerances(horizons)
    for horizon, tolerance in zip(horizons, tolerances):
        wanted = times[origin]+horizon*DAY
        right = min(int(np.searchsorted(times, wanted)), len(times)-1)
        left = max(right-1, 0)
        index = left if abs(times[left]-wanted) < abs(times[right]-wanted) else right
        if index <= origin or times[index] >= boundary or abs(times[index]-wanted) > tolerance*DAY:
            continue
        if target_admission_mask([horizon], canonical_animal, clock_context,
                                 origin_time=times[origin], target_times=[times[index]])[0]:
            return True
    return False


class _PastJumpQC:
    """Update observed speed diagnostics only through the current issue fix."""
    def __init__(self, lat, lon, times):
        self.lat, self.lon, self.times = lat, lon, times
        self.done = 0
        self.prefix = np.zeros(len(times), dtype="int64")

    def valid(self, origin):
        if origin < self.done:
            raise ValueError("Past-only QC origins must advance monotonically")
        if origin > self.done:
            a, b = self.done, origin
            speed = distance(self.lat[a+1:b+1], self.lon[a+1:b+1],
                             self.lat[a:b], self.lon[a:b])/np.maximum(np.diff(self.times[a:b+1]), 1.)
            self.prefix[a+1:b+1] = self.prefix[a]+np.cumsum(speed > 100.)
            self.done = origin
        left = int(np.searchsorted(self.times, self.times[origin]-7*DAY))
        return self.prefix[origin] == self.prefix[left]


def _support_comparison(parent, accounting, arrays):
    """Compare corrected development support, explicitly permitting retirement."""
    old_h = np.asarray(parent["manifest"]["horizons_days"])
    columns = np.array([int(np.flatnonzero(HORIZONS == h)[0]) for h in old_h])
    development = accounting.loc[~accounting.parent_split.isin(["test_individual", "test_year"])]
    retained = development.loc[development.child_case_id.notna()]
    with np.load(parent["paths"]["cases"], allow_pickle=False) as a:
        before = a["target_mask"][retained.parent_case_id.to_numpy(dtype=int)]
        original = a["target_mask"][development.parent_case_id.to_numpy(dtype=int)]
    after = arrays["target_mask"][retained.child_case_id.to_numpy(dtype=int)][:, columns]
    return {"horizons_days": old_h.tolist(), "parent_development_targets": original.sum(0).tolist(),
            "retained_case_original_targets": before.sum(0).tolist(),
            "retained_case_corrected_targets": after.sum(0).tolist(),
            "lost_on_retained_development_cases": (before & ~after).sum(0).tolist(),
            "gained_on_retained_development_cases": (~before & after).sum(0).tolist(),
            "retired_development_cases": int((development.status == "retired").sum()),
            "limitation": "Old spliced labels are superseded rather than preserved as biological truth. Reserved future coordinates and per-horizon support are not inspected or disclosed."}


def prepare(roots, output, *, parent_cache, repair_contract, parent_clock_audit=None,
            existing_manifest=None, train_end_year=2020, validation_year=2021, seed=1701,
            horizons=HORIZONS, progress=None):
    """Prepare the repaired protocol into a new immutable, sealed directory."""
    if validation_year != train_end_year+1:
        raise ValueError("Validation immediately follows the pinned training years")
    if not np.array_equal(np.asarray(horizons, dtype=float), HORIZONS):
        raise ValueError("The repaired protocol freezes its nine horizons")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new immutable repaired output directory")
    parent = previous._load_parent(parent_cache, train_end_year, validation_year)
    contract, repair_provenance = repair.load_contract(repair_contract, parent)
    paths, evidence = previous._input_paths(roots, parent, train_end_year, validation_year)
    if progress: progress("verified_sources", {"source_tables":len(paths)})
    frame, lineage, source_rules, frozen_inputs = _owned_observations(paths, evidence, contract, parent=parent)
    if progress: progress("owned_observations", {"rows":len(frame),"motion_groups":frame.animal.nunique()})
    exclusion, ambiguous = _exclusions(frame, contract)
    pins = repair.inherited_protections(parent, contract)
    bindings = [{"child_animal": child, "parent_animal": ancestor}
                for child, ancestors in pins["ancestors"].items() for ancestor in ancestors]
    clock_context, clock_provenance = _clock_context(
        frame, parent, bindings, evidence, parent_clock_audit, train_end_year, validation_year)
    ownership = repair.resolve_parent_origins(parent, lineage, contract, frozen_inputs=frozen_inputs)
    del lineage, frozen_inputs
    if progress: progress("parent_origin_ownership", {"parent_cases":len(ownership),
        "unresolved_origin_reasons":ownership.ownership_reason.loc[ownership.ownership_reason.ne("")].value_counts().to_dict()})
    forced = defaultdict(set)
    for row in ownership.loc[ownership.ownership_reason == ""].itertuples():
        forced[row.child_canonical_animal].add(pd.Timestamp(row.origin_time).value)
    previous_names = set(json.loads(Path(existing_manifest).read_text())["animals"]) if existing_manifest else set()
    previous_groups = {contract["exclusion_mapping"][n] for n in previous_names if n in contract["exclusion_mapping"]}
    group_ids = {name: i for i, name in enumerate(sorted(set(exclusion.values())))}
    animals = {name: group_ids[exclusion[name]] for name in sorted(frame.animal.unique())}
    cross_taxon = previous._cross_taxon_declarations(frame, exclusion, ambiguous, animals)
    species_names = sorted(frame.species.unique())
    species = {name: i+1 for i, name in enumerate(species_names)}
    studies = {name: i for i, name in enumerate(sorted(frame.study.unique()))}
    cutoff = pd.Timestamp(f"{train_end_year+1}-01-01", tz="UTC").timestamp()
    val_end = pd.Timestamp(f"{validation_year+1}-01-01", tz="UTC").timestamp()
    arrays_list = defaultdict(list)
    rows, animal_audit, failures, assignments = [], [], {}, {}
    skipped = Counter()
    clock_withheld = defaultdict(lambda: np.zeros(len(HORIZONS), dtype="int64"))
    for animal, g in frame.groupby("animal", sort=True):
        if progress and len(animal_audit)%100 == 0:
            progress("building_cases", {"motion_groups_processed":len(animal_audit),
                                       "cases":len(rows),"skipped":dict(skipped)})
        g = g.sort_values("timestamp", kind="stable")
        lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
        dates = pd.DatetimeIndex(g.timestamp); ns = dates.asi8; times = ns/1e9; days = dates.dayofyear.to_numpy()
        span = (times[-1]-times[0])/DAY
        stride = .25 if span < 14. else 1.
        group = exclusion[animal]
        fresh = group not in previous_groups and animal not in ambiguous
        fresh = pins["freshness"].get(group, fresh)
        if animal in ambiguous:
            fresh = False
        bucket = int(hashlib.sha256(f"migration-v1:{seed}:{group}".encode()).hexdigest()[:12], 16)%100
        role = "test_individual" if fresh and bucket < 15 else "val_individual" if (15 <= bucket < 30 if fresh else bucket < 20) else "temporal"
        role = pins["roles"].get(group, role); assignments[animal] = role
        animal_audit.append({"canonical_animal": animal, "individual": animals[animal], "species": g.species.iloc[0],
                             "study": g.study.iloc[0], "observations": len(g), "first": dates[0], "last": dates[-1],
                             "span_days": span, "observed_years": len(np.unique(dates.year)), "stride_days": stride,
                             "fresh_identity": fresh, "assignment": role})
        last_regular = -np.inf; qc = _PastJumpQC(lat, lon, times)
        inherited_times = forced.get(animal, set())
        for i in range(len(g)):
            inherited = ns[i] in inherited_times
            key = animal, int(ns[i])
            if i < MINIMUM_PRIOR_FIXES or times[i]-times[0] < MINIMUM_HISTORY_SPAN_DAYS*DAY:
                if inherited: failures[key] = "invalid_repaired_physical_past"
                continue
            regular = times[i]-last_regular >= stride*DAY
            if not inherited and not regular:
                continue
            hist, hm, static, hll, ht = features(lat, lon, times, dates, i, days)
            if hm.sum() < MINIMUM_HISTORY_LAGS:
                if inherited: failures[key] = "invalid_repaired_physical_past"
                continue
            if regular:
                last_regular = times[i]
            if role == "test_individual": split, boundary = 3, np.inf
            elif role == "val_individual":
                if times[i] >= val_end:
                    if inherited: failures[key] = "source_QC_incompatible"
                    skipped["development_origins_after_year_boundary"] += 1; continue
                split, boundary = 1, val_end
            elif times[i] < cutoff: split, boundary = 0, cutoff
            elif times[i] < val_end: split, boundary = 2, val_end
            elif fresh or group in pins["reserved_temporal"]: split, boundary = 4, np.inf
            else:
                if inherited: failures[key] = "source_QC_incompatible"
                skipped["previously_inspected_future_years"] += 1; continue
            if not qc.valid(i):
                if inherited: failures[key] = "source_QC_incompatible"
                skipped["past_only_speed_jump_cases"] += 1; continue
            sealed = split >= 3
            if sealed:
                if not has_future_timestamp(times, i, boundary=boundary, canonical_animal=animal, clock_context=clock_context):
                    if inherited: failures[key] = "no_valid_repaired_target_support"
                    skipped["no_observed_future_target"] += 1; continue
                # Do not derive future coordinates or a per-H support vector.
                mask = np.zeros(len(HORIZONS), dtype=bool)
                y = np.zeros((len(HORIZONS), 2), dtype="float32")
                target_ll = np.zeros((len(HORIZONS), 2)); target_time = np.zeros(len(HORIZONS)); proxy = "sealed"
            else:
                ix, mask = match_targets(times, i)
                mask &= times[ix] < boundary
                admission = target_admission_mask(HORIZONS, animal, clock_context, origin_time=times[i], target_times=times[ix])
                clock_withheld[g.species.iloc[0]] += mask & ~admission
                mask &= admission
                if not mask.any():
                    if inherited: failures[key] = "no_valid_repaired_target_support"
                    skipped["no_observed_future_target"] += 1; continue
                y = (project(lat[ix], lon[ix], lat[i], lon[i])/UNIT).astype("float32")
                target_ll = np.column_stack([lat[ix], lon[ix]]); target_time = times[ix].copy()
                proxy = "over_100km" if np.linalg.norm(y[mask], axis=-1).max()*UNIT/1000 > 100 else "under_100km"
                y[~mask] = 0; target_ll[~mask] = 0; target_time[~mask] = 0
            legacy_history = times[i]-times[0] >= 7*DAY and hm[LEGACY_LAG_COLUMNS].sum() >= 5
            long_history = (times[i]-times[0] >= 30*DAY and hm[LEGACY_LAG_COLUMNS].sum() >= 5
                            and hm[int(np.flatnonzero(LAGS == 7)[0])])
            long_support = "sealed" if sealed else "yes" if long_history and mask[HORIZONS >= 30].any() else "no"
            for name, value in {"history": hist, "history_mask": hm, "static": static, "species": species[g.species.iloc[0]],
                                "y": y, "target_mask": mask, "split": split, "individual": animals[animal],
                                "study": studies[g.study.iloc[0]], "origin": [lat[i], lon[i], times[i]],
                                "target_latlon": target_ll, "target_time": target_time,
                                "history_latlon": hll, "history_time": ht}.items():
                arrays_list[name].append(value)
            rows.append({"case_id": len(rows), "canonical_animal": animal, "individual": animals[animal],
                         "species": g.species.iloc[0], "study": g.study.iloc[0], "origin_time": dates[i], "year": dates.year[i],
                         "animal_year": f"{animal}:{dates.year[i]}", "split": next(n for n,v in SPLITS.items() if v == split),
                         "fresh_identity": fresh, "targets_sealed": sealed, "displacement_proxy": proxy,
                         "memory365_available": bool(hm[-1]), "forced_parent_origin": inherited, "regular_origin": regular,
                         "legacy_history_support": bool(legacy_history), "long_history_support": bool(long_history),
                         "long_migration_support_proxy": long_support})
    if not rows:
        raise ValueError("No eligible corrected physical movement cases")
    arrays = {k: np.asarray(v) for k,v in arrays_list.items()}; origins = pd.DataFrame(rows)
    train_species = {name: i+1 for i,name in enumerate(sorted(set(origins.loc[origins.split == "train", "species"])))}
    remap = {old: train_species.get(name, 0) for name,old in species.items()}
    arrays["species"] = np.asarray([remap[int(v)] for v in arrays["species"]], dtype="int64")
    metric_species = {name: i for i,name in enumerate(species_names)}
    arrays["metric_species"] = np.asarray([metric_species[n] for n in origins.species], dtype="int64")
    for key in ["individual", "study"]:
        arrays[key] = arrays[key].astype("int64")
    arrays["split"] = arrays["split"].astype("int8"); arrays["horizons"] = HORIZONS.copy()
    accounting = repair.account_parent_cases(parent, ownership, origins, contract, failures)
    comparison = _support_comparison(parent, accounting, arrays)
    sealed = arrays["split"] >= 3
    if arrays["target_mask"][sealed].any() or any(arrays[k][sealed].any() for k in ["y", "target_latlon", "target_time"]):
        raise ValueError("Corrected reserved targets must remain sealed and zero")
    support_rows = []
    for (split, taxon), selected in origins.loc[~origins.targets_sealed].groupby(["split", "species"]):
        ids = selected.case_id.to_numpy()
        for j,horizon in enumerate(HORIZONS):
            mask = arrays["target_mask"][ids,j]; group = selected.loc[mask]
            support_rows.append({"split": split, "species": taxon, "horizon_days": horizon, "cases": int(mask.sum()),
                                 "physical_animals": group.canonical_animal.nunique(), "exclusion_groups": group.individual.nunique(),
                                 "long_history_cases": int(group.long_history_support.sum()),
                                 "long_migration_support_proxy_cases": int((group.long_migration_support_proxy == "yes").sum())})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name+".pending-", dir=output.parent))
    try:
        np.savez_compressed(stage/"cases.npz", **arrays); origins.to_parquet(stage/"origins.parquet", index=False)
        accounting.to_parquet(stage/"parent-case-accounting.parquet", index=False)
        pd.DataFrame(animal_audit).to_parquet(stage/"animal-audit.parquet", index=False)
        pd.DataFrame(support_rows).to_csv(stage/"development-support.csv", index=False)
        origins.groupby(["split", "species"]).agg(cases=("case_id", "size"), animals=("canonical_animal", "nunique"),
                                                  exclusion_groups=("individual", "nunique")).reset_index().to_csv(stage/"cohort-counts.csv", index=False)
        audits = [repair_provenance]+([clock_provenance] if clock_provenance else [])
        audit_hashes = {r["sha256"] for r in audits}
        for kind, binding in contract.get("inputs", {}).items():
            if binding["sha256"] not in audit_hashes:
                audits.append({"kind":"physical_repair_"+kind, **binding})
                audit_hashes.add(binding["sha256"])
        recovery_manifest = contract.get("original_recovery",{}).get("manifest")
        if recovery_manifest and recovery_manifest["sha256"] not in audit_hashes:
            audits.append({"kind":"original_trajectory_recovery", **recovery_manifest})
        selected_rows = contract.get("historical_selected_rows")
        if selected_rows:
            audits.extend([{"kind":"historical_selected_rows", "path":selected_rows["path"], "sha256":selected_rows["sha256"]},
                           {"kind":"historical_selection_metadata", **selected_rows["metadata"]}])
            selected_metadata = json.loads(Path(selected_rows["metadata"]["path"]).read_text())
            for kind in ["replay_code", "parent_metadata", "certification"]:
                if selected_metadata.get(kind):
                    audits.append({"kind":"historical_selection_"+kind, **selected_metadata[kind]})
        manifest = {"protocol": PROTOCOL, "preparation_version": PROTOCOL,
                    "horizons_days": HORIZONS.tolist(), "horizons": HORIZONS.tolist(),
                    "assessment_target_matcher": "wildcast.multiscale_repaired.match_targets", "unit_metres": UNIT,
                    "history_names": HISTORY_NAMES, "static_names": STATIC_NAMES, "history_lags_days": LAGS.tolist(),
                    "minimum_history_lags": MINIMUM_HISTORY_LAGS, "minimum_history_prior_fixes": MINIMUM_PRIOR_FIXES,
                    "minimum_history_span_days": MINIMUM_HISTORY_SPAN_DAYS, "minimum_observed_future_targets": 1,
                    "species": train_species, "metric_species": metric_species, "animals": animals, "studies": studies,
                    "splits": SPLITS, "assignments": assignments, "seed": seed,
                    "train_end_year": train_end_year, "validation_year": validation_year,
                    "cases": len(origins), "case_counts": origins.split.value_counts().to_dict(),
                    "training_taxa_with_any_target": len(train_species),
                    "target_counts": {name: arrays["target_mask"][arrays["split"] == value].sum(0).tolist() for name,value in SPLITS.items()},
                    "development_support": "development-support.csv", "development_support_sha256": legacy.sha256(stage/"development-support.csv"),
                    "sampling_rule": {"long_track_stride_days": 1., "short_track_stride_days": .25, "short_track_span_below_days": 14.,
                                      "minimum_strictly_prior_fixes": 8, "minimum_past_span_hours": 6, "minimum_valid_history_slots": 2,
                                      "minimum_observed_future_targets": 1,
                                      "parent_origins": "Union only exact source-owned origins that retain valid physical past/future timestamp support; all others receive explicit retirement accounting",
                                      "track_span_scope": "Retrospective timestamp-only cadence policy; not a predictor"},
                    "target_rule": "Nearest actual observation; H<1d tolerance=min(1h,25%H), H>=1d legacy rule; pinned cutoff purge; no interpolation",
                    "target_tie_rule": "Equal distance chooses the right/later observation",
                    "observation_tie_rule": "Frozen per-source raw-ID ownership selectors; reviewed physical map only; stable canonical animal/time/original-ID order; first canonical/time observation",
                    "history_rule": "Actual at-or-before lag observations; short-lag quarter-lag tolerance capped1h; actual positive elapsed time for all velocities",
                    "source_QC_rule": "Original source QC plus issue-time past-only7d speed-jump diagnostic; no future path/error-based case filtering",
                    "reserved_eligibility_rule": "Timestamp-only scalar nearest-target availability, stopping on first admitted horizon; no future coordinates or per-horizon reserved support vector",
                    "reserved_test_targets_sealed": True, "sources": evidence,
                    "observation_source_rules": source_rules, "identity_mapping": contract["physical_mapping"],
                    "global_raw_source_interval_rules": contract.get("global_raw_source_interval_rules", []),
                    "global_raw_source_native_timestamp_rules": contract.get("global_raw_source_native_timestamp_rules", []),
                    "global_raw_source_record_policies": contract.get("global_raw_source_record_policies", []),
                    "identity_links": contract["retained_physical_primitives"],
                    "motion_only_duplicate_links": [r for r in contract["retained_physical_primitives"]
                                                    if r.get("type") == "complete_observed_track_duplicate"],
                    "motion_only_duplicate_rule": "Identical complete observed streams may share one coordinate history within these frozen inputs. This is not a permanent biological-marker identity assertion or permission to reconcile sex/life-stage metadata or transfer unrelated future deployments",
                    "identity_unit": "Reviewed physical coordinate histories; conservative exclusion groups remain shared for uncertain animals without coordinate merging",
                    "exclusion_groups": exclusion, "ambiguous_identity_records": sorted(ambiguous),
                    "cross_taxon_exclusion_groups": cross_taxon, **clock_context,
                    "parent_clock_audit": clock_provenance, "audits": audits,
                    "subday_clock_withheld_development_targets": {s:v.tolist() for s,v in clock_withheld.items() if v.any()},
                    "physical_repair": repair_provenance,
                    "existing_manifest": str(existing_manifest) if existing_manifest else None,
                    "existing_manifest_sha256": legacy.sha256(existing_manifest) if existing_manifest else None,
                    "cases_sha256": legacy.sha256(stage/"cases.npz"), "origins_sha256": legacy.sha256(stage/"origins.parquet"),
                    "skipped": dict(skipped),
                    "preparation_code_sha256": {name: legacy.sha256(Path(__file__).with_name(name)) for name in [
                        "multiscale_repaired.py", "migration_physical_repair.py", "migration_repaired_features.py",
                        "multiscale_data.py", "migration_data.py", "migration_clock.py", "parent_clock.py", "data.py"]},
                    "parent_cache": {"protocol": "wildcast-parent-physical-repair-accounted-v1", "path": parent["root"], **parent["hashes"],
                                     "physical_repair": repair_provenance, "parent_clock_audit": clock_provenance,
                                     "case_accounting": "parent-case-accounting.parquet",
                                     "case_accounting_sha256": legacy.sha256(stage/"parent-case-accounting.parquet"),
                                     "accounted_parent_cases": len(accounting), "case_status_counts": accounting.status.value_counts().to_dict(),
                                     "retirement_reason_counts": accounting.retirement_reason.dropna().value_counts().to_dict(),
                                     "shared_horizon_development_support": comparison,
                                     "reserved_protection": "Every historical exclusion/role/freshness link survives physical splitting. Every old case accounted exactly once; retained/reassigned cases keep split and cannot gain freshness; all reserved truth remains zero",
                                     "change_authorization": "Explicit corrected-physical-history protocol permits source-QC retirement of spliced parent cases and adds actual5m/15m history; frozen parent inputs remain unchanged"},
                    "freshness_status": "Candidate fresh only; uncertain/recovered identities cannot gain historical freshness",
                    "limitations": ["Sparse short-term tracking is not proof of seasonal migration skill",
                                    "Source clock standards do not establish device timing or spatial accuracy",
                                    "Coordinate coincidence alone never establishes individual identity",
                                    "All normalization remains training-only in the trainer",
                                    "No actual final assessment has been released by preparation"]}
        # Recheck every ancestry/evidence/source byte before atomic publication.
        for source in evidence:
            if legacy.sha256(source["path"]) != source["sha256"]:
                raise ValueError("Source observations changed during preparation")
        if legacy.sha256(repair_contract) != repair_provenance["sha256"]:
            raise ValueError("Physical repair artifact changed during preparation")
        repair.validate_contract(contract, parent)
        if clock_provenance:
            if legacy.sha256(parent_clock_audit) != clock_provenance["sha256"]:
                raise ValueError("Parent clock audit changed during preparation")
            previous.validate_parent_clock_evidence(parent_clock_audit, parent)
        repair.account_parent_cases(parent, ownership, origins, contract, failures)
        (stage/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")
        if output.exists():
            raise FileExistsError("Corrected output appeared during preparation")
        stage.rename(output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--parent-cache", required=True); parser.add_argument("--repair-contract", required=True)
    parser.add_argument("--parent-clock-audit"); parser.add_argument("--existing-manifest")
    parser.add_argument("--train-end-year", type=int, default=2020); parser.add_argument("--validation-year", type=int, default=2021)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    result = prepare(args.roots, args.output, parent_cache=args.parent_cache, repair_contract=args.repair_contract,
                     parent_clock_audit=args.parent_clock_audit, existing_manifest=args.existing_manifest,
                     train_end_year=args.train_end_year, validation_year=args.validation_year, seed=args.seed)
    print(json.dumps({k:result[k] for k in ["cases", "case_counts", "training_taxa_with_any_target"]}, indent=2))


if __name__ == "__main__":
    main()
