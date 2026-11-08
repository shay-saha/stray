"""Combine completed migration evaluations on identical frozen cases and draws.

This reads retained scores, not model weights or reserved data. It never averages
forecasts or mixes sampling protocols. Exactly identical baseline duplicates may
be deduplicated; learned model names must be unique across input reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wildcast.migration_evaluate import macro_scores, paired_animal_bootstrap


BASELINES = {"stationary", "movement", "crw", "seasonal", "seasonal_route"}
TARGET_KEYS = ["split", "case", "horizon_days"]
ORIGIN_KEYS = ["split", "case"]
IDENTITY = ["individual", "species", "study", "displacement_proxy"]
TIMING = ["actual_elapsed_days", "target_timing_error_hours"]
REQUIRED = {"model", *TARGET_KEYS, *IDENTITY, *TIMING, "endpoint_energy_km", "point_error_km",
            "representative_error_km", "radial_coverage50", "radial_coverage90", "within_1km", "within_10km", "within_100km"}
JOINT_REQUIRED = {"model", *ORIGIN_KEYS, *IDENTITY, "observed_horizons", "joint_chordal_energy_km"}
MATCH_METADATA = ("manifest_sha256", "cases_sha256", "case_ids_sha256", "samples", "seed", "batch_size", "splits", "cases", "observed_targets")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _inventory(root):
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _sorted(frame, keys):
    return frame.sort_values(keys).reset_index(drop=True).reindex(sorted(frame.columns), axis=1)


def _exact(left, right, keys, message):
    try:
        pd.testing.assert_frame_equal(_sorted(left, keys), _sorted(right, keys), check_exact=True,
                                      check_dtype=True, check_categorical=True)
    except AssertionError as exc:
        raise ValueError(message) from exc


def _model_provenance(metadata, name, checkpoints):
    if name in metadata.get("model_provenance", {}):
        return metadata["model_provenance"][name]
    return {"checkpoint": checkpoints.get(name),
            **{key: metadata.get(key) for key in ("code_sha256", "identity_unit", "samples", "seed",
                                                  "batch_size", "splits", "assessment", "metric_species_names", "notes")}}


def _read(root):
    metadata = json.loads((root / "metrics.json").read_text())
    for key in MATCH_METADATA:
        if key not in metadata:
            raise ValueError(f"Evaluation {root} lacks required {key}")
    if type(metadata["batch_size"]) is not int or metadata["batch_size"] < 1:
        raise ValueError("Evaluation batch_size must pin the positive RNG batch partition")
    frame = pd.read_parquet(root / "per-case.parquet")
    joint = pd.read_parquet(root / "joint-per-origin.parquet")
    if REQUIRED - set(frame) or JOINT_REQUIRED - set(joint):
        raise ValueError("Migration evaluation score/timing schema is incomplete")
    if frame.empty or joint.empty:
        raise ValueError("Cannot combine an empty migration evaluation")
    names = metadata.get("models", [])
    if len(names) != len(set(names)) or set(names) != set(frame.model) or set(names) != set(joint.model):
        raise ValueError("Migration model inventory differs from its score tables")
    if frame.duplicated(["model", *TARGET_KEYS]).any() or joint.duplicated(["model", *ORIGIN_KEYS]).any():
        raise ValueError("Duplicate model/case/horizon score rows")
    numeric = ["horizon_days", *TIMING, "endpoint_energy_km", "point_error_km", "representative_error_km"]
    if not np.isfinite(frame[numeric].to_numpy(float)).all() or not np.isfinite(joint.joint_chordal_energy_km).all():
        raise ValueError("Scored migration values and target timing must be finite")
    for column in ("radial_coverage50", "radial_coverage90", "within_1km", "within_10km", "within_100km"):
        if not np.isin(frame[column].to_numpy(), [False, True]).all():
            raise ValueError("Scored calibration/accuracy flags must be complete binary observations")
    if (frame.horizon_days <= 0).any() or (frame.actual_elapsed_days <= 0).any():
        raise ValueError("Scored migration horizons and elapsed times must be positive")
    expected_timing = (frame.actual_elapsed_days - frame.horizon_days) * 24
    if not np.allclose(expected_timing, frame.target_timing_error_hours, rtol=1e-12, atol=1e-8):
        raise ValueError("Target timing columns disagree")
    identity = [*IDENTITY, *(["prediction_species"] if "prediction_species" in frame else [])]
    if ("prediction_species" in frame) != ("prediction_species" in joint):
        raise ValueError("Endpoint/joint prediction-species metadata schemas differ")
    support = None; origin_support = None
    for name in names:
        per_model = frame.loc[frame.model == name]
        per_joint = joint.loc[joint.model == name]
        target = per_model[[*TARGET_KEYS, *identity, *TIMING]]
        origin = per_joint[[*ORIGIN_KEYS, *identity, "observed_horizons"]]
        actual_counts = per_model.groupby(ORIGIN_KEYS, observed=True).size().rename("observed_horizons").reset_index()
        expected_counts = per_joint[[*ORIGIN_KEYS, "observed_horizons"]]
        _exact(actual_counts, expected_counts, ORIGIN_KEYS, "Joint origins and endpoint target support differ")
        endpoint_identity = per_model[[*ORIGIN_KEYS, *identity]].drop_duplicates()
        if endpoint_identity.duplicated(ORIGIN_KEYS).any():
            raise ValueError("An origin changes identity or displacement stratum across horizons")
        _exact(endpoint_identity, per_joint[[*ORIGIN_KEYS, *identity]], ORIGIN_KEYS,
               "Endpoint/joint physical identity metadata differ")
        if support is None:
            support, origin_support = target, origin
        else:
            _exact(support, target, TARGET_KEYS, "Models do not have identical endpoint/timing support")
            _exact(origin_support, origin, ORIGIN_KEYS, "Models do not have identical joint-origin support")
    if origin_support.case.duplicated().any():
        raise ValueError("A case ID is assigned to multiple scored splits")
    ids = np.sort(origin_support.case.to_numpy(dtype="<i8"))
    actual_hash = hashlib.sha256(ids.tobytes()).hexdigest()
    if actual_hash != metadata["case_ids_sha256"] or len(ids) != metadata["cases"] or len(support) != metadata["observed_targets"]:
        raise ValueError("Scored support differs from the declared origin/target fingerprint")
    if set(metadata["splits"]) != set(origin_support.split):
        raise ValueError("Scored splits differ from their declared split list")
    # Verify the retained timing summary too; CSV floating serialization needs a
    # small numerical tolerance, while raw parquet timing matches exactly above.
    timing = support.groupby(["split", "horizon_days"], observed=True).target_timing_error_hours.agg(
        cases="size", mean_hours="mean", minimum_hours="min", maximum_hours="max").reset_index()
    retained = pd.read_csv(root / "target-timing.csv")
    left, right = timing.sort_values(["split", "horizon_days"]).reset_index(drop=True), retained.sort_values(["split", "horizon_days"]).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right) or not left.split.equals(right.split):
        raise ValueError("Retained target timing summary has different support")
    for column in left.columns[1:]:
        if not np.allclose(left[column], right[column], rtol=1e-12, atol=1e-10):
            raise ValueError("Retained target timing summary differs from scored targets")
    checkpoints = {}
    for record in metadata.get("checkpoints", []):
        name = record.get("name")
        if (name not in names or name in checkpoints or not record.get("path") or not record.get("sha256")
                or not record.get("model_kind") or not record.get("selection_version")):
            raise ValueError("Checkpoint provenance is missing, duplicated or inconsistent")
        checkpoints[name] = record
    if any(name not in BASELINES and name not in checkpoints for name in names):
        raise ValueError("A learned model has no checkpoint provenance")
    return metadata, frame, joint, support, origin_support, checkpoints


def combine(evaluations, output, reference="seasonal", bootstrap_replicates=500):
    if not evaluations or bootstrap_replicates < 1:
        raise ValueError("Provide evaluations and a positive bootstrap replicate count")
    roots = [Path(p).resolve() for p in evaluations]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate source evaluation directories")
    out = Path(output).resolve()
    if any(out == root or out.is_relative_to(root) for root in roots):
        raise ValueError("Combined output must be outside its source evaluations")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory to preserve completed comparisons")
    inventories = [_inventory(root) for root in roots]
    sources = [_read(root) for root in roots]
    canonical = sources[0][0]
    for metadata, _, _, support, origin, _ in sources[1:]:
        for key in MATCH_METADATA:
            if _canonical(metadata[key]) != _canonical(canonical[key]):
                raise ValueError(f"Cannot combine different evaluation {key}")
        _exact(sources[0][3], support, TARGET_KEYS, "Source reports have different endpoint/timing/identity support")
        _exact(sources[0][4], origin, ORIGIN_KEYS, "Source reports have different joint-origin support")
    assessments = [s[0].get("assessment") for s in sources]
    frozen = [str(s[0].get("scope", "")).startswith("Frozen local migration final assessment") or
              str(s[0].get("scope", "")).startswith("Combined frozen local migration final assessment") for s in sources]
    has_tests = any(str(s).startswith("test_") for s in canonical["splits"])
    if has_tests and (not all(frozen) or not all(a and a.get("selection_sha256") for a in assessments)
                      or len({_canonical(a) for a in assessments}) != 1):
        raise ValueError("Reserved test reports require the same frozen final assessment and selection hash")
    if any(frozen) != all(frozen):
        raise ValueError("Cannot mix development and frozen final assessment scopes")
    frames, joints, model_provenance, checkpoints, names = [], [], {}, {}, []
    original, checkpoint_by_path, checkpoint_by_hash = {}, {}, {}
    for metadata, frame, joint, _, _, source_checkpoints in sources:
        for name, record in source_checkpoints.items():
            path, digest = record.get("path"), record["sha256"]
            version = {key: record.get(key) for key in ("model_kind", "selection_version", "selected_epoch", "code_sha256")}
            if path in checkpoint_by_path and checkpoint_by_path[path] != digest:
                raise ValueError("A checkpoint path changed hash between source evaluations")
            if digest in checkpoint_by_hash and checkpoint_by_hash[digest] != version:
                raise ValueError("Checkpoint hash has inconsistent version/provenance")
            checkpoint_by_path[path], checkpoint_by_hash[digest] = digest, version
        for name in metadata["models"]:
            f, j = frame.loc[frame.model == name], joint.loc[joint.model == name]
            provenance = _model_provenance(metadata, name, source_checkpoints)
            if name in original:
                if name not in BASELINES or name in source_checkpoints:
                    raise ValueError("Learned model names must be unique across evaluations: " + name)
                if _canonical(provenance) != _canonical(model_provenance[name]):
                    raise ValueError("Duplicate baseline provenance differs: " + name)
                previous_f, previous_j = original[name]
                _exact(previous_f, f, ["model", *TARGET_KEYS], "Duplicate baseline rows differ: " + name)
                _exact(previous_j, j, ["model", *ORIGIN_KEYS], "Duplicate baseline joint rows differ: " + name)
                continue
            names.append(name); frames.append(f); joints.append(j)
            original[name] = (f, j); model_provenance[name] = provenance
            if name in source_checkpoints:
                checkpoints[name] = source_checkpoints[name]
    if reference not in names:
        raise ValueError("Comparison reference is absent from the merged models")
    frame, joint = pd.concat(frames, ignore_index=True), pd.concat(joints, ignore_index=True)
    overall, species, horizon = macro_scores(frame)
    seed = int(canonical["seed"])
    paired = paired_animal_bootstrap(frame, reference, seed, bootstrap_replicates)
    paired_joint = paired_animal_bootstrap(joint.rename(columns={"joint_chordal_energy_km": "endpoint_energy_km"}).assign(horizon_days=0.),
                                          reference, seed, bootstrap_replicates)
    timing = sources[0][3].groupby(["split", "horizon_days"], observed=True).target_timing_error_hours.agg(
        cases="size", mean_hours="mean", minimum_hours="min", maximum_hours="max").reset_index()
    errors = frame.groupby(["model", "split"]).point_error_km.quantile([.5, .9]).unstack().reset_index()
    errors.columns = ["model", "split", "median_point_error_km", "p90_point_error_km"]
    joint_macro = joint.groupby(["model", "split", "species", "individual"]).joint_chordal_energy_km.mean().groupby(
        ["model", "split", "species"]).mean().groupby(["model", "split"]).mean().reset_index()
    strata = frame.groupby(["model", "split", "displacement_proxy", "species", "individual", "horizon_days"]).endpoint_energy_km.mean().groupby(
        ["model", "split", "displacement_proxy", "species", "horizon_days"]).mean().groupby(
        ["model", "split", "displacement_proxy", "horizon_days"]).mean().reset_index()
    exports = {}
    for root in roots:
        path = root / "representative-routes.json"
        if not path.exists():
            continue
        for route in json.loads(path.read_text()).get("routes", []):
            key = (route["model"], route["split"], route["case"])
            if key in exports and _canonical(route) != _canonical(exports[key]):
                raise ValueError("Duplicate representative route differs across evaluations")
            exports[key] = route
    # Refuse partially rewritten inputs instead of silently combining a report
    # that was still being produced during these read-only calculations.
    for root, inventory in zip(roots, inventories):
        if _inventory(root) != inventory:
            raise ValueError("A source evaluation changed during combination")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "per-case.parquet", index=False)
    joint.to_parquet(out / "joint-per-origin.parquet", index=False)
    for name, value in (("scores", overall), ("per-species", species), ("per-horizon", horizon),
                        ("target-timing", timing), ("distance-quantiles", errors), ("joint-scores", joint_macro),
                        ("displacement-proxies", strata)):
        value.to_csv(out / (name + ".csv"), index=False)
    protocol = {key: canonical[key] for key in MATCH_METADATA}
    batch_sizes = [source[0].get("batch_size") for source in sources]
    protocol.update(scope=("Combined frozen local migration final assessment" if all(frozen) else
                           "Combined local migration development; no official benchmark or biological migration-label claim"),
                    assessment=assessments[0] if all(frozen) else None,
                    identity_unit=canonical.get("identity_unit"), data=canonical.get("data"),
                    metric_species_names=canonical.get("metric_species_names", {}), models=names,
                    checkpoints=list(checkpoints.values()), model_provenance=model_provenance,
                    samples=canonical["samples"], batch_size=batch_sizes[0] if all(v == batch_sizes[0] for v in batch_sizes) else None,
                    source_batch_sizes=batch_sizes, source_devices=[source[0].get("device") for source in sources], seed=seed,
                    scores=overall.to_dict("records"), joint_scores=joint_macro.to_dict("records"),
                    paired_animal_bootstrap=paired, paired_joint_animal_bootstrap=paired_joint,
                    source_evaluations=[{"path": str(root), "files_sha256": inventory} for root, inventory in zip(roots, inventories)],
                    retrieval={"source_report_diagnostics": [{"path": str(root), "diagnostics": source[0].get("retrieval")}
                                                              for root, source in zip(roots, sources)]},
                    code_sha256={Path(__file__).name: sha256(__file__), "migration_evaluate.py": sha256(Path(__file__).parent.parent / "wildcast/migration_evaluate.py")},
                    notes=list(canonical.get("notes", [])) + [
                        "Combined retained scores only; no model inference, retraining, location averaging, or new test-label release occurred.",
                        "Dataset fingerprints, origin/target/timing support, draw counts, seeds and RNG batch partitions match exactly across all sources.",
                        "Only exact baseline duplicates with identical provenance were deduplicated; learned model labels are unique.",
                        "All macro summaries and paired animal bootstrap intervals were recomputed from the merged per-case tables."])
    (out / "metrics.json").write_text(json.dumps(protocol, indent=2))
    (out / "representative-routes.json").write_text(json.dumps({"scope": "Combined evaluation overlays containing observed futures",
                                                               "routes": list(exports.values())}, indent=2))
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default="seasonal")
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    result = combine(**vars(parser.parse_args()))
    print(json.dumps({"models": result["models"], "cases": result["cases"], "observed_targets": result["observed_targets"]}))


if __name__ == "__main__":
    main()
