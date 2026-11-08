"""History-only migration forecasts with frozen training experience retrieval.

Reference validation/test labels are stripped before any predictor sees arrays.
The new query has no target values, masks or observed future coordinates.
"""
from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import unproject
from .migration_baselines import MigrationBaselines
from .migration_protocol import schema_for
from .migration_evaluate import load_cases, sha256
from .sources import dump

LOADERS = {"wildcast_migration_joint": ".migration_model",
           "wildcast_migration_environment_gate": ".migration_environment_gate",
           "wildcast_migration_context_joint": ".migration_context_joint",
           "wildcast_migration_retrieval": ".migration_retrieval_train",
           "wildcast_migration_moveformer": ".migration_moveformer",
           "wildcast_migration_moveformer_endpoint": ".migration_moveformer_endpoint",
           "wildcast_migration_regime": ".migration_regimes",
           "wildcast_migration_ensemble": ".migration_ensemble"}


def _observations(path, animal, issue_time):
    path = Path(path)
    source_hash = sha256(path)
    # A future malformed CSV value must not change how past numeric strings are
    # parsed through whole-column dtype inference.
    frame = (pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq")
             else pd.read_csv(path, dtype={"timestamp": "string", "lat": "string", "lon": "string"}))
    if sha256(path) != source_hash:
        raise ValueError("Observation source changed during reading; use a frozen GPS snapshot")
    if not {"timestamp", "lat", "lon"}.issubset(frame):
        raise ValueError("Forecast observations require timestamp, lat and lon columns")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    frame = frame.loc[frame.timestamp.notna()]
    cutoff = None
    if issue_time is not None:
        cutoff = pd.Timestamp(issue_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        if pd.isna(cutoff):
            raise ValueError("Forecast issue time must be a valid UTC-compatible timestamp")
        # Filter before numeric coercion, identity auto-selection or features.
        frame = frame.loc[frame.timestamp <= cutoff]
    selected = str(animal) if animal is not None else None
    if selected is not None:
        if "animal" not in frame:
            raise ValueError("Selecting an animal requires an animal column")
        frame = frame.loc[frame.animal.astype(str) == selected]
    elif "animal" in frame:
        names = frame.animal.dropna().astype(str).unique()
        if len(names) > 1:
            raise ValueError("Select one observed animal with the animal argument")
        selected = str(names[0]) if len(names) else None
        frame = frame.loc[frame.animal.notna()]
    frame = frame.copy()
    frame["lat"] = pd.to_numeric(frame.lat, errors="coerce").astype(float)
    frame["lon"] = pd.to_numeric(frame.lon, errors="coerce").astype(float)
    frame = frame.loc[frame.lat.between(-90, 90) & frame.lon.between(-180, 180)]
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid observed GPS history at or before issue time")
    # Parquet can preserve microsecond dtype; asi8 below always means ns.
    # Normalize only after issue-time filtering, so a future row cannot fail
    # conversion to the accepted history's nanosecond representation.
    frame["timestamp"] = frame.timestamp.dt.as_unit("ns")
    return frame, selected, cutoff, source_hash


def _load(checkpoint, device, hashes):
    checkpoint = Path(checkpoint)
    before = sha256(checkpoint)
    probe = torch.load(checkpoint, map_location="cpu", weights_only=True)
    kind = probe.get("model_kind")
    if kind not in LOADERS:
        raise ValueError("Unsupported migration checkpoint kind: " + str(kind))
    for key, expected in hashes.items():
        if probe.get(key) != expected:
            raise ValueError("Forecast checkpoint/reference " + key + " mismatch")
    del probe
    try:
        model, metadata = importlib.import_module(LOADERS[kind], package="wildcast").load_model(checkpoint, device=device)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise ValueError("Forecast package is missing a referenced cache/member asset; restore its frozen files and paths before inference: " + str(error)) from error
    if sha256(checkpoint) != before:
        raise ValueError("Forecast checkpoint changed during loading; use an immutable snapshot")
    for key, expected in hashes.items():
        if metadata.get(key) != expected:
            raise ValueError("Loaded forecast checkpoint/reference " + key + " mismatch")
    return model, metadata, kind, before


def _query_view(a, manifest, frame, species, canonical_animal, selected_animal):
    schema = schema_for(manifest, a, legacy_default=True)
    for key, expected in [("history_names", schema.HISTORY_NAMES), ("static_names", schema.STATIC_NAMES)]:
        if key in manifest and manifest[key] != expected:
            raise ValueError("Reference " + key + " differs from the raw GPS feature builder")
    if "history_lags_days" in manifest and not np.array_equal(np.asarray(manifest["history_lags_days"]), schema.LAGS):
        raise ValueError("Reference history lag schema differs from the raw GPS feature builder")
    if manifest.get("unit_metres", schema.UNIT) != schema.UNIT:
        raise ValueError("Reference displacement units differ from the raw GPS feature builder")
    dates = pd.DatetimeIndex(frame.timestamp)
    times = dates.asi8 / 1e9
    builder = getattr(schema, "query_features", schema.features)
    history, mask, static, history_latlon, history_time = builder(
        frame.lat.to_numpy(), frame.lon.to_numpy(), times, dates, len(frame) - 1)
    if a["history"].shape[1:] != history.shape or a["static"].shape[1:] != static.shape:
        raise ValueError("Reference history/static schema differs from the raw GPS feature builder")
    count, horizons = len(a["split"]), len(a["horizons"])
    train = a["split"] == 0
    view = {}
    # Allocate new arrays. Only training predictor/experience values survive;
    # held-out coordinates, labels and masks never reach the prediction API.
    for key in ["history", "history_mask", "static", "species", "metric_species", "y", "target_mask",
                "individual", "study", "origin", "target_latlon", "target_time", "history_latlon", "history_time"]:
        if key not in a:
            continue
        value = np.zeros((count + 1, *a[key].shape[1:]), dtype=a[key].dtype)
        value[:count][train] = a[key][train]
        view[key] = value
    # Non-training split codes are retained solely to exclude those empty rows.
    view["split"] = np.r_[a["split"], np.int8(2)].astype(np.int8)
    view["horizons"] = np.asarray(a["horizons"]).copy()
    view["history"][count], view["history_mask"][count], view["static"][count] = history, mask, static
    index = int(manifest.get("species", {}).get(species, 0))
    view["species"][count] = index
    requested_identity = str(canonical_animal) if canonical_animal is not None else selected_animal
    mapped = manifest.get("identity_mapping", {}).get(requested_identity, requested_identity)
    known = mapped in manifest.get("animals", {}) if mapped is not None else False
    identity = int(manifest["animals"][mapped]) if known else int(np.max(a["individual"], initial=-1)) + 1
    view["individual"][count] = identity
    matching = np.flatnonzero(a["individual"] == identity) if known else np.empty(0, dtype=int)
    view["study"][count] = int(a["study"][matching[0]]) if len(matching) else int(np.max(a["study"], initial=-1)) + 1
    if "metric_species" in view:
        groups = manifest.get("metric_species", {})
        view["metric_species"][count] = int(groups.get(species, int(np.max(a["metric_species"], initial=-1)) + 1))
    view["origin"][count] = [float(frame.lat.iloc[-1]), float(frame.lon.iloc[-1]), times[-1]]
    if "history_latlon" in view:
        view["history_latlon"][count] = history_latlon
    if "history_time" in view:
        view["history_time"][count] = history_time
    # Make future-free query support explicit even if reference storage changes.
    view["y"][count] = 0
    view["target_mask"][count] = False
    view["target_latlon"][count] = 0
    view["target_time"][count] = 0
    if view["y"].shape != (count + 1, horizons, 2):
        raise ValueError("Reference future horizon schema differs")
    identity_info = {"requested_canonical_animal": canonical_animal, "matched_canonical_animal": mapped if known else None,
                     "physical_identity_known": bool(known), "physical_identity_index": identity,
                     "identity_rule": "Exact frozen canonical name or recorded alias; otherwise a fresh exclusion ID, never fuzzy matching"}
    return view, count, dates, mask, history_latlon, history_time, identity_info


def forecast(data, checkpoint, observations, output, species, animal=None, canonical_animal=None,
             issue_time=None, samples=64, seed=2026, device="cpu"):
    """Forecast complete route samples without reading any raw future as input.

    Horizons begin at the last actual fix, which can precede issue_time. The
    reference cache supplies frozen training experiences only. This command
    makes no network calls and never fetches context fields automatically.
    """
    if samples < 2:
        raise ValueError("At least two complete path samples required")
    if not isinstance(species, str) or not species.strip():
        raise ValueError("A scientific species name is required")
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest, hashes = load_cases(data)
    frame, selected_animal, cutoff, source_hash = _observations(observations, animal, issue_time)
    model, metadata, kind, checkpoint_hash = _load(checkpoint, device, hashes)
    view, query, dates, mask, history_latlon, history_time, identity_info = _query_view(
        a, manifest, frame, species.strip(), canonical_animal, selected_animal)
    fitted = MigrationBaselines(view, manifest)
    ids = np.asarray([query], dtype=np.int64)
    try:
        if hasattr(model, "predict_cases"):
            paths = model.predict_cases(view, ids, samples, np.random.default_rng(seed), baselines=fitted)
        else:
            with torch.inference_mode():
                inputs = [torch.as_tensor(view[key][ids], device=device) for key in ["history", "history_mask", "static", "species"]]
                paths = model.sample_paths(*inputs, samples=samples, generator=torch.Generator(device=device).manual_seed(seed)).detach().cpu().numpy()
    except (FileNotFoundError, NotADirectoryError) as error:
        raise ValueError("Forecast package is missing a referenced member or context field cache; restore the recorded local assets. No network fallback is performed: " + str(error)) from error
    paths = np.asarray(paths, dtype=float)
    if paths.shape != (1, samples, len(a["horizons"]), 2) or not np.isfinite(paths).all():
        raise ValueError("Forecast model returned invalid complete path samples")
    lat, lon = float(frame.lat.iloc[-1]), float(frame.lon.iloc[-1])
    coordinates = unproject(paths[0] * 1_000_000, lat, lon)
    horizons = [float(h) for h in a["horizons"]]
    train = a["split"] == 0
    last_origin = float(a["origin"][train, 2].max())
    target_times = a["target_time"][train][a["target_mask"][train]]
    training_end = max(last_origin, float(target_times.max()) if len(target_times) else last_origin)
    replay = dates[-1].timestamp() < training_end
    if "validation_year" in manifest:
        development_end = pd.Timestamp(int(manifest["validation_year"]) + 1, 1, 1, tz="UTC")
        development_replay = dates[-1] < development_end
    else:
        development_end, development_replay = None, None
    generated_at = pd.Timestamp.now(tz="UTC")
    forecast_times = [dates[-1] + pd.Timedelta(days=h) for h in horizons]
    elapsed_horizons = [bool(t <= generated_at) for t in forecast_times]
    schema = schema_for(manifest, legacy_default=True)
    support_rule = {"minimum_history_lags": int(manifest.get("minimum_history_lags", 5)),
                    "minimum_history_prior_fixes": int(manifest.get("minimum_history_prior_fixes", 8)),
                    "minimum_history_span_days": float(manifest.get("minimum_history_span_days", 7))}
    training_support = (int(mask.sum()) >= support_rule["minimum_history_lags"]
                        and len(frame) - 1 >= support_rule["minimum_history_prior_fixes"]
                        and (dates[-1] - dates[0]).total_seconds() >= support_rule["minimum_history_span_days"] * 86400)
    observed = [{"timestamp": row.timestamp.isoformat(), "lat": float(row.lat), "lon": float(row.lon)} for row in frame.itertuples()]
    conditioned_hash = hashlib.sha256(json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    conditioning = [{"requested_lag_days": float(schema.LAGS[i]),
                     "timestamp": pd.Timestamp(history_time[i], unit="s", tz="UTC").isoformat(),
                     "lat": float(history_latlon[i, 0]), "lon": float(history_latlon[i, 1])} for i in np.flatnonzero(mask)]
    result = {"scope": "History-only forecast; no observed future coordinates or query targets",
        "output": str(output), "model_kind": kind, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "reference_cache": str(data), **hashes, "source_observations": str(observations), "source_file_sha256": source_hash,
        "conditioning_history_sha256": conditioned_hash, "code_sha256": sha256(__file__),
        "species": species.strip(), "prediction_species_index": int(view["species"][query]),
        "unknown_species_fallback": bool(view["species"][query] == 0), "selected_animal": selected_animal, **identity_info,
        "origin": {"timestamp": dates[-1].isoformat(), "lat": lat, "lon": lon},
        "requested_issue_time": cutoff.isoformat() if cutoff is not None else None,
        "generated_utc": generated_at.isoformat(), "horizons_days": horizons,
        "forecast_timestamps": [t.isoformat() for t in forecast_times], "horizons_already_elapsed_at_generation": elapsed_horizons,
        "sampled_latlon_paths": coordinates.tolist(), "samples": samples, "seed": seed,
        "observed_history": observed[-512:], "conditioning_history": conditioning,
        "history_available_lags": int(mask.sum()), "last_training_observation": pd.Timestamp(training_end, unit="s", tz="UTC").isoformat(),
        "history_support_matches_training": bool(training_support),
        "history_support_rule": support_rule, "feature_protocol": manifest.get("protocol", "wildcast-migration-v1"),
        "origin_precedes_training_end": bool(replay), "retrospective_replay": bool(replay or development_replay or any(elapsed_horizons)),
        "development_end_exclusive": development_end.isoformat() if development_end is not None else None,
        "operational_backtest": False, "selection_version": metadata.get("selection_version"),
        "retrieval_rule": "Training donors only; query marked familiar/year so every donor's complete observed future must end strictly before forecast origin",
        "notes": ["Raw observations after issue time are excluded before feature construction, numeric-coordinate filtering and automatic animal selection.",
                  "The reference predictor view strips all non-training labels, masks and coordinates; the appended query has no target values.",
                  "Every forecast horizon begins at the last actual observed fix, not a later requested issue time.",
                  "Identity is an exclusion/retrieval key, not an animal-ID prediction embedding.",
                  "Whole-path samples preserve the selected model's joint sampling law; predicted positions are not GPS observations or verified behavioural states.",
                  "Context models read packaged local fields with exact previous-year/nodata masks; no field acquisition or web service is invoked.",
                  "Origins before training/model selection or with already elapsed prediction times are retrospective replay; history-only inputs alone do not establish an operational backtest.",
                  "The history_support_rule records this protocol's minimum observed span, strictly prior fixes and available lags; inputs below these requirements can be sampled but their calibration has not been validated."]}
    dump(output, result)
    return {"output": str(output), "origin": result["origin"], "model_kind": kind,
            "samples": samples, "horizons_days": horizons, "retrospective_replay": result["retrospective_replay"]}
