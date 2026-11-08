"""Shared-path endpoint evaluation, physical audits and renderer-ready exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import ensemble_predictions, load_model
from .identity import verify_audit
from .model import energy_score
from .sources import dump
from .train import load_windows
from .trajectory import GRID_HOURS, STATE_NAMES, project_torch, step_durations


def forecast(root, checkpoint, output, window_ids, times=None, samples=64, seed=2026, device="auto"):
    """Forecast requested elapsed times using observed inputs only; no scoring."""
    if samples < 1 or not window_ids:
        raise ValueError("Positive sample count and explicit origin window IDs required")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest = load_windows(root)
    model, ck = load_model(checkpoint, device)
    if ck["manifest"]["windows_sha256"] != manifest["windows_sha256"]:
        raise ValueError("Forecast/source windows differ")
    ids = np.asarray(window_ids, dtype=int)
    if np.any(ids < 0) or np.any(ids >= len(a["meta"])):
        raise ValueError("Unknown origin window ID")
    requested = list(GRID_HOURS) if times is None else times
    if not requested or min(requested) <= 0 or max(requested) > 169 or any(v <= u for u, v in zip(requested, requested[1:])):
        raise ValueError("Forecast times must increase within the trained 0–169 hour range")
    # Retain intermediate modelling steps even when the caller requests only
    # distant endpoints; a sparse display request must not change the time grid
    # into a single untrained week-long movement increment.
    query = sorted(set(requested) | {h for h in GRID_HOURS if h <= max(requested)})
    aux = torch.as_tensor(a["aux"][ids], device=device, dtype=torch.float64)
    t = torch.tensor(query, device=device, dtype=torch.float32).expand(len(ids), -1)
    with torch.inference_mode():
        context = model.observed_context(a, ids, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            result = model.sample_paths(context, t, torch.ones_like(t, dtype=torch.bool), aux,
                                        aux[:, 4:6].float(), torch.as_tensor(a["anchors"][ids], device=device),
                                        torch.as_tensor(a["meta"][ids, 1], device=device, dtype=torch.long), samples,
                                        torch.Generator(device=device).manual_seed(seed),
                                        identity=model.identity_groups[torch.as_tensor(a["meta"][ids, 3], device=device, dtype=torch.long)])
    names = {v: k for k, v in manifest.get("species", {}).items()}
    records = []
    for j, window in enumerate(ids):
        records.append({"window": int(window), "origin_timestamp": float(aux[j, 2]),
                        "species": names.get(int(a["meta"][window, 1]), "unknown"),
                        "requested_times_hours": requested, "times_hours": [0., *query],
                        "paths_latlon": torch.cat([aux[j, None, None, :2].expand(samples, 1, -1), result["latlon"][j]], 1).cpu().tolist(),
                        "displacement_regime_probabilities": result["regime_probabilities"][j].cpu().tolist()})
    payload = {"schema": "wildcast.trajectories.v1", "coordinate_order": "latitude,longitude", "seed": seed,
               "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               "future_observations_included": False, "predicted_intermediate_points_are_observations": False,
               "regimes_are_biological_behaviour_labels": False, "origins": records}
    dump(Path(output), payload)
    return {"output": str(output), "origins": len(records), "samples": samples, "times_hours": query}


def summarize(frame):
    """Raw distance summaries remain separate from macro normalized scores."""
    records = []
    thresholds = (100, 1000, 10000)
    for (split, horizon), group in frame.groupby(["split", "horizon_hours"]):
        for model in ("trajectory", "baseline"):
            column = model + "_point_error_m"
            if column not in group:
                continue
            error = group[column].to_numpy()
            records.append({"split": split, "horizon_hours": int(horizon), "model": model,
                            "windows": len(group), "animals": int(group.animal.nunique()),
                            "mean_distance_error_m": float(np.mean(error)), "median_distance_error_m": float(np.median(error)),
                            "p90_distance_error_m": float(np.quantile(error, .9)),
                            **{f"within_{d}m_fraction": float(np.mean(error <= d)) for d in thresholds},
                            "radial_coverage90": float(group[model + "_coverage90"].mean())})
    return records


def evaluate(root, checkpoint, output, identity_audit, samples=64, batch_size=32, seed=2026,
             max_origins=0, device="auto", export_origins=20, baseline_checkpoints=None):
    from .raster_context import unproject_torch
    if samples < 2 or batch_size < 1:
        raise ValueError("At least two samples and positive batch size required")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest = load_windows(root)
    audit = json.loads(Path(identity_audit).read_text())
    verify_audit(audit, manifest, a)
    model, ck = load_model(checkpoint, device)
    if model.family not in ("wildcast_trajectory_flow", "wildcast_joint_prior_ensemble") or ck["manifest"]["windows_sha256"] != manifest["windows_sha256"]:
        raise ValueError("Trajectory checkpoint/source mismatch")
    baselines = [load_model(p, device)[0] for p in (baseline_checkpoints or [])]
    for path in baseline_checkpoints or []:
        original = torch.load(path, map_location="cpu", weights_only=True)
        if original["manifest"]["species"] != manifest["species"] or original["manifest"]["observations_sha256"] != manifest["observations_sha256"]:
            raise ValueError("Baseline/source mismatch")
        n = original["config"].get("n_drivers", 0)
        if original["manifest"].get("driver_names", []) != manifest.get("driver_names", [])[:n]:
            raise ValueError("Baseline driver schema differs")
    ids = np.flatnonzero(np.isin(a["meta"][:, 0], [1, 3]))
    ids = ids[~np.isin(ids, audit["excluded_windows"])]
    origins, first = np.unique(a["aux"][ids, 6].astype(int), return_index=True)
    representatives = ids[first]
    if max_origins and len(origins) > max_origins:
        pick = np.sort(np.random.default_rng(seed).choice(len(origins), max_origins, replace=False))
        origins, representatives = origins[pick], representatives[pick]
    by_origin = {}
    for i in ids:
        by_origin.setdefault(int(a["aux"][i, 6]), []).append(int(i))
    gen = torch.Generator(device=device).manual_seed(seed)
    baseline_gen = torch.Generator(device=device).manual_seed(seed)
    rows, joint, exports = [], [], []
    max_ratio, bad_steps, total_steps = 0., 0, 0
    with torch.inference_mode():
        for start in range(0, len(origins), batch_size):
            selected = representatives[start:start + batch_size]
            context = model.observed_context(a, selected, device)
            times = torch.tensor(GRID_HOURS, device=device).expand(len(selected), -1)
            mask = torch.ones_like(times, dtype=torch.bool)
            aux = torch.as_tensor(a["aux"][selected], device=device, dtype=torch.float64)
            species = torch.as_tensor(a["meta"][selected, 1], device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                result = model.sample_paths(context, times, mask, aux, aux[:, 4:6].float(),
                                            torch.as_tensor(a["anchors"][selected], device=device), species, samples, gen,
                                            identity=model.identity_groups[torch.as_tensor(a["meta"][selected, 3], device=device, dtype=torch.long)])
            previous = torch.cat([aux[:, None, None, :2].expand(-1, samples, 1, -1), result["latlon"][:, :, :-1]], 2)
            distances = project_torch(result["latlon"], previous).norm(dim=-1)
            bounds = model.speed_limits[species, None] * step_durations(times) * 3600 + model.config.get("gps_allowance_m", 100.)
            ratio = distances / bounds[:, None]
            max_ratio = max(max_ratio, float(ratio.max().cpu()))
            bad_steps += int((ratio > 1.00001).sum().cpu())
            total_steps += ratio.numel()
            # Select a single representative generated route, using predictions
            # only. The true future never participates in choosing a sample.
            path_features = (result["xy"] / times[:, None, :, None].sqrt()).flatten(2)
            medoid = torch.cdist(path_features, path_features).mean(-1).argmin(-1)
            batch_windows = [w for i in selected for w in by_origin[int(a["aux"][i, 6])]]
            baseline = ensemble_predictions(baselines, a, np.asarray(batch_windows), samples, device, baseline_gen) if baselines else None
            offset = 0
            for j, origin_window in enumerate(selected):
                windows = np.asarray(by_origin[int(a["aux"][origin_window, 6])])
                hs = [GRID_HOURS.index(float(v)) for v in a["meta"][windows, 4]]
                draws = result["xy"][j, :, hs].transpose(0, 1)
                truth = torch.as_tensor(a["y"][windows], device=device)
                scale = aux[j, 8]
                es = energy_score(draws, truth) * scale
                centre = draws.mean(1)
                radii = (draws - centre[:, None]).norm(dim=-1).quantile(.9, dim=1)
                centre_coords = unproject_torch(centre.double() * scale, aux[j, 0], aux[j, 1])
                truth_coords = unproject_torch(truth.double() * scale, aux[j, 0], aux[j, 1])
                error = project_torch(centre_coords, truth_coords).norm(dim=-1)
                route_error = project_torch(result["latlon"][j, medoid[j], hs], truth_coords).norm(dim=-1)
                if baseline is not None:
                    bdraw = baseline[offset:offset + len(windows)]
                    bes = energy_score(bdraw, truth) * scale
                    bc = bdraw.mean(1)
                    br = (bdraw - bc[:, None]).norm(dim=-1).quantile(.9, dim=1)
                    bcoords = unproject_torch(bc.double() * scale, aux[j, 0], aux[j, 1])
                    be = project_torch(bcoords, truth_coords).norm(dim=-1)
                split = "val_individual" if a["meta"][origin_window, 0] == 3 else "val_temporal"
                for k, window in enumerate(windows):
                    row = {"window": int(window), "split": split, "study": int(a["meta"][window, 2]),
                           "animal": int(a["meta"][window, 3]), "horizon_hours": int(a["meta"][window, 4]),
                           "stationary_es_m": float(truth[k].norm() * scale), "trajectory_es_m": float(es[k]),
                           "trajectory_point_error_m": float(error[k]), "trajectory_route_error_m": float(route_error[k]),
                           "trajectory_coverage90": bool((truth[k] - centre[k]).norm() <= radii[k])}
                    if baseline is not None:
                        row.update(baseline_es_m=float(bes[k]), baseline_point_error_m=float(be[k]),
                                   baseline_coverage90=bool((truth[k] - bc[k]).norm() <= br[k]))
                    rows.append(row)
                if len(windows) >= 2:
                    # Joint score at observed benchmark waypoints only. We do
                    # not claim dense-route ADE between unobserved positions.
                    paths = draws.transpose(0, 1).flatten(1) / len(windows) ** .5
                    actual = truth.flatten() / len(windows) ** .5
                    accuracy = (paths - actual).norm(dim=-1).mean()
                    pair = torch.cdist(paths[None], paths[None])[0]
                    score = accuracy - .5 * pair.sum() / (samples * (samples - 1))
                    joint.append({"origin_window": int(origin_window), "split": split, "study": int(a["meta"][origin_window, 2]),
                                  "observed_waypoints": len(windows), "joint_energy_m": float(score * scale),
                                  "stationary_joint_energy_m": float(actual.norm() * scale),
                                  "representative_route_waypoint_ADE_m": float(route_error.mean())})
                if len(exports) < export_origins:
                    history_xy = a["x"][origin_window, :, :2]
                    history_xy = np.sign(history_xy) * np.expm1(abs(history_xy))
                    history_coords = unproject_torch(torch.as_tensor(history_xy, device=device).double() * scale, aux[j, 0], aux[j, 1])
                    species_names = {v: k for k, v in manifest.get("species", {}).items()}
                    animal_names = {v: k for k, v in manifest.get("animals", {}).items()}
                    exports.append({"origin_window": int(origin_window), "split": split,
                                    "species": species_names.get(int(a["meta"][origin_window, 1]), "unknown"),
                                    "animal": animal_names.get(int(a["meta"][origin_window, 3]), "unknown"),
                                    "origin_timestamp": float(aux[j, 2]), "origin_latlon": aux[j, :2].cpu().tolist(),
                                    "observed_history": [{"elapsed_hours": float(-np.expm1(a["x"][origin_window, k, 5])),
                                                          "latlon": history_coords[k].cpu().tolist()} for k in range(len(history_coords))],
                                    "times_hours": [0., *GRID_HOURS], "state_names": STATE_NAMES,
                                    "state_probabilities": result["regime_probabilities"][j].cpu().tolist(),
                                    "paths_latlon": torch.cat([aux[j, None, None, :2].expand(min(samples, 16), 1, -1),
                                                               result["latlon"][j, :16]], 1).cpu().tolist(),
                                    "full_sample_count": samples,
                                    "representative_index_in_full_sample_bank": int(medoid[j]),
                                    "representative_path_latlon": torch.cat([aux[j, None, :2], result["latlon"][j, medoid[j]]], 0).cpu().tolist(),
                                    "observed_future": [{"elapsed_hours": float((a["aux"][w, 3] - aux[j, 2].cpu().item()) / 3600),
                                                         "latlon": truth_coords[k].cpu().tolist()} for k, w in enumerate(windows)]})
                offset += len(windows)
            if start % (batch_size * 30) == 0:
                print(f"Trajectory evaluation {start}/{len(origins)} origins", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No validation cases to evaluate")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "per-window.parquet", index=False)
    pd.DataFrame(joint).to_parquet(out / "joint-waypoints.parquet", index=False)
    cells = frame.groupby(["split", "study", "horizon_hours"])[[c for c in frame if c.endswith("_es_m")]].mean()
    normalized = cells.div(cells.stationary_es_m.clip(lower=1), axis=0).groupby("split").mean()
    cells.to_csv(out / "per-study-horizon.csv")
    accuracy = summarize(frame)
    pd.DataFrame(accuracy).to_csv(out / "distance-metrics.csv", index=False)
    report = {"scope": "Audited local validation development; no official MoveBench or independent-test claim",
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "baseline_checkpoints": [{"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in baseline_checkpoints or []],
              "identity_audit_sha256": hashlib.sha256(Path(identity_audit).read_bytes()).hexdigest(),
              "windows_sha256": manifest["windows_sha256"], "samples": samples, "seed": seed,
              "max_origins": max_origins, "origins": len(origins), "cases": len(frame),
              "case_ids_sha256": hashlib.sha256(np.sort(frame.window.to_numpy()).tobytes()).hexdigest(),
              "macro_normalized_energy": normalized.to_dict(orient="index"), "distance_metrics": accuracy,
              "physical_audit": {"generated_steps": total_steps, "steps_outside_envelope": bad_steps,
                                 "maximum_bound_ratio": max_ratio, "gps_allowance_m_per_interval": model.config.get("gps_allowance_m", 100.)},
              "notes": ["One sampled joint trajectory bank is reused across all horizons of an origin.",
                        "Point errors use the sample mean in the origin's projection, scored using great-circle distance.",
                        "Representative routes are sample medoids selected without observing the true future.",
                        "Within-distance rates are fixed at 100 m, 1 km and 10 km; these are not universal biological accuracy thresholds.",
                        "Joint energy and waypoint ADE use observed sparse waypoints only, not interpolated ground truth.",
                        "Speed envelopes are training-derived assumptions; habitat embeddings do not enforce hard terrain barriers.",
                        "Radial coverage is not highest-density-region coverage."]}
    dump(out / "metrics.json", report)
    dump(out / "trajectories.json", {"schema": "wildcast.trajectories.v1", "coordinate_order": "latitude,longitude",
                                      "predicted_intermediate_points_are_observations": False, "origins": exports})
    return report
