"""Compare old forecasters and whole-model MoEs on identical audited cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import load_model, ensemble_predictions
from .identity import case_fingerprint, verify_audit
from .model import energy_score
from .moe import taxonomy_groups
from .raster_context import unproject_torch
from .sources import dump
from .train import load_windows
from .trajectory import project_torch


def evaluate(root, experiments, output, identity_audit, samples=64, batch_size=256,
             seed=2026, device="auto", precision="bf16"):
    if samples < 2:
        raise ValueError("At least two independent forecast samples required")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    audit = json.loads(Path(identity_audit).read_text())
    specs = json.loads(Path(experiments).read_text())
    reference, reference_manifest = load_windows(root)
    fingerprint = case_fingerprint(reference)
    verify_audit(audit, reference_manifest, reference)
    ids = np.flatnonzero(np.isin(reference["meta"][:, 0], [1, 3]))
    ids = ids[~np.isin(ids, audit["excluded_windows"])]
    grouping = taxonomy_groups(reference_manifest, "data/combined-v2/environment/taxonomy")
    frame0 = pd.DataFrame({"window": ids, "split": np.where(reference["meta"][ids, 0] == 1, "val_temporal", "val_individual"),
                           "study": reference["meta"][ids, 2], "animal": reference["meta"][ids, 3],
                           "species": reference["meta"][ids, 1], "horizon_hours": reference["meta"][ids, 4],
                           "family": [grouping["families"][grouping["species_family"][s]] for s in reference["meta"][ids, 1]],
                           "stationary_es_m": np.linalg.norm(reference["y"][ids], axis=-1) * reference["aux"][ids, 8]})
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    summary, experiment_map = {}, {}
    loaded_root, a, manifest = root, reference, reference_manifest
    for name, spec in specs.items():
        data_root = spec.get("data", root)
        if data_root != loaded_root:
            a, manifest = load_windows(data_root)
            loaded_root = data_root
        if manifest["species"] != reference_manifest["species"] or case_fingerprint(a) != fingerprint:
            raise ValueError("Comparison cases or species IDs differ")
        for key in ["anchors", "memory"]:
            if not np.array_equal(a[key], reference[key]):
                raise ValueError("Comparison changed observed " + key)
        verify_audit(audit, manifest, a)
        paths = spec["checkpoints"]
        loaded = [load_model(path, device) for path in paths]
        for _, ck in loaded:
            if ck["manifest"]["windows_sha256"] != manifest["windows_sha256"]:
                raise ValueError("A comparison model uses a different feature schema")
        generator = torch.Generator(device=device).manual_seed(seed)
        es, errors, coverage, usage = [], [], [], []
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                take = ids[start:start + batch_size]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16" and device.startswith("cuda")):
                    draws = ensemble_predictions([m for m, _ in loaded], a, take, samples, device, generator)
                truth = torch.as_tensor(a["y"][take], device=device)
                aux = torch.as_tensor(a["aux"][take], device=device, dtype=torch.float64)
                es.append((energy_score(draws, truth) * aux[:, 8]).cpu().numpy())
                centre = draws.mean(1)
                radii = (draws - centre[:, None]).norm(dim=-1).quantile(.9, dim=1)
                coverage.append(((truth - centre).norm(dim=-1) <= radii).cpu().numpy())
                predicted = unproject_torch(centre.double() * aux[:, 8:9], aux[:, 0], aux[:, 1])
                observed = unproject_torch(truth.double() * aux[:, 8:9], aux[:, 0], aux[:, 1])
                errors.append(project_torch(predicted, observed).norm(dim=-1).cpu().numpy())
                if len(loaded) == 1 and getattr(loaded[0][0], "family", "") == "wildcast_full_network_moe":
                    usage.append(loaded[0][0].weights(a, take, device).cpu().numpy())
        frame = frame0.copy()
        frame["forecast_es_m"] = np.concatenate(es)
        frame["point_error_m"] = np.concatenate(errors)
        frame["radial_coverage90"] = np.concatenate(coverage)
        target = out / name; target.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target / "per-window.parquet", index=False)
        cells = frame.groupby(["split", "study", "horizon_hours"])[["stationary_es_m", "forecast_es_m"]].mean()
        macro = (cells.forecast_es_m / cells.stationary_es_m.clip(lower=1)).groupby("split").mean().to_dict()
        family_cells = frame.groupby(["split", "family", "study", "horizon_hours"])[["stationary_es_m", "forecast_es_m"]].mean()
        (family_cells.forecast_es_m / family_cells.stationary_es_m.clip(lower=1)).groupby(["split", "family"]).mean().rename("normalized_energy").to_csv(target / "family-scores.csv")
        distances = []
        for (split, horizon), group in frame.groupby(["split", "horizon_hours"]):
            error = group.point_error_m
            distances.append({"split": split, "horizon_hours": int(horizon), "windows": len(group),
                              "median_error_m": float(error.median()), "mean_error_m": float(error.mean()),
                              "p90_error_m": float(error.quantile(.9)),
                              **{f"within_{d}m_fraction": float((error <= d).mean()) for d in [100, 300, 1000, 10000]}})
        pd.DataFrame(distances).to_csv(target / "distance-metrics.csv", index=False)
        checkpoints = [{"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        "epoch": ck["epoch"], "model_family": ck.get("model_family", "wildcast_transformer"),
                        "parameters": ck["training"]["parameters"]} for path, (_, ck) in zip(paths, loaded)]
        report = {"name": name, "scope": "Audited local validation development, not independent test or official MoveBench",
                  "macro_normalized_energy": macro, "distance_metrics": distances, "cases": len(ids),
                  "case_fingerprint": fingerprint, "case_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                  "samples": samples, "batch_size": batch_size, "seed": seed, "precision": precision,
                  "data": str(data_root), "windows_sha256": manifest["windows_sha256"], "checkpoints": checkpoints,
                  "mean_specialist_weight": float(np.concatenate(usage)[:, 1].mean()) if usage else None,
                  "notes": ["Same audited cases, horizon definitions and metric; each checkpoint receives its own exact trained feature schema.",
                            "Point errors use spherical distances from the predictive sample mean; these are separate from distribution energy scores.",
                            "Family-specific checkpoints use family validation for selection. All results remain development comparisons.",
                            "Fixed 64-draw evaluation seeds do not imply perfectly coupled samples across differing architectures."]}
        dump(target / "metrics.json", report)
        summary[name] = macro
        experiment_map[name] = {"path": str(target), "column": "forecast_es_m"}
        print(json.dumps({"name": name, "scores": macro}), flush=True)
        del loaded
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    dump(out / "summary.json", summary)
    dump(out / "experiments.json", experiment_map)
    return summary
