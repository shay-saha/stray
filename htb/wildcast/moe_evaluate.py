"""Coupled full validation of family experts, shared controls and sparse routing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluate import load_model
from .identity import case_fingerprint
from .model import energy_score
from .moe import CategoryMoE
from .raster_context import unproject_torch
from .sources import dump
from .train import load_windows
from .trajectory import project_torch


def evaluate(root, cache, checkpoints, output, samples=64, batch_size=512, seed=2026, device="auto"):
    if samples < 2:
        raise ValueError("At least two independent samples required")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest = load_windows(root)
    cache, out = Path(cache), Path(output)
    info = json.loads((cache / "manifest.json").read_text())
    cache_hash = hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()
    if info["windows_sha256"] != manifest["windows_sha256"] or info["case_fingerprint"] != case_fingerprint(a):
        raise ValueError("MoE evaluator and cached cases differ")
    for p, expected in info["files_sha256"].items():
        if hashlib.sha256((cache / p).read_bytes()).hexdigest() != expected:
            raise ValueError("MoE evaluation cache checksum mismatch")
    ids = np.load(cache / "ids.npy")
    rows = np.flatnonzero(np.isin(a["meta"][ids, 0], [1, 3]))
    ids = ids[rows]
    f = torch.as_tensor(np.array(np.load(cache / "features.npy", mmap_mode="r")[rows]), device=device)
    p = torch.as_tensor(np.array(np.load(cache / "base.npy", mmap_mode="r")[rows]), device=device)
    species = torch.as_tensor(a["meta"][ids, 1], device=device, dtype=torch.long)
    truth = torch.as_tensor(a["y"][ids], device=device)
    aux = torch.as_tensor(a["aux"][ids], device=device, dtype=torch.float64)
    initial = torch.load(cache / "bases.pt", map_location="cpu", weights_only=True)
    baseline = CategoryMoE({**initial["config"], "mode": "shared"}).to(device).eval()
    for base, state in zip(baseline.bases, initial["states"]):
        base.load_state_dict(state)
    models = [("frozen-ensemble", baseline, None)]
    for checkpoint in checkpoints:
        model, ck = load_model(checkpoint, device)
        if model.family != "wildcast_category_moe" or ck["training"]["cache_manifest_sha256"] != cache_hash:
            raise ValueError("Expert checkpoint uses another feature bank")
        if ck["manifest"]["windows_sha256"] != manifest["windows_sha256"]:
            raise ValueError("Expert checkpoint source windows differ")
        name = Path(checkpoint).parent.name
        if any(name == n for n, _, _ in models):
            raise ValueError("Duplicate experiment labels")
        models.append((name, model, ck))
    summary, experiment_map = {}, {}
    base_rows = pd.DataFrame({"window": ids, "split": np.where(a["meta"][ids, 0] == 1, "val_temporal", "val_individual"),
                             "study": a["meta"][ids, 2], "animal": a["meta"][ids, 3],
                             "species": a["meta"][ids, 1], "horizon_hours": a["meta"][ids, 4],
                             "family": [initial["config"]["families"][initial["config"]["species_family"][i]] for i in a["meta"][ids, 1]],
                             "stationary_es_m": np.linalg.norm(a["y"][ids], axis=-1) * a["aux"][ids, 8]})
    for name, model, ck in models:
        generator = torch.Generator(device=device).manual_seed(seed)
        es, error, coverage = [], [], []
        usage = torch.zeros((len(model.experts),), device=device)
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                sl = slice(start, start + batch_size)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    draws = model.sample_from_features(f[sl], p[sl], species[sl], samples, generator)
                es.append((energy_score(draws, truth[sl]) * aux[sl, 8]).cpu().numpy())
                centre = draws.mean(1)
                radii = (draws - centre[:, None]).norm(dim=-1).quantile(.9, dim=1)
                coverage.append(((truth[sl] - centre).norm(dim=-1) <= radii).cpu().numpy())
                predicted = unproject_torch(centre.double() * aux[sl, 8:9], aux[sl, 0], aux[sl, 1])
                observed = unproject_torch(truth[sl].double() * aux[sl, 8:9], aux[sl, 0], aux[sl, 1])
                error.append(project_torch(predicted, observed).norm(dim=-1).cpu().numpy())
                usage += model.last_usage.sum(0)
        frame = base_rows.copy()
        frame["forecast_es_m"] = np.concatenate(es)
        frame["point_error_m"] = np.concatenate(error)
        frame["radial_coverage90"] = np.concatenate(coverage)
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target / "per-window.parquet", index=False)
        cells = frame.groupby(["split", "study", "horizon_hours"])[["stationary_es_m", "forecast_es_m"]].mean()
        scores = cells.forecast_es_m / cells.stationary_es_m.clip(lower=1)
        macro = scores.groupby("split").mean().to_dict()
        families = frame.groupby(["split", "family", "study", "horizon_hours"])[["stationary_es_m", "forecast_es_m"]].mean()
        family_scores = (families.forecast_es_m / families.stationary_es_m.clip(lower=1)).groupby(["split", "family"]).mean()
        family_scores.rename("normalized_energy").to_csv(target / "family-scores.csv")
        distances = []
        for (split, horizon), g in frame.groupby(["split", "horizon_hours"]):
            e = g.point_error_m
            distances.append({"split": split, "horizon_hours": int(horizon), "windows": len(g),
                              "mean_error_m": float(e.mean()), "median_error_m": float(e.median()), "p90_error_m": float(e.quantile(.9)),
                              **{f"within_{d}m_fraction": float((e <= d).mean()) for d in [100, 300, 1000, 10000]}})
        pd.DataFrame(distances).to_csv(target / "distance-metrics.csv", index=False)
        report = {"scope": "Audited local validation development, not official MoveBench or independent test",
                  "name": name, "macro_normalized_energy": macro, "samples": samples, "seed": seed, "cases": len(frame),
                  "case_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "cache_manifest_sha256": cache_hash,
                  "identity_audit_sha256": info["identity_audit_sha256"], "distance_metrics": distances,
                  "mean_expert_weight": {model.config["families"][i]: float(v) for i, v in enumerate((usage / len(ids)).cpu())},
                  "training_support": info["training_support"],
                  "checkpoint": str(next((p for p in checkpoints if Path(p).parent.name == name), "frozen ensemble from cache")),
                  "checkpoint_sha256": next((hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in checkpoints if Path(p).parent.name == name), None),
                  "base_checkpoints": info.get("base_checkpoints", []),
                  "selected_epoch": ck["epoch"] if ck else None, "training": ck["training"] if ck else None,
                  "notes": ["Identical cases, feature bank, loss and hierarchical common random draws across routing variants.",
                            "Frozen encoders and shared/ensemble knowledge; family-specific heads are the specialization being tested.",
                            "Parameter counts differ; include the wider shared head as a capacity control.",
                            "Training-support counts describe taxonomy-matched observations; shared heads can receive all training families.",
                            "Point errors are great-circle distances from the sample mean. Rate thresholds are fixed in advance.",
                            "Training and selection use no reserved fresh-test observations."]}
        dump(target / "metrics.json", report)
        summary[name] = macro
        experiment_map[name] = {"path": str(target), "column": "forecast_es_m"}
        print(json.dumps({"model": name, "macro_normalized_energy": macro}), flush=True)
    dump(out / "summary.json", summary)
    dump(out / "experiments.json", experiment_map)
    return summary
