from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import CorrelatedRandomWalk, AnalogueForecaster
from .data import SPLITS, unproject
from .model import MovementTransformer, energy_score, sample
from .sources import dump
from .train import load_windows


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("model_family") == "wildcast_full_network_moe":
        from .full_moe import FullNetworkMoE
        model = FullNetworkMoE(ckpt["config"]).to(device)
    elif ckpt.get("model_family") == "wildcast_category_moe":
        from .moe import CategoryMoE
        model = CategoryMoE(ckpt["config"]).to(device)
    elif ckpt.get("model_family") == "wildcast_joint_prior_ensemble":
        from .trajectory import JointTrajectoryEnsemble
        model = JointTrajectoryEnsemble(ckpt["config"]).to(device)
    elif ckpt.get("model_family") == "wildcast_trajectory_flow":
        from .trajectory import TrajectoryFlow
        model = TrajectoryFlow(ckpt["config"]).to(device)
    elif ckpt.get("model_family") == "wildcast_destination_habitat":
        from .habitat import HabitatForecast
        model = HabitatForecast(ckpt["config"]).to(device)
    elif ckpt.get("model_family") == "wildcast_routed_ensemble":
        from .router import RoutedForecast
        model = RoutedForecast(ckpt["config"],ckpt["analogue"]).to(device)
    elif ckpt.get("model_family") == "moveformer_author_core_local_adapter":
        from .moveformer import LocalMoveFormer
        model = LocalMoveFormer(**ckpt["config"]).to(device)
    else:
        model = MovementTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, ckpt


def predictions(model, a, ix, count, device, generator):
    if hasattr(model,"predict_samples"):
        return model.predict_samples(a,ix,count,device,generator)
    x = torch.from_numpy(a["x"][ix]).to(device)
    sp = torch.from_numpy(a["meta"][ix, 1].astype(np.int64)).to(device)
    h = torch.from_numpy(a["meta"][ix, 4].astype(np.float32)).to(device)
    v = torch.from_numpy(a["aux"][ix, 4:6].astype(np.float32)).to(device)
    dr = torch.from_numpy(a["drivers"][ix]).to(device) if "drivers" in a else None
    if getattr(model, "family", None) == "moveformer_author_core_local_adapter":
        return model.sample_forecast(x, sp, h, v, dr, count, generator)
    anc = torch.from_numpy(a["anchors"][ix]).to(device) if "anchors" in a else None
    memory = torch.from_numpy(a["memory"][ix]).to(device) if "memory" in a else None
    return sample(model(x, sp, h, v, dr, anc, memory), count, generator)


def ensemble_predictions(models, a, ix, count, device, generator):
    if len(models) == 1:
        return predictions(models[0], a, ix, count, device, generator)
    draws = torch.stack([predictions(m, a, ix, count, device, generator) for m in models], dim=2)
    member = torch.randint(len(models), (len(ix), count, 1, 1), device=device, generator=generator).expand(-1, -1, -1, 2)
    return draws.gather(2, member).squeeze(2)


def evaluate(root, checkpoints, output, samples=64, batch_size=256, seed=2026, max_per_split=0, device="auto", splits=None,
             identity_audit=None,fresh_only=False):
    if samples < 2:
        raise ValueError("At least two forecast samples required")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest = load_windows(root)
    excluded=set();audit_info=None
    if identity_audit:
        audit_info=json.loads(Path(identity_audit).read_text())
        from .identity import verify_audit
        verify_audit(audit_info,manifest,a)
        excluded=set(audit_info["excluded_windows"])
    fresh={manifest["studies"][s] for s in manifest.get("fresh_studies",[])} if fresh_only else None
    if fresh_only and not fresh:raise ValueError("No fresh study cohort declared")
    models = [load_model(p, device) for p in checkpoints]
    checkpoint_info = [{"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                        "epoch": ck["epoch"], "seed": ck["training"]["seed"], "parameters": ck["training"]["parameters"],
                        "model_family": ck.get("model_family", "wildcast_transformer")}
                       for p, (_, ck) in zip(checkpoints, models)]
    for _, ck in models:
        if ck["manifest"]["windows_sha256"] != manifest["windows_sha256"]:
            raise ValueError("Checkpoint and evaluation window fingerprints differ")
    train = a["meta"][:, 0] == SPLITS["train"]
    crw = CorrelatedRandomWalk().fit(a["y"][train], a["meta"][train], a["aux"][train, 4:6])
    analogue = AnalogueForecaster().fit(a["x"][train], a["y"][train], a["meta"][train], a["aux"][train, 4:6])
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = []
    with torch.inference_mode():
        for split, code in SPLITS.items():
            if split == "train" or (splits is not None and split not in splits):
                continue
            ids = np.flatnonzero(a["meta"][:, 0] == code)
            if excluded:ids=ids[~np.isin(ids,list(excluded))]
            if fresh is not None:ids=ids[np.isin(a["meta"][ids,2],list(fresh))]
            if max_per_split and len(ids) > max_per_split:
                ids = np.sort(rng.choice(ids, max_per_split, replace=False))
            for start in range(0, len(ids), batch_size):
                ix = ids[start:start + batch_size]
                truth = torch.from_numpy(a["y"][ix]).to(device)
                neural = ensemble_predictions([m for m, _ in models], a, ix, samples, device, gen)
                # All members get equal sample weight; no test-set ensemble tuning.
                random_walk = torch.from_numpy(crw.sample(a["meta"][ix], a["aux"][ix, 4:6], samples, rng)).to(device)
                analog = torch.from_numpy(analogue.sample(a["x"][ix], a["meta"][ix], a["aux"][ix, 4:6], samples, rng)).to(device)
                zero = torch.linalg.vector_norm(truth, dim=-1).cpu().numpy()
                cv = a["aux"][ix, 4:6] * a["meta"][ix, 4:5]
                metrics = {"dont_move": zero,
                           "constant_velocity": np.linalg.norm(cv - a["y"][ix], axis=-1),
                           "crw": energy_score(random_walk, truth).cpu().numpy(),
                           "analogue": energy_score(analog, truth).cpu().numpy(),
                           "transformer": energy_score(neural, truth).cpu().numpy()}
                if len(models) == 1 and getattr(models[0][0], "family", None) == "moveformer_author_core_local_adapter":
                    # Isolate the learned selection benefit from the movement
                    # proposal itself. Independent RNG preserves shared baselines.
                    prng = np.random.default_rng(seed + 40000 + int(ix[0]))
                    prior = torch.from_numpy(models[0][0].proposal.sample(a["meta"][ix], a["aux"][ix, 4:6], samples, prng)).to(device)
                    metrics["moveformer_proposal"] = energy_score(prior, truth).cpu().numpy()
                # Fixed equal mixture. Its weight is declared before inspecting
                # tests; validation reports permit comparison with each member.
                hybrid = torch.cat([analog[:, :samples // 2], neural[:, :samples // 2]], dim=1)
                metrics["hybrid"] = energy_score(hybrid, truth).cpu().numpy()
                centre = neural.mean(1)
                radii = torch.linalg.vector_norm(neural - centre[:, None], dim=-1)
                error = torch.linalg.vector_norm(truth - centre, dim=-1)
                coverage50 = (error <= radii.quantile(.5, dim=1)).cpu().numpy()
                coverage90 = (error <= radii.quantile(.9, dim=1)).cpu().numpy()
                hcentre = hybrid.mean(1)
                hradii = torch.linalg.vector_norm(hybrid - hcentre[:, None], dim=-1)
                herror = torch.linalg.vector_norm(truth - hcentre, dim=-1)
                hcoverage50 = (herror <= hradii.quantile(.5, dim=1)).cpu().numpy()
                hcoverage90 = (herror <= hradii.quantile(.9, dim=1)).cpu().numpy()
                for k, idx in enumerate(ix):
                    row = {"window": int(idx), "split": split, "study": int(a["meta"][idx, 2]),
                           "animal": int(a["meta"][idx, 3]), "horizon_hours": int(a["meta"][idx, 4]),
                           "radial_coverage50": bool(coverage50[k]), "radial_coverage90": bool(coverage90[k]),
                           "hybrid_radial_coverage50": bool(hcoverage50[k]), "hybrid_radial_coverage90": bool(hcoverage90[k])}
                    scale = a["aux"][idx, 8] if a["aux"].shape[1] > 8 else manifest["scale_metres"]
                    row.update({f"{name}_es_m": float(value[k] * scale) for name, value in metrics.items()})
                    rows.append(row)
            print(f"Evaluated {split}: {len(ids)} windows", flush=True)
    f = pd.DataFrame(rows)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out / "per-window.parquet", index=False)
    cells = f.groupby(["split", "study", "horizon_hours"], as_index=False).agg(
        **{col: (col, "mean") for col in f if col.endswith("_es_m")},
        windows=("window", "size"), animals=("animal", "nunique"),
        radial_coverage50=("radial_coverage50", "mean"), radial_coverage90=("radial_coverage90", "mean"),
        hybrid_radial_coverage50=("hybrid_radial_coverage50", "mean"), hybrid_radial_coverage90=("hybrid_radial_coverage90", "mean"))
    for name in [c.removesuffix("_es_m") for c in cells if c.endswith("_es_m")]:
        cells[name + "_normalized_es"] = cells[name + "_es_m"] / cells.dont_move_es_m.clip(lower=1.)
    cells.to_csv(out / "per-study-horizon.csv", index=False)
    means = cells.groupby("split")[[c for c in cells if c.endswith("_normalized_es") or "radial_coverage" in c]].mean()
    result = {"protocol": manifest["protocol"], "samples_requested": samples,
              "samples_neural": samples, "checkpoints": [str(p) for p in checkpoints],
              "checkpoint_info": checkpoint_info,
              "windows_sha256": manifest["windows_sha256"], "seed": seed, "max_per_split": max_per_split,
              "driver_features": len(manifest.get("driver_names", [])),
              "identity_audit_sha256":hashlib.sha256(Path(identity_audit).read_bytes()).hexdigest() if identity_audit else None,
              "identity_exclusions":audit_info.get("excluded_by_split") if audit_info else None,
              "fresh_only":fresh_only,"fresh_studies":sorted(fresh) if fresh is not None else None,
              "weather_enabled": manifest.get("weather_enabled", False),
              "vegetation": {k: v for k, v in manifest.get("vegetation", {}).items() if k != "source"},
              "satellite": {k: v for k, v in manifest.get("satellite", {}).items() if k != "source"},
              "macro_by_split": means.to_dict(orient="index"),
              "notes": ["Local split results cannot be compared numerically with official MoveBench scores.",
                        "Distances use a spherical equidistant local projection; not the unavailable official evaluator.",
                        "Radial coverage is around the sample mean, not highest-density-region coverage.",
                        "Unprefixed radial_coverage fields describe the transformer ensemble; hybrid_radial_coverage fields describe the fixed mixture.",
                        "Input provenance is recorded in the checkpoint and dataset manifest; historical environmental release vintages are not established."]}
    # Paired hierarchical bootstrap: resample studies, then animals within each
    # selected study. Preserve their windows/horizons together. Negative is better.
    cis = {}
    bootstrap_rng = np.random.default_rng(seed + 10000)
    for split, g in f.groupby("split"):
        names = ["transformer", "analogue", "hybrid", "crw", "dont_move"]
        columns = [name + "_es_m" for name in names]
        blocks = []
        for study, gg in g.groupby("study"):
            animals = sorted(gg.animal.unique()); horizons = sorted(gg.horizon_hours.unique())
            aid, hid = {a: i for i, a in enumerate(animals)}, {h: i for i, h in enumerate(horizons)}
            values = np.zeros((len(animals), len(horizons), len(names)))
            counts = np.zeros((len(animals), len(horizons)))
            for (animal, horizon), cell in gg.groupby(["animal", "horizon_hours"]):
                values[aid[animal], hid[horizon]] = cell[columns].sum().to_numpy()
                counts[aid[animal], hid[horizon]] = len(cell)
            blocks.append((values, counts))
        draws = []
        for _ in range(300):
            ds = []
            for study_ix in bootstrap_rng.integers(0, len(blocks), len(blocks)):
                values, counts = blocks[study_ix]
                weights = bootstrap_rng.multinomial(len(values), np.full(len(values), 1 / len(values)))
                sums = np.tensordot(weights, values, axes=(0, 0))
                sizes = weights @ counts
                valid = sizes > 0
                denom = np.maximum(sums[valid, -1], sizes[valid])
                delta = (sums[valid, :3] - sums[valid, 3:4]) / denom[:, None]
                ds.extend(delta.tolist())
            draws.append(np.mean(ds, axis=0))
        intervals = np.quantile(draws, [.025, .975], axis=0)
        cis[split] = {"paired_delta_vs_crw_ci95": {name: intervals[:, i].tolist() for i, name in enumerate(names[:3])},
                      "bootstrap_unit": "study, then animal within study", "replicates": 300}
    result["uncertainty"] = cis
    dump(out / "metrics.json", result)
    return result


def export(root, checkpoint=None, output="runs/forecasts.json", count=100, samples=64, split="test_individual", seed=2026, forecaster="transformer", checkpoints=None):
    torch.set_num_threads(min(8, torch.get_num_threads()))
    a, manifest = load_windows(root)
    paths = checkpoints or [checkpoint]
    loaded = [load_model(p, "cpu") for p in paths]
    models = [m for m, ck in loaded]
    for _, ck in loaded:
        if ck["manifest"]["windows_sha256"] != manifest["windows_sha256"]:
            raise ValueError("Checkpoint/data mismatch")
    ids = np.flatnonzero(a["meta"][:, 0] == SPLITS[split])
    # Reproducible evenly spaced cases cover animals and horizons, with no
    # selection based on outcome or forecast accuracy.
    if len(ids) > count:
        ids = ids[np.linspace(0, len(ids) - 1, count, dtype=int)]
    gen = torch.Generator().manual_seed(seed)
    obs = pd.read_parquet(Path(root) / "processed" / "observations.parquet")
    train = a["meta"][:, 0] == SPLITS["train"]
    rng = np.random.default_rng(seed)
    crw = CorrelatedRandomWalk().fit(a["y"][train], a["meta"][train], a["aux"][train, 4:6]) if forecaster == "crw" else None
    analogue = AnalogueForecaster().fit(a["x"][train], a["y"][train], a["meta"][train], a["aux"][train, 4:6]) if forecaster in ["analogue", "hybrid"] else None
    records = []
    with torch.inference_mode():
        for idx in ids:
            ix = np.asarray([idx])
            scale = a["aux"][idx, 8] if a["aux"].shape[1] > 8 else manifest["scale_metres"]
            if forecaster == "crw":
                cloud = crw.sample(a["meta"][ix], a["aux"][ix, 4:6], samples, rng)[0]
            elif forecaster == "analogue":
                cloud = analogue.sample(a["x"][ix], a["meta"][ix], a["aux"][ix, 4:6], samples, rng)[0]
            elif forecaster == "hybrid":
                n = samples // 2
                neural = ensemble_predictions(models, a, ix, samples - n, "cpu", gen)[0].numpy()
                analog = analogue.sample(a["x"][ix], a["meta"][ix], a["aux"][ix, 4:6], n, rng)[0]
                cloud = np.concatenate([neural, analog])
            else:
                cloud = ensemble_predictions(models, a, ix, samples, "cpu", gen)[0].numpy()
            cloud = cloud * scale
            lat, lon, t, target_t = a["aux"][idx, :4]
            origin_row, target_row = a["aux"][idx, 6:8].astype(int)
            g = obs.loc[obs.animal == obs.loc[origin_row, "animal"]]
            g = g.loc[g.segment == obs.loc[origin_row, "segment"]]
            history = g.loc[g.index <= origin_row].tail(manifest["context"])
            future = g.loc[(g.index > origin_row) & (g.index <= target_row)]
            def points(frame):
                return [{"t": float(np.float64(x.timestamp.value) / 1e9), "timestamp_utc": x.timestamp.isoformat(),
                         "lat": x.lat, "lon": x.lon} for x in frame.itertuples()]
            records.append({"window": int(idx), "animal": obs.loc[origin_row, "animal"], "species": obs.loc[origin_row, "species"],
                "forecast_origin": {"t": t, "lat": lat, "lon": lon}, "target_t": target_t,
                "horizon_hours": int(a["meta"][idx, 4]), "history": points(history),
                "observed_future": points(future), "forecast_endpoints_latlon": unproject(cloud, lat, lon).tolist(),
                "truth_latlon": unproject(a["y"][idx] * scale, lat, lon).tolist()})
    dump(output, {"schema": "wildcast.forecasts.v1", "protocol": manifest["protocol"], "split": split,
                  "checkpoints": [str(p) for p in paths], "forecaster": forecaster, "coordinate_order": "latitude,longitude", "time_unit": "unix seconds UTC (source timezone caveat applies)",
                  "forecast_type": "endpoint distributions; connecting samples is not a validated trajectory forecast",
                  "records": records})
    return {"output": str(output), "forecasts": len(records)}
