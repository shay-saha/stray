"""Observed-only, phase-purged future sequences and frozen causal context cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import SPLITS, distance, project
from .identity import case_fingerprint, verify_audit
from .sources import dump
from .train import load_windows
from .trajectory import GRID_HOURS, TrajectoryFlow


def phases(observations, manifest):
    phase = np.full(len(observations), SPLITS["train"], dtype=np.int8)
    for name, ix in observations.groupby("animal", sort=False).groups.items():
        ix = np.asarray(ix)
        assignment = manifest["animal_assignments"][name]
        if assignment == "temporal":
            phase[ix[int(len(ix) * .65):int(len(ix) * .8)]] = SPLITS["val_temporal"]
            phase[ix[int(len(ix) * .8):]] = SPLITS["test_temporal"]
        else:
            phase[ix] = SPLITS[assignment]
    return phase


def observed_sequences(observations, a, manifest, max_steps=20, excluded_windows=()):
    """Select actual future GPS records; never interpolate labels or cross phases."""
    if max_steps < 4:
        raise ValueError("At least four future slots required")
    f = observations.copy()
    f["phase"] = phases(f, manifest)
    allowed = np.isin(a["meta"][:, 0], [0, 1, 3])
    excluded_origins = set(a["aux"][np.asarray(excluded_windows, dtype=int), 6].astype(int))
    allowed &= ~np.isin(a["aux"][:, 6].astype(int), list(excluded_origins))
    ids = np.flatnonzero(allowed)
    origins, where = np.unique(a["aux"][ids, 6].astype(int), return_index=True)
    representatives = ids[where]
    position = {int(v): i for i, v in enumerate(origins)}
    times = np.zeros((len(origins), max_steps), np.float32)
    xy = np.zeros((len(origins), max_steps, 2), np.float32)
    rows = np.full((len(origins), max_steps), -1, np.int64)
    mask = np.zeros_like(times, dtype=bool)
    targets = {}
    for window in ids:
        targets.setdefault(int(a["aux"][window, 6]), set()).add(int(a["aux"][window, 7]))
    for _, g in f.groupby(["animal", "segment", "phase"], sort=False):
        indices = g.index.to_numpy()
        ts = g.timestamp.astype("int64").to_numpy() / 1e9
        lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
        for i, origin in enumerate(indices):
            if int(origin) not in position:
                continue
            r = position[int(origin)]
            end = np.searchsorted(ts, ts[i] + 169 * 3600, side="right")
            future = np.arange(i + 1, end)
            if not len(future):
                continue
            desired = ts[i] + np.asarray(GRID_HOURS) * 3600
            right = np.searchsorted(ts, desired).clip(i + 1, end - 1)
            left = (right - 1).clip(i + 1, end - 1)
            choices = np.where(abs(ts[right] - desired) < abs(ts[left] - desired), right, left)
            # Preserve all actual endpoint labels used by the canonical evaluator.
            essential = [j for j in future if int(indices[j]) in targets[int(origin)]]
            selected = sorted(set(essential + choices.tolist()))
            if len(selected) > max_steps:
                optional = [j for j in selected if j not in essential]
                pick = np.linspace(0, len(optional) - 1, max_steps - len(essential)).round().astype(int)
                selected = sorted(essential + [optional[j] for j in pick])
            n = len(selected)
            times[r, :n] = (ts[selected] - ts[i]) / 3600
            # Padding times repeat the final time and are always masked out.
            times[r, n:] = times[r, n - 1]
            xy[r, :n] = project(lat[selected], lon[selected], lat[i], lon[i]) / a["aux"][representatives[r], 8]
            rows[r, :n] = indices[selected]
            mask[r, :n] = True
    valid = mask.any(-1)
    return {"window_ids": representatives[valid], "times": times[valid], "future_xy": xy[valid],
            "future_rows": rows[valid], "mask": mask[valid]}


def speed_limits(observations, manifest):
    """Fit conservative interval-speed envelopes on training observations only."""
    f = observations.copy()
    f["phase"] = phases(f, manifest)
    train = f[f.phase == SPLITS["train"]]
    speeds = {}
    for _, g in train.groupby(["animal", "segment"], sort=False):
        if len(g) < 2:
            continue
        dt = g.timestamp.diff().dt.total_seconds().to_numpy()[1:]
        d = distance(g.lat.to_numpy()[1:], g.lon.to_numpy()[1:], g.lat.to_numpy()[:-1], g.lon.to_numpy()[:-1])
        value = d / np.maximum(dt, 1)
        speeds.setdefault(g.species.iloc[0], []).extend(value[np.isfinite(value)].tolist())
    limits = np.full(len(manifest["species"]) + 1, 100., np.float32)
    details = {}
    for name, species in manifest["species"].items():
        values = np.asarray(speeds.get(name, []))
        if len(values):
            limits[species] = np.clip(np.quantile(values, .999) * 3, 5, 100)
        details[name] = {"limit_mps": float(limits[species]), "training_intervals": len(values)}
    return limits.tolist(), details


def shared_experiences(a, sequences, context, identity_groups, entries_per_animal=8):
    """Training-only episodic values; select by date, never by forecast errors."""
    train = np.flatnonzero(a["meta"][sequences["window_ids"], 0] == 0)
    animal = a["meta"][sequences["window_ids"], 3]
    selected = []
    for value in np.unique(animal[train]):
        rows = train[animal[train] == value]
        positions = np.linspace(0, len(rows) - 1, min(entries_per_animal, len(rows))).round().astype(int)
        selected.extend(rows[positions].tolist())
    selected = np.asarray(selected, int)
    ids = sequences["window_ids"][selected]
    values = np.zeros((len(selected), 16), np.float32)
    for j, row in enumerate(selected):
        valid = sequences["mask"][row]
        t, xy = sequences["times"][row, valid], sequences["future_xy"][row, valid]
        for k, h in enumerate((1., 12., 24., 168.)):
            nearest = np.argmin(abs(t - h))
            if abs(t[nearest] - h) <= min(1., h * .25):
                v = xy[nearest] / h ** .5
                values[j, k * 3:k * 3 + 2] = np.sign(v) * np.log1p(abs(v))
                values[j, k * 3 + 2] = 1
        step = np.diff(xy, axis=0, prepend=np.zeros((1, 2)))
        dt = np.diff(t, prepend=0).clip(.001)
        r = np.linalg.norm(step, axis=-1) / np.sqrt(dt)
        labels = (r > .25).astype(int) + (r > 2.).astype(int)
        values[j, 12:15] = np.bincount(labels, minlength=3) / len(labels)
        values[j, 15] = np.log1p(t[-1])
    return {"bank_keys": torch.from_numpy(np.asarray(context[selected]).copy()), "bank_values": torch.from_numpy(values),
            "bank_species": torch.from_numpy(a["meta"][ids, 1].astype(np.int64)),
            "bank_identity": torch.tensor([identity_groups[i] for i in a["meta"][ids, 3]], dtype=torch.long)}, ids


def prepare(root, checkpoint, output, identity_audit, fields=None, max_steps=20, batch_size=256, device="auto", taxonomy_root=None):
    from .raster_context import SatelliteFields
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest = load_windows(root)
    audit = json.loads(Path(identity_audit).read_text())
    verify_audit(audit, manifest, a)
    observations = pd.read_parquet(Path(root) / "processed/observations.parquet")
    data = observed_sequences(observations, a, manifest, max_steps, audit["excluded_windows"])
    limits, limit_report = speed_limits(observations, manifest)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck.get("model_family", "wildcast_transformer") != "wildcast_transformer":
        raise ValueError("Trajectory context requires a MovementTransformer checkpoint")
    # Species/identity/schema must agree even when driver-enriched files differ.
    for key in ("species", "animals", "observations_sha256", "context"):
        if ck["manifest"][key] != manifest[key]:
            raise ValueError(f"Backbone and trajectory data disagree on {key}")
    if ck["manifest"].get("driver_names") != manifest.get("driver_names", [])[:ck["config"]["n_drivers"]]:
        raise ValueError("Backbone driver ordering differs")
    field = SatelliteFields.from_cache(fields) if fields else None
    names = [name for name, i in sorted(manifest["animals"].items(), key=lambda item: item[1])]
    canonical = [audit["fold_groups"].get(name, name) for name in names]
    vocabulary = {name: i for i, name in enumerate(sorted(set(canonical)))}
    identities = [vocabulary[name] for name in canonical]
    taxonomy = Path(taxonomy_root) if taxonomy_root else Path(root).parent / "combined-v2/environment/taxonomy"
    records = {}
    taxon_sources = {}
    for name in manifest["species"]:
        path = taxonomy / (name.replace(" ", "_") + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            records[name] = record.get("response", {})
            taxon_sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    class_names = {value: i + 1 for i, value in enumerate(sorted({v["class"] for v in records.values() if v.get("class")}))}
    order_names = {value: i + 1 for i, value in enumerate(sorted({v["order"] for v in records.values() if v.get("order")}))}
    classes, orders = [0] * len(limits), [0] * len(limits)
    for name, i in manifest["species"].items():
        classes[i] = class_names.get(records.get(name, {}).get("class"), 0)
        orders[i] = order_names.get(records.get(name, {}).get("order"), 0)
    config = {"backbone": ck["config"], "n_drivers": len(manifest["driver_names"]),
              "horizon_driver_indices": [i for i, v in enumerate(manifest["driver_names"]) if "future_" in v or "horizon" in v],
              "speed_limits": limits, "gps_allowance_m": 100., "grid_hours": list(GRID_HOURS),
              "field_metadata": field.metadata if field is not None else [], "width": 128, "layers": 3,
              "ode_steps": 8, "dropout": .05, "identity_groups": identities,
              "species_classes": classes, "species_orders": orders}
    model = TrajectoryFlow(config, field).to(device).eval()
    model.backbone.load_state_dict(ck["model"])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    context = np.lib.format.open_memmap(out / "context.npy", mode="w+", dtype=np.float32,
                                      shape=(len(data["window_ids"]), model.backbone.head[0].normalized_shape[0] + config["n_drivers"]))
    with torch.inference_mode():
        for start in range(0, len(context), batch_size):
            ids = data["window_ids"][start:start + batch_size]
            context[start:start + len(ids)] = model.observed_context(a, ids, device).cpu().numpy()
            if start % (batch_size * 50) == 0:
                print(f"Trajectory context {start}/{len(context)}", flush=True)
    context.flush()
    bank, bank_ids = shared_experiences(a, data, context, identities)
    config["memory_entries"] = len(bank_ids)
    model.cpu()
    for key, value in bank.items():
        setattr(model, key, value)
    np.savez_compressed(out / "sequences.npz", **data)
    torch.save({"config": config, "model": model.cpu().state_dict()}, out / "initial.pt")
    counts = {k: int((a["meta"][data["window_ids"], 0] == v).sum()) for k, v in SPLITS.items()}
    report = {"source_data": str(root), "case_fingerprint": case_fingerprint(a), "windows_sha256": manifest["windows_sha256"],
              "identity_audit_sha256": hashlib.sha256(Path(identity_audit).read_bytes()).hexdigest(),
              "backbone_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(), "config": config,
              "origins": len(context), "origin_counts": counts, "future_steps": int(data["mask"].sum()),
              "sequences_sha256": hashlib.sha256((out / "sequences.npz").read_bytes()).hexdigest(),
              "context_sha256": hashlib.sha256((out / "context.npy").read_bytes()).hexdigest(),
              "speed_limits": limit_report,
              "shared_memory": {"entries": len(bank_ids), "source_window_ids": bank_ids.tolist(),
                                "source_splits": ["train"], "entries_per_animal": 8,
                                "query_exclusion": "All entries from the canonical physical query identity are masked in training and inference.",
                                "values": "Observed training endpoint displacements with presence masks and weak displacement-regime frequencies"},
              "taxonomy_sources_sha256": taxon_sources, "class_vocabulary": class_names, "order_vocabulary": order_names,
              "protocol": "Train/validation only. Actual observed future coordinates and timestamps, same animal/segment/phase. No interpolated labels or test forecasts.",
              "limitations": ["Training-derived speed envelopes are modelling assumptions, not established physiological limits.",
                              "Previous-year habitat embeddings supply context, not hard coastline/fence masks.",
                              "Displacement-regime targets are weak proxies, not biological behaviour labels."]}
    dump(out / "manifest.json", report)
    return report
