"""Conditional flow matching on actual irregularly sampled future sequences."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .identity import case_fingerprint
from .model import energy_score
from .sources import dump
from .train import load_windows
from .trajectory import GRID_HOURS, TrajectoryFlow, decode_path, encode_steps, geographic_steps


def load_cache(root, cache):
    cache = Path(cache)
    a, manifest = load_windows(root)
    info = json.loads((cache / "manifest.json").read_text())
    if info["windows_sha256"] != manifest["windows_sha256"] or info["case_fingerprint"] != case_fingerprint(a):
        raise ValueError("Trajectory cache does not match source windows")
    for filename, key in (("sequences.npz", "sequences_sha256"), ("context.npy", "context_sha256")):
        if hashlib.sha256((cache / filename).read_bytes()).hexdigest() != info[key]:
            raise ValueError(f"Trajectory cache checksum mismatch: {filename}")
    with np.load(cache / "sequences.npz") as f:
        sequences = {key: f[key] for key in f.files}
    sequences["context"] = np.load(cache / "context.npy", mmap_mode="r")
    return a, manifest, sequences, info


def batch_tensors(a, seq, rows, device):
    ids = seq["window_ids"][rows]
    return (torch.as_tensor(np.asarray(seq["context"][rows]).copy(), device=device),
            torch.as_tensor(seq["times"][rows], device=device),
            torch.as_tensor(seq["mask"][rows], device=device),
            torch.as_tensor(a["aux"][ids], device=device, dtype=torch.float64),
            torch.as_tensor(a["aux"][ids, 4:6], device=device, dtype=torch.float32),
            torch.as_tensor(a["anchors"][ids], device=device),
            torch.as_tensor(a["meta"][ids, 1], device=device, dtype=torch.long))


def validation_cases(a, seq, maximum, seed=2718):
    ids = seq["window_ids"]
    eligible = np.flatnonzero(np.isin(a["meta"][ids, 0], [1, 3]))
    if maximum and len(eligible) > maximum:
        # Proportional stratification with a floor per split/study, deterministic
        # across experiments. This is a selection subset, not the final report.
        rng = np.random.default_rng(seed)
        keys = a["meta"][ids[eligible]][:, [0, 2]]
        groups, inverse = np.unique(keys, axis=0, return_inverse=True)
        pieces = []
        for j in range(len(groups)):
            members = eligible[inverse == j]
            n = min(len(members), max(1, maximum // len(groups)))
            pieces.append(rng.choice(members, n, replace=False))
        eligible = np.sort(np.concatenate(pieces))
    origin_to_row = {int(a["aux"][ids[r], 6]): int(r) for r in eligible}
    cases = np.asarray([i for i in np.flatnonzero(np.isin(a["meta"][:, 0], [1, 3]))
                        if int(a["aux"][i, 6]) in origin_to_row], dtype=int)
    return eligible, cases, origin_to_row


def validate(model, a, seq, rows, cases, origin_to_row, batch_size, samples, ode_steps, device):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(2026)
    case_by_origin = {}
    for window in cases:
        case_by_origin.setdefault(int(a["aux"][window, 6]), []).append(window)
    scores = np.full(len(a["meta"]), np.nan)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            context, _, _, aux, velocity, anchors, species = batch_tensors(a, seq, selected, device)
            times = torch.tensor(GRID_HOURS, device=device).expand(len(selected), -1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                result = model.sample_paths(context, times, torch.ones_like(times, dtype=torch.bool), aux,
                                            velocity, anchors, species, samples, gen, ode_steps,
                                            identity=model.identity_groups[torch.as_tensor(a["meta"][seq["window_ids"][selected], 3], device=device, dtype=torch.long)])
            for j, row in enumerate(selected):
                origin = int(a["aux"][seq["window_ids"][row], 6])
                windows = np.asarray(case_by_origin[origin])
                horizons = [GRID_HOURS.index(float(v)) for v in a["meta"][windows, 4]]
                samples_xy = result["xy"][j, :, horizons].transpose(0, 1)
                truth = torch.as_tensor(a["y"][windows], device=device)
                scores[windows] = energy_score(samples_xy, truth).cpu().numpy() * a["aux"][windows, 8]
    meta = a["meta"][cases]
    _, inverse = np.unique(meta[:, [0, 2, 4]], axis=0, return_inverse=True)
    stationary = np.linalg.norm(a["y"][cases], axis=-1) * a["aux"][cases, 8]
    values = scores[cases]
    if not np.isfinite(values).all():
        raise FloatingPointError("Missing/nonfinite trajectory validation predictions")
    return float(np.mean([values[inverse == j].mean() / max(stationary[inverse == j].mean(), 1.)
                          for j in np.unique(inverse)]))


def train(root, cache, output, epochs=30, batch_size=192, width=128, layers=3, lr=.0003,
          seed=42, patience=7, device="auto", validation_origins=2048, validation_samples=8,
          ode_steps=6, max_train=0, state_weight=.05, reconstruction_weight=.05, prior_kind="copula",
          objective="flow", training_samples=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest, seq, info = load_cache(root, cache)
    initial = torch.load(Path(cache) / "initial.pt", map_location="cpu", weights_only=True)
    config = {**initial["config"], "width": width, "layers": layers, "ode_steps": ode_steps, "prior_kind": prior_kind}
    model = TrajectoryFlow(config).to(device)
    reusable = {k: v for k, v in initial["model"].items() if k.startswith(("backbone.", "fields.", "speed_limits", "bank_"))}
    model.load_state_dict(reusable, strict=False)
    train_rows = np.flatnonzero(a["meta"][seq["window_ids"], 0] == 0)
    rng = np.random.default_rng(seed)
    if max_train and len(train_rows) > max_train:
        train_rows = np.sort(rng.choice(train_rows, max_train, replace=False))
    val_rows, val_cases, mapping = validation_cases(a, seq, validation_origins)
    if not len(train_rows) or not len(val_rows):
        raise ValueError("Both training and validation origins required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # GPU-preload the modest fixed corpus; context generation is frozen/cached.
    dataset = batch_tensors(a, seq, train_rows, device)
    target_xy = torch.as_tensor(seq["future_xy"][train_rows], device=device)
    train_identity = model.identity_groups[torch.as_tensor(a["meta"][seq["window_ids"][train_rows], 3], device=device, dtype=torch.long)]
    with torch.no_grad():
        physical_steps = geographic_steps(target_xy, dataset[3])
        targets = encode_steps(physical_steps, dataset[1], dataset[3][:, 8], model.speed_limits[dataset[-1]],
                               config["gps_allowance_m"]) * dataset[2][..., None]
        dt = torch.diff(dataset[1], prepend=torch.zeros_like(dataset[1][:, :1]), dim=-1).clamp_min(1e-5)
        radii = physical_steps.norm(dim=-1) / (dataset[3][:, 8:9] * dt.sqrt()).float()
        state_labels = (radii > .25).long() + (radii > 2.).long()
        bounds = model.speed_limits[dataset[-1], None] * dt * 3600 + config["gps_allowance_m"]
        outside = ((physical_steps.norm(dim=-1) > bounds) & dataset[2]).sum().item()
    _, inverse, counts = np.unique(a["meta"][seq["window_ids"][train_rows], 2], return_inverse=True, return_counts=True)
    probability = 1 / counts[inverse]
    probability = torch.tensor(probability / probability.sum(), device=device)
    endpoint_truth = endpoint_mask = endpoint_weights = None
    if objective == "energy":
        if training_samples < 2:
            raise ValueError("Energy training requires at least two independent path samples")
        origin_lookup = {int(a["aux"][seq["window_ids"][r], 6]): j for j, r in enumerate(train_rows)}
        ep = np.zeros((len(train_rows), 4, 2), np.float32)
        em = np.zeros((len(train_rows), 4), bool)
        for w in np.flatnonzero(a["meta"][:, 0] == 0):
            row = origin_lookup.get(int(a["aux"][w, 6]))
            if row is not None:
                h = (1, 12, 24, 168).index(int(a["meta"][w, 4]))
                ep[row, h] = a["y"][w]
                em[row, h] = True
        ew = np.zeros_like(em, dtype=np.float32)
        studies = a["meta"][seq["window_ids"][train_rows], 2]
        scales = a["aux"][seq["window_ids"][train_rows], 8]
        for study in np.unique(studies):
            for h in range(4):
                valid = (studies == study) & em[:, h]
                if valid.any():
                    denominator = max(float((np.linalg.norm(ep[valid, h], axis=-1) * scales[valid]).mean()), 1.)
                    ew[valid, h] = scales[valid] / denominator
        endpoint_truth = torch.as_tensor(ep, device=device)
        endpoint_mask = torch.as_tensor(em, device=device)
        endpoint_weights = torch.as_tensor(ew, device=device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1))
    training = {"seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in parameters), "epochs": epochs, "batch_size": batch_size,
                "lr": lr, "training_origins": len(train_rows), "validation_origins": len(val_rows),
                "validation_cases": len(val_cases), "validation_samples": validation_samples,
                "validation_case_ids_sha256": hashlib.sha256(val_cases.tobytes()).hexdigest(),
                "cache_manifest_sha256": hashlib.sha256((Path(cache) / "manifest.json").read_bytes()).hexdigest(),
                "identity_audit_sha256": info["identity_audit_sha256"], "target_steps_outside_envelope": outside,
                "state_weight": state_weight, "reconstruction_weight": reconstruction_weight,
                "objective": objective, "training_samples": training_samples,
                "selection": "Fixed train/validation-only origin subset, macro normalized endpoint energy; no fresh test predictions",
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    dump(out / "config.json", {"model": config, "training": training})

    def save(epoch, score):
        torch.save({"model_family": model.family, "config": config, "model": model.state_dict(),
                    "training": training, "manifest": manifest, "epoch": epoch, "val_score": score}, out / "best.pt.tmp")
        (out / "best.pt.tmp").replace(out / "best.pt")

    best = validate(model, a, seq, val_rows, val_cases, mapping, min(batch_size, 64), validation_samples, ode_steps, device)
    save(0, best)
    print(json.dumps({"epoch": 0, "val_macro_normalized_energy": best, "note": f"Frozen {prior_kind} movement prior"}), flush=True)
    history, stale = [], 0
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        draws = torch.multinomial(probability, len(train_rows), replacement=True)
        losses = []
        for indices in draws.split(batch_size):
            context, times, mask, aux, velocity, anchors, species = [value[indices] for value in dataset]
            if objective == "energy":
                # Predict before accessing labels. This is a proper sample-based
                # endpoint energy objective, without denoising teacher inputs.
                grid = torch.tensor(GRID_HOURS, device=device).expand(len(indices), -1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    forecast = model.sample_paths(context, grid, torch.ones_like(grid, dtype=torch.bool), aux,
                                                  velocity, anchors, species, training_samples, None,
                                                  min(ode_steps, 2), identity=train_identity[indices])
                horizon_slots = [GRID_HOURS.index(h) for h in (1., 12., 24., 168.)]
                points = forecast["xy"][:, :, horizon_slots].transpose(1, 2).flatten(0, 1)
                score = energy_score(points, endpoint_truth[indices].flatten(0, 1)).reshape(-1, 4)
                valid = endpoint_mask[indices]
                loss = ((score * endpoint_weights[indices]).sum(-1) / valid.sum(-1).clamp_min(1)).mean()
                # Keep species adjustments small until direct forecasting
                # evidence supports departing from shared movement knowledge.
                loss = loss + .0001 * model.species_adapter.weight.square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite trajectory energy loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.detach())
                continue
            truth = targets[indices]
            prior = model.prior(context, times, mask, aux, velocity, anchors, species, 1, None)
            t = torch.rand(len(indices), device=device) * .98 + .01
            z = (1 - t[:, None, None, None]) * prior + t[:, None, None, None] * truth[:, None]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                tokens, states = model.prepare_condition(context, times, aux, mask, prior, species, train_identity[indices])
                vector = model.vector_field(z, prior, t, tokens, states, mask)
            wanted = truth[:, None] - prior
            fm = ((vector - wanted).square().mean(-1)[:, 0] * mask).sum() / mask.sum()
            ce = F.nll_loss(states.clamp_min(1e-7).log().flatten(0, 1), state_labels[indices].flatten(), reduction="none").reshape_as(mask)
            state_loss = (ce * mask).sum() / mask.sum()
            loss = fm + state_weight * state_loss
            if reconstruction_weight:
                estimate = z + (1 - t[:, None, None, None]) * vector
                positions, _ = decode_path(estimate, times, aux, model.speed_limits[species], config["gps_allowance_m"])
                residual = torch.asinh(positions[:, 0]) - torch.asinh(target_xy[indices])
                reconstruction = (F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none").mean(-1) * mask).sum() / mask.sum()
                loss = loss + reconstruction_weight * reconstruction
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite trajectory training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        score = validate(model, a, seq, val_rows, val_cases, mapping, min(batch_size, 64), validation_samples, ode_steps, device)
        row = {"epoch": epoch + 1, "train_loss": float(torch.stack(losses).mean().cpu()),
               "val_macro_normalized_energy": score, "seconds": round(time.monotonic() - started, 2)}
        history.append(row)
        dump(out / "history.json", history)
        print(json.dumps(row), flush=True)
        if score < best:
            best, stale = score, 0
            save(epoch + 1, score)
        else:
            stale += 1
        schedule.step()
        if stale >= patience:
            break
    return {"output": str(out), "best_validation_score": best, "epochs": len(history),
            "trainable_parameters": training["trainable_parameters"]}
