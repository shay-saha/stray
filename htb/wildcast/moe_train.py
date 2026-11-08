"""Controlled endpoint specialization: fixed backbone, targets, loss and cases."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .identity import case_fingerprint
from .model import energy_score, mixture_energy_score, nll
from .moe import CategoryMoE
from .sources import dump
from .train import load_windows


def train(root, cache, output, mode="family", hidden=96, epochs=25, patience=5, batch_size=512,
          lr=.0001, seed=42, nll_weight=.01, regularization=.001, device="auto", max_train=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    cache, out = Path(cache), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    a, manifest = load_windows(root)
    info = json.loads((cache / "manifest.json").read_text())
    if info["windows_sha256"] != manifest["windows_sha256"] or info["case_fingerprint"] != case_fingerprint(a):
        raise ValueError("MoE cache and source cases differ")
    for filename, expected in info["files_sha256"].items():
        if hashlib.sha256((cache / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("MoE cache checksum mismatch: " + filename)
    initial = torch.load(cache / "bases.pt", map_location="cpu", weights_only=True)
    config = {**initial["config"], "mode": mode, "hidden": hidden}
    model = CategoryMoE(config).to(device)
    for base, state in zip(model.bases, initial["states"]):
        base.load_state_dict(state)
    ids = np.load(cache / "ids.npy")
    features = np.load(cache / "features.npy", mmap_mode="r")
    base = np.load(cache / "base.npy", mmap_mode="r")
    train_rows = np.flatnonzero(a["meta"][ids, 0] == 0)
    val_rows = np.flatnonzero(np.isin(a["meta"][ids, 0], [1, 3]))
    if max_train and len(train_rows) > max_train:
        train_rows = np.sort(np.random.default_rng(seed).choice(train_rows, max_train, replace=False))
    if not len(train_rows) or not len(val_rows):
        raise ValueError("Train and validation cases required")
    scales = a["aux"][:, 8]
    _, tinverse, counts = np.unique(a["meta"][ids[train_rows]][:, [2, 4]], axis=0, return_inverse=True, return_counts=True)
    loss_weights = np.ones(len(train_rows), np.float32)
    displacement = np.linalg.norm(a["y"][ids[train_rows]], axis=-1) * scales[ids[train_rows]]
    for cell in np.unique(tinverse):
        mask = tinverse == cell
        loss_weights[mask] = scales[ids[train_rows[mask]]] / max(float(displacement[mask].mean()), 1.)

    def preload(rows):
        selected = ids[rows]
        return (torch.as_tensor(np.array(features[rows]), device=device), torch.as_tensor(np.array(base[rows]), device=device),
                torch.as_tensor(a["meta"][selected, 1], device=device, dtype=torch.long),
                torch.as_tensor(a["y"][selected], device=device))

    train_data, val_data = preload(train_rows), preload(val_rows)
    ew = torch.tensor(loss_weights, device=device)
    probability = torch.tensor(1 / counts[tinverse], device=device, dtype=torch.float32)
    probability /= probability.sum()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1))
    _, vinverse = np.unique(a["meta"][ids[val_rows]][:, [0, 2, 4]], axis=0, return_inverse=True)
    vd = np.linalg.norm(a["y"][ids[val_rows]], axis=-1) * scales[ids[val_rows]]
    config_train = {"seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr, "mode": mode,
                    "train_cases": len(train_rows), "validation_cases": len(val_rows), "nll_weight": nll_weight,
                    "regularization": regularization, "parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in parameters),
                    "cache_manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest(),
                    "identity_audit_sha256": info["identity_audit_sha256"],
                    "objective": "Study/horizon-balanced proper stratified endpoint energy + NLL + residual shrinkage",
                    "selection": "Same complete audited validation cases and 32 coupled forecast draws for every variant",
                    "training_support": info["training_support"],
                    "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    dump(out / "config.json", {"model": config, "training": config_train})

    def validate():
        model.eval()
        generator = torch.Generator(device=device).manual_seed(2026)
        values, usage = [], torch.zeros(len(model.experts), device=device)
        with torch.inference_mode():
            for start in range(0, len(val_rows), batch_size):
                f, p, species, truth = [t[start:start + batch_size] for t in val_data]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    samples = model.sample_from_features(f, p, species, 32, generator)
                values.append(energy_score(samples, truth).cpu().numpy())
                usage += model.last_usage.sum(0)
        scores = np.concatenate(values) * scales[ids[val_rows]]
        cells = [scores[vinverse == k].mean() / max(vd[vinverse == k].mean(), 1.) for k in np.unique(vinverse)]
        return float(np.mean(cells)), (usage / len(val_rows)).cpu().tolist()

    def save(epoch, score):
        torch.save({"model_family": model.family, "config": config, "model": model.state_dict(), "training": config_train,
                    "manifest": manifest, "epoch": epoch, "val_score": score}, out / "best.pt.tmp")
        (out / "best.pt.tmp").replace(out / "best.pt")

    best, usage = validate()
    save(0, best)
    dump(out / "initial.json", {"val_macro_normalized_energy": best, "mean_expert_weight": usage})
    print(json.dumps({"epoch": 0, "val_macro_normalized_energy": best, "mean_expert_weight": usage}), flush=True)
    history, stale = [], 0
    batch_generator = torch.Generator(device=device).manual_seed(seed + 200)
    loss_generator = torch.Generator(device=device).manual_seed(seed + 350)
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        draws = torch.multinomial(probability, len(train_rows), replacement=True, generator=batch_generator)
        losses = []
        for selected in draws.split(batch_size):
            f, p, species, truth = [value[selected] for value in train_data]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                params = model(f, p, species)
            loss = (mixture_energy_score(params, truth, repeats=1, generator=loss_generator) * ew[selected]).mean()
            if nll_weight:
                loss = loss + nll_weight * nll(params, truth).mean()
            loss = loss + regularization * model.last_regularization
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite category-expert loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        score, usage = validate()
        row = {"epoch": epoch + 1, "loss": float(torch.stack(losses).mean().cpu()),
               "val_macro_normalized_energy": score, "mean_expert_weight": usage,
               "seconds": round(time.monotonic() - started, 2)}
        history.append(row)
        dump(out / "history.json", history)
        print(json.dumps(row), flush=True)
        if score < best:
            best, stale = score, 0
            save(epoch + 1, best)
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            break
    return {"output": str(out), "best_validation_score": best, "epochs": len(history),
            "trainable_parameters": config_train["trainable_parameters"]}
