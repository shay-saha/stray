"""Train the upstream MoveFormer core on the same frozen local windows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .baselines import CorrelatedRandomWalk
from .data import SPLITS
from .model import energy_score
from .moveformer import LocalMoveFormer, EmpiricalStepProposal, proposal_records, provenance
from .sources import dump
from .train import DeviceBatches, load_windows


def train(root, output, epochs=100, patience=12, batch_size=512, width=128, layers=6, heads=8,
          lr=.00005, seed=42, candidates=100, validation_candidates=256, device="auto", max_train=0,
          proposal_kind="empirical", variable_context=False, warmup_steps=1000):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest = load_windows(root)
    ti = np.flatnonzero(a["meta"][:, 0] == SPLITS["train"])
    vi = np.flatnonzero(np.isin(a["meta"][:, 0], [SPLITS["val_temporal"], SPLITS["val_individual"]]))
    # The proposal is fitted on precisely the same training targets as CRW.
    proposal = (EmpiricalStepProposal() if proposal_kind == "empirical" else CorrelatedRandomWalk()).fit(a["y"][ti], a["meta"][ti], a["aux"][ti, 4:6])
    if max_train and len(ti) > max_train:
        ti = np.sort(np.random.default_rng(seed).choice(ti, max_train, replace=False))
    model = LocalMoveFormer(len(manifest["species"]), n_features=a["x"].shape[-1], n_drivers=a["drivers"].shape[-1],
                           width=width, layers=layers, heads=heads, proposal=proposal_records(proposal), variable_context=variable_context).to(device)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    config = {"model_family": model.family, "model": model.config, "seed": seed, "epochs": epochs,
              "patience": patience, "batch_size": batch_size, "lr": lr, "parameters": sum(p.numel() for p in model.parameters()),
              "windows_sha256": manifest["windows_sha256"], "upstream": provenance(), "candidates": candidates,
              "validation_candidates": validation_candidates, "max_train": max_train,
              "proposal_kind": proposal_kind, "variable_context": variable_context,
              "warmup_steps": warmup_steps,
              "training_windows": len(ti), "validation_windows": len(vi), "torch_version": str(torch.__version__),
              "precision": "bf16 network, fp32 softmax and energy" if device.startswith("cuda") else "fp32",
              "selection": "Lowest same-case study/horizon/split macro normalized energy on validation. Also save best conditional cross-entropy.",
              "proposal": "Fixed training-only species/horizon CRW; fixed candidate pools across epochs; target appended only in loss, never in forecast support.",
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    if proposal_kind == "empirical":
        config["upstream"]["evaluation"] = config["upstream"]["evaluation"].replace("CRW proposal", "empirical log-distance/turn-quantile proposal")
        config["proposal"] = config["proposal"].replace("CRW", "empirical-quantile proposal")
    dump(out / "config.json", config)
    print(f"Author core initialized: {config['parameters']} parameters; creating candidate pools", flush=True)

    def dataset(ids, n, pool_seed):
        q = proposal.sample(a["meta"][ids], a["aux"][ids, 4:6], n, np.random.default_rng(pool_seed))
        # Dataset creation uses observed features for proposals. Held-out labels
        # are carried separately solely for loss/score calculation.
        return TensorDataset(torch.from_numpy(a["x"][ids]), torch.from_numpy(a["meta"][ids, 1].astype(np.int64)),
            torch.from_numpy(a["meta"][ids, 4].astype(np.float32)), torch.from_numpy(a["aux"][ids, 4:6].astype(np.float32)),
            torch.from_numpy(a["drivers"][ids]), torch.from_numpy(q), torch.from_numpy(a["y"][ids]))
    _, inv, counts = np.unique(a["meta"][ti][:, [2, 4]], axis=0, return_inverse=True, return_counts=True)
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(1. / counts[inv], dtype=torch.double), len(ti),
        replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DeviceBatches(dataset(ti, candidates, 1776), batch_size, device, sampler)
    val_loader = DeviceBatches(dataset(vi, validation_candidates, 1777), batch_size, device)
    scales = a["aux"][vi, 8]
    dist = np.linalg.norm(a["y"][vi], axis=-1) * scales
    _, cells = np.unique(a["meta"][vi][:, [0, 2, 4]], axis=0, return_inverse=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.98)
    best, best_ce, stale = float("inf"), float("inf"), 0
    history = []
    bf16 = device.startswith("cuda")
    global_step = 0

    def checkpoint(path, epoch, score):
        tmp = out / (path + ".tmp")
        torch.save({"model_family": model.family, "model": model.state_dict(), "config": model.config,
                    "training": config, "manifest": manifest, "epoch": epoch, "val_score": score}, tmp)
        tmp.replace(out / path)

    for epoch in range(epochs):
        start = time.monotonic(); model.train(); losses = []
        for x, sp, h, v, dr, negative, truth in loader:
            for group in optimizer.param_groups:
                group["lr"] = lr * .98 ** epoch * min(1., (global_step + 1) / max(warmup_steps, 1))
            optimizer.zero_grad(set_to_none=True)
            # Upstream learns selection from the observed endpoint plus decoys.
            points = torch.cat([truth[:, None], negative], dim=1)
            # Same unknown-species dropout used by our local transformer.
            sp = sp.masked_fill(torch.rand(sp.shape, device=device) < .1, 0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                _, loss = model(x, sp, h, v, dr, points)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite author-core loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(loss.detach())
            global_step += 1
        model.eval(); scores, ce, ess = [], [], []
        gen = torch.Generator(device=device).manual_seed(seed + 10000)
        with torch.inference_mode():
            for x, sp, h, v, dr, points, truth in val_loader:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    state = model.encode(x, sp, h, dr)
                    logits, _ = model.score_encoded(state, points, v)
                    _, conditional_ce = model.score_encoded(state, torch.cat([truth[:, None], points[:, :candidates]], dim=1), v)
                # Crucially the truth-containing choice set is NOT sampled.
                prob = logits.softmax(-1)
                chosen = torch.multinomial(prob, 32, replacement=True, generator=gen)
                samples = points.gather(1, chosen[..., None].expand(-1, -1, 2))
                scores.append(energy_score(samples, truth)); ce.append(conditional_ce.detach())
                ess.append(1 / prob.square().sum(-1))
        values = torch.cat(scores).cpu().numpy() * scales
        score = float(np.mean([values[cells == k].mean() / max(dist[cells == k].mean(), 1.) for k in np.unique(cells)]))
        ce_value = float(torch.stack(ce).mean().cpu())
        row = {"epoch": epoch + 1, "train_loss": float(torch.stack(losses).mean().cpu()),
               "val_macro_normalized_energy": score, "val_choice_loss": ce_value,
               "proposal_effective_candidates": float(torch.cat(ess).mean().cpu()), "seconds": round(time.monotonic() - start, 2)}
        history.append(row); dump(out / "history.json", history); print(json.dumps(row), flush=True)
        if score < best:
            best, stale = score, 0; checkpoint("best.pt", epoch + 1, best)
        else:
            stale += 1
        if ce_value < best_ce:
            best_ce = ce_value; checkpoint("best-choice.pt", epoch + 1, score)
        schedule.step()
        if stale >= patience:
            break
    return {"output": str(out), "best_validation_score": best, "parameters": config["parameters"], "epochs": len(history)}
