from __future__ import annotations

import json
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import SPLITS
from .model import MovementTransformer, energy_score, mixture_energy_score, nll, sample
from .sources import dump


def load_windows(root):
    path = Path(root) / "processed"
    a = np.load(path / "windows.npz")
    return {k: a[k] for k in a.files}, json.loads((path / "windows.json").read_text())


def tensors(a, ix, loss_weights=None):
    drivers = a["drivers"][ix] if "drivers" in a else np.empty((len(ix), 0), dtype=np.float32)
    anchors = a["anchors"][ix] if "anchors" in a else np.empty((len(ix), 0, 2), dtype=np.float32)
    memory = a["memory"][ix] if "memory" in a else np.empty((len(ix), 0, 0), dtype=np.float32)
    return TensorDataset(torch.from_numpy(a["x"][ix]), torch.from_numpy(a["y"][ix]),
        torch.from_numpy(a["meta"][ix, 1].astype(np.int64)), torch.from_numpy(a["meta"][ix, 4].astype(np.float32)),
        torch.from_numpy(a["aux"][ix, 4:6].astype(np.float32)), torch.from_numpy(drivers), torch.from_numpy(anchors),
        torch.from_numpy(loss_weights[ix].astype(np.float32) if loss_weights is not None else np.ones(len(ix), dtype=np.float32)),
        torch.from_numpy(memory))


class DeviceBatches:
    """Avoid per-row Python collation when the small tabular corpus fits VRAM."""
    def __init__(self, dataset, batch_size, device, sampler=None):
        self.values = tuple(t.to(device) for t in dataset.tensors)
        self.batch_size, self.sampler, self.device = batch_size, sampler, device

    def __iter__(self):
        if self.sampler is None:
            for start in range(0, len(self.values[0]), self.batch_size):
                yield tuple(t[start:start + self.batch_size] for t in self.values)
        else:
            indices = torch.as_tensor(list(self.sampler), dtype=torch.long, device=self.device)
            for ids in indices.split(self.batch_size):
                yield tuple(t[ids] for t in self.values)


def train(root, output, epochs=15, batch_size=256, width=128, layers=3, components=8,
          lr=.0003, seed=42, max_train=0, device="auto", patience=5, workers=0, energy_weight=.05, student_df=4,
          driver_groups="all", anchors=False, init_checkpoint=None, nll_weight=1., stratified_energy=False,
          precision="fp32", energy_repeats=2, preload=False, long_memory=False, memory_fusion="concat", driver_base_features=0,
          horizon_adapter=False, regime_weight=0., freeze_backbone=False, identity_audit=None,
          scope_family=None, taxonomy_root="data/combined-v2/environment/taxonomy", weight_decay=.01,
          environment_lr=None, environment_gate_lr=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    a, manifest = load_windows(root)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    train_ix = np.flatnonzero(a["meta"][:, 0] == SPLITS["train"])
    rng = np.random.default_rng(seed)
    if max_train and len(train_ix) > max_train:
        train_ix = rng.choice(train_ix, max_train, replace=False)
    val_ix = np.flatnonzero(np.isin(a["meta"][:, 0], [SPLITS["val_temporal"], SPLITS["val_individual"]]))
    if identity_audit:
        from .identity import verify_audit
        audit=json.loads(Path(identity_audit).read_text());verify_audit(audit,manifest,a)
        val_ix=val_ix[~np.isin(val_ix,audit["excluded_windows"])]
    if scope_family:
        from .moe import taxonomy_groups
        taxonomy = taxonomy_groups(manifest, taxonomy_root)
        if scope_family not in taxonomy["families"] or scope_family == "shared":
            raise ValueError("Unknown specialist family: " + scope_family)
        family = taxonomy["families"].index(scope_family)
        family_ids = np.asarray(taxonomy["species_family"])[a["meta"][:, 1]]
        train_ix = train_ix[family_ids[train_ix] == family]
        val_ix = val_ix[family_ids[val_ix] == family]
        if freeze_backbone:
            raise ValueError("Whole-model specialists must not freeze their encoders")
    if not len(train_ix) or not len(val_ix):
        raise ValueError("Training and validation windows are both required")
    vloader = DataLoader(tensors(a, val_ix), batch_size=batch_size, num_workers=workers)
    groups = set(driver_groups.split(","))
    names = manifest.get("driver_names", [])
    driver_ids = [i for i, name in enumerate(names) if "all" in groups or
                  ("weather" in groups and name.startswith("weather_")) or
                  ("vegetation" in groups and name.startswith("vegetation_")) or
                  ("satellite" in groups and name.startswith("satellite_")) or
                  ("memory" in groups and (name.startswith(("memory_", "past_", "log_memory", "log_context")))) or
                  ("traits" in groups and name.startswith(("trait_", "taxonomy_"))) or
                  ("clock" in groups and not name.startswith(("satellite_", "vegetation_", "weather_", "memory_", "past_", "log_memory", "log_context", "trait_", "taxonomy_")))]
    model = MovementTransformer(len(manifest["species"]), width=width, layers=layers, components=components,
                                context=manifest["context"], n_features=a["x"].shape[-1], student_df=student_df,
                                n_drivers=len(driver_ids), n_anchors=manifest.get("n_anchors", 0) if anchors else 0,
                                driver_indices=driver_ids if names else None,
                                memory_features=a["memory"].shape[-1] if long_memory else 0,
                                memory_length=a["memory"].shape[1] if long_memory else 0, memory_fusion=memory_fusion,
                                driver_base_features=driver_base_features, horizon_adapter=horizon_adapter, regime_head=regime_weight > 0).to(device)
    if init_checkpoint:
        ck = torch.load(init_checkpoint, map_location=device, weights_only=True)
        current = model.state_dict()
        reusable = {k: v for k, v in ck["model"].items() if k in current and current[k].shape == v.shape}
        if ck["manifest"]["species"] != manifest["species"]:
            reusable.pop("species.weight", None)
        model.load_state_dict(reusable, strict=False)
        with torch.no_grad():
            model.species.weight[0].copy_(ck["model"]["species.weight"][0])
            for name, idx in manifest["species"].items():
                if name in ck["manifest"]["species"]:
                    model.species.weight[idx].copy_(ck["model"]["species.weight"][ck["manifest"]["species"][name]])
        print(f"Initialized {len(reusable)} compatible tensors from {init_checkpoint}", flush=True)
    if freeze_backbone:
        if not init_checkpoint:
            raise ValueError("Freezing the backbone requires an initialized checkpoint")
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith(("horizon_adapter.", "regime_head.", "environment_encoder.", "environment_gate")))
        if not any(p.requires_grad for p in model.parameters()):
            raise ValueError("No new trainable adapters selected")
    parameter_groups = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            rate = environment_gate_lr if name == "environment_gate" else environment_lr if name.startswith("environment_encoder.") else lr
            parameter_groups.append({"params": [parameter], "lr": lr if rate is None else rate})
    optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1))
    config = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed, "max_train": max_train,
              "device": device, "parameters": sum(p.numel() for p in model.parameters()), "model": model.config,
              "energy_weight": energy_weight, "windows_sha256": manifest["windows_sha256"],
              "protocol": manifest["protocol"], "torch_version": str(torch.__version__),
              "driver_groups": driver_groups, "init_checkpoint": init_checkpoint,
              "driver_names": [names[i] for i in driver_ids], "nll_weight": nll_weight,
              "stratified_energy": stratified_energy, "energy_repeats": energy_repeats, "precision": precision,
              "preload": preload,
              "freeze_backbone": freeze_backbone, "regime_weight": regime_weight,
              "scope_family": scope_family, "training_cases":len(train_ix), "validation_cases":len(val_ix),
              "weight_decay":weight_decay,
              "environment_lr":environment_lr,"environment_gate_lr":environment_gate_lr,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "identity_audit_sha256":hashlib.sha256(Path(identity_audit).read_bytes()).hexdigest() if identity_audit else None,
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    dump(out / "config.json", config)
    best, stale, history = float("inf"), 0, []
    # Balance training/validation by study x horizon x split instead of allowing
    # the longest collar deployment to determine every selection decision.
    meta_train = a["meta"][train_ix]
    _, inverse, counts = np.unique(meta_train[:, [2, 4]], axis=0, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(train_ix), replacement=True,
                                                    generator=torch.Generator().manual_seed(seed))
    scales_all = a["aux"][:, 8] if a["aux"].shape[1] > 8 else np.full(len(a["meta"]), manifest["scale_metres"])
    loss_weights = np.ones(len(a["meta"]), dtype=np.float32)
    if stratified_energy:
        train_distance = np.linalg.norm(a["y"][train_ix], axis=-1) * scales_all[train_ix]
        for k in range(len(counts)):
            ids = train_ix[inverse == k]
            loss_weights[ids] = scales_all[ids] / max(float(train_distance[inverse == k].mean()), 1.)
    loader = DataLoader(tensors(a, train_ix, loss_weights), batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=device == "cuda")
    if preload:
        loader = DeviceBatches(tensors(a, train_ix, loss_weights), batch_size, device, sampler)
        vloader = DeviceBatches(tensors(a, val_ix), batch_size, device)
    val_keys = a["meta"][val_ix][:, [0, 2, 4]]
    _, vinverse = np.unique(val_keys, axis=0, return_inverse=True)
    def validate():
        model.eval(); scores, val_nll = [], []
        gen = torch.Generator(device=device).manual_seed(seed + 10000)
        with torch.inference_mode():
            for batch in vloader:
                x, y, sp, h, v, dr, anc, ew, mem = [t.to(device) for t in batch]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16" and device.startswith("cuda")):
                    params = model(x, sp, h, v, dr, anc, mem)
                scores.append(energy_score(sample(params, 32, gen), y)); val_nll.append(nll(params, y))
        values = torch.cat(scores).cpu().numpy()
        scales = a["aux"][val_ix, 8] if a["aux"].shape[1] > 8 else np.full(len(val_ix), manifest["scale_metres"])
        values *= scales
        dist = np.linalg.norm(a["y"][val_ix], axis=-1) * scales
        cells = [values[vinverse == k].mean() / max(dist[vinverse == k].mean(), 1.) for k in np.unique(vinverse)]
        return float(np.mean(cells)), float(torch.cat(val_nll).mean().cpu())

    def save_checkpoint(epoch, score):
        torch.save({"model": model.state_dict(), "config": model.config, "training": config,
                    "manifest": manifest, "epoch": epoch, "val_score": score}, out / "best.pt.tmp")
        (out / "best.pt.tmp").replace(out / "best.pt")

    best_trained = float("inf")
    reference_parameters = {name:p.detach().clone() for name,p in model.named_parameters() if name.startswith(("input.","encoder.","memory_encoder.","memory_attention.","head."))}
    gradient_audit = {}

    if init_checkpoint:
        best, initial_nll = validate()
        save_checkpoint(0, best)
        print(json.dumps({"epoch": 0, "val_macro_normalized_energy": best, "val_nll": initial_nll,
                          "note": "Preserve initialization if fine tuning cannot improve validation."}), flush=True)
    for epoch in range(epochs):
        start = time.monotonic(); model.train(); losses = []
        if freeze_backbone:
            # Frozen dropout must not randomly change the forecast being adapted.
            model.eval()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            x, y, sp, h, v, dr, anc, ew, mem = [t.to(device, non_blocking=True) for t in batch]
            # Unknown-species dropout teaches a fallback for unseen taxa.
            sp = sp.masked_fill(torch.rand(sp.shape, device=device) < .1, 0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=precision == "bf16" and device.startswith("cuda")):
                params = model(x, sp, h, v, dr, anc, mem)
            loss = nll_weight * nll(params, y).mean()
            if regime_weight:
                radius = torch.linalg.vector_norm(y, dim=-1)
                labels = (radius > 1.).long() + (radius > torch.maximum(torch.ones_like(h), 2 * h.sqrt())).long()
                loss = loss + regime_weight * torch.nn.functional.cross_entropy(model.last_regime_logits, labels)
            if energy_weight:
                es = mixture_energy_score(params, y, energy_repeats) if stratified_energy else energy_score(sample(params, 8), y)
                loss = loss + energy_weight * (es * ew).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; inspect QC and reduce learning rate")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if not gradient_audit:
                for prefix in ["input.","encoder.","species.","driver_encoder.","memory_encoder.","memory_attention.","head."]:
                    group=[p for name,p in model.named_parameters() if name.startswith(prefix)]
                    gradient_audit[prefix]={"parameters":sum(p.numel() for p in group),
                        "all_trainable":all(p.requires_grad for p in group),
                        "nonzero_gradient_tensors":sum(p.grad is not None and bool(p.grad.abs().sum()>0) for p in group)}
                dump(out/"gradient-audit.json",gradient_audit)
            optimizer.step(); losses.append(loss.detach())
        train_seconds = time.monotonic() - start
        score, mean_nll = validate()
        row = {"epoch": epoch + 1, "train_loss": float(torch.stack(losses).mean().cpu()), "val_macro_normalized_energy": score,
               "val_nll": mean_nll, "seconds": round(time.monotonic() - start, 2),
               "train_examples_per_second": round(len(train_ix) / train_seconds, 1),
               "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.startswith("cuda") else None}
        history.append(row); print(json.dumps(row), flush=True)
        dump(out / "history.json", history)
        if score < best_trained:
            best_trained = score
            torch.save({"model":model.state_dict(),"config":model.config,"training":config,"manifest":manifest,
                        "epoch":epoch+1,"val_score":score},out/"best-trained.pt.tmp")
            (out/"best-trained.pt.tmp").replace(out/"best-trained.pt")
        if score < best:
            best = score; stale = 0
            save_checkpoint(epoch + 1, best)
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            break
    changes={prefix:sum(not torch.equal(p.detach(),reference_parameters[name]) for name,p in model.named_parameters()
                      if name in reference_parameters and name.startswith(prefix))
             for prefix in ["input.","encoder.","memory_encoder.","memory_attention.","head."]}
    selected = torch.load(out / "best-trained.pt", map_location=device, weights_only=True)["model"] if history else {}
    selected_changes = {prefix:sum(not torch.equal(selected[name], reference_parameters[name]) for name in reference_parameters
                                   if name.startswith(prefix)) for prefix in changes} if history else {}
    dump(out/"parameter-change-audit.json",{"changed_tensors_by_module":changes,"selected_checkpoint_changed_tensors_by_module":selected_changes,"training_steps_performed":len(history)>0,
          "scope_family":scope_family,"best_post_update_score":best_trained,"best_including_initialization":best})
    return {"output": str(out), "best_validation_score": best, "parameters": config["parameters"]}
