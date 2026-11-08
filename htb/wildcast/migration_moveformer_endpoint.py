"""Author-recipe endpoint MoveFormer reference for the local migration task.

Pinned author encoder/attention/SelectionHead methods and empirical inverse
log-distance/turn-quantile sampling are unchanged. Local input projections,
calendar splits, past-only scale and horizon adapters are explicit adaptations.
Positive endpoints enter TRAINING choice sets only. Inference uses continuous
proposal candidates, never observed future locations. Independent horizon draws
define a product-joint baseline; learned route coherence is not claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .migration_moveformer import author_provenance as pinned_author_provenance
from .moveformer import EmpiricalStepProposal, candidate_features, load_proposal, proposal_records


KIND = "wildcast_migration_moveformer_endpoint"
SELECTION_VERSION = "migration-moveformer-endpoint-physical-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def causal_scale(a, ids):
    """Past-only 1/7-day displacement rates, in 1000 km/day.

    The 1 km/day numerical floor is an assumption, not a biological maximum.
    Missing 1/7-day history falls back to the most recent observed past rate.
    """
    history, mask = a["history"][ids], a["history_mask"][ids]
    scale = np.full(len(ids), .001, np.float32)
    heading = np.zeros((len(ids), 2), np.float32)
    for row in range(len(ids)):
        valid = np.flatnonzero(mask[row] & (history[row, :, 2] > 0))
        if not len(valid):
            continue
        rates = []
        for lag in (1., 7.):
            chosen = valid[np.abs(history[row, valid, 2] - lag).argmin()]
            age = float(history[row, chosen, 2])
            if abs(age - lag) <= max(.25, .2 * lag):
                rate = -history[row, chosen, :2] / max(age, .01)
                rates.append(rate)
        if not rates:
            chosen = valid[history[row, valid, 2].argmin()]
            rates = [-history[row, chosen, :2] / max(float(history[row, chosen, 2]), .01)]
        rates = np.asarray(rates)
        scale[row] = max(float(np.sqrt(np.mean(np.square(rates).sum(-1)))), .001)
        moving = np.flatnonzero(np.linalg.norm(rates, axis=-1) > 1e-8)
        if len(moving):
            heading[row] = rates[moving[0]] / scale[row]
    return scale, heading


def proposal_metadata(species, horizon):
    result = np.zeros((len(species), 5), dtype=np.float64)
    result[:, 1] = species
    result[:, 4] = np.rint(np.asarray(horizon) * 24)
    return result


def fit_proposal(a):
    ids = np.flatnonzero((a["split"] == 0) & a["target_mask"].any(-1))
    if not len(ids):
        raise ValueError("Endpoint proposal requires observed training targets")
    scale, velocity = causal_scale(a, ids)
    rows, horizons = np.nonzero(a["target_mask"][ids])
    y = a["y"][ids[rows], horizons] / scale[rows, None]
    meta = proposal_metadata(a["species"][ids[rows]], a["horizons"][horizons])
    if not np.isfinite(y).all():
        raise ValueError("Observed training endpoints must be finite")
    proposal = EmpiricalStepProposal().fit(y, meta, velocity[rows])
    # A pooled per-horizon fallback is fitted from the same training labels,
    # making unknown-species dropout use the same q as unknown inference.
    pooled_meta = meta.copy(); pooled_meta[:, 1] = 0
    pooled = EmpiricalStepProposal().fit(y, pooled_meta, velocity[rows])
    proposal.records.update({key: value for key, value in pooled.records.items() if key[0] == 0})
    return proposal, {"training_origins": int(len(ids)), "training_targets": int(len(rows)),
                       "training_species": sorted(np.unique(a["species"][ids]).tolist()),
                       "unknown_fallback": "Pooled training-only empirical proposal separately at each horizon"}


class EndpointMoveFormer(nn.Module):
    def __init__(self, n_species, horizons, history_features=10, static_features=14,
                 history_length=25, width=128, layers=6, heads=8, proposal=None,
                 inference_candidates=1024, variable_context=True, species_dropout=.1, neural_precision="bf16"):
        super().__init__()
        if neural_precision not in ("bf16", "float32"):
            raise ValueError("Choose bf16 or float32 neural precision")
        if inference_candidates < 5:
            raise ValueError("The unchanged author SelectionHead needs at least five candidates")
        (Core, Head, AttentionLayers), upstream = pinned_author_provenance()
        self.config = dict(n_species=int(n_species), horizons=list(map(float, horizons)),
                           history_features=int(history_features), static_features=int(static_features),
                           history_length=int(history_length), width=int(width), layers=int(layers), heads=int(heads),
                           proposal=proposal, inference_candidates=int(inference_candidates),
                           variable_context=bool(variable_context), species_dropout=float(species_dropout),
                           neural_precision=neural_precision)
        self.proposal = load_proposal(proposal) if proposal else None
        self.upstream = {**upstream,
            "adapter": "Local 25-lag observed history, species/static origin features, horizon conditioning and past-only movement scale; no environment or pretrained weights.",
            "objective": "Original positive-vs-proposal conditional cross-entropy; observed endpoint inserted only in TRAINING choice sets.",
            "evaluation": "Continuous empirical log-distance/turn proposal, 1024 candidates per horizon by default; never includes query truth. Independent horizons form an explicit product-joint law.",
            "scale": "RMS observed 1-day/7-day displacement rates, fallback to most recent past rate, 1 km/day floor. No query future normalization.",
            "species_dropout": "Training choice-set creation drops the species for both proposal and classifier once; encoder does not drop it a second time.",
            "scope": "Pinned author-recipe local-task adaptation; not an official paper/MoveBench/90-day reproduction."}
        for name, count in (("history", history_features), ("static", static_features)):
            self.register_buffer(name + "_mean", torch.zeros(count))
            self.register_buffer(name + "_std", torch.ones(count))
        head = Head(candidate_encoder_layers=[nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)],
                    embedding_in_keys=["candidate"], in_features=width, hidden_features=width)
        self.core = Core(encoder=AttentionLayers(dim=width, depth=layers, heads=heads, causal=True), prediction_head=head,
                         disc_feat_dims={"species": n_species},
                         cont_feat_dims={"history": history_features, "static": static_features, "horizon": 3, "candidate": 5},
                         keys_in=["history", "species", "static", "horizon"], emb_dim=width, learnable_nan=False)

    @torch.no_grad()
    def set_normalization(self, history_mean, history_std, static_mean, static_std):
        for name, value in (("history_mean", history_mean), ("history_std", history_std),
                            ("static_mean", static_mean), ("static_std", static_std)):
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32, device=self.history_mean.device))

    def encode(self, history, history_mask, static, species, horizon):
        if history.shape[1:] != (self.config["history_length"], self.config["history_features"]):
            raise ValueError("Endpoint MoveFormer history schema mismatch")
        observed = torch.where(history_mask[..., None], history, self.history_mean)
        observed = ((observed - self.history_mean) / self.history_std).clamp(-20, 20).flip(1)
        mask = history_mask.bool().flip(1)
        batch = len(history)
        observed = torch.cat((torch.zeros(batch, 1, observed.shape[-1], device=history.device), observed), 1)
        mask = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=history.device), mask), 1)
        length = observed.shape[1]
        static = ((static - self.static_mean) / self.static_std).clamp(-20, 20)
        h = torch.log1p(horizon)
        features = {"history": observed, "static": static[:, None].expand(-1, length, -1),
                    "species": species[:, None].expand(-1, length),
                    "horizon": torch.stack((h, h.sin(), h.cos()), -1)[:, None].expand(-1, length, -1), "mask": mask}
        kwargs = {}
        if self.training and self.config["variable_context"]:
            window = torch.randint(1, length + 1, (), device=history.device)
            blocks = (length + window - 1) // window
            block_mask = torch.eye(blocks, dtype=torch.bool, device=history.device)
            block_mask = block_mask.repeat_interleave(window, 0).repeat_interleave(window, 1)[:length, :length]
            block_mask[:, 0] = True
            kwargs["attn_mask"] = block_mask
        _, encoded = self.core(features, run_prediction=False, encoder_kwargs=kwargs)
        last = torch.arange(length, device=history.device).expand(batch, -1).masked_fill(~mask, -1).max(-1).values
        return encoded[torch.arange(batch, device=history.device), last][:, None]

    def score(self, history, history_mask, static, species, horizon, points, velocity):
        state = self.encode(history, history_mask, static, species, horizon)
        embeddings = {"candidate": self.core.cont_proj["candidate"](candidate_features(points, velocity)[:, None])}
        support = {"mask": torch.ones(len(history), 1, dtype=torch.bool, device=history.device)}
        logits, loss, _ = self.core.prediction_head(support, embeddings, state)
        return logits[:, 0].float(), loss

    def forward(self, *args, **kwargs):
        return self.score(*args, **kwargs)

    @torch.no_grad()
    def predict_cases(self, a, ids, samples, rng=None, baselines=None, candidates=None):
        return self.predict_endpoints(a, ids, self.config["horizons"], samples, rng, candidates)

    @torch.no_grad()
    def predict_endpoints(self, a, ids, horizons, samples=64, rng=None, candidates=None):
        horizons = np.asarray(horizons, dtype=float)
        if (horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all()
                or any(not np.isclose(h, self.config["horizons"], atol=1e-8, rtol=1e-7).any() for h in horizons)):
            raise ValueError("Requested endpoint horizons must belong to the frozen MoveFormer model")
        if self.proposal is None:
            raise ValueError("A frozen training-fitted endpoint proposal is required")
        if samples < 1:
            raise ValueError("At least one endpoint sample required")
        count = int(candidates or self.config["inference_candidates"])
        if count < 5:
            raise ValueError("At least five inference candidates required by the author head")
        self.eval()
        ids = np.asarray(ids, dtype=np.int64)
        rng = rng or np.random.default_rng(2026)
        scale, velocity = causal_scale(a, ids)
        device = self.history_mean.device
        inputs = [torch.as_tensor(a[key][ids], device=device) for key in ("history", "history_mask", "static", "species")]
        vel = torch.as_tensor(velocity, device=device)
        result = np.empty((len(ids), samples, len(horizons), 2), np.float32)
        for column, horizon in enumerate(horizons):
            meta = proposal_metadata(a["species"][ids], np.full(len(ids), horizon))
            points = self.proposal.sample(meta, velocity, count, rng)
            times = torch.full((len(ids),), horizon, dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and self.config["neural_precision"] == "bf16"):
                logits, _ = self.score(*inputs, times, torch.as_tensor(points, device=device), vel)
            weights = logits.softmax(-1).cpu().numpy()
            chosen = np.stack([rng.choice(count, samples, p=w.astype(float) / w.sum(dtype=float)) for w in weights])
            result[:, :, column] = points[np.arange(len(ids))[:, None], chosen] * scale[:, None, None]
        return result


def make_training_batch(model, a, ids, horizon_indices, candidates, rng, device):
    """The only function that inserts query truth; rejects nontraining origins."""
    ids, horizon_indices = np.asarray(ids), np.asarray(horizon_indices)
    if np.any(a["split"][ids] != 0) or not a["target_mask"][ids, horizon_indices].all():
        raise ValueError("Positive choice sets require observed TRAINING endpoints only")
    if candidates < 4:
        raise ValueError("At least four noise candidates required by the unchanged author head")
    scale, velocity = causal_scale(a, ids)
    species = a["species"][ids].copy()
    species[rng.random(len(ids)) < model.config["species_dropout"]] = 0
    horizons = a["horizons"][horizon_indices]
    meta = proposal_metadata(species, horizons)
    negative = model.proposal.sample(meta, velocity, candidates, rng)
    positive = a["y"][ids, horizon_indices] / scale[:, None]
    points = np.concatenate((positive[:, None], negative), 1).astype(np.float32)
    return {"history": torch.as_tensor(a["history"][ids], device=device),
            "history_mask": torch.as_tensor(a["history_mask"][ids], device=device),
            "static": torch.as_tensor(a["static"][ids], device=device),
            "species": torch.as_tensor(species, dtype=torch.long, device=device),
            "horizon": torch.as_tensor(horizons, dtype=torch.float32, device=device),
            "points": torch.as_tensor(points, device=device), "velocity": torch.as_tensor(velocity, device=device)}


def selection_origins(a, maximum=4096, seed=1701):
    ids = np.flatnonzero(np.isin(a["split"], [1, 2]) & a["target_mask"].any(-1))
    if maximum <= 0 or len(ids) <= maximum:
        return ids
    species = a.get("metric_species", a["species"])
    groups = {}
    for case in ids:
        groups.setdefault((int(a["split"][case]), int(species[case]), int(a["individual"][case])), []).append(int(case))
    rng = np.random.default_rng(seed)
    blocks = [rng.permutation(group).tolist() for group in groups.values()]
    rng.shuffle(blocks)
    chosen = []
    while len(chosen) < maximum and blocks:
        remaining = []
        for block in blocks:
            if len(chosen) == maximum:
                break
            chosen.append(block.pop())
            if block:
                remaining.append(block)
        blocks = remaining
    return np.sort(chosen).astype(np.int64)


@torch.no_grad()
def validate(model, a, ids, samples=32, candidates=256, batch_size=64, seed=2026):
    from .migration_evaluate import macro_scores, score_paths
    rows, joint = [], []
    species = a.get("metric_species", a["species"])
    for start in range(0, len(ids), batch_size):
        selected = ids[start:start + batch_size]
        paths = model.predict_cases(a, selected, samples, np.random.default_rng(seed + start), candidates=candidates)
        scores = score_paths(paths, a["target_latlon"][selected], a["target_mask"][selected], a["origin"][selected],
                             device=str(model.history_mean.device))
        for row, case in enumerate(selected):
            split = "val_individual" if a["split"][case] == 1 else "val_year"
            base = {"model": "endpoint_moveformer", "split": split, "species": int(species[case]), "individual": int(a["individual"][case])}
            joint.append({**base, "energy": float(scores["joint_chordal_energy_km"][row])})
            for h in np.flatnonzero(a["target_mask"][case]):
                rows.append({**base, "horizon_days": float(a["horizons"][h]),
                             "endpoint_energy_km": float(scores["endpoint_energy_km"][row, h])})
    if not rows:
        raise ValueError("Endpoint MoveFormer requires validation targets")
    overall, _, _ = macro_scores(pd.DataFrame(rows))
    joint_summary = pd.DataFrame(joint).groupby(["split", "species", "individual"]).energy.mean().groupby(
        ["split", "species"]).mean().groupby("split").mean()
    return {"selection_endpoint_energy_km": float(overall.endpoint_energy_km.mean()),
            "splits": {row.split: {"endpoint_energy_km": float(row.endpoint_energy_km),
                                     "product_joint_energy_km": float(joint_summary.loc[row.split])}
                       for row in overall.itertuples()}}


def train(data, output, epochs=40, batch_size=256, width=128, layers=6, heads=8,
          lr=.00005, warmup_steps=1000, seed=42, device="auto", patience=10, candidates=100,
          inference_candidates=1024, validation_candidates=256, validation_samples=32,
          max_validation_origins=4096, validation_seed=1701, validation_batch_size=64,
          variable_context=True, species_dropout=.1, weight_decay=0.,
          training_horizons_hours=None, neural_precision="bf16"):
    from .migration_evaluate import load_cases
    from .migration_train import load_training_data
    if epochs < 0 or batch_size < 1 or candidates < 4 or not 0 <= species_dropout <= 1:
        raise ValueError("Invalid endpoint MoveFormer training options")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    # Reuse the calendar/identity/sealed-target training integrity audit.
    _, _, audited_hashes = load_training_data(data)
    a, manifest, hashes = load_cases(data)
    if hashes != audited_hashes or not manifest.get("reserved_test_targets_sealed", False):
        raise ValueError("Endpoint reference requires an unchanged sealed development cache")
    if training_horizons_hours is not None:
        from .migration_endpoint_evaluate import horizon_columns
        columns = horizon_columns(a["horizons"], np.asarray(training_horizons_hours, dtype=float) / 24)
        # Subset only this in-memory task after the full frozen-cache audit.
        a = {**a, "horizons": a["horizons"][columns]}
        for key in ("y", "target_mask", "target_latlon", "target_time"):
            a[key] = a[key][:, columns]
    if neural_precision == "float32":
        torch.set_float32_matmul_precision("high")
    train_ids = np.flatnonzero((a["split"] == 0) & a["target_mask"].any(-1))
    val_ids = selection_origins(a, max_validation_origins, validation_seed)
    proposal, proposal_audit = fit_proposal(a)
    n_species = max(manifest.get("species", {}).values(), default=int(a["species"].max())) + 1
    model = EndpointMoveFormer(n_species, a["horizons"], a["history"].shape[-1], a["static"].shape[-1], a["history"].shape[1],
                               width, layers, heads, proposal_records(proposal), inference_candidates,
                               variable_context, species_dropout, neural_precision).to(device)
    observed = a["history"][train_ids][a["history_mask"][train_ids]]
    static = a["static"][train_ids]
    model.set_normalization(observed.mean(0), np.maximum(observed.std(0), 1e-3), static.mean(0), np.maximum(static.std(0), 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    _, inverse, counts = np.unique(a["individual"][train_ids], return_inverse=True, return_counts=True)
    probability = 1. / counts[inverse]; probability /= probability.sum()
    rng = np.random.default_rng(seed)
    out = Path(output)
    if (out / "best.pt").exists():
        raise FileExistsError("Use a new output path to preserve endpoint reference selection")
    out.mkdir(parents=True, exist_ok=True)
    selection_hash = hashlib.sha256(np.asarray(val_ids, dtype="<i8").tobytes()).hexdigest()
    config = {"scope": "Local author-recipe migration endpoint reference; not an official published reproduction",
              "model_kind": KIND, "selection_version": SELECTION_VERSION, **hashes, "model": model.config,
              "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr, "warmup_steps": warmup_steps,
              "patience": patience, "noise_candidates": candidates, "inference_candidates": inference_candidates,
              "validation_candidates": validation_candidates, "validation_samples": validation_samples,
              "validation_batch_size": validation_batch_size, "max_validation_origins": max_validation_origins,
              "validation_origin_ids_sha256": selection_hash, "validation_origin_ids": val_ids.tolist(),
              "validation_subset_rule": "Fixed metadata-only round-robin over split/species/physical-group blocks; seed declared before fitting",
              "validation_subset_seed": validation_seed, "validation_origins": int(len(val_ids)),
              "selection": "Minimum fixed-subset endpoint energy; equal physical group within species, then species/horizons/splits. Product-joint score diagnostic only.",
              "sampling": "Animal-balanced origins; independently choose one uniformly observed horizon per origin each epoch",
              "proposal_audit": proposal_audit, "upstream": model.upstream, "species_dropout": species_dropout,
              "weight_decay": weight_decay, "parameters": sum(p.numel() for p in model.parameters()),
              "precision": neural_precision + " neural computation on CUDA; float32 choice loss",
              "supported_horizons_days": a["horizons"].tolist(), "device": str(device),
              "code_sha256": sha256(__file__)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    history, best, best_epoch, stale, step = [], float("inf"), 0, 0, 0
    began = time.monotonic()
    for epoch in range(epochs + 1):
        total, batches = 0., 0
        if epoch:
            model.train()
            selected = rng.choice(train_ids, len(train_ids), replace=True, p=probability)
            for start in range(0, len(selected), batch_size):
                ids = selected[start:start + batch_size]
                h = np.asarray([rng.choice(np.flatnonzero(a["target_mask"][case])) for case in ids])
                batch = make_training_batch(model, a, ids, h, candidates, rng, device)
                rate = lr * min(1., (step + 1) / max(1, warmup_steps)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * (epoch - 1) / max(1, epochs))))
                for group in optimizer.param_groups: group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(device).startswith("cuda") and neural_precision == "bf16"):
                    _, loss = model.score(**batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite author conditional endpoint loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); total += float(loss.detach()); batches += 1; step += 1
        metrics = validate(model, a, val_ids, validation_samples, validation_candidates, validation_batch_size, 2026)
        value = metrics["selection_endpoint_energy_km"]
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite endpoint reference validation score")
        row = {"epoch": epoch, "train_choice_loss": total / batches if batches else None,
               "elapsed_seconds": time.monotonic() - began, **metrics}
        history.append(row); (out / "history.json").write_text(json.dumps(history, indent=2))
        improved = value < best - 1e-4
        if improved: best, best_epoch, stale = value, epoch, 0
        elif epoch: stale += 1
        checkpoint = {"model_kind": KIND, "selection_version": SELECTION_VERSION, "model": model.state_dict(),
                      "config": model.config, "manifest": manifest, **hashes, "epoch": epoch,
                      "metrics": metrics, "training": config, "upstream": model.upstream,
                      "optimizer": optimizer.state_dict(), "code_sha256": config["code_sha256"]}
        if not epoch: torch.save(checkpoint, out / "initial.pt")
        if improved: torch.save(checkpoint, out / "best.pt")
        torch.save(checkpoint, out / "last.pt")
        print(json.dumps({"epoch": epoch, "train_choice_loss": row["train_choice_loss"],
                          "selection_endpoint_energy_km": value, "best_km": best,
                          "seconds": round(row["elapsed_seconds"], 1)}), flush=True)
        if epoch and stale >= patience and step >= warmup_steps:
            break
    result = {"scope": config["scope"], "selected_epoch": best_epoch, "epochs_completed": history[-1]["epoch"],
              "best_endpoint_selection_km": best, "selection_version": SELECTION_VERSION,
              "validation_origin_ids_sha256": selection_hash, "validation_origins": len(val_ids),
              "parameters": config["parameters"], **hashes}
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    if ck.get("model_kind") != KIND:
        raise ValueError("Not a migration empirical-endpoint MoveFormer checkpoint")
    model = EndpointMoveFormer(**ck["config"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    if ck.get("upstream", {}).get("source_sha256") != model.upstream["source_sha256"]:
        raise ValueError("Endpoint MoveFormer pinned source provenance mismatch")
    return model, ck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True); parser.add_argument("--output", required=True)
    for name, default in (("epochs", 40), ("batch-size", 256), ("width", 128), ("layers", 6), ("heads", 8),
                          ("warmup-steps", 1000), ("patience", 10), ("seed", 42), ("candidates", 100),
                          ("inference-candidates", 1024), ("validation-candidates", 256), ("validation-samples", 32),
                          ("max-validation-origins", 4096), ("validation-seed", 1701), ("validation-batch-size", 64)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("lr", .00005), ("species-dropout", .1), ("weight-decay", 0.)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variable-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--training-horizons-hours", nargs="+", type=float)
    parser.add_argument("--neural-precision", choices=("bf16", "float32"), default="bf16")
    print(json.dumps(train(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
