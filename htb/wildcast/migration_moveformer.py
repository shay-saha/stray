"""Pinned author-core MoveFormer adapted to whole migration-route selection.

The author encoder, attention implementation, and SelectionHead are unchanged.
Input projections, causal lag windows, seasonal route proposals, and the proper
energy training objective adapt them to this new task. This is not an official
published 90-day reproduction and uses no public pretrained weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .moveformer import PROJECT, author_classes, provenance


KIND = "wildcast_migration_moveformer"


def author_provenance():
    expected = {"third_party/moveformer/geo_transformers/models/any_horizon_forecast_transformer.py":
                    "9454f64bec902143633b981fdb64375551148b7522910550f8cd94e10b4db639",
                "third_party/x-transformers/x_transformers/x_transformers.py":
                    "e2871f4b9e9c85ed108b25972ba378bf783e0ebd440cc79343616fc0b973274e"}
    from .migration_retrieval_train import sha256
    for relative, digest in expected.items():
        path = PROJECT / relative
        if not path.exists():
            raise RuntimeError("Pinned MoveFormer and author x-transformers sources are required")
        if sha256(path) != digest:
            raise RuntimeError("MoveFormer source differs from the pinned author protocol")
    try:
        classes = author_classes()
    except ImportError as exc:
        raise RuntimeError("MoveFormer attention dependencies are missing; install the project's moveformer extra") from exc
    record = provenance()
    record.update(adapter="Migration history at 25 causal elapsed-time lags, species/static origin inputs and whole-route seasonal candidate features; optional identical previous-year context.",
                  evaluation="Whole routes sampled from the same fixed training-only seasonal bank used by the residual contender. Query truth never enters forecast support.",
                  objective="Exact finite-route-distribution spherical endpoint/joint energy plus KL to the proposal; replaces the original observed-step conditional cross-entropy.",
                  variable_context="Author block-mask recipe, with an always-visible virtual origin query token and explicit observed-history masks.",
                  scope="Author-core adaptation to a new migration task, not an official published 90-day reproduction.")
    return classes, record


class MigrationMoveFormer(nn.Module):
    model_kind = KIND
    initial_law = "Randomly initialized author-core route selection; no public pretrained weights"

    def __init__(self, query_features, candidate_features, n_species, history_features=10,
                 static_features=14, history_length=25, candidates=64, seed=1701,
                 width=128, layers=6, heads=8, variable_context=True,
                 species_dropout=.1, context=None):
        super().__init__()
        if candidates < 5:
            raise ValueError("The unchanged author SelectionHead requires at least five candidates")
        (Core, Head, AttentionLayers), self.upstream = author_provenance()
        self.config = dict(query_features=int(query_features), candidate_features=int(candidate_features),
                           n_species=int(n_species), history_features=int(history_features),
                           static_features=int(static_features), history_length=int(history_length),
                           candidates=int(candidates), seed=int(seed), width=int(width), layers=int(layers),
                           heads=int(heads), variable_context=bool(variable_context),
                           species_dropout=float(species_dropout), context=context)
        for name, size in (("query", query_features), ("candidate", candidate_features),
                           ("history", history_features), ("static", static_features)):
            self.register_buffer(name + "_mean", torch.zeros(size))
            self.register_buffer(name + "_std", torch.ones(size))
        self._route_context = None
        head = Head(candidate_encoder_layers=[nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)],
                    embedding_in_keys=["candidate"], in_features=width, hidden_features=width)
        self.core = Core(encoder=AttentionLayers(dim=width, depth=layers, heads=heads, causal=True), prediction_head=head,
                         disc_feat_dims={"species": n_species},
                         cont_feat_dims={"history": history_features, "static": static_features,
                                         "candidate": candidate_features},
                         keys_in=["history", "species", "static"], emb_dim=width, learnable_nan=False)

    @torch.no_grad()
    def set_history_normalization(self, history_mean, history_std, static_mean, static_std):
        for name, value in (("history_mean", history_mean), ("history_std", history_std),
                            ("static_mean", static_mean), ("static_std", static_std)):
            getattr(self, name).copy_(torch.as_tensor(value, device=self.history_mean.device))

    def encode_history(self, history, history_mask, static, species):
        if history.shape[1:] != (self.config["history_length"], self.config["history_features"]):
            raise ValueError("MoveFormer history differs from the checkpoint schema")
        if self.training and self.config["species_dropout"]:
            species = torch.where(torch.rand(species.shape, device=species.device) < self.config["species_dropout"], 0, species)
        observed = torch.where(history_mask[..., None], history, self.history_mean)
        observed = ((observed - self.history_mean) / self.history_std).clamp(-20, 20).flip(1)
        mask = history_mask.bool().flip(1)
        batch = len(history)
        # A virtual query token carries origin-time context and prevents empty
        # causal attention rows; it is not an invented observed GPS location.
        observed = torch.cat((torch.zeros(batch, 1, observed.shape[-1], device=history.device), observed), 1)
        mask = torch.cat((torch.ones(batch, 1, dtype=torch.bool, device=history.device), mask), 1)
        t = observed.shape[1]
        static = ((static - self.static_mean) / self.static_std).clamp(-20, 20)
        features = {"history": observed, "static": static[:, None].expand(-1, t, -1),
                    "species": species[:, None].expand(-1, t), "mask": mask}
        kwargs = {}
        if self.training and self.config["variable_context"]:
            length = torch.randint(1, t + 1, (), device=history.device)
            blocks = (t + length - 1) // length
            attention_mask = torch.eye(blocks, dtype=torch.bool, device=history.device)
            attention_mask = attention_mask.repeat_interleave(length, 0).repeat_interleave(length, 1)[:t, :t]
            # Keep the origin query token visible through missing-history blocks.
            attention_mask[:, 0] = True
            kwargs["attn_mask"] = attention_mask
        _, encoded = self.core(features, run_prediction=False, encoder_kwargs=kwargs)
        index = torch.arange(t, device=history.device).expand(batch, -1).masked_fill(~mask, -1).max(-1).values
        return encoded[torch.arange(batch, device=history.device), index][:, None]

    def logits(self, history, history_mask, static, species, candidate):
        state = self.encode_history(history, history_mask, static, species)
        features = ((candidate - self.candidate_mean) / self.candidate_std).clamp(-20, 20)
        if self.config["context"]:
            offset = candidate.shape[-1] - 132
            features = torch.cat((features[..., :offset],
                                  features[..., offset:offset + 64] * (candidate[..., offset + 128:offset + 129] > 0),
                                  features[..., offset + 64:offset + 128] * (candidate[..., offset + 131:offset + 132] > 0),
                                  features[..., offset + 128:]), dim=-1)
        embeddings = {"candidate": self.core.cont_proj["candidate"](features[:, None])}
        support = {"mask": torch.ones(len(history), 1, dtype=torch.bool, device=history.device)}
        values, _, _ = self.core.prediction_head(support, embeddings, state)
        return values[:, 0].float()

    def weights_from_batch(self, batch):
        logits = self.logits(batch["history"], batch["history_mask"], batch["static"],
                             batch["species"], batch["candidate_features"])
        return (logits + batch["base_weights"].clamp_min(1e-20).log()).softmax(-1)

    def forward(self, history, history_mask, static, species, candidate):
        return self.logits(history, history_mask, static, species, candidate)

    @torch.no_grad()
    def predict_cases(self, a, ids, samples, rng=None, baselines=None):
        from .migration_candidate_cache import cached_candidates
        if samples < 1:
            raise ValueError("At least one whole-route sample required")
        self.eval()
        ids = np.asarray(ids)
        paths, _, features, _ = cached_candidates(a, ids, self.config["candidates"], self.config["seed"], baselines)
        if self.config["context"]:
            from .migration_route_context import RouteContext
            spec = self.config["context"]
            if self._route_context is None:
                self._route_context = RouteContext(spec["fields_root"], expected=spec["fingerprint"])
            extra = (np.zeros((*paths.shape[:2], 132), np.float32) if spec.get("mode") == "masked"
                     else self._route_context.features(a["origin"][ids], paths))
            if spec.get("mode") == "availability":
                extra[..., :128] = 0.
            features = np.concatenate((features, extra), -1)
        device = self.history_mean.device
        values = [torch.as_tensor(a[key][ids], device=device) for key in ("history", "history_mask", "static", "species")]
        logits = self.logits(*values, torch.as_tensor(features, device=device))
        weights = logits.softmax(-1).cpu().numpy()
        rng = rng or np.random.default_rng(2026)
        chosen = np.stack([rng.choice(paths.shape[1], samples, p=w.astype(float) / w.sum(dtype=float)) for w in weights])
        return paths[np.arange(len(ids))[:, None], chosen]


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    if ck.get("model_kind") != KIND:
        raise ValueError("Not a migration author-core MoveFormer checkpoint")
    model = MigrationMoveFormer(**ck["config"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    if ck.get("upstream", {}).get("source_sha256") != model.upstream["source_sha256"]:
        raise ValueError("MoveFormer checkpoint source provenance mismatch")
    return model, ck


def train(data, bank, output, epochs=40, batch_size=128, width=128, layers=6, heads=8,
          lr=.00005, warmup_steps=1000, variable_context=True, **kwargs):
    from .migration_retrieval_train import train_bank
    return train_bank(data, bank, output, epochs=epochs, batch_size=batch_size, lr=lr,
                      model_class="moveformer", width=width, layers=layers, heads=heads,
                      warmup_steps=warmup_steps, variable_context=variable_context, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "bank", "output"):
        parser.add_argument("--" + name, required=True)
    for name, default in (("epochs", 40), ("batch-size", 128), ("width", 128), ("layers", 6),
                          ("heads", 8), ("warmup-steps", 1000), ("patience", 6), ("seed", 42)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("lr", .00005), ("kl-weight", .02), ("weight-decay", .01), ("species-dropout", .1)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context"); parser.add_argument("--context-mode", choices=("actual", "masked", "availability"), default="actual")
    parser.add_argument("--variable-context", action=argparse.BooleanOptionalAction, default=True)
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
