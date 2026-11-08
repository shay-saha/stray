"""Author MoveFormer core with an explicit adapter to our local data protocol.

The three author classes are compiled unchanged from the pinned MIT source.
Lightning's base is replaced by nn.Module; our trainer handles orchestration.
The author's attention fork and candidate-selection head are used directly.
This is not a reproduction of the unavailable MoveBench implementation.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baselines import CorrelatedRandomWalk

PROJECT = Path(__file__).resolve().parent.parent
AUTHOR_COMMIT = "8af592274b19b98bdd74f9f3a1317556337b1eff"
ATTENTION_COMMIT = "41eec21aca36c4f0497a5eb1dd8bd65cab7b3c56"


@lru_cache(maxsize=1)
def author_classes():
    fork = PROJECT / "third_party/x-transformers"
    source = PROJECT / "third_party/moveformer/geo_transformers/models/any_horizon_forecast_transformer.py"
    if not source.exists() or not fork.exists():
        raise RuntimeError("Pinned MoveFormer and x-transformers source checkouts are required")
    expected = {source: "9454f64bec902143633b981fdb64375551148b7522910550f8cd94e10b4db639",
                fork / "x_transformers/x_transformers.py": "e2871f4b9e9c85ed108b25972ba378bf783e0ebd440cc79343616fc0b973274e"}
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != digest for p, digest in expected.items()):
        raise RuntimeError("Upstream source differs from pinned comparison protocol")
    sys.path.insert(0, str(fork))
    from x_transformers.x_transformers import AttentionLayers, Attention
    import x_transformers.x_transformers as attention_module
    if Path(attention_module.__file__).resolve() != (fork / "x_transformers/x_transformers.py").resolve():
        raise RuntimeError("Another x-transformers implementation is already imported")
    tree = ast.parse(source.read_text())
    required = {"PredictionHead", "AnyHorizonForecastTransformer", "SelectionHead"}
    nodes = [v for v in tree.body if isinstance(v, ast.ClassDef) and v.name in required]
    if {v.name for v in nodes} != required:
        raise RuntimeError("Upstream class definitions changed")
    scope = {"__name__": "wildcast.moveformer_author", "np": np, "torch": torch, "nn": nn, "F": F,
             "pl": SimpleNamespace(LightningModule=nn.Module), "Dict": Dict, "List": List, "Optional": Optional,
             "AttentionLayers": AttentionLayers, "Attention": Attention}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope["AnyHorizonForecastTransformer"], scope["SelectionHead"], AttentionLayers


def provenance():
    files = [PROJECT / "third_party/moveformer/geo_transformers/models/any_horizon_forecast_transformer.py",
             PROJECT / "third_party/x-transformers/x_transformers/x_transformers.py"]
    return {"author_repository": "https://github.com/cifkao/moveformer", "author_commit": AUTHOR_COMMIT,
            "attention_repository": "https://github.com/cifkao/x-transformers", "attention_commit": ATTENTION_COMMIT,
            "source_sha256": {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "pretrained_weights": False,
            "overlap_audit": "Released checkpoints use Movebank+ungulate data. Exact training-animal exclusion could not be established from repository metadata; all public pretrained weights are excluded.",
            "adapter": "Local 32-fix input windows; normalized observed history, learned species token, full available origin drivers and horizon. Candidate features are normalized displacement, length and turn. No candidate-destination environmental rasters are supplied to either model.",
            "framework": "Unchanged author encoder/selection forward methods; nn.Module base replaces Lightning orchestration; pinned author attention fork.",
            "evaluation": "Samples from training-fitted CRW proposal reweighted by learned utility. No observed endpoint is inserted during inference. Local continuous coordinates, not official MoveBench raster evaluation."}


class EmpiricalStepProposal:
    """Author's inverse-quantile log-distance/turn recipe, fitted locally.

    Distances are in our causal context-normalized coordinates; models are fit
    per species and horizon rather than the original fixed-12h taxon data.
    No artificial point mass at zero is added to a continuous choice model.
    """
    def fit(self, y, meta, velocity):
        self.records = {}
        keys = sorted(set((int(s), int(h)) for s,h in meta[:, [1,4]])) + [(-1,-1)]
        for sp,h in keys:
            ix = np.ones(len(y),bool) if sp == -1 else ((meta[:,1] == sp) & (meta[:,4] == h))
            delta, v = y[ix], velocity[ix]
            r = np.linalg.norm(delta,axis=-1)
            valid = np.isfinite(r) & (r > 0)
            if not valid.any():
                logq = np.full(101,np.log(1e-6))
            else:
                logq = np.quantile(np.log(r[valid]),np.linspace(0,1,101))
            angle = np.arctan2(delta[:,1],delta[:,0]) - np.arctan2(v[:,1],v[:,0])
            angle = (angle + np.pi) % (2*np.pi) - np.pi
            moving = valid & (np.linalg.norm(v,axis=-1) > 1e-8)
            aq = np.quantile(angle[moving],np.linspace(0,1,101)) if moving.any() else np.linspace(-np.pi,np.pi,101)
            self.records[sp,h] = (logq,aq)
        return self

    def sample(self, meta, velocity, count, rng):
        output = np.zeros((len(meta),count,2),np.float32)
        for sp,h in sorted(set((int(s),int(h)) for s,h in meta[:,[1,4]])):
            ix = np.flatnonzero((meta[:,1] == sp) & (meta[:,4] == h))
            logq,aq = self.records.get((sp,h),self.records[-1,-1])
            # Same inverse quantile interpolation as author's StepSampler.
            r = np.exp(np.interp(rng.random((len(ix),count)),np.linspace(0,1,len(logq)),logq))
            angle = np.interp(rng.random((len(ix),count)),np.linspace(0,1,len(aq)),aq)
            angle += np.arctan2(velocity[ix,1],velocity[ix,0])[:,None]
            no_heading = np.linalg.norm(velocity[ix],axis=-1) <= 1e-8
            angle[no_heading] = rng.uniform(-np.pi,np.pi,(int(no_heading.sum()),count))
            output[ix] = r[...,None] * np.stack([np.cos(angle),np.sin(angle)],axis=-1)
        return output


def candidate_features(points, velocity):
    length = torch.linalg.vector_norm(points, dim=-1)
    norm = torch.linalg.vector_norm(velocity, dim=-1).clamp_min(1e-8)
    direction = velocity / norm[:, None]
    direction = torch.where((norm > 1e-7)[:, None], direction, torch.tensor([1., 0.], device=points.device))
    unit = points / length.clamp_min(1e-8)[..., None]
    cos = (unit * direction[:, None]).sum(-1)
    sin = unit[..., 1] * direction[:, None, 0] - unit[..., 0] * direction[:, None, 1]
    return torch.cat([torch.asinh(points), torch.log1p(length)[..., None], cos[..., None], sin[..., None]], dim=-1)


def proposal_records(proposal):
    if isinstance(proposal, EmpiricalStepProposal):
        return {"kind":"author_empirical_quantiles", "records":[[s,h,a.tolist(),b.tolist()] for (s,h),(a,b) in sorted(proposal.records.items())]}
    return [[int(s), int(h), *[float(v) for v in values]] for (s, h), values in sorted(proposal.params.items())]


def load_proposal(records):
    if isinstance(records, dict) and records.get("kind") == "author_empirical_quantiles":
        result = EmpiricalStepProposal()
        result.records = {(int(s),int(h)):(np.asarray(a),np.asarray(b)) for s,h,a,b in records["records"]}
        return result
    result = CorrelatedRandomWalk()
    result.params = {(int(v[0]), int(v[1])): tuple(v[2:]) for v in records}
    return result


class LocalMoveFormer(nn.Module):
    family = "moveformer_author_core_local_adapter"

    def __init__(self, n_species, n_features=10, n_drivers=296, width=128, layers=6, heads=8, proposal=None,
                 inference_candidates=1024, variable_context=False):
        super().__init__()
        self.config = dict(n_species=n_species, n_features=n_features, n_drivers=n_drivers, width=width, layers=layers,
                           heads=heads, proposal=proposal, inference_candidates=inference_candidates, variable_context=variable_context)
        self.variable_context = variable_context
        self.inference_candidates = inference_candidates
        self.proposal = load_proposal(proposal) if proposal else None
        Core, Head, AttentionLayers = author_classes()
        head = Head(candidate_encoder_layers=[nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)],
                    embedding_in_keys=["candidate"], in_features=width, hidden_features=width)
        self.core = Core(encoder=AttentionLayers(dim=width, depth=layers, heads=heads, causal=True), prediction_head=head,
                         disc_feat_dims={"species": n_species + 1},
                         cont_feat_dims={"history": n_features, "drivers": n_drivers, "horizon": 3, "candidate": 5},
                         keys_in=["history", "species", "drivers", "horizon"], emb_dim=width, learnable_nan=False)

    def encode(self, x, species, horizon, drivers):
        b, t = x.shape[:2]
        h = torch.log1p(horizon)
        features = {"history": x, "species": species[:, None].expand(-1, t),
                    "drivers": drivers[:, None].expand(-1, t, -1),
                    "horizon": torch.stack([h, torch.sin(h), torch.cos(h)], -1)[:, None].expand(-1, t, -1),
                    "mask": torch.ones((b, t), dtype=torch.bool, device=x.device)}
        encoder_kwargs = {}
        if self.training and self.variable_context:
            # Exact block-mask construction from author's var_len_training_v2.
            length = torch.randint(1,t+1,(),device=x.device)
            n_blocks = (t + length - 1) // length
            attention_mask = torch.eye(n_blocks,dtype=torch.bool,device=x.device)
            attention_mask = attention_mask.repeat_interleave(length,0).repeat_interleave(length,1)[:t,:t]
            encoder_kwargs["attn_mask"] = attention_mask
        _, state = self.core(features, run_prediction=False, encoder_kwargs=encoder_kwargs)
        return state[:, -1:]

    def score_encoded(self, state, points, velocity):
        embeddings = {"candidate": self.core.cont_proj["candidate"](candidate_features(points, velocity)[:, None])}
        features = {"mask": torch.ones((len(points), 1), dtype=torch.bool, device=points.device)}
        logits, loss, _ = self.core.prediction_head(features, embeddings, state)
        return logits[:, 0].float(), loss

    def forward(self, x, species, horizon, velocity, drivers, points):
        return self.score_encoded(self.encode(x, species, horizon, drivers), points, velocity)

    def sample_forecast(self, x, species, horizon, velocity, drivers, count, generator):
        if self.proposal is None:
            raise ValueError("Training-fitted movement proposal required")
        # Sampling depends solely on observed inputs and the training proposal.
        seed = int(torch.randint(2**31, (), generator=generator, device=x.device).item())
        rng = np.random.default_rng(seed)
        meta = np.zeros((len(x), 5)); meta[:, 1] = species.cpu().numpy(); meta[:, 4] = horizon.cpu().numpy()
        points = torch.from_numpy(self.proposal.sample(meta, velocity.cpu().numpy(), self.inference_candidates, rng)).to(x.device)
        logits, _ = self(x, species, horizon, velocity, drivers, points)
        probabilities = logits.softmax(-1)
        chosen = torch.multinomial(probabilities, count, replacement=True, generator=generator)
        return points.gather(1, chosen[..., None].expand(-1, -1, 2))
