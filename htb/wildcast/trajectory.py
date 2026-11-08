"""Joint, irregular-time spherical trajectory flow conditioned on observed history.

The frozen endpoint model supplies a destination-informed bridge prior. A joint
conditional flow transports that prior to future step distributions. All future
tokens can attend to each other; only observed history enters the conditioning.
Movement states are displacement proxies, not annotated biological behaviours.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .model import MovementTransformer, sample
from .raster_context import SatelliteFields, unproject_torch

RADIUS = 6_371_008.8
GRID_HOURS = (1., 2., 3., 4., 6., 8., 12., 18., 24., 36., 48., 72., 96., 120., 144., 168.)
STATE_NAMES = ("low_displacement", "intermediate_displacement", "high_displacement")


def project_torch(coords, origin):
    lat, lon = torch.deg2rad(coords).unbind(-1)
    lat0, lon0 = torch.deg2rad(origin).unbind(-1)
    dl = (lon - lon0 + math.pi) % (2 * math.pi) - math.pi
    east = torch.cos(lat) * torch.sin(dl)
    north = torch.cos(lat0) * torch.sin(lat) - torch.sin(lat0) * torch.cos(lat) * torch.cos(dl)
    dot = torch.sin(lat0) * torch.sin(lat) + torch.cos(lat0) * torch.cos(lat) * torch.cos(dl)
    norm = torch.sqrt(east.square() + north.square() + 1e-24)
    angle = torch.atan2(norm, dot.clamp(-1, 1))
    factor = RADIUS * angle / norm
    return torch.stack([east * factor, north * factor], -1)


def step_durations(times):
    return torch.diff(times, dim=-1, prepend=torch.zeros_like(times[..., :1])).clamp_min(1e-5)


def constrain_steps(raw, times, scale, speed_limit, gps_allowance=100.):
    """Smooth radial speed cap in metres; exact zero stays exactly zero.

    Limits are conservative training-derived assumptions. The GPS allowance is
    positional uncertainty per observed interval, not additional biological speed.
    """
    dt = step_durations(times)
    desired = torch.sinh(raw.clamp(-12, 12)) * (scale[..., None] * dt.sqrt())[..., None]
    bound = speed_limit[..., None] * dt * 3600 + gps_allowance
    radius = torch.linalg.vector_norm(desired, dim=-1)
    ratio = radius / bound.clamp_min(1e-6)
    multiplier = torch.where(ratio > 1e-5, torch.tanh(ratio) / ratio.clamp_min(1e-8), 1 - ratio.square() / 3)
    return desired * multiplier[..., None]


def encode_steps(step_metres, times, scale, speed_limit, gps_allowance=100.):
    dt = step_durations(times)
    bound = speed_limit[..., None] * dt * 3600 + gps_allowance
    radius = torch.linalg.vector_norm(step_metres, dim=-1)
    ratio = radius / bound.clamp_min(1e-6)
    clipped = ratio.clamp(max=.9999)
    factor = torch.where(ratio > 1e-5, torch.atanh(clipped) / ratio.clamp_min(1e-8), 1 + ratio.square() / 3)
    desired = step_metres * factor[..., None]
    return torch.asinh(desired / (scale[..., None] * dt.sqrt())[..., None].clamp_min(1e-6)).to(step_metres.dtype)


def decode_path(raw, times, aux, speed_limit, gps_allowance=100.):
    """Integrate geodesic steps on the sphere, including dateline crossings.

    raw [B,M,T,2], times [B,T], aux [B,9]. Returns offsets in
    the origin's equidistant projection and geographic coordinates [lat,lon].
    """
    # Double precision is essential: centimetre-to-metre increments disappear
    # when added to continental coordinates in float32.
    steps = constrain_steps(raw.double(), times[:, None].double(), aux[:, None, 8].double(),
                            speed_limit[:, None].double(), gps_allowance)
    here = aux[:, None, :2].double().expand(-1, raw.shape[1], -1)
    coords = []
    for k in range(raw.shape[2]):
        here = unproject_torch(steps[:, :, k], here[..., 0], here[..., 1])
        coords.append(here)
    coords = torch.stack(coords, 2)
    xy = project_torch(coords, aux[:, None, None, :2].double()) / aux[:, None, None, 8:9]
    return xy.float(), coords


def geographic_steps(future_xy, aux):
    coords = unproject_torch(future_xy.double() * aux[:, None, 8:9], aux[:, None, 0], aux[:, None, 1])
    previous = torch.cat([aux[:, None, :2], coords[:, :-1]], 1)
    return project_torch(coords, previous).float()


def time_features(times, aux):
    dt = step_durations(times)
    solar = 2 * math.pi * ((aux[:, 2:3] / 3600 + times + aux[:, 1:2] / 15) % 24) / 24
    season = 2 * math.pi * (aux[:, 2:3] / 86400 + times / 24) / 365.2425
    return torch.stack([torch.log1p(times), torch.log1p(dt), times / 168,
                        torch.sin(solar), torch.cos(solar), torch.sin(season), torch.cos(season),
                        (times / times.amax(-1, keepdim=True).clamp_min(1e-5))], -1).float()


class TrajectoryFlow(nn.Module):
    family = "wildcast_trajectory_flow"

    def __init__(self, config, fields=None):
        super().__init__()
        self.config = config
        self.backbone = MovementTransformer(**config["backbone"])
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        ctx = self.backbone.head[0].normalized_shape[0] + config["n_drivers"]
        width = config.get("width", 128)
        self.condition = nn.Sequential(nn.LayerNorm(ctx), nn.Linear(ctx, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.query = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width))
        self.flow_time = nn.Sequential(nn.Linear(17, width), nn.SiLU(), nn.Linear(width, width))
        self.flow_input = nn.Linear(7, width)  # current 2, prior 2, state probabilities 3
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, config.get("dropout", .05),
                                          batch_first=True, norm_first=True, activation="gelu")
        self.flow = nn.TransformerEncoder(layer, config.get("layers", 3), enable_nested_tensor=False)
        self.output = nn.Linear(width, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.state_emission = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.state_initial = nn.Linear(width, 3)
        self.transition_rates = nn.Parameter(torch.full((3, 3), -2.))
        species_count = config["backbone"]["n_species"] + 1
        self.species_adapter = nn.Embedding(species_count, width * 2)
        self.species_transition = nn.Embedding(species_count, 9)
        nn.init.zeros_(self.species_adapter.weight)
        nn.init.zeros_(self.species_transition.weight)
        self.register_buffer("speed_limits", torch.tensor(config["speed_limits"], dtype=torch.float32))
        self.register_buffer("species_classes", torch.tensor(config.get("species_classes", [0] * species_count), dtype=torch.long))
        self.register_buffer("species_orders", torch.tensor(config.get("species_orders", [0] * species_count), dtype=torch.long))
        self.register_buffer("identity_groups", torch.tensor(config.get("identity_groups", []), dtype=torch.long))
        n = config.get("memory_entries", 0)
        self.register_buffer("bank_keys", torch.zeros(n, ctx))
        self.register_buffer("bank_values", torch.zeros(n, 16))
        self.register_buffer("bank_species", torch.zeros(n, dtype=torch.long))
        self.register_buffer("bank_identity", torch.zeros(n, dtype=torch.long))
        self.experience_key = nn.Sequential(nn.LayerNorm(ctx), nn.Linear(ctx, 64))
        self.experience_value = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, width))
        self.experience_gate = nn.Parameter(torch.zeros(()))
        self.species_memory_bias = nn.Parameter(torch.tensor(2.))
        self.order_memory_bias = nn.Parameter(torch.tensor(1.))
        self.fields = fields if fields is not None else (SatelliteFields(config["field_metadata"]) if config.get("field_metadata") else None)
        if self.fields is not None:
            self.landscape = nn.Sequential(nn.Linear(130, width), nn.GELU(), nn.Linear(width, width))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def observed_context(self, a, ids, device):
        x = torch.as_tensor(a["x"][ids], device=device)
        sp = torch.as_tensor(a["meta"][ids, 1], device=device, dtype=torch.long)
        dr = torch.as_tensor(a["drivers"][ids], device=device).clone()
        # A single origin must have the same representation at every horizon.
        for i in self.config.get("horizon_driver_indices", []):
            dr[:, i] = 0
        v = torch.as_tensor(a["aux"][ids, 4:6], device=device, dtype=torch.float32)
        anchors = torch.as_tensor(a["anchors"][ids], device=device)
        mem = torch.as_tensor(a["memory"][ids], device=device)
        horizon = torch.full((len(ids),), 168., device=device)
        with torch.no_grad():
            context = self.backbone.context_features(x, sp, horizon, v, dr, anchors, mem)
        return torch.cat([context.float(), dr.float()], -1)

    def retrieve_experience(self, context, species, identity):
        if not len(self.bank_keys):
            return torch.zeros_like(self.condition(context))
        if identity is None:
            raise ValueError("Shared memory requires the canonical query identity for exclusion")
        q = torch.nn.functional.normalize(self.experience_key(context).float(), dim=-1)
        k = torch.nn.functional.normalize(self.experience_key(self.bank_keys).float(), dim=-1)
        scores = q @ k.T * 8
        scores = scores + self.species_memory_bias * (species[:, None] == self.bank_species)
        same_order = (self.species_orders[species, None] == self.species_orders[self.bank_species]) & (self.species_orders[species, None] != 0)
        scores = scores + self.order_memory_bias * same_order
        valid = identity[:, None] != self.bank_identity
        # Cross-species transfer remains within a known taxonomic class. Unknown
        # taxonomy falls back to learned similarity, never to target labels.
        qc, kc = self.species_classes[species, None], self.species_classes[self.bank_species]
        valid &= (qc == kc) | (qc == 0) | (kc == 0)
        values, selected = scores.masked_fill(~valid, -1e9).topk(min(8, len(self.bank_keys)), dim=-1)
        usable = valid.gather(1, selected)
        weights = values.softmax(-1) * usable
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        memories = self.bank_values[selected]
        mean = (weights[..., None] * memories).sum(1)
        variance = (weights[..., None] * (memories - mean[:, None]).square()).sum(1)
        output = self.experience_value(torch.cat([mean, (variance + 1e-8).sqrt()], -1))
        return output * usable.any(-1, keepdim=True)

    def state_probabilities(self, condition, query, times, mask, species):
        # Continuous-time transition generator: row-stochastic transition
        # matrices adapt to the actual duration, rather than an arbitrary tick.
        offsets = self.species_transition(species).reshape(-1, 3, 3)
        rates = torch.nn.functional.softplus(self.transition_rates.float()[None] + offsets.float()) * (1 - torch.eye(3, device=times.device))
        generator = rates - torch.diag_embed(rates.sum(-1))
        transitions = torch.matrix_exp(generator[:, None] * (step_durations(times) / 24)[..., None, None])
        emission = self.state_emission(condition[:, None] + query).float().softmax(-1)
        p = self.state_initial(condition).float().softmax(-1)
        states = []
        for j in range(times.shape[1]):
            proposed = torch.einsum("bi,bij->bj", p, transitions[:, j]) * emission[:, j]
            proposed = proposed / proposed.sum(-1, keepdim=True).clamp_min(1e-8)
            p = torch.where(mask[:, j:j + 1], proposed, p)
            states.append(p)
        return torch.stack(states, 1)

    def prior(self, context, times, mask, aux, velocity, anchors, species, count, generator):
        with torch.no_grad():
            if self.config.get("prior_kind", "bridge") == "copula":
                return self.copula_prior(context, times, mask, aux, velocity, anchors, species, count, generator)
            h = times.masked_fill(~mask, 0).amax(-1).clamp_min(.01)
            base = context[:, :self.backbone.head[0].normalized_shape[0]].clone()
            k = self.backbone.config["width"] * 2 + 24
            logh = torch.log1p(h)
            base[:, k:k + 3] = torch.stack([logh, logh.sin(), logh.cos()], -1)
            params = self.backbone.parameters_from_context(base, h, velocity, anchors)
            goal = sample(params, count, generator)
            dt = step_durations(times) * mask
            fraction = dt / h[:, None]
            eps = torch.randn((len(times), count, times.shape[1], 2), device=times.device, generator=generator)
            noise = eps * dt[:, None, :, None].sqrt()
            noise = noise - fraction[:, None, :, None] * noise.sum(2, keepdim=True)
            amplitude = (.2 + .15 * goal.norm(dim=-1) / h[:, None].sqrt()).clamp(.05, 2.)
            increments = goal[:, :, None] * fraction[:, None, :, None] + amplitude[:, :, None, None] * noise
            points = increments.cumsum(2)
            # The bridge is a prior, not a claimed observed intermediate route.
            coords = unproject_torch(points.double() * aux[:, None, None, 8:9],
                                     aux[:, None, None, 0], aux[:, None, None, 1])
            previous = torch.cat([aux[:, None, None, :2].expand(-1, count, 1, -1), coords[:, :, :-1]], 2)
            steps = project_torch(coords, previous).float()
            return encode_steps(steps, times[:, None], aux[:, None, 8], self.speed_limits[species, None],
                                self.config.get("gps_allowance_m", 100.)) * mask[:, None, :, None]

    def copula_prior(self, context, times, mask, aux, velocity, anchors, species, count, generator):
        """Couple the frozen model's horizon marginals before enforcing motion.

        Shared mixture quantiles and correlated noise give a connected starting
        law without discarding the backbone's short-horizon information. This
        copula is an initialization assumption; the flow learns path dependence.
        """
        b, length = times.shape
        base, h = self.horizon_context(context, times, aux)
        params = self.backbone.parameters_from_context(base, h,
                    velocity[:, None].expand(-1, length, -1).flatten(0, 1),
                    anchors[:, None].expand(-1, length, -1, -1).flatten(0, 1))
        logits, mu, sigma, rho = params[:4]
        components = logits.shape[-1]
        cumulative = logits.float().softmax(-1).reshape(b, length, components).cumsum(-1)
        uniform = torch.rand(b, count, device=times.device, generator=generator)
        choice = (uniform[:, None, :, None] > cumulative[:, :, None]).sum(-1).clamp_max(components - 1)
        index = choice[..., None].expand(-1, -1, -1, 2)
        mean = mu.reshape(b, length, components, 2).gather(2, index).transpose(1, 2)
        std = sigma.reshape(b, length, components, 2).gather(2, index).transpose(1, 2)
        correlation = rho.reshape(b, length, components).gather(2, choice).transpose(1, 2)
        shared = torch.randn(b, count, 1, 2, device=times.device, generator=generator)
        independent = torch.randn(b, count, length, 2, device=times.device, generator=generator)
        eps = (shared + .2 * independent) / 1.04 ** .5
        noise = torch.stack([eps[..., 0], correlation * eps[..., 0] + (1 - correlation.square()).sqrt() * eps[..., 1]], -1)
        if len(params) == 5:
            df = params[4]
            chi = torch.randn(b, count, 1, df, device=times.device, generator=generator).square().sum(-1).clamp_min(1e-8)
            noise = noise / (chi / df).sqrt()[..., None]
        points = mean + std * noise
        coords = unproject_torch(points.double() * aux[:, None, None, 8:9], aux[:, None, None, 0], aux[:, None, None, 1])
        previous = torch.cat([aux[:, None, None, :2].expand(-1, count, 1, -1), coords[:, :, :-1]], 2)
        physical = project_torch(coords, previous).float()
        return encode_steps(physical, times[:, None], aux[:, None, 8], self.speed_limits[species, None],
                            self.config.get("gps_allowance_m", 100.)) * mask[:, None, :, None]

    def horizon_context(self, context, times, aux):
        """Restore deterministic query-time cues in the frozen endpoint head."""
        b, length = times.shape
        core = self.backbone.head[0].normalized_shape[0]
        base = context[:, None, :core].expand(-1, length, -1).reshape(b * length, core).clone()
        h = times.flatten().clamp_min(.01)
        k = self.backbone.config["width"] * 2 + 24
        logh = h.log1p()
        base[:, k:k + 3] = torch.stack([logh, logh.sin(), logh.cos()], -1)
        if self.config.get("future_photoperiod"):
            import numpy as np
            import pandas as pd
            from .drivers import daylength
            spec = self.config["future_photoperiod"]
            seconds = aux[:, 2].detach().cpu().numpy()
            doy = pd.to_datetime(seconds, unit="s", utc=True).dayofyear.to_numpy() + (seconds % 86400) / 86400
            light = daylength(aux[:, 0].cpu().numpy()[:, None], doy[:, None] + times.cpu().numpy() / 24) / 24
            dr = context[:, None, core:].expand(-1, length, -1).reshape(b * length, -1).clone()
            dr[:, spec["index"]] = torch.as_tensor(np.clip((light - spec["median"]) / spec["iqr"], -12, 12).flatten(),
                                                  device=times.device, dtype=dr.dtype)
            if self.backbone.driver_indices is not None:
                dr = dr[:, self.backbone.driver_indices]
            encoded = self.backbone.driver_encoder(dr[:, :self.backbone.driver_base_features])
            if self.backbone.driver_base_features < self.backbone.n_drivers:
                encoded = encoded + self.backbone.environment_gate.tanh() * self.backbone.environment_encoder(dr[:, self.backbone.driver_base_features:])
            start = self.backbone.config["width"] * 2 + 29
            base[:, start:start + self.backbone.config["width"]] = encoded
        return base, h

    def prepare_condition(self, context, times, aux, mask, prior, species, identity=None):
        condition = self.condition(context)
        gain, shift = self.species_adapter(species).chunk(2, -1)
        condition = condition * (1 + .25 * gain.tanh()) + shift
        condition = condition + self.experience_gate.tanh() * self.retrieve_experience(context, species, identity)
        query = self.query(time_features(times, aux))
        states = self.state_probabilities(condition, query, times, mask, species)
        tokens = query + condition[:, None]
        if self.fields is not None:
            with torch.no_grad():
                # Sample-dependent spatial context, strictly previous-year data.
                points, _ = decode_path(prior, times, aux, self.speed_limits[species], self.config.get("gps_allowance_m", 100.))
                import pandas as pd
                year = torch.as_tensor(pd.to_datetime(aux[:, 2].detach().cpu().numpy(), unit="s", utc=True).year.to_numpy() - 1,
                                       device=aux.device)
                n = prior.shape[1]
                features = self.fields.candidate_features(points.flatten(1, 2), aux, year).reshape(len(times), n, times.shape[1], 130)
            tokens = tokens[:, None] + self.landscape(features)
        else:
            tokens = tokens[:, None].expand(-1, prior.shape[1], -1, -1)
        return tokens, states

    def vector_field(self, z, prior, flow_time, tokens, states, mask):
        b, m, length, _ = z.shape
        freq = 2 ** torch.arange(8, device=z.device, dtype=torch.float32) * math.pi
        phase = flow_time[:, None] * freq
        ft = self.flow_time(torch.cat([flow_time[:, None], phase.sin(), phase.cos()], -1))
        state = states[:, None].expand(-1, m, -1, -1)
        h = self.flow_input(torch.cat([z, prior, state], -1)) + tokens + ft[:, None, None]
        padding = ~mask[:, None].expand(-1, m, -1).reshape(b * m, length)
        h = self.flow(h.reshape(b * m, length, -1), src_key_padding_mask=padding)
        return self.output(h).reshape(b, m, length, 2).float() * mask[:, None, :, None]

    def sample_paths(self, context, times, mask, aux, velocity, anchors, species, count=16, generator=None, steps=None, identity=None):
        if count < 1 or times.ndim != 2 or mask.shape != times.shape or not mask.any(-1).all():
            raise ValueError("Positive sample count and at least one valid time per origin required")
        if ((torch.diff(times, dim=-1) <= 0) & mask[:, 1:]).any() or ((times <= 0) & mask).any():
            raise ValueError("Future query times must be positive and strictly increasing")
        prior = self.prior(context, times, mask, aux, velocity, anchors, species, count, generator)
        if self.config.get("frozen_prior_only", False):
            points, coords = decode_path(prior, times, aux, self.speed_limits[species], self.config.get("gps_allowance_m", 100.))
            previous = torch.cat([aux[:, None, None, :2].expand(-1, count, 1, -1), coords[:, :, :-1]], 2)
            physical = project_torch(coords, previous).norm(dim=-1)
            normalized = physical / (aux[:, None, 8:9] * step_durations(times)[:, None].sqrt())
            regime = (normalized > .25).long() + (normalized > 2.).long()
            probability = torch.nn.functional.one_hot(regime, 3).float().mean(1)
            return {"xy": points, "latlon": coords, "regime_probabilities": probability, "latent_steps": prior}
        tokens, states = self.prepare_condition(context, times, aux, mask, prior, species, identity)
        z = prior
        steps = self.config.get("ode_steps", 8) if steps is None else steps
        if steps < 1:
            raise ValueError("At least one integration step required")
        # Midpoint ODE solver; integration resolution differs from GPS time steps.
        for j in range(steps):
            t = torch.full((len(times),), j / steps, device=times.device)
            first = self.vector_field(z, prior, t, tokens, states, mask)
            middle = z + first / (2 * steps)
            second = self.vector_field(middle, prior, t + .5 / steps, tokens, states, mask)
            z = (z + second / steps).clamp(-10, 10)
        points, coords = decode_path(z, times, aux, self.speed_limits[species], self.config.get("gps_allowance_m", 100.))
        return {"xy": points, "latlon": coords, "regime_probabilities": states, "latent_steps": z}

    def predict_samples(self, a, ids, count, device, generator):
        # Same fixed joint grid for every horizon. Dedicated trajectory evaluation
        # reuses each origin's sampled path across horizons, avoiding redraws.
        times = torch.tensor(self.config.get("grid_hours", GRID_HOURS), device=device).expand(len(ids), -1)
        aux = torch.as_tensor(a["aux"][ids], device=device, dtype=torch.float64)
        species = torch.as_tensor(a["meta"][ids, 1], device=device, dtype=torch.long)
        context = self.observed_context(a, ids, device)
        result = self.sample_paths(context, times, torch.ones_like(times, dtype=torch.bool), aux,
                                   aux[:, 4:6].float(), torch.as_tensor(a["anchors"][ids], device=device),
                                   species, count, generator,
                                   identity=self.identity_groups[torch.as_tensor(a["meta"][ids, 3], device=device, dtype=torch.long)] if len(self.identity_groups) else None)
        horizons = torch.as_tensor(a["meta"][ids, 4], device=device)
        selected = (times - horizons[:, None]).abs().argmin(-1)
        if not torch.allclose(times[torch.arange(len(ids), device=device), selected], horizons.float()):
            raise ValueError("Requested horizon absent from the frozen trajectory grid")
        return result["xy"][torch.arange(len(ids), device=device), :, selected]


class JointTrajectoryEnsemble(nn.Module):
    """Controlled joint baseline: choose a pretrained member once per whole path.

    Its marginal forecasts inherit the existing endpoint ensemble before motion
    limits. Dependence is a declared copula assumption, not newly learned biology.
    """
    family = "wildcast_joint_prior_ensemble"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.members = nn.ModuleList([TrajectoryFlow(c) for c in config["members"]])
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def speed_limits(self):
        return self.members[0].speed_limits

    @property
    def identity_groups(self):
        return self.members[0].identity_groups

    def observed_context(self, a, ids, device):
        return [m.observed_context(a, ids, device) for m in self.members]

    def sample_paths(self, context, times, mask, aux, velocity, anchors, species, count=16,
                     generator=None, steps=None, identity=None):
        choice = torch.randint(len(self.members), (len(times), count), device=times.device, generator=generator)
        draws = [m.sample_paths(c, times, mask, aux, velocity, anchors, species, count, generator,
                                steps=steps, identity=identity) for m, c in zip(self.members, context)]
        result = {}
        for key in ("xy", "latlon", "latent_steps"):
            bank = torch.stack([v[key] for v in draws], 2)
            index = choice[:, :, None, None, None].expand(-1, -1, 1, times.shape[1], 2)
            result[key] = bank.gather(2, index).squeeze(2)
        result["regime_probabilities"] = torch.stack([v["regime_probabilities"] for v in draws]).mean(0)
        return result

    predict_samples = TrajectoryFlow.predict_samples


def build_prior_ensemble(root, cache, checkpoints, output):
    import hashlib
    import json
    from pathlib import Path
    from .sources import dump
    from .train import load_windows
    from .identity import case_fingerprint
    a, manifest = load_windows(root)
    info = json.loads((Path(cache) / "manifest.json").read_text())
    if info["case_fingerprint"] != case_fingerprint(a):
        raise ValueError("Joint baseline/cache cases differ")
    records = [torch.load(p, map_location="cpu", weights_only=True) for p in checkpoints]
    configs = []
    for ck in records:
        for key in ("species", "animals", "observations_sha256", "context"):
            if ck["manifest"][key] != manifest[key]:
                raise ValueError(f"Joint baseline source mismatch: {key}")
        n = ck["config"]["n_drivers"]
        if ck["manifest"]["driver_names"] != manifest["driver_names"][:n]:
            raise ValueError("Joint baseline driver schema differs")
        configs.append({**info["config"], "backbone": ck["config"], "memory_entries": 0,
                        "field_metadata": [], "prior_kind": "copula", "frozen_prior_only": True})
        if "future_photoperiod_at_origin" in manifest["driver_names"]:
            i = manifest["driver_names"].index("future_photoperiod_at_origin")
            configs[-1]["future_photoperiod"] = {"index": i, "median": manifest["driver_median"][i], "iqr": manifest["driver_iqr"][i]}
    config = {"members": configs, "grid_hours": list(GRID_HOURS), "gps_allowance_m": info["config"]["gps_allowance_m"]}
    model = JointTrajectoryEnsemble(config)
    for member, ck in zip(model.members, records):
        member.backbone.load_state_dict(ck["model"])
    training = {"seed": 0, "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": 0,
                "base_checkpoints": [{"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in checkpoints],
                "identity_audit_sha256": info["identity_audit_sha256"],
                "note": "Frozen endpoint models with shared-quantile path copulas and speed envelopes; no new optimization or test selection"}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_family": model.family, "config": config, "model": model.state_dict(),
                "training": training, "manifest": manifest, "epoch": 0, "val_score": None}, out / "best.pt")
    dump(out / "config.json", training)
    return {"output": str(out), **training}
