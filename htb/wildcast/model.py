"""History-conditioned multimodal endpoint distributions."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MovementTransformer(nn.Module):
    def __init__(self, n_species, n_features=10, width=128, layers=3, heads=4, components=8, dropout=.15, context=32, student_df=0,
                 n_drivers=0, n_anchors=0, driver_indices=None, memory_features=0, memory_length=0, memory_fusion="concat",
                 driver_base_features=0, horizon_adapter=False, regime_head=False):
        super().__init__()
        self.config = dict(n_species=n_species, n_features=n_features, width=width, layers=layers,
                           heads=heads, components=components, dropout=dropout, context=context, student_df=student_df,
                           n_drivers=n_drivers, n_anchors=n_anchors, driver_indices=driver_indices,
                           memory_features=memory_features, memory_length=memory_length, memory_fusion=memory_fusion,
                           driver_base_features=driver_base_features, horizon_adapter=horizon_adapter, regime_head=regime_head)
        if student_df and (student_df < 3 or not isinstance(student_df, int)):
            raise ValueError("Student degrees of freedom must be an integer >=3, or 0 for Gaussian")
        self.student_df = student_df
        self.components = components
        self.n_drivers, self.n_anchors = n_drivers, n_anchors
        if n_anchors > components:
            raise ValueError("Need at least as many mixture components as movement anchors")
        self.driver_indices = driver_indices
        self.memory_features = memory_features
        self.memory_fusion = memory_fusion
        self.driver_base_features = driver_base_features or n_drivers
        if n_drivers and not 0 < self.driver_base_features <= n_drivers:
            raise ValueError("Invalid base driver dimension")
        self.input = nn.Linear(n_features, width)
        self.position = nn.Parameter(torch.randn(1, context, width) * .02)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.species = nn.Embedding(n_species + 1, 24)
        if n_drivers:
            self.driver_encoder = nn.Sequential(nn.LayerNorm(self.driver_base_features), nn.Linear(self.driver_base_features, width), nn.GELU(),
                                                nn.Dropout(dropout), nn.Linear(width, width))
            if self.driver_base_features < n_drivers:
                extra = n_drivers - self.driver_base_features
                self.environment_encoder = nn.Sequential(nn.LayerNorm(extra), nn.Linear(extra, width), nn.GELU(),
                                                         nn.Dropout(dropout), nn.Linear(width, width))
                self.environment_gate = nn.Parameter(torch.zeros(1))
        if memory_features:
            self.memory_input = nn.Linear(memory_features, width)
            self.memory_position = nn.Parameter(torch.randn(1, memory_length, width) * .02)
            memory_layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout, batch_first=True, activation="gelu", norm_first=True)
            self.memory_encoder = nn.TransformerEncoder(memory_layer, 1, enable_nested_tensor=False)
            self.memory_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            if memory_fusion == "residual":
                self.memory_gate = nn.Parameter(torch.zeros(1))
        head_width = width * (2 + bool(n_drivers) + bool(memory_features and memory_fusion == "concat")) + 29
        self.head = nn.Sequential(nn.LayerNorm(head_width), nn.Linear(head_width, width * 2),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, components * 6))
        if horizon_adapter:
            self.horizon_adapter = nn.ModuleList([nn.Linear(head_width, components * 6) for _ in range(4)])
            for layer in self.horizon_adapter:
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        if regime_head:
            if not n_anchors:
                raise ValueError("Displacement regimes require anchored mixture components")
            self.regime_head = nn.Linear(head_width, 3)
            nn.init.zeros_(self.regime_head.weight); nn.init.zeros_(self.regime_head.bias)
        self.last_regime_logits = None

    def context_features(self, x, species, horizon, velocity, drivers=None, anchors=None, memory=None):
        # All tokens are observed before the forecast origin. Bidirectional
        # attention within this observed context is causal with respect to target.
        h = self.encoder(self.input(x) + self.position[:, :x.shape[1]])
        parts = [h[:, -1], h.mean(1), self.species(species),
                       torch.log1p(horizon)[:, None], torch.sin(torch.log1p(horizon))[:, None],
                       torch.cos(torch.log1p(horizon))[:, None], torch.asinh(velocity)]
        if self.n_drivers:
            if drivers is None:
                raise ValueError("This model requires migration drivers")
            if self.driver_indices is not None:
                drivers = drivers[:, self.driver_indices]
            encoded_drivers = self.driver_encoder(drivers[:, :self.driver_base_features])
            if self.driver_base_features < self.n_drivers:
                encoded_drivers = encoded_drivers + torch.tanh(self.environment_gate) * self.environment_encoder(drivers[:, self.driver_base_features:])
            parts.append(encoded_drivers)
        if self.memory_features:
            if memory is None:
                raise ValueError("This model requires a sparse observed memory stream")
            padding = memory[..., 0] < .5
            encoded = self.memory_encoder(self.memory_input(memory) + self.memory_position, src_key_padding_mask=padding)
            attended, _ = self.memory_attention(h[:, -1:], encoded, encoded, key_padding_mask=padding, need_weights=False)
            if self.memory_fusion == "residual":
                parts[0] = parts[0] + torch.tanh(self.memory_gate) * attended[:, 0]
            else:
                parts.append(attended[:, 0])
        return torch.cat(parts, dim=-1)

    def forward(self, x, species, horizon, velocity, drivers=None, anchors=None, memory=None):
        h = self.context_features(x, species, horizon, velocity, drivers, anchors, memory)
        return self.parameters_from_context(h, horizon, velocity, anchors)

    def parameters_from_context(self, h, horizon, velocity, anchors=None):
        """Decode a cached observed-history representation without re-encoding it."""
        raw_flat = self.head(h)
        if self.config["horizon_adapter"]:
            horizons = torch.tensor([1., 12., 24., 168.], device=h.device)
            code = (horizon[:, None] - horizons).abs().argmin(-1)
            corrections = torch.stack([layer(h) for layer in self.horizon_adapter], dim=1)
            raw_flat = raw_flat + corrections[torch.arange(len(h), device=h.device), code]
        raw = raw_flat.reshape(-1, self.components, 6).float()
        # Sqrt-time scaling gives a useful initial random-walk prior; learned
        # means/scales can depart from it at every horizon.
        scale = torch.sqrt(horizon).clamp_min(1)[:, None, None]
        logits = raw[..., 0]
        if self.config["regime_head"]:
            self.last_regime_logits = self.regime_head(h).float()
            # These are endpoint-displacement regimes, not observed biological
            # behaviours: compact origin mode, historical anchors, free modes.
            groups = torch.tensor([0] + [1] * (self.n_anchors - 1) + [2] * (self.components - self.n_anchors), device=h.device)
            logits = logits + self.last_regime_logits[:, groups]
        means = raw[..., 1:3] * scale
        sigma = (F.softplus(raw[..., 3:5]) + .005) * scale
        if self.n_anchors:
            if anchors is None or anchors.shape[1] < self.n_anchors:
                raise ValueError("This model requires causal movement anchors")
            bases = anchors[:, :self.n_anchors]
            if self.n_anchors == 5:
                bases = torch.cat([bases[:, :4], (velocity * horizon.clamp(max=12)[:, None])[:, None]], dim=1)
            zeros = torch.zeros((len(h), self.components - self.n_anchors, 2), device=h.device, dtype=h.dtype)
            bases = torch.cat([bases, zeros], dim=1)
            means = means + bases
            # A compact resting component should not diffuse merely because
            # the forecast horizon is long. State identity remains latent.
            means = torch.cat([raw[:, :1, 1:3] * .1, means[:, 1:]], dim=1)
            sigma = torch.cat([(F.softplus(raw[:, :1, 3:5]) + .005) * .1, sigma[:, 1:]], dim=1)
        rho = .95 * torch.tanh(raw[..., 5])
        params = (logits, means, sigma, rho)
        return (*params, self.student_df) if self.student_df else params


def nll(params, y):
    logits, mu, sigma, rho = params[:4]
    z = (y[:, None] - mu) / sigma
    one_minus = (1 - rho.square()).clamp_min(.01)
    quadratic = (z[..., 0].square() + z[..., 1].square() - 2 * rho * z[..., 0] * z[..., 1]) / one_minus
    if len(params) == 5:
        df = params[4]
        normalizer = math.lgamma((df + 2) / 2) - math.lgamma(df / 2) - math.log(df * math.pi)
        logp = normalizer - sigma.log().sum(-1) - .5 * one_minus.log() - ((df + 2) / 2) * torch.log1p(quadratic / df)
    else:
        logp = -math.log(2 * math.pi) - sigma.log().sum(-1) - .5 * one_minus.log() - .5 * quadratic
    return -torch.logsumexp(logits.log_softmax(-1) + logp, -1)


def sample(params, count, generator=None):
    logits, mu, sigma, rho = params[:4]
    k = torch.multinomial(logits.softmax(-1), count, replacement=True, generator=generator)
    ix = k[..., None].expand(-1, -1, 2)
    m = mu.gather(1, ix); s = sigma.gather(1, ix); r = rho.gather(1, k)
    eps = torch.randn(m.shape, device=m.device, generator=generator)
    z = torch.stack([eps[..., 0], r * eps[..., 0] + (1 - r.square()).sqrt() * eps[..., 1]], -1)
    if len(params) == 5:
        df = params[4]
        chi = torch.randn((*m.shape[:2], df), device=m.device, generator=generator).square().sum(-1).clamp_min(1e-8)
        z = z / (chi / df).sqrt()[..., None]
    return m + s * z


def energy_score(samples, truth):
    """Unbiased U-statistic estimator of ES; O(B M²) reference implementation."""
    m = samples.shape[1]
    if m < 2:
        raise ValueError("Energy score requires at least two Monte Carlo samples")
    accuracy = torch.linalg.vector_norm(samples - truth[:, None], dim=-1).mean(1)
    distances = torch.cdist(samples, samples)
    # GPU matmul-based cdist can yield tiny nonzero self-distances.
    pairs = (distances.sum((1, 2)) - distances.diagonal(dim1=1, dim2=2).sum(1)) / (m * (m - 1))
    return accuracy - .5 * pairs


def component_samples(params, repeats=2, generator=None):
    """Stratified reparameterized draws from EVERY mixture component."""
    logits, mu, sigma, rho = params[:4]
    shape = (len(mu), mu.shape[1], repeats, 2)
    eps = torch.randn(shape, device=mu.device, generator=generator)
    r = rho[..., None]
    z = torch.stack([eps[..., 0], r * eps[..., 0] + (1 - r.square()).sqrt() * eps[..., 1]], -1)
    if len(params) == 5:
        df = params[4]
        chi = torch.randn((*shape[:-1], df), device=mu.device, generator=generator).square().sum(-1).clamp_min(1e-8)
        z = z / (chi / df).sqrt()[..., None]
    draws = (mu[:, :, None] + sigma[:, :, None] * z).flatten(1, 2)
    weights = logits.softmax(-1).repeat_interleave(repeats, dim=1) / repeats
    return draws, weights


def mixture_energy_score(params, truth, repeats=2, generator=None):
    """Unbiased differentiable ES estimate, including gradients to mode weights.

    Independent draw sets avoid self-pair bias. Stratification integrates the
    discrete component distribution exactly rather than sampling its labels.
    """
    x, w = component_samples(params, repeats, generator)
    xp, _ = component_samples(params, repeats, generator)
    accuracy = .5 * (w * (torch.linalg.vector_norm(x - truth[:, None], dim=-1) +
                           torch.linalg.vector_norm(xp - truth[:, None], dim=-1))).sum(1)
    dispersion = torch.einsum("bi,bij,bj->b", w, torch.cdist(x, xp), w)
    return accuracy - .5 * dispersion
