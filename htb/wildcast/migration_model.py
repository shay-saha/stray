"""Small probabilistic *joint* migration forecasts.

The route factorization is a modelling hypothesis, not a biological label.  Each
mixture component has one destination and a smooth route mean.  Samples select
one component for the entire journey and use temporally correlated Gaussian
noise.  The likelihood and sampler describe exactly the same distribution.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ("seasonal_static", "history_direct", "goal_route", "residence_route")


class MigrationModel(nn.Module):
    def __init__(self, history_features, static_features, n_species, horizons,
                 history_length=32, width=128, layers=2, components=6,
                 variant="goal_route", dropout=.1, min_scale=.0005, species_dropout=.1,
                 history_newest_first=True, covariance_jitter=None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if width % 4 or width < 8:
            raise ValueError("width must be a multiple of four and at least eight")
        if variant == "residence_route" and components < 2:
            raise ValueError("residence_route needs a residence and an already-moving component")
        if not 0 <= species_dropout <= 1:
            raise ValueError("species_dropout must lie between zero and one")
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive")
        if covariance_jitter is not None and (not math.isfinite(covariance_jitter) or covariance_jitter <= 0):
            raise ValueError("covariance_jitter must be finite and positive")
        times = torch.as_tensor(horizons, dtype=torch.float32)
        if times.ndim != 1 or not len(times) or not torch.all(times > 0) or not torch.all(times[1:] > times[:-1]):
            raise ValueError("horizons must be positive and strictly increasing")
        self.config = dict(history_features=int(history_features), static_features=int(static_features),
                           n_species=int(n_species), horizons=times.tolist(), history_length=int(history_length),
                           width=int(width), layers=int(layers), components=int(components),
                           variant=variant, dropout=float(dropout), min_scale=float(min_scale),
                           species_dropout=float(species_dropout), history_newest_first=bool(history_newest_first))
        # Preserve historical checkpoint configs and seeded laws by default.
        # Subdaily recipes can explicitly choose metre-scale numerical floors;
        # those floors are modelling choices, not measured sensor precision.
        self.covariance_jitter = 1e-8 if covariance_jitter is None else float(covariance_jitter)
        if covariance_jitter is not None:
            self.config["covariance_jitter"] = self.covariance_jitter
        self.variant, self.components, self.n_horizons = variant, int(components), len(times)
        self.min_scale = float(min_scale)
        self.register_buffer("horizons", times)
        self.register_buffer("history_mean", torch.zeros(history_features))
        self.register_buffer("history_std", torch.ones(history_features))
        self.register_buffer("static_mean", torch.zeros(static_features))
        self.register_buffer("static_std", torch.ones(static_features))
        self.species = nn.Embedding(n_species, width)
        self.static_encoder = nn.Sequential(nn.Linear(static_features, width), nn.GELU(), nn.LayerNorm(width))
        if variant != "seasonal_static":
            self.history_input = nn.Linear(history_features, width)
            self.position = nn.Parameter(torch.randn(history_length + 2, width) * .02)
            self.start = nn.Parameter(torch.zeros(width))
            self.summary = nn.Parameter(torch.zeros(width))
            layer = nn.TransformerEncoderLayer(width, 4, width * 3, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.history_encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.fusion = nn.Sequential(nn.Linear(width * (2 if variant == "seasonal_static" else 3), width),
                                    nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout),
                                    nn.Linear(width, width), nn.GELU())
        self.logit_head = nn.Linear(width, components)
        self.mean_head = nn.Linear(width, components * (8 if variant in ("goal_route", "residence_route") else 2 * len(times)))
        self.scale_head = nn.Linear(width, components * 2 * len(times))
        self.dependence_head = nn.Linear(width, components * 2)
        # Small initial conditional deviations around a broad, valid path law.
        for head in (self.logit_head, self.mean_head, self.scale_head, self.dependence_head):
            nn.init.normal_(head.weight, std=.01)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.scale_head.bias, -2.)
        if variant == "residence_route":
            self._initialize_residence_timing()

    @torch.no_grad()
    def _initialize_residence_timing(self):
        """Spread departures while leaving one moving component at time zero.

        The zero-mean component is an uncertainty distribution, not an exact
        stationary path. Means/uncertainty retain the same parameter budget as
        goal_route. Only this new variant changes its timing initialization.
        """
        last = self.config["horizons"][-1]
        minimum = min(self.config["horizons"][0] / last, .25)
        bias = self.mean_head.bias.reshape(self.components, 8)
        # An already-moving route can learn a short duration independently.
        initial_duration = max(self.config["horizons"][min(2, self.n_horizons - 1)] / last, 2 * minimum)
        fraction = min(max((initial_duration - minimum) / (1 - minimum), .02), .95)
        bias[1, 3] = math.log(fraction / (1 - fraction))
        if self.components > 2:
            anchors = self.horizons[:-1] if self.n_horizons > 1 else self.horizons
            indices = torch.linspace(0, len(anchors) - 1, self.components - 2).round().long()
            departure = .5 * anchors[indices] / last
            fraction = (departure / (1 - minimum)).clamp(.001, .98)
            bias[2:, 2] = torch.logit(fraction)
            # Interior horizons then provide timing gradients for several
            # components instead of placing all routes past the last waypoint.
            bias[2:, 3] = math.log(.75 / .25)

    def _residence_means(self, raw, return_timing=False):
        """Exactly anchored prefix means, followed by a smooth destination route.

        A sampled component owns one mean departure and duration for its whole
        path. With r=clip((t-d)/u,0,1), the mean is
        smoothstep(r)*destination + 4*r²*(1-r)²*(A+(2*r-1)*B).
        Covariance is deliberately unchanged. These are mathematical mean
        timings, not biological departure or arrival labels.

        Exact gating creates timing-gradient dead zones: if every observed
        horizon lies before departure or after arrival, d/u cannot be learned
        from those means. Initial timing bands mitigate this; they cannot reveal
        timing inside an unobserved horizon gap.
        """
        maximum = self.config["horizons"][-1]
        minimum = min(self.config["horizons"][0] / maximum, .25)
        departure = (1 - minimum) * raw[..., 2].sigmoid()
        departure = torch.cat((torch.zeros_like(departure[:, :2]), departure[:, 2:]), dim=1)
        duration = minimum + (1 - departure - minimum) * raw[..., 3].sigmoid()
        time = (self.horizons / maximum)[None, None, :, None]
        progress = ((time - departure[..., None, None]) / duration[..., None, None]).clamp(0, 1)
        smooth = progress.square() * (3 - 2 * progress)
        destination = raw[..., :2].unsqueeze(-2)
        curve = raw[..., 4:6].unsqueeze(-2) + (2 * progress - 1) * raw[..., 6:8].unsqueeze(-2)
        means = smooth * destination + 4 * progress.square() * (1 - progress).square() * curve
        means = torch.cat((torch.zeros_like(means[:, :1]), means[:, 1:]), dim=1)
        if return_timing:
            departure_days = departure * maximum
            # The anchored component has no mean departure inside this window.
            departure_days = torch.cat((torch.full_like(departure_days[:, :1], maximum + self.config["horizons"][0]),
                                        departure_days[:, 1:]), dim=1)
            return means, departure_days, duration * maximum
        return means

    @torch.no_grad()
    def set_normalization(self, history_mean, history_std, static_mean, static_std):
        for name, value in (("history_mean", history_mean), ("history_std", history_std),
                            ("static_mean", static_mean), ("static_std", static_std)):
            buffer = getattr(self, name)
            buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device))
        if not torch.all(self.history_std > 0) or not torch.all(self.static_std > 0):
            raise ValueError("normalization standard deviations must be positive")

    @torch.no_grad()
    def initialize_scales(self, scales):
        """Set initial marginal uncertainty using training targets only."""
        scales = torch.as_tensor(scales, dtype=self.scale_head.bias.dtype,
                                 device=self.scale_head.bias.device).reshape(1, -1)
        positive = (scales - self.min_scale).clamp(min=.001)
        inverse = positive + torch.log(-torch.expm1(-positive))
        self.scale_head.bias.copy_(inverse.expand(self.components, -1).reshape(-1))

    def _encode(self, history, history_mask, static, species):
        if history.shape[1] != self.config["history_length"]:
            raise ValueError("history length differs from the checkpoint schema")
        if species.min() < 0 or species.max() >= self.config["n_species"]:
            raise ValueError("species index outside checkpoint schema")
        if self.training and self.config["species_dropout"]:
            hidden = torch.rand(species.shape, device=species.device) < self.config["species_dropout"]
            species = torch.where(hidden, 0, species)
        stat = ((static - self.static_mean) / self.static_std).clamp(-20, 20)
        parts = [self.static_encoder(stat), self.species(species)]
        if self.variant != "seasonal_static":
            # The migration cache lists requested lags from zero to one year.
            # Encode them in chronological order for causal attention.
            if self.config["history_newest_first"]:
                history, history_mask = history.flip(1), history_mask.flip(1)
            hist = torch.where(history_mask[..., None], history, self.history_mean)
            hist = ((hist - self.history_mean) / self.history_std).clamp(-20, 20)
            hist = self.history_input(hist)
            batch = len(history)
            # An always-visible start token prevents all-masked attention rows.
            # The final token can attend to all observed history; no input is future.
            tokens = torch.cat((self.start.expand(batch, 1, -1), hist,
                                self.summary.expand(batch, 1, -1)), dim=1)
            tokens = tokens + self.position
            padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=hist.device),
                                 ~history_mask.bool(),
                                 torch.zeros(batch, 1, dtype=torch.bool, device=hist.device)), dim=1)
            causal = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1],
                                          dtype=torch.bool, device=hist.device), diagonal=1)
            encoded = self.history_encoder(tokens, mask=causal, src_key_padding_mask=padding)
            parts.append(encoded[:, -1])
        return self.fusion(torch.cat(parts, dim=-1))

    def forward(self, history, history_mask, static, species):
        z = self._encode(history, history_mask, static, species)
        batch, count, horizons = len(z), self.components, self.n_horizons
        raw = self.mean_head(z).reshape(batch, count, -1)
        if self.variant == "goal_route":
            t = (self.horizons / self.horizons[-1])[None, None, :, None]
            destination = raw[..., :2, None].transpose(-1, -2)
            departure = raw[..., 2:3, None].sigmoid().transpose(-1, -2)
            duration = (.05 + .95 * raw[..., 3:4, None].sigmoid()).transpose(-1, -2)
            base = torch.sigmoid(-departure / duration)
            progress = (torch.sigmoid((t - departure) / duration) - base)
            progress = progress / (torch.sigmoid((1 - departure) / duration) - base).clamp_min(1e-5)
            curve_a, curve_b = raw[..., 4:6, None].transpose(-1, -2), raw[..., 6:8, None].transpose(-1, -2)
            means = progress * destination + t * (1 - t) * (curve_a + (2 * t - 1) * curve_b)
        elif self.variant == "residence_route":
            means = self._residence_means(raw)
        else:
            means = raw.reshape(batch, count, horizons, 2)
        scale = F.softplus(self.scale_head(z).reshape(batch, count, horizons * 2)) + self.min_scale
        dependence = self.dependence_head(z).reshape(batch, count, 2)
        length = F.softplus(dependence[..., 0]) + .1
        rho = .95 * dependence[..., 1].tanh()
        logtime = torch.log1p(self.horizons)
        distance = (logtime[:, None] - logtime[None, :]).abs()
        temporal = torch.exp(-distance / length[..., None, None])
        spatial = torch.stack((torch.ones_like(rho), rho, rho, torch.ones_like(rho)), dim=-1)
        spatial = spatial.reshape(batch, count, 2, 2)
        correlation = torch.einsum("bkhj,bkac->bkhajc", temporal, spatial).reshape(batch, count, 2 * horizons, 2 * horizons)
        covariance = correlation * scale[..., :, None] * scale[..., None, :]
        covariance = covariance + torch.eye(2 * horizons, device=z.device, dtype=z.dtype) * self.covariance_jitter
        return {"logits": self.logit_head(z), "means": means, "covariance": covariance}

    def sample_paths(self, history, history_mask, static, species, samples, generator=None):
        return sample_paths(self(history, history_mask, static, species), samples, generator)


def sample_paths(parameters, samples, generator=None):
    if samples <= 0:
        raise ValueError("samples must be positive")
    means = parameters["means"]
    batch, _, horizons, _ = means.shape
    component = torch.multinomial(parameters["logits"].softmax(-1), samples,
                                  replacement=True, generator=generator)
    rows = torch.arange(batch, device=means.device)[:, None]
    chosen_mean = means.flatten(-2)[rows, component]
    cholesky = torch.linalg.cholesky(parameters["covariance"])[rows, component]
    noise = torch.randn(batch, samples, 2 * horizons, device=means.device,
                        dtype=means.dtype, generator=generator)
    return (chosen_mean + torch.einsum("bsij,bsj->bsi", cholesky, noise)).reshape(batch, samples, horizons, 2)


def masked_joint_nll(parameters, target, target_mask):
    """Exact marginal mixture NLL, integrating out every missing waypoint.

    Missing dimensions become independent N(0,1) dummy variables. Their log
    determinants and residuals vanish, and their normalizing constants are
    omitted. This is equivalent to taking each observed covariance submatrix.
    """
    mask = target_mask.bool().repeat_interleave(2, dim=-1)
    means = parameters["means"].flatten(-2)
    clean = torch.where(mask, target.flatten(-2), 0.)
    if not torch.isfinite(clean).all():
        raise ValueError("observed migration targets must be finite")
    residual = torch.where(mask[:, None], clean[:, None] - means, 0.)
    covariance = parameters["covariance"] * (mask[:, None, :, None] & mask[:, None, None, :])
    covariance = covariance + torch.diag_embed((~mask).to(covariance.dtype))[:, None]
    chol = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(chol, residual[..., None], upper=False).squeeze(-1)
    logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    logp = -.5 * (whitened.square().sum(-1) + logdet + mask.sum(-1)[:, None] * math.log(2 * math.pi))
    value = -torch.logsumexp(parameters["logits"].log_softmax(-1) + logp, dim=-1)
    return torch.where(mask.any(-1), value, value * 0.)


def mixture_energy(parameters, target, target_mask, repeats=2, generator=None):
    """Proper endpoint and joint-path energy estimates with probability gradients.

    Independent stratified banks avoid self-distance bias and categorical
    sampling in the loss. Mixture probabilities weight all component pairs.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    means = parameters["means"]
    batch, components, horizons, _ = means.shape
    chol = torch.linalg.cholesky(parameters["covariance"])
    banks = []
    for _ in range(2):
        eps = torch.randn(batch, components, repeats, horizons * 2,
                          device=means.device, dtype=means.dtype, generator=generator)
        bank = means.flatten(-2)[:, :, None] + torch.einsum("bkij,bkrj->bkri", chol, eps)
        banks.append(bank.reshape(batch, components * repeats, horizons, 2))
    weight = parameters["logits"].softmax(-1).repeat_interleave(repeats, dim=-1) / repeats
    mask = target_mask.bool()
    clean = torch.where(mask[..., None], target, 0.)
    count = mask.sum(-1).clamp_min(1)
    endpoint_first = ((banks[0] - clean[:, None]).norm(dim=-1) * weight[..., None]).sum(1)
    a = banks[0].transpose(1, 2)
    b = banks[1].transpose(1, 2)
    endpoint_pairs = torch.cdist(a, b)
    pair_weight = weight[:, None, :, None] * weight[:, None, None, :]
    endpoint = ((endpoint_first - .5 * (endpoint_pairs * pair_weight).sum((-1, -2))) * mask).sum(-1) / count
    flat_a = (banks[0] * mask[:, None, :, None]).flatten(-2) / count.sqrt()[:, None, None]
    flat_b = (banks[1] * mask[:, None, :, None]).flatten(-2) / count.sqrt()[:, None, None]
    truth = clean.flatten(-2) / count.sqrt()[:, None]
    first = ((flat_a - truth[:, None]).norm(dim=-1) * weight).sum(-1)
    pairs = torch.cdist(flat_a, flat_b)
    path = first - .5 * (pairs * weight[:, :, None] * weight[:, None, :]).sum((-1, -2))
    return {"endpoint_energy": endpoint, "path_energy": path}


def load_model(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    if ckpt.get("model_kind") != "wildcast_migration_joint":
        raise ValueError("not a WildCast joint migration checkpoint")
    model = MigrationModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
