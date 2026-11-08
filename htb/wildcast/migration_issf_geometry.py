"""Device-vectorized evaluation of the unchanged iSSF cell quadrature."""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def gamma_vm_grid(parameters, cell_metres, bearing, known, window, chunk=16):
    device = cell_metres.device
    parameters = torch.as_tensor(parameters, dtype=torch.float64, device=device)
    cell_metres = cell_metres.double()
    bearing = torch.as_tensor(bearing, dtype=torch.float64, device=device)
    known = torch.as_tensor(known, dtype=torch.bool, device=device)
    axis = torch.arange(window, dtype=torch.float64, device=device)
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    offsets = (torch.arange(4, dtype=torch.float64, device=device) + .5) / 4 - .5
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    unit_x = col[..., None] - window // 2 + ox.flatten()
    unit_y = window // 2 - row[..., None] + oy.flatten()
    theta = (torch.arange(256, dtype=torch.float64, device=device) + .5) * (2 * math.pi / 256)
    central_unit = .5 / torch.maximum(theta.cos().abs(), theta.sin().abs())
    output = torch.empty((len(parameters), window, window), dtype=torch.float64, device=device)
    for start in range(0, len(parameters), chunk):
        end = min(start + chunk, len(parameters))
        shape, scale, mu, kappa = parameters[start:end].unbind(-1)
        direction = torch.where(known[start:end], bearing[start:end], 0.)
        mu = torch.where(known[start:end], mu, 0.)
        kappa = torch.where(known[start:end], kappa, 0.)
        cell = cell_metres[start:end]
        x, y = unit_x[None] * cell[:, None, None, None], unit_y[None] * cell[:, None, None, None]
        radial, angle = torch.hypot(x, y), torch.atan2(y, x)
        angular_norm = math.log(2 * math.pi) + torch.special.i0e(kappa).log() + kappa.abs()
        norm = torch.lgamma(shape) + shape * scale.log() + angular_norm
        density = ((shape[:, None, None, None] - 2) * radial.log() - radial / scale[:, None, None, None]
                   + kappa[:, None, None, None] * (angle - direction[:, None, None, None] - mu[:, None, None, None]).cos()
                   - norm[:, None, None, None])
        masses = density.logsumexp(-1) + 2 * cell[:, None, None].log() - math.log(16)
        limit = cell[:, None] * central_unit
        angular = (kappa[:, None] * (theta - direction[:, None] - mu[:, None]).cos() - angular_norm[:, None]).exp()
        central = (torch.special.gammainc(shape[:, None], limit / scale[:, None]) * angular).sum(-1) * (2 * math.pi / 256)
        masses[:, window // 2, window // 2] = central.clamp_min(torch.finfo(torch.float64).tiny).log()
        output[start:end] = masses - masses.logsumexp((1, 2))[:, None, None]
    return output


def predictor_features(a, ids, heads, parameters, cell, bearing, known, window, variant, environment, scalars):
    """Same movement/time/environment products and ±40 clip, on the model device."""
    device = cell.device
    axis = torch.arange(window, dtype=torch.float64, device=device)
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    unit = torch.stack([col - window // 2, window // 2 - row], -1).reshape(-1, 2)
    xy = unit[None] * cell[:, None, None]
    radius = torch.linalg.vector_norm(xy, dim=-1)
    theta = torch.atan2(xy[..., 1], xy[..., 0])
    params = torch.as_tensor(parameters, dtype=torch.float64, device=device)
    shape, scale = params[:, 0:1], params[:, 1:2]
    bearing = torch.as_tensor(bearing, dtype=torch.float64, device=device)
    known = torch.as_tensor(known, dtype=torch.bool, device=device)
    turn = (theta - bearing[:, None]).cos() * known[:, None]
    movement = torch.stack([radius / (shape * scale),
        (radius.clamp_min(.1).log() - torch.special.digamma(shape) - scale.log()) / torch.special.polygamma(1, shape).sqrt(),
        torch.where(radius > 0, turn, 0.)], -1)
    temporal = torch.as_tensor(scalars[:, :4], dtype=torch.float64, device=device)[:, None].expand(-1, window * window, -1)
    environment = torch.as_tensor(environment, dtype=torch.float64, device=device)
    parts = [movement, environment, temporal, known[:, None, None].expand(-1, window * window, 1)]
    if variant == "linear":
        for left, right in ((movement, environment), (temporal, environment), (temporal, movement)):
            parts.append((left[..., :, None] * right[..., None, :]).flatten(-2))
    return torch.cat(parts, -1).clamp(-40, 40).float()
