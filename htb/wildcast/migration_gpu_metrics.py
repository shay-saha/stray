"""Float64 distribution scores for already projected geographic forecasts.

Projection, point forecasts, coverage and route selection remain in the shared
evaluator. Only sample-pair distance calculations move to this backend.
"""
from __future__ import annotations

import math

import numpy as np
import torch

EARTH_KM = 6371.0088


def distribution_scores(coords, truth, mask, origin, device="cuda"):
    """Return endpoint ES, joint ES and truth-free sample-medoid costs."""
    with torch.inference_mode():
        radians = math.pi / 180
        c = torch.as_tensor(coords, dtype=torch.float64, device=device) * radians
        y = torch.as_tensor(truth, dtype=torch.float64, device=device) * radians
        observed = torch.as_tensor(mask, dtype=torch.bool, device=device)
        n = c.shape[1]

        def distance(left, right):
            dlat = left[..., 0] - right[..., 0]
            dlon = torch.remainder(left[..., 1] - right[..., 1] + math.pi, 2 * math.pi) - math.pi
            hav = ((dlat / 2).sin().square() + left[..., 0].cos() * right[..., 0].cos()
                   * (dlon / 2).sin().square()).clamp(0, 1)
            return 2 * EARTH_KM * torch.atan2(hav.sqrt(), (1 - hav).sqrt())

        pair = distance(c[:, :, None], c[:, None, :])
        endpoint = distance(c, y[:, None]).mean(1) - pair.sum((1, 2)) / (2 * n * (n - 1))
        endpoint = torch.where(observed, endpoint, torch.full_like(endpoint, torch.nan))
        medoid_cost = pair.mean(2).sum(-1)
        del pair

        def xyz(value):
            lat, lon = value.unbind(-1)
            return EARTH_KM * torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], -1)

        # Translation preserves the spherical Cartesian score and avoids
        # cancellation for metre-scale forecasts near Earth-sized coordinates.
        anchor = xyz(torch.as_tensor(np.asarray(origin)[:, :2], dtype=torch.float64, device=device) * radians)
        weight = (observed.to(torch.float64) / observed.sum(-1, keepdim=True)).sqrt()[..., None]
        paths = ((xyz(c) - anchor[:, None, None]) * weight[:, None]).flatten(-2)
        target = ((xyz(y) - anchor[:, None]) * weight).flatten(-2)
        pair = torch.cdist(paths, paths)
        pair.diagonal(dim1=-2, dim2=-1).zero_()
        joint = (paths - target[:, None]).norm(dim=-1).mean(1) - pair.sum((1, 2)) / (2 * n * (n - 1))
        return tuple(value.cpu().numpy() for value in (endpoint, joint, medoid_cost))
