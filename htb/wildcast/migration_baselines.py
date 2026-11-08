"""Training-only migration baselines with explicit causal route retrieval.

All predictions are joint draws in 1000 km east/north at the query origin.
Missing donor targets are completed as *predictions*, never promoted to labels.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _velocity(a, ids):
    """Most recent observed past displacement divided by its actual elapsed time."""
    h, mask = a["history"][ids], a["history_mask"][ids]
    result = np.zeros((len(ids), 2), dtype=float)
    for row in range(len(ids)):
        valid = np.flatnonzero(mask[row] & (h[row, :, 2] > 0))
        if len(valid):
            j = valid[np.argmin(h[row, valid, 2])]
            # The selector already requires positive actual elapsed time. A
            # 14.4-minute floor would bias new 5-minute observed velocities.
            result[row] = -h[row, j, :2] / float(h[row, j, 2])
    return result


def _features(a, ids):
    origin = a["origin"][ids]
    lat, lon = np.deg2rad(origin[:, :2]).T
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    season = 2 * np.pi * (origin[:, 2] / 86400 / 365.2425)
    return np.column_stack([xyz / .2, np.sin(season) / .5, np.cos(season) / .5,
                            np.clip(_velocity(a, ids), -.5, .5) / .1])


class MigrationBaselines:
    """Fit exclusively to split zero, retaining donor identities and valid times."""

    def __init__(self, a, manifest=None):
        self.a = a
        self.horizons = np.asarray(a["horizons"], dtype=float)
        self.train = np.flatnonzero(a["split"] == 0)
        if not len(self.train):
            raise ValueError("Baselines require observed training cases")
        self.diffusion = {}
        # A species-conditioned displacement diffusion; no validation residuals.
        for species in [-1, *np.unique(a["species"][self.train]).tolist()]:
            ids = self.train if species == -1 else self.train[a["species"][self.train] == species]
            mask = a["target_mask"][ids]
            displacement = a["y"][ids]
            variance = (displacement ** 2).sum(-1) / self.horizons[None].clip(.01)
            values = variance[mask]
            self.diffusion[int(species)] = max(float(np.median(values)) / 2, 1e-6) if len(values) else 1e-4
        self.features = _features(a, self.train)
        self.trees = {-1: (cKDTree(self.features), self.train)}
        for species in np.unique(a["species"][self.train]):
            selected = np.flatnonzero(a["species"][self.train] == species)
            self.trees[int(species)] = (cKDTree(self.features[selected]), self.train[selected])
        valid_time = np.where(a["target_mask"][self.train], a["target_time"][self.train], -np.inf)
        self.donor_end = np.maximum(valid_time.max(-1), a["origin"][self.train, 2])
        self.end_by_id = dict(zip(self.train.tolist(), self.donor_end.tolist()))

    def movement(self, ids, samples, rng):
        """An integrated Ornstein-Uhlenbeck correlated-velocity movement law.

        The observed velocity decays over a fixed three-day timescale. Exact
        Gaussian interval transitions correlate displacement and velocity noise.
        Its diffusion is fitted on training cases only; the timescale is declared
        in advance rather than selected using evaluated targets.
        """
        velocity = np.repeat(_velocity(self.a, ids)[:, None], samples, axis=1)
        dt = np.diff(np.r_[0., self.horizons])
        diffusion = np.asarray([self.diffusion.get(int(s), self.diffusion[-1]) for s in self.a["species"][ids]])
        tau = 3.
        result = np.zeros((len(ids), samples, len(dt), 2), dtype=float)
        position = np.zeros_like(velocity)
        for h, interval in enumerate(dt):
            phi = np.exp(-interval / tau)
            var_v = diffusion / (2 * tau) * (1 - phi ** 2)
            var_x = diffusion * (interval - 2 * tau * (1 - phi) + tau / 2 * (1 - phi ** 2))
            covariance = diffusion / 2 * (1 - phi) ** 2
            z_v, z_x = rng.standard_normal((2, len(ids), samples, 2))
            sd_v = np.sqrt(np.maximum(var_v, 1e-20))
            correlated = covariance / sd_v
            noise_x = correlated[:, None, None] * z_v + np.sqrt(np.maximum(var_x - correlated ** 2, 0))[:, None, None] * z_x
            position = position + tau * (1 - phi) * velocity + noise_x
            velocity = phi * velocity + sd_v[:, None, None] * z_v
            result[:, :, h] = position
        return result.astype(np.float32)

    def eligible_donors(self, query, max_candidates=1024):
        """Return eligible nearest routes, with same-species preference.

        Unseen-individual assessment excludes the physical query identity.
        Familiar/year assessment allows it only when the WHOLE observed donor
        future ends before the query origin. All donors in year assessment must
        likewise precede the origin, preventing later-year population leakage.
        """
        species = int(self.a["species"][query])
        feat = _features(self.a, np.asarray([query]))[0]
        familiar = int(self.a["split"][query]) in (2, 4)
        identity = self.a["individual"][query]
        origin_time = self.a["origin"][query, 2]
        for group in [species, -1] if species in self.trees else [-1]:
            tree, ids = self.trees[group]
            costs, where = tree.query(feat, k=min(max_candidates, len(ids)))
            donors = ids[np.atleast_1d(where)]
            costs = np.atleast_1d(costs)
            same = self.a["individual"][donors] == identity
            ends = np.asarray([self.end_by_id[int(d)] for d in donors])
            valid = self.a["target_mask"][donors].any(-1)
            valid &= (ends < origin_time) if familiar else ~same
            if valid.any():
                return donors[valid], costs[valid]
        return np.empty(0, dtype=int), np.empty(0)

    def seasonal(self, ids, samples, rng):
        """Draw a single eligible donor per whole route, with stochastic bridges.

        Observed donor endpoints anchor a correlated movement bridge. A missing
        donor endpoint is predicted by that bridge; beyond its final observation
        the movement baseline continues. Neither zeros nor another donor's
        endpoints replace missing labels. Returned metadata makes this explicit.
        """
        result = self.movement(ids, samples, rng)
        completion = np.ones((len(ids), samples, len(self.horizons)), dtype=bool)
        donor_ids = np.full((len(ids), samples), -1, dtype=np.int64)
        grid = np.r_[0., self.horizons]
        for row, query in enumerate(ids):
            donors, costs = self.eligible_donors(int(query))
            if not len(donors):
                continue
            weights = np.exp(-np.minimum((costs - costs.min()) ** 2 / 2, 700))
            weights /= weights.sum()
            chosen = rng.choice(donors, samples, p=weights)
            donor_ids[row] = chosen
            for draw, donor in enumerate(chosen):
                valid = self.a["target_mask"][donor]
                times = np.r_[0., self.horizons[valid]]
                points = np.vstack([np.zeros(2), self.a["y"][donor, valid]])
                noise = np.vstack([np.zeros(2), result[row, draw].copy()])
                # Bridge a whole fallback path through this donor's actual
                # observed endpoint support. Interpolation is forecast synthesis.
                for axis in range(2):
                    baseline = np.interp(self.horizons, times, points[:, axis])
                    pinned_noise = np.interp(times, grid, noise[:, axis])
                    noise_trend = np.interp(self.horizons, times, pinned_noise)
                    result[row, draw, :, axis] = baseline + noise[1:, axis] - noise_trend
                completion[row, draw] = ~valid
        return result, {"donor_ids": donor_ids, "completed_mask": completion,
                        "fallback_routes": int((donor_ids < 0).sum())}

    def predict(self, name, ids, samples, rng):
        if samples < 2:
            raise ValueError("At least two samples required")
        if name == "stationary":
            return np.zeros((len(ids), samples, len(self.horizons), 2), dtype=np.float32), {}
        if name in ("movement", "crw"):
            return self.movement(ids, samples, rng), {}
        if name in ("seasonal", "seasonal_route"):
            return self.seasonal(ids, samples, rng)
        raise ValueError("Unknown migration baseline: " + name)
