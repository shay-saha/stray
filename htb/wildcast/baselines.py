"""Train-only empirical movement kernel baselines."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class CorrelatedRandomWalk:
    """Gamma step length / von Mises angle relative to last observed heading.

    This is an endpoint kernel fitted separately for species and horizon. It is
    an independent local baseline, not a reproduction of MoveBench's CRW code.
    """
    def fit(self, y, meta, velocity):
        self.params = {}
        keys = [(int(s), int(h)) for s, h in meta[:, [1, 4]]]
        for key in sorted(set(keys)) + [(-1, -1)]:
            ix = np.ones(len(y), bool) if key == (-1, -1) else (meta[:, 1] == key[0]) & (meta[:, 4] == key[1])
            delta, v = y[ix], velocity[ix]
            lengths = np.linalg.norm(delta, axis=-1)
            moving = lengths > .005
            if moving.sum() < 3:
                shape, scale = 1., .01
            else:
                # Method of moments is robust to optimizer failures at zero.
                a = lengths[moving]
                shape = float(np.clip(a.mean() ** 2 / max(a.var(), 1e-6), .1, 100))
                scale = float(a.mean() / shape)
            turns = np.arctan2(delta[:, 1], delta[:, 0]) - np.arctan2(v[:, 1], v[:, 0])
            turns = turns[moving & (np.linalg.norm(v, axis=-1) > .005)]
            if len(turns):
                z = np.exp(1j * turns).mean()
                r, angle = abs(z), np.angle(z)
                kappa = 2 * r + r ** 3 + 5 * r ** 5 / 6 if r < .53 else -.4 + 1.39 * r + .43 / (1 - r) if r < .85 else 1 / max(r ** 3 - 4 * r ** 2 + 3 * r, 1e-6)
            else:
                angle, kappa = 0., 0.
            self.params[key] = (shape, scale, float(angle), float(min(kappa, 50)), float((~moving).mean()))
        return self

    def sample(self, meta, velocity, count, rng):
        output = np.zeros((len(meta), count, 2), dtype=np.float32)
        for key in set((int(s), int(h)) for s, h in meta[:, [1, 4]]):
            ix = np.flatnonzero((meta[:, 1] == key[0]) & (meta[:, 4] == key[1]))
            shape, scale, angle, kappa, p_still = self.params.get(key, self.params[-1, -1])
            lengths = rng.gamma(shape, scale, (len(ix), count))
            lengths[rng.random(lengths.shape) < p_still] = 0
            theta = rng.vonmises(angle, kappa, lengths.shape)
            theta += np.arctan2(velocity[ix, 1], velocity[ix, 0])[:, None]
            zero = np.linalg.norm(velocity[ix], axis=-1) <= .005
            theta[zero] = rng.uniform(-np.pi, np.pi, (int(zero.sum()), count))
            output[ix] = lengths[..., None] * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        return output


def rotate(xy, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([xy[..., 0] * c + xy[..., 1] * s, -xy[..., 0] * s + xy[..., 1] * c], axis=-1)


def analogue_features(x, velocity):
    xy = np.sign(x[..., :2]) * np.expm1(np.minimum(np.abs(x[..., :2]), 15))
    theta = np.arctan2(velocity[:, 1], velocity[:, 0])
    canonical = rotate(xy, theta[:, None])
    offsets = sorted(set(max(0, x.shape[1] - k) for k in (2, 4, 8, 16, 32)))
    positions = canonical[:, offsets].reshape(len(x), -1)
    steps = np.diff(canonical, axis=1)
    summaries = np.concatenate([steps.mean(1), steps.std(1), np.abs(steps).mean(1),
        np.linalg.norm(velocity, axis=-1, keepdims=True), x[:, -1, 4:5], x[:, -1, 6:10]], axis=-1)
    return np.concatenate([np.arcsinh(positions), np.arcsinh(summaries[:, :7]), summaries[:, 7:]], axis=-1), theta


class AnalogueForecaster:
    """Distributional nearest-history retrieval with rotation-equivariant targets.

    No held-out fixes enter the search index. Species/horizon-specific indices
    back off to a cross-species index at the same forecast horizon.
    """
    def __init__(self, neighbors=64):
        self.neighbors = neighbors

    def fit(self, x, y, meta, velocity):
        features, theta = analogue_features(x, velocity)
        targets = rotate(y, theta)
        self.indices = {}
        keys = set((int(s), int(h)) for s, h in meta[:, [1, 4]])
        keys.update((-1, int(h)) for h in np.unique(meta[:, 4]))
        for sp, h in keys:
            ix = (meta[:, 4] == h) & ((meta[:, 1] == sp) if sp != -1 else True)
            f = features[ix]
            loc = np.median(f, axis=0)
            scale = np.maximum(np.quantile(f, .75, axis=0) - np.quantile(f, .25, axis=0), .1)
            self.indices[sp, h] = (cKDTree((f - loc) / scale), targets[ix], loc, scale)
        return self

    def sample(self, x, meta, velocity, count, rng):
        f, theta = analogue_features(x, velocity)
        out = np.zeros((len(x), count, 2), dtype=np.float32)
        for sp, h in set((int(s), int(h)) for s, h in meta[:, [1, 4]]):
            ix = np.flatnonzero((meta[:, 1] == sp) & (meta[:, 4] == h))
            tree, targets, loc, scale = self.indices.get((sp, h), self.indices[-1, h])
            k = min(self.neighbors, len(targets))
            _, neighbours = tree.query((f[ix] - loc) / scale, k=k, workers=2)
            neighbours = np.asarray(neighbours).reshape(len(ix), k)
            choices = rng.integers(0, k, (len(ix), count))
            chosen = neighbours[np.arange(len(ix))[:, None], choices]
            values = targets[chosen]
            out[ix] = rotate(values, -theta[ix, None])
        return out
