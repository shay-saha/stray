"""Bounded memoization of the unchanged deterministic seasonal candidate bank.

One fitted baseline owns one last-batch entry, shared by residual/author route
selectors. Mutable NumPy inputs are content-fingerprinted on each lookup. Only
query predictors and training experiences enter that fingerprint; held-out
future values, coordinates and observation masks are never consulted.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np


_ATTRIBUTE = "_migration_last_candidate_bank"
_PREDICTORS = ("history", "history_mask", "origin", "species", "individual", "split")
_TRAIN_EXPERIENCES = ("y", "target_mask", "target_time")
_ARRAYS = (*_PREDICTORS, "static", "horizons", *_TRAIN_EXPERIENCES)


def _update(digest, value):
    value = np.asarray(value)
    digest.update(repr((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes(order="C"))


def _rows(digest, value, rows):
    # Avoid allocating a full copy of a large training-history tensor.
    for start in range(0, len(rows), 4096):
        _update(digest, value[rows[start:start + 4096]])


def _fingerprint(a, ids, candidates, seed, fitted):
    digest = hashlib.sha256()
    _update(digest, ids)
    digest.update(repr((int(candidates), int(seed))).encode())
    train = np.asarray(fitted.train)
    _update(digest, train)
    _update(digest, a["horizons"])
    _update(digest, fitted.horizons)
    digest.update(repr(sorted(fitted.diffusion.items())).encode())
    # End times are fitted causal donor metadata. They can be edited separately
    # from target_time, so include their actual values too.
    _update(digest, np.fromiter(fitted.end_by_id.keys(), dtype=np.int64))
    _update(digest, np.fromiter(fitted.end_by_id.values(), dtype=np.float64))
    for name in _PREDICTORS:
        digest.update(name.encode())
        # These candidate functions use east/north/actual-age only. Other
        # history fields still reach each neural model directly, outside this
        # bank. Hashing unused donor features would waste most lookup time.
        value = a[name][..., :3] if name == "history" else a[name]
        _rows(digest, value, train)
        _rows(digest, value, ids)
    _rows(digest, a["static"], ids)
    for name in _TRAIN_EXPERIENCES:
        digest.update(name.encode())
        _rows(digest, a[name], train)
    return digest.digest()


def _clone(result):
    # Return independent writable arrays, as the original generator does. A
    # caller modifying its output cannot poison another model's cached bank.
    paths, query, features, info = result
    return paths.copy(), query.copy(), features.copy(), copy.deepcopy(info)


def cached_candidates(a, ids, candidates, seed, baselines=None):
    """Reuse only an exactly matching, unmodified last development/query batch.

    Training-query generation retains its original identity-excluded diffusion
    path and bypasses memoization, avoiding any extra read of its own labels.
    A baseline fitted to another array dictionary also bypasses memoization.
    """
    from .migration_retrieval_train import generate_candidates

    ids = np.asarray(ids)
    if (baselines is None or getattr(baselines, "a", None) is not a
        or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
        or np.any(a["split"][ids] == 0)):
        if baselines is not None:
            setattr(baselines, _ATTRIBUTE, None)
        return generate_candidates(a, ids, candidates, seed, baselines)
    ids = ids.astype(np.int64, copy=False)
    arrays = tuple(a[name] for name in _ARRAYS)
    fingerprint = _fingerprint(a, ids, candidates, seed, baselines)
    previous = getattr(baselines, _ATTRIBUTE, None)
    if (previous is not None and previous["source"] is a and previous["generator"] is generate_candidates
        and all(left is right for left, right in zip(previous["arrays"], arrays))
        and previous["fingerprint"] == fingerprint):
        return _clone(previous["result"])
    result = generate_candidates(a, ids, candidates, seed, baselines)
    # Do not publish a cache entry if a caller changed dependencies while the
    # original generator was running. This adds no restrictions to that API.
    if (_fingerprint(a, ids, candidates, seed, baselines) == fingerprint
        and all(a[name] is value for name, value in zip(_ARRAYS, arrays))):
        setattr(baselines, _ATTRIBUTE, {"source": a, "arrays": arrays, "generator": generate_candidates,
                                       "fingerprint": fingerprint, "result": _clone(result)})
    else:
        setattr(baselines, _ATTRIBUTE, None)
    return result
