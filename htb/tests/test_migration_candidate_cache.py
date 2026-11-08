"""Shared memoization preserves whole-route laws and causal input boundaries."""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from wildcast.migration_baselines import MigrationBaselines
from wildcast.migration_candidate_cache import cached_candidates
from wildcast import migration_retrieval_train as retrieval


@pytest.fixture
def corpus():
    n = 8
    history = np.zeros((n, 4, 10), np.float32)
    history[..., 2] = [0., 1., 7., 30.]
    history[..., 0] = [0., -.01, -.07, -.3]
    horizons = np.asarray([1., 7.], np.float32)
    origin = np.column_stack((np.zeros(n), np.zeros(n), np.r_[np.arange(4) + 1000, np.arange(4) + 3000] * 86400))
    target_mask = np.ones((n, 2), bool); target_mask[1, 1] = False; target_mask[6:] = False
    y = np.zeros((n, 2, 2), np.float32)
    y[..., 0] = (np.arange(n)[:, None] + 1) * horizons[None] * .005
    return {"history": history, "history_mask": np.ones((n, 4), bool),
            "static": np.zeros((n, 14), np.float32), "horizons": horizons,
            "origin": origin, "species": np.ones(n, np.int64), "individual": np.arange(n),
            "split": np.asarray([0, 0, 0, 0, 1, 2, 3, 4], np.int8),
            "y": y, "target_mask": target_mask,
            "target_time": origin[:, 2:3] + horizons * 86400}


def equal(left, right):
    for lvalue, rvalue in zip(left[:3], right[:3]):
        np.testing.assert_array_equal(lvalue, rvalue)
    assert left[3].keys() == right[3].keys()
    for key in left[3]:
        np.testing.assert_array_equal(left[3][key], right[3][key])


def counted(monkeypatch):
    original = retrieval.generate_candidates
    calls = []
    def wrapper(*args, **kwargs):
        calls.append(tuple(np.asarray(args[1])))
        return original(*args, **kwargs)
    monkeypatch.setattr(retrieval, "generate_candidates", wrapper)
    return calls, original


def test_exact_last_batch_only_across_ids_seeds_and_candidate_counts(corpus, monkeypatch):
    a = corpus; fitted = MigrationBaselines(a)
    calls, original = counted(monkeypatch)
    for ids, count, seed in [([4, 5], 8, 17), ([5, 4], 8, 17), ([4], 8, 17),
                             ([4], 8, 42), ([4], 12, 42), ([4, 5], 8, 17)]:
        ids = np.asarray(ids)
        expected = original(a, ids, count, seed, fitted)
        equal(cached_candidates(a, ids, count, seed, fitted), expected)
        equal(cached_candidates(a, ids.copy(), count, seed, fitted), expected)
    assert len(calls) == 6  # The earlier [4,5] entry was evicted, not retained.


@pytest.mark.parametrize("mutation", ["query-history", "train-history", "query-static", "train-route",
                                     "train-mask", "train-time", "species", "identity", "origin", "array-replacement", "diffusion"])
def test_inplace_changes_and_equal_valued_array_replacement_invalidate(corpus, monkeypatch, mutation):
    a = corpus; fitted = MigrationBaselines(a); ids = np.asarray([4, 5])
    calls, original = counted(monkeypatch)
    cached_candidates(a, ids, 8, 17, fitted)
    if mutation == "query-history": a["history"][4, 1, 0] *= 2
    elif mutation == "train-history": a["history"][0, 1, 0] *= 2
    elif mutation == "query-static": a["static"][4, 0] = 1
    elif mutation == "train-route": a["y"][0, 0, 0] += .1
    elif mutation == "train-mask": a["target_mask"][1, 1] = True
    elif mutation == "train-time": a["target_time"][0, 1] += 1
    elif mutation == "species": a["species"][0] = 0
    elif mutation == "identity": a["individual"][0] = 4
    elif mutation == "origin": a["origin"][4, 0] += 2
    elif mutation == "array-replacement": a["history"] = a["history"].copy()
    else: fitted.diffusion[1] *= 2
    equal(cached_candidates(a, ids, 8, 17, fitted), original(a, ids, 8, 17, fitted))
    assert len(calls) == 2


class TrainingOnly(np.ndarray):
    """Fail if either the generator or its cache looks at held-out truth."""
    def __getitem__(self, key):
        row = key[0] if isinstance(key, tuple) else key
        rows = np.arange(len(self))[row]
        assert np.all(rows < 4), "Held-out future accessed"
        return super().__getitem__(key)


class Forbidden:
    def __array__(self, *args, **kwargs):
        raise AssertionError("Future-coordinate array accessed")
    def __getitem__(self, key):
        raise AssertionError("Future-coordinate array accessed")


def test_cache_never_reads_query_future_and_ignores_heldout_outcome_mutation(corpus, monkeypatch):
    a = corpus; fitted = MigrationBaselines(a)
    for key in ("y", "target_mask", "target_time"):
        a[key] = a[key].view(TrainingOnly)
    a["target_latlon"] = Forbidden()
    calls, original = counted(monkeypatch)
    ids = np.asarray([4, 5, 6, 7])
    first = cached_candidates(a, ids, 8, 17, fitted)
    for key in ("y", "target_mask", "target_time"):
        a[key].view(np.ndarray)[4:] = 0
    second = cached_candidates(a, ids, 8, 17, fitted)
    equal(first, second)
    assert len(calls) == 1


def test_outputs_cannot_poison_private_cache_and_sources_never_cross(corpus, monkeypatch):
    a = corpus; fitted = MigrationBaselines(a); ids = np.asarray([4, 5])
    calls, original = counted(monkeypatch)
    expected = original(a, ids, 8, 17, fitted)
    result = cached_candidates(a, ids, 8, 17, fitted)
    for value in (*result[:3], *result[3].values()): value.flat[0] = 99
    equal(cached_candidates(a, ids, 8, 17, fitted), expected)
    other = {key: value.copy() for key, value in a.items()}
    equal(cached_candidates(other, ids, 8, 17, fitted), original(other, ids, 8, 17, fitted))
    assert len(calls) == 2
    assert fitted._migration_last_candidate_bank is None


def test_irrelevant_donor_history_channels_do_not_invalidate_exact_bank(corpus, monkeypatch):
    a = corpus; fitted = MigrationBaselines(a); ids = np.asarray([4, 5])
    calls, original = counted(monkeypatch)
    first = cached_candidates(a, ids, 8, 17, fitted)
    a["history"][:4, :, 3:] = 12345
    equal(cached_candidates(a, ids, 8, 17, fitted), original(a, ids, 8, 17, fitted))
    equal(cached_candidates(a, ids, 8, 17, fitted), first)
    assert len(calls) == 1


def test_training_queries_bypass_memoization(corpus, monkeypatch):
    a = corpus; fitted = MigrationBaselines(a); ids = np.asarray([0, 4])
    calls, original = counted(monkeypatch)
    expected = original(a, ids, 8, 17, fitted)
    equal(cached_candidates(a, ids, 8, 17, fitted), expected)
    equal(cached_candidates(a, ids, 8, 17, fitted), expected)
    assert len(calls) == 2 and fitted._migration_last_candidate_bank is None


def test_mutation_during_generation_is_not_published_as_a_cache_hit(corpus, monkeypatch):
    a = corpus; fitted = MigrationBaselines(a); ids = np.asarray([4, 5])
    original = retrieval.generate_candidates
    calls = []
    def changing(*args, **kwargs):
        calls.append(True)
        a["static"][4, 0] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(retrieval, "generate_candidates", changing)
    cached_candidates(a, ids, 8, 17, fitted)
    assert fitted._migration_last_candidate_bank is None
    cached_candidates(a, ids, 8, 17, fitted)
    assert len(calls) == 2 and fitted._migration_last_candidate_bank is None


def test_predictors_share_cache_and_preserve_exact_paths(corpus, monkeypatch):
    pytest.importorskip("entmax")
    from wildcast.migration_moveformer import MigrationMoveFormer
    a = corpus; ids = np.asarray([4, 5]); fitted = MigrationBaselines(a)
    _, query, features, _ = retrieval.generate_candidates(a, ids, 8, 1701, fitted)
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        torch.manual_seed(4)
        residual = retrieval.RetrievalReweighter(query.shape[-1], features.shape[-1], 2, candidates=8, hidden=8, dropout=0.)
        author = MigrationMoveFormer(query.shape[-1], features.shape[-1], 2, history_length=4, candidates=8, width=8, layers=1, heads=2)
        expected = [model.predict_cases(a, ids, 12, np.random.default_rng(2026)) for model in (residual, author)]
        calls, _ = counted(monkeypatch)
        for model, truth in zip((residual, author, residual, author), expected * 2):
            result = model.predict_cases(a, ids, 12, np.random.default_rng(2026), baselines=fitted)
            np.testing.assert_array_equal(result, truth)
        assert len(calls) == 1
    finally:
        torch.set_num_threads(previous)


def test_frozen_bank_algorithm_and_baseline_source_remain_identical():
    source = "\n".join(inspect.getsource(f) for f in (retrieval.query_features, retrieval.candidate_features, retrieval.generate_candidates))
    assert hashlib.sha256(source.encode()).hexdigest() == "e185f32a55a151170c4f40e6568f0e1ebcb03b437dee3749939446d5fb476818"
    baseline = Path(retrieval.__file__).with_name("migration_baselines.py")
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == "20397428f624d037855ac07db6430eedb647745143561dba196dfc31c5fe9077"
