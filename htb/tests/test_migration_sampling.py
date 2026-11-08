import hashlib
import json

import numpy as np
import pytest
import torch

from wildcast.migration_evaluate import forecast_latlon
from wildcast.migration_model import load_model
from wildcast.migration_train import (load_training_data, sampling_probabilities, train)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def imbalance():
    animals = np.repeat(np.arange(5), [10, 2, 1, 3, 1])
    metric = np.where(animals <= 2, 0, animals - 2)
    a = {"individual": np.r_[animals, [0, 0, 0, 0]],
         "metric_species": np.r_[metric, [999, 999, 999, 999]],
         "species": np.zeros(len(animals) + 4, np.int64),  # Unknown predictor tokens must not collapse reporting taxa.
         "split": np.r_[np.zeros(len(animals), np.int8), [1, 2, 3, 4]]}
    return a, np.arange(len(animals))


def test_animal_default_is_bitwise_previous_sampler_and_collars_do_not_dominate():
    a, ids = imbalance()
    _, inverse, counts = np.unique(a["individual"][ids], return_inverse=True, return_counts=True)
    old = 1. / counts[inverse]; old /= old.sum()
    current = sampling_probabilities(a, ids)
    np.testing.assert_array_equal(current, old)
    for animal in range(5):
        assert current[a["individual"][ids] == animal].sum() == pytest.approx(.2)


@pytest.mark.parametrize("scheme", ["species_animal", "species_sqrt"])
def test_species_sampling_mass_and_long_collar_balance(scheme):
    a, ids = imbalance(); probability = sampling_probabilities(a, ids, scheme)
    expected = np.ones(3) if scheme == "species_animal" else np.sqrt([3., 1., 1.])
    expected /= expected.sum()
    for species in range(3):
        assert probability[a["metric_species"][ids] == species].sum() == pytest.approx(expected[species])
    # All three common-species animals have equal mass despite 10/2/1 origins.
    masses = [probability[a["individual"][ids] == animal].sum() for animal in range(3)]
    np.testing.assert_allclose(masses, expected[0] / 3)
    assert probability[-1] > .2  # Single-animal rare taxon gains mass over the animal-only sampler.


def test_sampling_is_seeded_and_rejects_cross_species_or_nontraining_groups():
    a, ids = imbalance(); p = sampling_probabilities(a, ids, "species_animal")
    first = np.random.default_rng(7).choice(ids, 100, replace=True, p=p)
    second = np.random.default_rng(7).choice(ids, 100, replace=True, p=p)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="TRAINING"):
        sampling_probabilities(a, np.r_[ids, len(ids)], "animal")
    a["metric_species"][1] = 7
    with pytest.raises(ValueError, match="conflicting"):
        sampling_probabilities(a, ids, "species_animal")


@pytest.mark.parametrize("scheme", ["animal", "species_animal", "species_sqrt"])
def test_declared_uncertainty_block_preserves_identity_and_balances_taxon_strata(scheme):
    a = {"individual":np.array([0,0,0,1,2,2,3]), "metric_species":np.array([0,0,1,0,1,1,1]),
         "species":np.zeros(7,dtype=int), "split":np.zeros(7,dtype="int8")}
    ids = np.arange(7)
    before = {key:value.copy() for key,value in a.items()}
    with pytest.raises(ValueError,match="conflicting"):
        sampling_probabilities(a,ids,scheme)
    p = sampling_probabilities(a,ids,scheme,allowed_cross_taxon_exclusion_ids={0})
    if scheme == "animal":
        for identity in range(4):
            assert p[a["individual"] == identity].sum() == pytest.approx(.25)
    else:
        masses = np.ones(2) if scheme == "species_animal" else np.sqrt([2,3])
        masses /= masses.sum()
        for taxon,count in enumerate([2,3]):
            assert p[a["metric_species"] == taxon].sum() == pytest.approx(masses[taxon])
            for identity in np.unique(a["individual"][a["metric_species"] == taxon]):
                chosen=(a["individual"] == identity)&(a["metric_species"] == taxon)
                assert p[chosen].sum() == pytest.approx(masses[taxon]/count)
    for key in a:np.testing.assert_array_equal(a[key],before[key])


@pytest.mark.parametrize("scheme", ["species_animal", "species_sqrt"])
def test_disjoint_taxon_sampling_remains_bitwise_previous_law(scheme):
    a,ids=imbalance()
    animals,inverse,counts=np.unique(a["individual"][ids],return_inverse=True,return_counts=True)
    labels=np.array([a["metric_species"][ids][inverse == row][0] for row in range(len(animals))])
    _,si,n=np.unique(labels,return_inverse=True,return_counts=True)
    old=1./counts[inverse]
    old/=n[si[inverse]] if scheme == "species_animal" else np.sqrt(n[si[inverse]])
    old/=old.sum()
    np.testing.assert_array_equal(sampling_probabilities(a,ids,scheme),old)


def fixture_data(root):
    root.mkdir()
    rng = np.random.default_rng(12)
    animals = np.array([0] * 5 + [1] * 3 + [2] * 2 + [3, 4] + [10, 11, 0, 3, 20, 0])
    n = len(animals); h = np.array([1., 3., 7.], np.float32)
    actual = np.array([0] * 10 + [1, 2] + [0, 1, 0, 1, 2, 0])
    a = {"history": rng.normal(size=(n, 4, 10)).astype(np.float32), "history_mask": np.ones((n, 4), bool),
         "static": rng.normal(size=(n, 3)).astype(np.float32), "species": actual.astype(np.int64) + 1,
         "metric_species": actual.astype(np.int64), "y": rng.normal(size=(n, 3, 2)).astype(np.float32) * .03,
         "target_mask": np.ones((n, 3), bool), "split": np.array([0] * 12 + [1, 1, 2, 2, 3, 4], np.int8),
         "individual": animals.astype(np.int64), "study": np.zeros(n, np.int64), "horizons": h,
         "origin": np.column_stack((np.zeros(n), np.zeros(n), np.full(n, 1_600_000_000.)))}
    a["target_time"] = a["origin"][:, 2:3] + h * 86400
    a["target_latlon"] = forecast_latlon(a["y"][:, None], a["origin"])[:, 0]
    a["target_mask"][-2:] = False; a["y"][-2:] = np.nan; a["target_latlon"][-2:] = np.nan
    np.savez_compressed(root / "cases.npz", **a)
    manifest = {"cases_sha256": hashlib.sha256((root / "cases.npz").read_bytes()).hexdigest(),
                "species": {"Taxon A": 1, "Taxon B": 2, "Taxon C": 3}, "reserved_test_targets_sealed": True}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return a


def fit_options():
    return dict(batch_size=4, width=16, layers=1, components=2, device="cpu", dropout=0.,
                validation_samples=4, energy_repeats=1)


def test_continuation_epoch_zero_preserves_weights_normalizers_and_forecasts(tmp_path):
    a = fixture_data(tmp_path / "data")
    train(tmp_path / "data", tmp_path / "base", epochs=1, **fit_options())
    source = tmp_path / "base/last.pt"
    before, source_ck = load_model(source)
    assert source_ck["optimizer"]["state"]  # Source really has optimizer history.
    train(tmp_path / "data", tmp_path / "continue", epochs=0, sampling="species_animal",
          init_checkpoint=source, **fit_options())
    after, ck = load_model(tmp_path / "continue/initial.pt")
    for name in before.state_dict():
        torch.testing.assert_close(before.state_dict()[name], after.state_dict()[name], atol=0, rtol=0)
    inputs = [torch.as_tensor(a[key][[12, 14]]) for key in ("history", "history_mask", "static", "species")]
    left = before.sample_paths(*inputs, 16, torch.Generator().manual_seed(9))
    right = after.sample_paths(*inputs, 16, torch.Generator().manual_seed(9))
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert not ck["optimizer"]["state"]
    assert ck["training"]["initialization"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert ck["training"]["sampling"] == "species_animal"


def test_initialization_rejects_cache_and_model_schema_mismatch(tmp_path):
    fixture_data(tmp_path / "data")
    train(tmp_path / "data", tmp_path / "base", epochs=0, **fit_options())
    source = tmp_path / "base/best.pt"
    ck = torch.load(source, weights_only=True); ck["cases_sha256"] = "other-cache"
    torch.save(ck, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="cases_sha256"):
        train(tmp_path / "data", tmp_path / "wrong-cache", epochs=0,
              init_checkpoint=tmp_path / "wrong.pt", **fit_options())
    options = fit_options(); options["width"] = 32
    with pytest.raises(ValueError, match="config/schema"):
        train(tmp_path / "data", tmp_path / "wrong-shape", epochs=0, init_checkpoint=source, **options)


def write_filter(root, data, train_mask, val_mask, provenance=None):
    root.mkdir()
    np.savez_compressed(root / "filters.npz", train_filter=train_mask, val_filter=val_mask)
    _, _, hashes = load_training_data(data)
    metadata = {**hashes, "filters_sha256": hashlib.sha256((root / "filters.npz").read_bytes()).hexdigest(),
                "provenance": provenance or {"causal": True, "threshold_fit_split": "train", "uses_future_targets": False,
                                              "rule": "Declared synthetic past-history activity criterion"}}
    (root / "filter.json").write_text(json.dumps(metadata))


def test_causal_filter_limits_fitting_and_selection_before_epoch_zero(tmp_path):
    a = fixture_data(tmp_path / "data")
    train_mask = np.zeros(len(a["split"]), bool); train_mask[:2] = True
    val_mask = np.zeros_like(train_mask); val_mask[[12, 14]] = True
    write_filter(tmp_path / "filter", tmp_path / "data", train_mask, val_mask)
    train(tmp_path / "data", tmp_path / "filtered", epochs=0, case_filter=tmp_path / "filter", **fit_options())
    m, ck = load_model(tmp_path / "filtered/best.pt")
    expected = a["history"][:2].reshape(-1, 10).mean(0)
    np.testing.assert_allclose(m.history_mean.numpy(), expected, atol=1e-7)
    assert ck["training"]["train_cases"] == 2
    assert ck["training"]["selection_population"] == "Declared causal regime subset only"
    assert ck["validation_case_ids_sha256"] == hashlib.sha256(np.array([12, 14], dtype="<i8").tobytes()).hexdigest()
    assert ck["case_filter"]["selected_validation_origins"] == 2
    assert sum(s["cases"] for s in ck["metrics"]["splits"].values()) == 2


def test_filter_rejects_reserved_masks_and_future_selected_rules(tmp_path):
    a = fixture_data(tmp_path / "data")
    train_mask = a["split"] == 0; val_mask = np.isin(a["split"], [1, 2]); val_mask[-1] = True
    write_filter(tmp_path / "bad-mask", tmp_path / "data", train_mask, val_mask)
    with pytest.raises(ValueError, match="reserved test"):
        train(tmp_path / "data", tmp_path / "bad-fit", epochs=0, case_filter=tmp_path / "bad-mask", **fit_options())
    val_mask[-1] = False
    write_filter(tmp_path / "future", tmp_path / "data", train_mask, val_mask,
                  {"causal": True, "threshold_fit_split": "train", "uses_future_targets": True, "rule": "Future displacement"})
    with pytest.raises(ValueError, match="causal provenance"):
        train(tmp_path / "data", tmp_path / "future-fit", epochs=0, case_filter=tmp_path / "future", **fit_options())
