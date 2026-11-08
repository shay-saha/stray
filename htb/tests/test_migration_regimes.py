"""Past-only filters, immutable regime packages and correct joint sampling laws."""
import hashlib
import json

import numpy as np
import pytest
import torch

from wildcast import migration_regimes as regimes
from wildcast.migration_model import MigrationModel, sample_paths


@pytest.fixture(autouse=True)
def small_cpu():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def inputs(tmp_path):
    count = 16
    history = np.zeros((count, 2, 3), dtype="float32")
    distances = np.array([*range(1, 11), 11, 1, 99, 99, 1000, 1000], dtype="float32")
    history[..., 0] = distances[:, None] / 1000
    history[..., 2] = [7, 30]
    mask = np.ones((count, 2), bool); mask[12, 0] = False
    split = np.array([0] * 10 + [1, 1, 2, 2, 3, 4], dtype="int8")
    target_mask = np.ones((count, 2), bool); target_mask[split >= 3] = False
    a = {"history": history, "history_mask": mask, "species": np.array([1] * 5 + [2] * 5 + [1, 1, 2, 0, 1, 2]),
         "metric_species": np.array([10] * 5 + [20] * 5 + [10, 10, 20, 30, 10, 20]),
         "individual": np.arange(count), "split": split, "static": np.zeros((count, 2), "float32"),
         "horizons": np.array([1, 7], "float32"), "study": np.zeros(count, "int64"),
         "origin": np.column_stack([np.full(count, 10.), np.full(count, 20.), np.full(count, 1.6e9)]),
         "target_mask": target_mask, "y": np.zeros((count, 2, 2), "float32"),
         "target_latlon": np.zeros((count, 2, 2)), "target_time": np.zeros((count, 2))}
    a["target_time"][:] = a["origin"][:, 2:3] + a["horizons"] * 86400
    a["target_time"][~target_mask] = 0
    data = tmp_path / "cache"; data.mkdir()
    np.savez_compressed(data / "cases.npz", **a)
    manifest = {"cases_sha256": regimes.sha256(data / "cases.npz"),
                "history_names": ["east_1000km", "north_1000km", "actual_age_days"],
                "history_lags_days": [7, 30], "species": {"one": 1, "two": 2},
                "metric_species": {"one": 10, "two": 20, "unknown": 30}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    hashes = {"cases_sha256": regimes.sha256(data / "cases.npz"), "manifest_sha256": regimes.sha256(data / "manifest.json")}
    return a, manifest, hashes, data


def test_rules_ignore_future_values_and_use_unknown_global_fallback(inputs):
    a, manifest, hashes, _ = inputs
    rules = regimes.fit_rules(a, manifest, hashes)
    changed = {k: v.copy() for k, v in a.items()}
    for key in ("y", "target_latlon", "target_time"):
        changed[key][:] = np.nan
    changed["target_mask"][:] = False
    changed["history"][changed["split"] != 0, :, 0] *= 100
    second = regimes.fit_rules(changed, manifest, hashes)
    assert rules["thresholds_km"] == second["thresholds_km"]
    assert rules["global_thresholds_km"] == second["global_thresholds_km"]
    only_predictors = {k: a[k] for k in ("history", "history_mask", "species")}
    result = regimes.classify(only_predictors, np.arange(10, 14), rules)
    assert result["high"].tolist() == [True, False, False, True]
    assert result["missing"].tolist() == [False, False, True, False]
    assert result["global_fallback"].tolist() == [False, False, False, True]


def test_threshold_scale_and_equal_animal_weighting(inputs):
    a, manifest, hashes, _ = inputs
    first = regimes.fit_rules(a, manifest, hashes)
    scaled = {k: v.copy() for k, v in a.items()}
    scaled["history"][..., :2] *= 3
    second = regimes.fit_rules(scaled, manifest, hashes)
    for key in first["thresholds_km"]:
        np.testing.assert_allclose(np.array(first["thresholds_km"][key]) * 3, second["thresholds_km"][key], rtol=1e-6)
    repeated = {k: np.concatenate([v, v[:1].repeat(7, axis=0)]) if k != "horizons" else v for k, v in a.items()}
    repeated_rules = regimes.fit_rules(repeated, manifest, hashes)
    assert repeated_rules["thresholds_km"] == first["thresholds_km"]
    assert repeated_rules["global_thresholds_km"] == first["global_thresholds_km"]


def test_prepare_exact_filter_interface_excludes_sealed_cases(inputs, tmp_path):
    a, _, hashes, data = inputs
    output = tmp_path / "filters"
    metadata = regimes.prepare(data, output)
    assert all(metadata[key] == value for key, value in hashes.items())
    assert metadata["provenance"]["causal"] is True
    assert metadata["provenance"]["uses_future_targets"] is False
    with np.load(output / "filters.npz", allow_pickle=False) as masks:
        for key in ("train_filter", "val_filter"):
            assert masks[key].dtype == bool and masks[key].shape == a["split"].shape
            assert not masks[key][a["split"] >= 3].any()
        assert masks["train_filter"].sum() == 2
        assert masks["val_filter"].sum() == 2
    with pytest.raises(FileExistsError, match="immutable"):
        regimes.prepare(data, output)


class FixedJoint(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.register_buffer("offset", torch.tensor(float(offset)))

    def forward(self, history, mask, static, species):
        batch = len(history)
        means = torch.tensor([0., 10.], device=history.device)[None, :, None, None].expand(batch, 2, 2, 2) + self.offset
        covariance = torch.eye(4, device=history.device)[None, None].expand(batch, 2, 4, 4) * .0001
        return {"means": means, "logits": torch.zeros(batch, 2, device=history.device), "covariance": covariance}


def test_initial_identity_and_whole_path_routing(inputs):
    a, manifest, hashes, _ = inputs
    rules = regimes.fit_rules(a, manifest, hashes)
    base = FixedJoint(0.)
    ids = np.arange(10, 14)
    control = regimes.MigrationRegime(base, rules, active_fallback="base")
    initial = regimes.MigrationRegime(base, rules, active=FixedJoint(0.))
    first = control.predict_cases(a, ids, 100, np.random.default_rng(4))
    np.testing.assert_array_equal(first, initial.predict_cases(a, ids, 100, np.random.default_rng(4)))
    specialist = regimes.MigrationRegime(base, rules, active=FixedJoint(100.))
    result = specialist.predict_cases(a, ids, 100, np.random.default_rng(4))
    np.testing.assert_array_equal(result[[1, 2]], first[[1, 2]])
    assert np.all(result[[0, 3]] > 99)
    assert np.all((result[[0, 3], :, 0, 0] > 105) == (result[[0, 3], :, 1, 0] > 105))
    class Seasonal:
        def predict(self, name, cases, samples, rng):
            assert name == "seasonal" and list(cases) == [10, 13]
            return np.full((len(cases), samples, 2, 2), 200.), {}
    fallback = regimes.MigrationRegime(base, rules)
    result = fallback.predict_cases(a, ids, 10, np.random.default_rng(1), Seasonal())
    assert np.all(result[[0, 3]] == 200.)


def test_conditional_covariance_law_preserves_component_means_and_logits():
    means = torch.tensor([[[[0., 0.], [0., 0.]], [[100., 100.], [100., 100.]]]])
    covariance = torch.eye(4)[None, None].expand(1, 2, 4, 4).clone()
    covariance[..., 0, 2] = covariance[..., 2, 0] = .6
    parameters = {"means": means, "logits": torch.zeros(1, 2), "covariance": covariance}
    scaled = regimes.scale_joint_parameters(parameters, 3.)
    assert scaled["means"] is means and scaled["logits"] is parameters["logits"]
    torch.testing.assert_close(scaled["covariance"], covariance * 9)
    draws = sample_paths(scaled, 30000, torch.Generator().manual_seed(2))[0]
    assert torch.all((draws[:, 0, 0] > 50) == (draws[:, 1, 0] > 50))
    conditional = draws[draws[:, 0, 0] < 50].flatten(-2)
    torch.testing.assert_close(torch.cov(conditional.T), covariance[0, 0] * 9, atol=.3, rtol=.07)


def test_member_class_rules_and_cache_hash_guards(inputs, tmp_path):
    _, _, hashes, data = inputs
    filters = tmp_path / "filters"; regimes.prepare(data, filters)
    model = MigrationModel(3, 2, 3, [1., 7.], history_length=2, width=16, layers=1, components=2, dropout=0.)
    member = tmp_path / "member.pt"
    torch.save({"model_kind": "wildcast_migration_joint", "config": model.config, "model": model.state_dict(), **hashes}, member)
    output = tmp_path / "router"
    regimes.assemble(data, filters, member, output, active_fallback="base")
    loaded, ck = regimes.load_model(output / "best.pt")
    assert isinstance(loaded, regimes.MigrationRegime) and ck["model_kind"] == regimes.KIND
    with member.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="member checksum"):
        regimes.load_model(output / "best.pt")
    wrong = tmp_path / "wrong.pt"
    torch.save({"model_kind": "wildcast_migration_ensemble", **hashes}, wrong)
    with pytest.raises(ValueError, match="must be joint"):
        regimes.assemble(data, filters, wrong, tmp_path / "invalid-router")


def test_coupled_scale_bank_changes_conditional_noise_only():
    means = torch.tensor([[[[0., 0.], [0., 0.]], [[100., 100.], [100., 100.]]]])
    parameters = {"means": means, "logits": torch.zeros(1, 2),
                  "covariance": torch.eye(4)[None, None].expand(1, 2, 4, 4)}
    bank = regimes.scaled_joint_bank(parameters, 20000, generator=torch.Generator().manual_seed(7))
    reference = sample_paths(parameters, 20000, torch.Generator().manual_seed(7))
    torch.testing.assert_close(bank[0], reference)
    component_mean = torch.where(bank[0][..., :1, :1] > 50, 100., 0.).expand_as(bank[0])
    for scale, draws in zip(regimes.SCALE_GRID, bank):
        torch.testing.assert_close(draws - component_mean, (bank[0] - component_mean) * scale, atol=4e-5, rtol=1e-5)
        assert torch.all((draws[..., 0, 0] > 50) == (draws[..., 1, 0] > 50))
        conditional = draws[0, draws[0, :, 0, 0] < 50].flatten(-2)
        torch.testing.assert_close(torch.cov(conditional.T), torch.eye(4) * scale ** 2, atol=.15 * scale ** 2, rtol=.06)


def _three_group_validation(inputs):
    a, manifest, _, data = inputs
    # Synthetic development origins; every query is HIGH under frozen TRAIN rules.
    a["history"][10:14, :, 0] = .1
    a["history_mask"][10:14] = True
    a["individual"][10:14] = [100, 100, 101, 102]
    manifest["animals"] = {}
    for identity, fold in zip((100, 101, 102), range(3)):
        for candidate in range(100):
            label = f"synthetic-{identity}-{candidate}"
            digest = hashlib.sha256(f"migration-ensemble:1701:{label}".encode()).hexdigest()
            if int(digest[:16], 16) % 3 == fold:
                manifest["animals"][label] = identity
                break
        else:
            raise AssertionError("Unable to construct deterministic synthetic folds")
    np.savez_compressed(data / "cases.npz", **a)
    manifest["cases_sha256"] = regimes.sha256(data / "cases.npz")
    (data / "manifest.json").write_text(json.dumps(manifest))
    hashes = {"manifest_sha256": regimes.sha256(data / "manifest.json"), "cases_sha256": regimes.sha256(data / "cases.npz")}
    return a, manifest, hashes, data


def test_crossfit_scale_choices_do_not_read_held_group_outcomes(inputs):
    a, manifest, _, _ = _three_group_validation(inputs)
    ids = np.arange(10, 14)
    endpoint = np.broadcast_to(np.arange(5.)[:, None, None], (5, 4, 2)).copy()
    joint = endpoint[..., 0].copy()
    selected, folds, records = regimes.crossfit_scales(a, manifest, ids, endpoint, joint)
    assert folds.tolist() == [0, 0, 1, 2]
    assert len(records) == 3 and np.all(selected == 0)
    changed_e, changed_j = endpoint.copy(), joint.copy()
    changed_e[:, folds == 0] = np.arange(5.)[::-1, None, None] * 100
    changed_j[:, folds == 0] = np.arange(5.)[::-1, None] * 100
    second, _, _ = regimes.crossfit_scales(a, manifest, ids, changed_e, changed_j)
    np.testing.assert_array_equal(second[folds == 0], selected[folds == 0])
    assert np.all(second[folds != 0] == 4)
    for invalid in (np.array([10, 10, 12, 13]), np.array([-1, 11, 12, 13]), np.array([10, 11, 12, 14])):
        with pytest.raises(ValueError):
            regimes.crossfit_scales(a, manifest, invalid, endpoint, joint)
    invalid_costs = endpoint.copy(); invalid_costs[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        regimes.crossfit_scales(a, manifest, ids, invalid_costs, joint)


def test_synthetic_calibration_requires_explicit_finalization_and_preserves_cache(inputs, tmp_path):
    a, _, hashes, data = _three_group_validation(inputs)
    filters = tmp_path / "filters"; regimes.prepare(data, filters)
    model = MigrationModel(3, 2, 3, [1., 7.], history_length=2, width=16, layers=1, components=2, dropout=0.)
    member = tmp_path / "member.pt"
    torch.save({"model_kind": "wildcast_migration_joint", "config": model.config, "model": model.state_dict(), **hashes}, member)
    calibration = tmp_path / "calibration"
    report = regimes.calibrate(data, member, filters, calibration, samples=8, batch_size=2)
    assert report["reserved_cases_scored"] == 0 and report["validation_high_origins"] == 4
    assert report["family_selected"] is False and report["final_scale_fitted"] is False
    assert not (calibration / "scale.json").exists()
    with np.load(calibration / "validation-costs.npz", allow_pickle=False) as bank:
        assert bank["case_ids"].tolist() == [10, 11, 12, 13]
        assert not np.isin(a["split"][bank["case_ids"]], [3, 4]).any()
    finalized = tmp_path / "finalized"
    scale = regimes.finalize_scale(calibration, finalized)
    assert scale["family_selected"] is True and scale["active_scale"] in regimes.SCALE_GRID
    assert regimes.sha256(data / "cases.npz") == hashes["cases_sha256"]
    assert regimes.sha256(data / "manifest.json") == hashes["manifest_sha256"]
    output = tmp_path / "router"
    regimes.assemble(data, filters, member, output, active=member, calibration=finalized)
    _, ck = regimes.load_model(output / "best.pt")
    assert ck["active_scale"] == scale["active_scale"]
    with pytest.raises(FileExistsError, match="immutable"):
        regimes.calibrate(data, member, filters, calibration, samples=8)
    with pytest.raises(FileExistsError, match="immutable"):
        regimes.finalize_scale(calibration, finalized)
    (finalized / "scale.json").write_text("{}")
    with pytest.raises(ValueError, match="calibration checksum"):
        regimes.load_model(output / "best.pt")


def test_finalize_rejects_mutated_calibration_costs(inputs, tmp_path):
    _, _, hashes, data = _three_group_validation(inputs)
    filters = tmp_path / "filters"; regimes.prepare(data, filters)
    model = MigrationModel(3, 2, 3, [1., 7.], history_length=2, width=16, layers=1, components=2, dropout=0.)
    member = tmp_path / "member.pt"
    torch.save({"model_kind": "wildcast_migration_joint", "config": model.config, "model": model.state_dict(), **hashes}, member)
    calibration = tmp_path / "calibration"
    regimes.calibrate(data, member, filters, calibration, samples=4, batch_size=4)
    with (calibration / "validation-costs.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="cost checksum"):
        regimes.finalize_scale(calibration, tmp_path / "final")
