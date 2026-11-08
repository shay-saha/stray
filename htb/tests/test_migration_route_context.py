import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from wildcast.data import project
from wildcast.migration_evaluate import forecast_latlon
from wildcast.migration_route_context import RouteContext, prepare_context
from wildcast.migration_retrieval_train import load_model, prepare_bank, train_bank


def fields(root, nodata=False):
    root.mkdir()
    records = []
    for year, value in ((2019, 64), (2020, 127)):
        values = np.full((2, 2, 64), value, np.int8)
        if nodata and year == 2019:
            values[0, 1] = -128
        path = root / f"tile{year}.npz"
        np.savez_compressed(path, values=values, bounds=[0., 0., 1., 1.], year=year)
        records.append({"key": path.stem, "year": year, "bounds": [0., 0., 1., 1.],
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / "manifest.json").write_text(json.dumps({"records": records}))


def test_exact_previous_year_and_no_outside_edge_clamping(tmp_path):
    fields(tmp_path / "fields", nodata=True)
    sampler = RouteContext(tmp_path / "fields")
    points = np.array([[.75, .25], [.75, .75], [.75, 1.0001], [.75, -.0001], [0., .25]])
    values, present = sampler.lookup(points, np.full(5, 2019))
    assert present.tolist() == [True, False, False, False, False]
    np.testing.assert_allclose(values[0], (64 / 127.5) ** 2, atol=1e-7)
    assert not values[1:].any()
    # A later tile must never replace missing data from the allowed year.
    future, mask = sampler.lookup(points[:2], np.array([2018, 2018]))
    assert not mask.any() and not future.any()
    later, mask = sampler.lookup(points[:2], np.array([2020, 2020]))
    assert mask.all() and (later > values[:2]).all()


def test_route_summary_masks_and_utc_year_vintage(tmp_path):
    fields(tmp_path / "fields", nodata=True)
    sampler = RouteContext(tmp_path / "fields")
    now = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    origin = np.array([[.75, .25, now], [.75, .25, now - 1]])
    destinations = np.array([[[[.75, .25], [.75, 1.1]]], [[[.75, .25], [.75, 1.1]]]])
    paths = project(destinations[..., 0], destinations[..., 1], origin[:, None, None, 0], origin[:, None, None, 1]) / 1e6
    values, mask = sampler.features(origin, paths, return_masks=True)
    assert values.shape == (2, 1, 132)
    np.testing.assert_allclose(values[0, 0, :64], (64 / 127.5) ** 2, atol=1e-7)
    assert not values[0, 0, 64:128].any()
    np.testing.assert_array_equal(values[0, 0, 128:], [.5, 0, 1, 0])
    assert not values[1].any()  # 2019 origin permits only the absent 2018 tile.
    assert mask[0, 0].tolist() == [True, False]


def test_field_bytes_metadata_and_fingerprint_are_verified(tmp_path):
    fields(tmp_path / "fields")
    sampler = RouteContext(tmp_path / "fields")
    with pytest.raises(ValueError, match="fingerprint"):
        RouteContext(tmp_path / "fields", expected="wrong")
    with (tmp_path / "fields/tile2019.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        RouteContext(tmp_path / "fields")
    assert sampler.fingerprint


def write_data(root):
    root.mkdir()
    n, h = 6, np.array([1., 3., 7.], np.float32)
    now = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    a = {"history": np.zeros((n, 3, 10), np.float32), "history_mask": np.ones((n, 3), bool),
         "static": np.zeros((n, 14), np.float32), "species": np.ones(n, np.int64),
         "metric_species": np.ones(n, np.int64), "y": np.zeros((n, 3, 2), np.float32),
         "target_mask": np.ones((n, 3), bool), "split": np.array([0, 0, 0, 1, 2, 3], np.int8),
         "individual": np.array([1, 2, 3, 4, 1, 5], np.int64), "study": np.zeros(n, np.int64), "horizons": h}
    a["origin"] = np.column_stack((np.full(n, .75), np.full(n, .25), now + np.array([0, 10, 20, 31, 91, 150]) * 86400))
    a["target_time"] = a["origin"][:, 2:3] + h * 86400
    a["history"][:, :, 2] = [0, 1, 7]
    a["history"][:, :, 0] = [0., -.01, -.07]
    a["y"][:, :, 0] = h * .01
    a["target_latlon"] = forecast_latlon(a["y"][:, None], a["origin"])[:, 0]
    a["target_mask"][-1] = False
    a["y"][-1] = 0; a["target_time"][-1] = 0; a["target_latlon"][-1] = 0
    np.savez_compressed(root / "cases.npz", **a)
    (root / "manifest.json").write_text(json.dumps({"cases_sha256": hashlib.sha256((root / "cases.npz").read_bytes()).hexdigest(),
        "species": {"Animalus test": 1}, "reserved_test_targets_sealed": True}))
    return a


def test_context_and_masked_control_share_bank_capacity_and_initial_law(tmp_path):
    old = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        a = write_data(tmp_path / "data")
        fields(tmp_path / "fields")
        prepare_bank(tmp_path / "data", tmp_path / "bank", candidates=8, seed=11)
        report = prepare_context(tmp_path / "data", tmp_path / "bank", tmp_path / "fields", tmp_path / "context")
        assert report["reserved_test_cases_prepared"] == 0
        assert len(report["coverage_by_split"]["0"]["waypoint_coverage_by_horizon"]) == 3
        models = []
        for mode in ("actual", "masked", "availability"):
            train_bank(tmp_path / "data", tmp_path / "bank", tmp_path / mode,
                       context=tmp_path / "context", context_mode=mode, epochs=0, hidden=16, batch_size=4, device="cpu")
            models.append(load_model(tmp_path / mode / "best.pt")[0])
        assert len({sum(p.numel() for p in m.parameters()) for m in models}) == 1
        assert len({m.config["candidate_features"] for m in models}) == 1
        first = models[0].predict_cases(a, np.array([3, 4]), 12, np.random.default_rng(5))
        second = models[1].predict_cases(a, np.array([3, 4]), 12, np.random.default_rng(5))
        np.testing.assert_array_equal(first, second)
        third = models[2].predict_cases(a, np.array([3, 4]), 12, np.random.default_rng(5))
        np.testing.assert_array_equal(first, third)
        assert models[2].config["context"]["mode"] == "availability"
        offset = models[2].config["candidate_features"] - 132
        assert not models[2].candidate_mean[offset:offset + 128].any()
    finally:
        torch.set_num_threads(old)
