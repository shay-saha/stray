import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from wildcast import migration_deepssf, migration_issf, migration_model, multiscale_data
from wildcast.data import unproject
from wildcast.migration_endpoint_evaluate import evaluate, horizon_columns
from wildcast.migration_evaluate import load_cases, sha256


def reference(root):
    root.mkdir()
    h = multiscale_data.HORIZONS.copy()
    dates = pd.to_datetime(["2019-01-01", "2019-06-01", "2019-10-01", "2021-07-01", "2022-01-01"], utc=True)
    origin = np.column_stack([np.full(5, 10.), np.full(5, 179.99), dates.asi8/1e9])
    y = np.zeros((5, 9, 2), "float32")
    y[:, :, 0] = h * .001
    y[2, 2, 0] = 10.  # A distant outcome must remain in every comparison.
    coords = unproject(y*1e6, origin[:, None, 0], origin[:, None, 1])
    mask = np.ones((5, 9), bool)
    mask[2, 1] = False
    mask[4] = False
    times = origin[:, 2:3]+h*86400
    times[2, 0] += 8*60
    y[~mask] = 0;coords[~mask] = 0;times[~mask] = 0
    history = np.zeros((5, 27, 10), "float32")
    history[:, :, 3] = np.arange(1, 6)[:, None]
    np.savez_compressed(root/"cases.npz", history=history, history_mask=np.ones((5, 27), bool),
        static=np.zeros((5, 18), "float32"), species=np.array([1, 2, 1, 1, 2]),
        metric_species=np.array([0, 1, 0, 0, 1]), individual=np.array([0, 1, 2, 0, 3]),
        study=np.arange(5), split=np.array([0, 0, 1, 2, 3], "int8"), origin=origin,
        y=y, target_mask=mask, target_time=times, target_latlon=coords, horizons=h)
    animals = {"a": 0, "b": 1, "c": 2, "reserved": 3}
    manifest = {"protocol":multiscale_data.PROTOCOL, "cases":5, "horizons_days":h.tolist(),
        "assessment_target_matcher":"wildcast.multiscale_data.match_targets",
        "history_names":multiscale_data.HISTORY_NAMES, "static_names":multiscale_data.STATIC_NAMES,
        "history_lags_days":multiscale_data.LAGS.tolist(), "unit_metres":1e6,
        "minimum_history_lags":2, "minimum_history_prior_fixes":8, "minimum_history_span_days":.25,
        "minimum_observed_future_targets":1, "train_end_year":2020, "validation_year":2021,
        "cutoff_boundaries":["2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"],
        "animals":animals, "species":{"one":1, "two":2}, "metric_species":{"one":0,"two":1},
        "subday_clock_excluded_animals":[], "subday_clock_modes":{a:"declared_utc" for a in animals},
        "reserved_test_targets_sealed":True, "cases_sha256":sha256(root/"cases.npz")}
    (root/"manifest.json").write_text(json.dumps(manifest))
    return load_cases(root)


def checkpoint(root, kind, hashes):
    root.mkdir()
    path = root/"best.pt"
    metadata = {"model_kind":kind, **hashes}
    torch.save(metadata, path)
    return path, metadata


def assert_future_free(a, ids):
    held = a["split"] != 0
    for key in ("y", "target_mask", "target_time", "target_latlon"):
        assert not np.any(a[key][held])
    other = held.copy();other[ids] = False
    assert not np.any(a["history"][other])
    assert not np.any(a["origin"][other])
    assert np.any(a["history"][ids])
    assert a["target_mask"][a["split"] == 0].all()


@pytest.mark.parametrize("short_module", [migration_deepssf, migration_issf])
def test_short_and_route_references_share_exact_endpoint_support_without_future_inputs(tmp_path, monkeypatch, short_module):
    data = tmp_path/"data"
    a, _, hashes = reference(data)
    before = {p.name:p.read_bytes() for p in data.iterdir()}
    seen = []

    class Short:
        def predict_endpoints(self, view, ids, horizons, samples, rng):
            assert_future_free(view, ids)
            np.testing.assert_array_equal(view["horizons"], multiscale_data.HORIZONS)
            np.testing.assert_array_equal(horizons, multiscale_data.HORIZONS[:3])
            seen.extend(ids.tolist())
            return np.zeros((len(ids), samples, 3, 2))

    class Route:
        def predict_cases(self, view, ids, samples, rng, baselines):
            assert_future_free(view, ids)
            return np.zeros((len(ids), samples, 9, 2))

    short, short_metadata = checkpoint(tmp_path/"short", short_module.MODEL_KIND, hashes)
    route, route_metadata = checkpoint(tmp_path/"route", "wildcast_migration_joint", hashes)
    monkeypatch.setattr(short_module, "load_model", lambda path,device:(Short(),short_metadata))
    monkeypatch.setattr(migration_model, "load_model", lambda path,device:(Route(),route_metadata))
    output = tmp_path/"comparison"
    report = evaluate(data, output, [short, route], baselines=["stationary"], samples=4,
                      batch_size=1, bootstrap_replicates=3)
    assert seen == [2, 3]
    assert report["cases"] == 2 and report["observed_targets"] == 5
    assert report["stored_cache_horizons_days"] == multiscale_data.HORIZONS.tolist()
    frame = pd.read_parquet(output/"per-case.parquet")
    assert len(frame) == 15
    assert frame.groupby("model").size().to_dict() == {"route":5,"short":5,"stationary":5}
    assert frame.point_error_km.max() > 9900
    assert frame.loc[frame.horizon_days == 1/24, "target_timing_error_hours"].max() == pytest.approx(8/60)
    assert set(frame.identity_in_training) == {False,True}
    assert not any("joint" in key for key in report)
    assert all(sha256(output/name) == digest for name,digest in report["output_sha256"].items())
    assert not list(output.glob("*route*"))
    assert {p.name:p.read_bytes() for p in data.iterdir()} == before
    assert report["case_ids_sha256"] == hashlib.sha256(np.array([2,3],dtype="<i8").tobytes()).hexdigest()
    with pytest.raises(FileExistsError,match="immutable"):
        evaluate(data,output)


def test_horizon_selection_does_not_change_eligibility_in_other_tasks(tmp_path):
    reference(tmp_path/"data")
    report = evaluate(tmp_path/"data", tmp_path/"twelve", baselines=["stationary"],
                      horizons_days=[.5], samples=3, bootstrap_replicates=2)
    assert report["cases"] == 1 and report["observed_targets"] == 1
    assert report["horizons_days"] == [.5]
    frame = pd.read_parquet(tmp_path/"twelve/per-case.parquet")
    assert frame.case.tolist() == [3]


def test_reserved_truth_is_never_opened_for_development_or_without_frozen_assessment(tmp_path):
    reference(tmp_path/"data")
    for kwargs in ({"splits":["test_individual"]},{"assessment":tmp_path/"nonexistent"}):
        with pytest.raises(ValueError,match="Reserved splits"):
            evaluate(tmp_path/"data",tmp_path/"out",**kwargs)
    assert not (tmp_path/"out").exists()


def test_wrong_checkpoint_provenance_rejected_before_forecasting(tmp_path):
    _, _, hashes = reference(tmp_path/"data")
    path, _ = checkpoint(tmp_path/"model", "wildcast_migration_joint", {**hashes,"cases_sha256":"wrong"})
    with pytest.raises(ValueError,match="hash mismatch"):
        evaluate(tmp_path/"data", tmp_path/"out", [path], baselines=[])
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("horizons", [[1/24,1/24],[0.],[float("nan")],[2.]])
def test_missing_or_invalid_exact_horizons_are_rejected(horizons):
    with pytest.raises(ValueError):
        horizon_columns(multiscale_data.HORIZONS,horizons)
