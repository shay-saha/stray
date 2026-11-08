import numpy as np
import pandas as pd
import pytest

from wildcast.vegetation import FEATURES, composite_dates, patch_features, tile_indices


def test_composites_are_completed_before_origin_even_across_leap_year():
    times = pd.to_datetime(["2020-01-02", "2020-03-01", "2021-01-01"], utc=True).astype("int64") / 1e9
    dates = composite_dates(times, 32)
    assert np.all(dates.asi8 / 1e9 + 16 * 86400 < times)
    assert np.all((dates.dayofyear - 1) % 16 == 0)
    with pytest.raises(ValueError):
        composite_dates(times, 0)


def test_modis_tile_projection_and_qa_do_not_turn_clouds_into_greenness():
    h, v = tile_indices(np.asarray([49.05]), np.asarray([8.05]))
    assert (h[0], v[0]) == (18, 4)
    ndvi = np.full((51, 51), 8000); evi = np.full((51, 51), 4000)
    good = patch_features(ndvi, evi, np.zeros((51, 51)))
    assert good[FEATURES.index("ndvi_centre")] == pytest.approx(.8)
    assert good[FEATURES.index("good_fraction51")] == 1
    cloudy = patch_features(ndvi, evi, np.full((51, 51), 3))
    assert np.all(cloudy[:14] == 0)
    assert cloudy[FEATURES.index("centre_present")] == 0
    assert cloudy[FEATURES.index("cloud_fraction51")] == 1


def test_missing_directional_support_does_not_create_a_false_gradient():
    qa = np.zeros((51, 51)); qa[:, :26] = -1
    features = patch_features(np.full((51, 51), 5000), np.full((51, 51), 2500), qa)
    assert features[FEATURES.index("grid_x_difference_present")] == 0
    assert features[FEATURES.index("ndvi_grid_x_difference51")] == 0
    assert features[FEATURES.index("ndvi_mean51")] == pytest.approx(.5)


def test_alphaearth_dequantization_masks_missing_vectors():
    from wildcast.satellite import dequantize
    a = np.asarray([[127, -127, 0], [-128, -128, -128]], dtype=np.int8)
    values, present = dequantize(a)
    np.testing.assert_array_equal(present, [True, False])
    np.testing.assert_allclose(values[0], [(127 / 127.5) ** 2, -(127 / 127.5) ** 2, 0])
    np.testing.assert_array_equal(values[1], 0)


def test_new_environment_branch_starts_at_exact_pretrained_forecast():
    import torch
    from wildcast.model import MovementTransformer
    kwargs = dict(n_species=2, width=32, layers=1, n_anchors=5, student_df=4)
    base = MovementTransformer(**kwargs, n_drivers=4).eval()
    extended = MovementTransformer(**kwargs, n_drivers=10, driver_base_features=4).eval()
    extended.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(3, 32, 10); drivers = torch.randn(3, 10); anchors = torch.randn(3, 5, 2)
    args = (x, torch.tensor([0, 1, 2]), torch.tensor([1., 24., 168.]), torch.zeros(3, 2))
    with torch.inference_mode():
        p = base(*args, drivers[:, :4], anchors); q = extended(*args, drivers, anchors)
    for left, right in zip(p[:4], q[:4]):
        torch.testing.assert_close(left, right)


def test_vegetation_join_rejects_future_composite_and_preserves_targets(tmp_path):
    import json
    from wildcast.vegetation import add_vegetation
    source = tmp_path / "source"; processed = source / "processed"; processed.mkdir(parents=True)
    cache = tmp_path / "cache"; cache.mkdir()
    t = pd.Timestamp("2020-07-01", tz="UTC").timestamp()
    aux = np.zeros((2, 9)); aux[:, :3] = [49.05, 8.05, t]; aux[:, 6] = [0, 1]
    meta = np.zeros((2, 5)); meta[1, 0] = 2
    target = np.asarray([[1., 2.], [3., 4.]])
    np.savez_compressed(processed / "windows.npz", x=np.zeros((2, 1, 1)), y=target, meta=meta, aux=aux, drivers=np.zeros((2, 1), np.float32))
    (processed / "windows.json").write_text(json.dumps({"observations_sha256": "example", "windows_sha256": "parent",
        "driver_names": ["old"], "driver_median": [0], "driver_iqr": [1]}))
    (cache / "source.json").write_text(json.dumps({"observations_sha256": "example"}))
    date = composite_dates([t], 32).strftime("%Y-%m-%d")[0]
    p = cache / f"h18v04_{date}.npz"
    values = np.zeros((2, len(FEATURES)), np.float32)
    np.savez_compressed(p, rows=np.arange(2), values=values, end=t+1)
    with pytest.raises(ValueError, match="Future"):
        add_vegetation(source, tmp_path / "bad", cache)
    np.savez_compressed(p, rows=np.arange(2), values=values, end=t-20*86400)
    add_vegetation(source, tmp_path / "good", cache)
    with np.load(tmp_path / "good/processed/windows.npz") as a:
        np.testing.assert_array_equal(a["y"], target)
        np.testing.assert_array_equal(a["aux"], aux)
        np.testing.assert_array_equal(a["meta"], meta)
