"""Dated covariate integrity, missingness controls and sealed-case exclusion."""
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from wildcast import migration_human_pressure as human


def timestamp(year, month=7, day=1):
    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "sources"; root.mkdir()
    records = []
    for epoch, value in ((2000, 200_000), (2005, 800_000)):
        raster = root / f"epoch-{epoch}.tif"
        values = np.full((64, 64), value, dtype="uint32")
        values[32, 32] = 0  # Valid undeveloped pixel, never missing.
        values[10:13, 10:13] = 4294967295
        with rasterio.open(raster, "w", driver="GTiff", width=64, height=64, count=1, dtype="uint32",
                           crs="ESRI:54009", transform=from_origin(-32000, 32000, 1000, 1000), nodata=4294967295) as ds:
            ds.write(values, 1)
        archive = root / f"epoch-{epoch}.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.write(raster, raster.name)
        records.append({"epoch": epoch, "path": str(archive), "sha256": human.sha256(archive),
                        "raster": str(raster), "url": f"https://example.test/GHSL/{epoch}"})
    (root / "sources.json").write_text(json.dumps({"records": records}))
    return root


def coordinates(row, col):
    lon, lat = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True).transform(-32000 + (col + .5) * 1000, 32000 - (row + .5) * 1000)
    return [lat, lon]


@pytest.fixture
def cache(tmp_path):
    root = tmp_path / "cache"; root.mkdir()
    n = 5
    origin = np.asarray([[*coordinates(20, 20), timestamp(2006)]] * n)
    history = np.zeros((n, 2, 3), "float32"); history[..., 2] = [0, 365]
    mask = np.ones((n, 2), bool); mask[1, 1] = False
    split = np.arange(n, dtype="int8")
    targets = np.ones((n, 2), bool); targets[split >= 3] = False
    a = {"history": history, "history_mask": mask, "origin": origin, "split": split,
         "static": np.zeros((n, 1), "float32"), "species": np.ones(n, "int64"),
         "metric_species": np.ones(n, "int64"), "individual": np.arange(n), "study": np.zeros(n, "int64"),
         "horizons": np.asarray([1., 90.]), "target_mask": targets, "target_latlon": np.zeros((n, 2, 2)),
         "target_time": np.zeros((n, 2)), "y": np.zeros((n, 2, 2), "float32")}
    a["target_time"][:] = origin[:, 2:3] + a["horizons"] * 86400
    a["target_time"][~targets] = 0
    np.savez_compressed(root / "cases.npz", **a)
    manifest = {"cases_sha256": human.sha256(root / "cases.npz"), "history_names": ["east_1000km", "north_1000km", "actual_age_days"]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, a, manifest


def test_nominal_epoch_is_strictly_prior_and_projections_are_rejected():
    years = np.asarray([1999, 2000, 2001, 2005, 2006, 2020, 2021, 2025, 2031])
    assert human.prior_epoch(years).tolist() == [-1, -1, 2000, 2000, 2005, 2015, 2020, 2020, 2020]
    with pytest.raises(ValueError, match="Projected"):
        human.prior_epoch(years, [2000, 2025])


def test_observed_zero_is_distinct_from_nodata_and_no_coordinate_clamping(sources):
    with human.HumanPressure(sources, tile_size=16, max_tiles=1) as context:
        point = np.asarray([coordinates(32, 32), coordinates(11, 11), coordinates(20, 20), [91., 0.], [40., 20.]])
        result = context.lookup(point, timestamp(2005))
        assert result["epoch"].tolist()[:3] == [2000] * 3
        assert result["values"][0, 0] == 0 and result["available"][0, 0]
        assert not result["available"][1, 0] and result["coverage"][1, 0] == 0
        assert result["available"][1, 1] and 0 < result["coverage"][1, 1] < 1
        np.testing.assert_allclose(result["values"][1, 1], .2, atol=1e-6)
        np.testing.assert_allclose(result["values"][2], .2, atol=1e-6)
        assert not result["available"][[3, 4]].any()
        assert np.all((result["values"] >= 0) & (result["values"] <= 1))
        assert len(context.tiles) <= 1
        assert not result["release_prior_calendar_year"].any()
        assert result["retrospective_reconstruction"][:3].all()


def test_three_matched_encoder_modes_and_release_vintage(sources):
    with human.HumanPressure(sources) as context:
        lookup = context.lookup(np.asarray([coordinates(20, 20)]), timestamp(2026))
        actual = human.encode_lookup(lookup, "actual")
        control = human.encode_lookup(lookup, "availability")
        masked = human.encode_lookup(lookup, "masked")
        assert actual.shape == control.shape == masked.shape == (1, len(human.FEATURE_NAMES))
        np.testing.assert_allclose(actual[0, :3], .8)
        assert np.all(control[:, :3] == 0)
        np.testing.assert_array_equal(control[:, 3:], actual[:, 3:])
        assert np.all(masked == 0)
        assert lookup["release_prior_calendar_year"].all()
        with pytest.raises(ValueError, match="mode"):
            human.encode_lookup(lookup, "invented")


@pytest.mark.parametrize("when", [timestamp(1999), timestamp(2000), timestamp(2001),
                                  timestamp(2005), timestamp(2006), timestamp(2026), float("nan")])
@pytest.mark.parametrize("native_bytes", [0, 64 * 64 * 4])
def test_native_only_lookup_is_bitwise_equal_at_edges_nodata_and_epoch_boundaries(sources, when, native_bytes):
    points = np.asarray([coordinates(r, c) for r,c in [(0,0),(15,15),(30,30),(32,32),(63,63),(11,11)]]
                        + [[91.,0.],[40.,20.],[float("nan"),0.]])
    with human.HumanPressure(sources, tile_size=30, max_tiles=1, native_cache_bytes=native_bytes) as context:
        full = context.lookup(points, when)
        native = context.lookup(points, when, native_only=True)
        for name in full:
            expected = full[name][..., :1] if name in {"values","available","coverage"} else full[name]
            np.testing.assert_array_equal(native[name], expected)
        assert len(context.tiles) <= 1
        assert sum(a.nbytes for a in context.native_arrays.values()) <= native_bytes


def test_archive_and_extracted_raster_hashes_are_both_verified(sources):
    record = json.loads((sources / "sources.json").read_text())["records"][0]
    with open(record["raster"], "ab") as stream:
        stream.write(b"changed-extracted-raster")
    with pytest.raises(ValueError, match="archive member"):
        human.HumanPressure(sources)
    with open(record["path"], "ab") as stream:
        stream.write(b"changed-archive")
    with pytest.raises(ValueError, match="archive checksum"):
        human.HumanPressure(sources)


def test_origin_history_are_future_invariant_and_use_actual_history_time(cache, sources):
    _, a, manifest = cache
    ids = np.arange(3)
    first = human.predictor_positions(a, manifest, ids)
    changed = {k: v.copy() for k, v in a.items()}
    for key in ("y", "target_latlon", "target_time"):
        changed[key][:] = np.nan
    changed["target_mask"][:] = False
    changed["history"][1, 1] = np.nan  # Masked rows must be ignored entirely.
    second = human.predictor_positions(changed, manifest, ids)
    for left, right in zip(first, second):
        np.testing.assert_allclose(left, right, equal_nan=True)
    with human.HumanPressure(sources) as context:
        origin, coordinates_, times = first
        now = context.lookup(origin[:, :2], origin[:, 2])
        past = context.lookup(coordinates_, times)
        assert (now["epoch"] == 2005).all()
        assert past["epoch"][0].tolist() == [2005, 2000]
        assert not human.encode_lookup(past)[1, 1].any()


def test_candidate_routes_freeze_maps_at_issue_time(sources):
    with human.HumanPressure(sources) as context:
        origin = np.asarray([[*coordinates(20, 20), timestamp(2005, 12, 31)]])
        paths = np.zeros((1, 3, 2, 2))
        features = context.route_features(origin, paths)
        # Some waypoint horizons could lie in2006; all still use the issue map.
        np.testing.assert_allclose(features[..., :3], .2, atol=1e-6)
        later = origin.copy(); later[:, 2] = timestamp(2006)
        np.testing.assert_allclose(context.route_features(later, paths)[..., :3], .8, atol=1e-6)
        assert np.all(context.route_features(origin, paths, mode="masked") == 0)


def test_prepare_is_development_only_immutable_and_loader_checks_provenance(cache, sources, tmp_path):
    data, a, _ = cache
    original = human.sha256(data / "cases.npz")
    output = tmp_path / "features"
    summary = human.prepare(data, sources, output, batch_size=2)
    assert summary["reserved_origins_prepared"] == 0 and summary["development_origins"] == 3
    assert summary["provenance"]["uses_future_targets"] is False
    assert summary["provenance"]["historical_product_may_use_later_imagery"] is True
    assert summary["provenance"]["operational_historical_backtest"] is False
    ids, encoded, _ = human.load_features(data, output)
    assert ids.tolist() == [0, 1, 2] and encoded["history"].shape == (3, 2, len(human.FEATURE_NAMES))
    assert human.sha256(data / "cases.npz") == original
    assert human.sha256(output / "preparation-source.py") == summary["code_sha256"]
    with pytest.raises(FileExistsError, match="immutable"):
        human.prepare(data, sources, output)
    with (output / "features.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        human.load_features(data, output)


def test_position_budget_fails_before_writing(cache, sources, tmp_path):
    data, _, _ = cache
    output = tmp_path / "over-budget"
    with pytest.raises(ValueError, match="quota"):
        human.prepare(data, sources, output, max_positions=1)
    assert not output.exists()
