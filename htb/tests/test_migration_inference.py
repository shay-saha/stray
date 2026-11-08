import importlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from wildcast.data import unproject
from wildcast.migration_evaluate import load_cases, sha256
from wildcast.migration_inference import LOADERS, forecast


def reference(root, multiscale=False):
    root.mkdir()
    from wildcast import multiscale_data
    if multiscale == "repaired":
        from wildcast import multiscale_repaired as multiscale_data
    horizons = multiscale_data.HORIZONS.copy() if multiscale else np.asarray([1., 7.], np.float32)
    history_length, static_length = (len(multiscale_data.LAGS), len(multiscale_data.STATIC_NAMES)) if multiscale else (25, 14)
    seconds = np.asarray([pd.Timestamp(t, tz="UTC").timestamp() for t in ["2020-01-01", "2023-01-01", "2021-01-01", "2025-01-01"]])
    y = np.full((4, len(horizons), 2), .01, np.float32)
    mask = np.ones((4, len(horizons)), bool); mask[-1] = False
    # Reserved coordinates/labels are deliberately poisoned and must be stripped.
    y[-1] = np.nan
    origin = np.column_stack([np.zeros(4), np.zeros(4), seconds])
    latlon = unproject(y.astype(float) * 1_000_000, 0., 0.)
    time = seconds[:, None] + horizons * 86400
    time[-1] = 0
    np.savez_compressed(root / "cases.npz", history=np.zeros((4, history_length, 10), np.float32),
        history_mask=np.ones((4, history_length), bool), static=np.zeros((4, static_length), np.float32),
        species=np.ones(4, np.int64), metric_species=np.ones(4, np.int64), y=y,
        target_mask=mask, split=np.asarray([0, 0, 1, 3], np.int8), individual=np.asarray([10, 10, 20, 30]),
        study=np.asarray([1, 1, 2, 3]), origin=origin, target_latlon=latlon, target_time=time,
        horizons=horizons, history_time=np.zeros((4, history_length)), history_latlon=np.zeros((4, history_length, 2)))
    lags = [0, .25, .5, 1, 2, 3, 4, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
    metadata = {"cases_sha256": sha256(root / "cases.npz"), "species": {"Known species": 1},
                "metric_species": {"Known species": 1}, "animals": {"known": 10, "validation": 20, "reserved": 30},
                "identity_mapping": {"known-alias": "known"}, "history_lags_days": lags,
                "train_end_year": 2023, "validation_year": 2024, "reserved_test_targets_sealed": True}
    if multiscale:
        metadata.update(protocol=multiscale_data.PROTOCOL, history_lags_days=multiscale_data.LAGS.tolist(),
                        history_names=multiscale_data.HISTORY_NAMES, static_names=multiscale_data.STATIC_NAMES,
                        minimum_history_lags=2, minimum_history_prior_fixes=8, minimum_history_span_days=.25,
                        minimum_observed_future_targets=1, unit_metres=multiscale_data.UNIT,
                        horizons_days=multiscale_data.HORIZONS.tolist(),
                        assessment_target_matcher=multiscale_data.__name__+".match_targets",
                        cutoff_boundaries=["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
                        subday_clock_excluded_animals=[],
                        subday_clock_modes={name: "declared_utc" for name in metadata["animals"]})
        if multiscale == "repaired":
            # Inference needs frozen inline declarations; original field/raw
            # roots are deliberately unavailable in a portable forecast bundle.
            proof={"kind":"synthetic_repair","path":"/not-packaged/repair.json","sha256":"a"*64}
            metadata.update(physical_repair=proof,parent_cache={
                "protocol":"wildcast-parent-physical-repair-accounted-v1","physical_repair":proof,
                "accounted_parent_cases":4,"case_status_counts":{"retained_unaffected":4},
                "case_accounting_sha256":"b"*64},sources=[{"path":"/not-packaged/source.parquet","sha256":"c"*64}],
                observation_source_rules=[{"sha256":"c"*64,"allowed_raw_animals":["known-alias"],"excluded_raw_animals":[]}])
    (root / "manifest.json").write_text(json.dumps(metadata))
    _, _, hashes = load_cases(root)
    return hashes


def raw_history(path, poisoned=False):
    dates = pd.date_range("2022-01-01", periods=25, freq="D", tz="UTC")
    frame = pd.DataFrame({"timestamp": dates, "lat": 30. + np.arange(25) * .01,
                          "lon": (179.7 + np.arange(25) * .05 + 180) % 360 - 180, "animal": "local"})
    if poisoned:
        frame["lat"], frame["lon"] = frame.lat.astype(object), frame.lon.astype(object)
        frame.loc[frame.index >= 20, "lat"] = "not a measured latitude"
        frame.loc[frame.index >= 20, "lon"] = "not a measured longitude"
        frame.loc[frame.index >= 20, "animal"] = "a different future animal"
    frame.to_csv(path, index=False)


class Generic:
    def predict_cases(self, arrays, ids, samples, rng, baselines):
        query = int(ids[0])
        assert query == 4 and arrays["split"][query] == 2
        nontrain = arrays["split"] != 0
        assert not arrays["target_mask"][nontrain].any()
        assert np.all(arrays["y"][nontrain] == 0)
        assert np.all(arrays["target_latlon"][nontrain] == 0)
        assert np.all(arrays["target_time"][nontrain] == 0)
        assert arrays["individual"][query] == 10
        # Same known physical identity is permitted only for past completed donors.
        donors, _ = baselines.eligible_donors(query)
        assert donors.tolist() == [0]
        return baselines.predict("seasonal", ids, samples, rng)[0]


class Joint:
    def sample_paths(self, history, history_mask, static, species, samples, generator):
        assert torch.isfinite(history).all() and species.tolist() == [1]
        noise = torch.randn(1, samples, 2, 2, generator=generator) * .0001
        return noise + static[:, None, :2, None]


def checkpoint(path, hashes, kind, model, monkeypatch):
    metadata = {"model_kind": kind, **hashes, "selection_version": "test-prospective"}
    torch.save(metadata, path)
    module = importlib.import_module(LOADERS[kind], package="wildcast")
    monkeypatch.setattr(module, "load_model", lambda *args, **kwargs: (model, metadata))
    return metadata


@pytest.mark.parametrize("kind", list(LOADERS))
def test_future_poison_cannot_change_forecast_features_or_export_truth(tmp_path, monkeypatch, kind):
    hashes = reference(tmp_path / "data")
    model = Joint() if kind == "wildcast_migration_joint" else Generic()
    path = tmp_path / "model.pt"
    checkpoint(path, hashes, kind, model, monkeypatch)
    raw_history(tmp_path / "one.csv")
    raw_history(tmp_path / "two.csv", poisoned=True)
    for name in ["one", "two"]:
        forecast(tmp_path / "data", path, tmp_path / (name + ".csv"), tmp_path / (name + ".json"),
                 "Known species", canonical_animal="known-alias", issue_time="2022-01-20", samples=4)
    a, b = [json.loads((tmp_path / (name + ".json")).read_text()) for name in ["one", "two"]]
    assert a["sampled_latlon_paths"] == b["sampled_latlon_paths"]
    assert a["observed_history"] == b["observed_history"]
    assert a["conditioning_history"] == b["conditioning_history"]
    assert a["conditioning_history_sha256"] == b["conditioning_history_sha256"]
    assert a["source_file_sha256"] != b["source_file_sha256"]
    assert a["physical_identity_known"] and a["matched_canonical_animal"] == "known"
    assert a["retrospective_replay"] and not a["operational_backtest"]
    assert len(a["observed_history"]) == 20
    assert all(pd.Timestamp(row["timestamp"]) <= pd.Timestamp("2022-01-20", tz="UTC") for row in a["conditioning_history"])
    assert "observed_future" not in a and "target_latlon" not in a and "y" not in a


def test_unknown_species_and_fresh_identity_use_declared_fallback(tmp_path, monkeypatch):
    hashes = reference(tmp_path / "data")

    class Unknown:
        def predict_cases(self, a, ids, samples, rng, baselines):
            assert a["species"][ids].tolist() == [0]
            assert a["individual"][ids].tolist() == [31]
            assert not a["target_mask"][ids].any()
            return np.zeros((1, samples, 2, 2))

    path = tmp_path / "model.pt"; checkpoint(path, hashes, "wildcast_migration_retrieval", Unknown(), monkeypatch)
    raw_history(tmp_path / "observed.csv")
    output = tmp_path / "forecast.json"
    forecast(tmp_path / "data", path, tmp_path / "observed.csv", output, "Unseen species", samples=4)
    result = json.loads(output.read_text())
    assert result["unknown_species_fallback"] and not result["physical_identity_known"]
    assert result["prediction_species_index"] == 0


def test_invalid_schema_cache_and_checkpoint_hashes_are_rejected(tmp_path, monkeypatch):
    hashes = reference(tmp_path / "data")
    path = tmp_path / "model.pt"; checkpoint(path, hashes, "wildcast_migration_joint", Joint(), monkeypatch)
    bad = tmp_path / "bad.csv"; pd.DataFrame({"timestamp": ["2022-01-01"], "lat": [0.]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="timestamp, lat and lon"):
        forecast(tmp_path / "data", path, bad, tmp_path / "out.json", "Known species")
    raw_history(tmp_path / "one.csv")
    torch.save({"model_kind": "wildcast_migration_joint", "cases_sha256": "wrong", "manifest_sha256": "wrong"}, path)
    with pytest.raises(ValueError, match="checkpoint/reference"):
        forecast(tmp_path / "data", path, tmp_path / "one.csv", tmp_path / "out.json", "Known species")
    with (tmp_path / "data" / "cases.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        forecast(tmp_path / "data", path, tmp_path / "one.csv", tmp_path / "out.json", "Known species")


def test_missing_ensemble_assets_have_clear_package_error(tmp_path, monkeypatch):
    hashes = reference(tmp_path / "data")
    path = tmp_path / "ensemble.pt"
    torch.save({"model_kind": "wildcast_migration_ensemble", **hashes}, path)
    module = importlib.import_module("wildcast.migration_ensemble")

    def absent(*args, **kwargs):
        raise FileNotFoundError("/original-server/immutable/member.pt")

    monkeypatch.setattr(module, "load_model", absent)
    raw_history(tmp_path / "one.csv")
    with pytest.raises(ValueError, match="missing a referenced cache/member asset"):
        forecast(tmp_path / "data", path, tmp_path / "one.csv", tmp_path / "out.json", "Known species")


def test_actual_joint_checkpoint_runs_from_synthetic_past_history(tmp_path):
    from wildcast.migration_model import MigrationModel
    hashes = reference(tmp_path / "data")
    torch.manual_seed(4)
    model = MigrationModel(10, 14, 2, [1, 7], history_length=25,
                           width=8, layers=1, components=2, dropout=0.)
    path = tmp_path / "joint.pt"
    torch.save({"model_kind": "wildcast_migration_joint", "model": model.state_dict(),
                "config": model.config, "manifest": {"species": {"Known species": 1}}, **hashes}, path)
    raw_history(tmp_path / "one.csv")
    result = forecast(tmp_path / "data", path, tmp_path / "one.csv", tmp_path / "out.json", "Known species",
                      canonical_animal="known", issue_time="2022-01-20", samples=3)
    payload = json.loads((tmp_path / "out.json").read_text())
    assert result["model_kind"] == "wildcast_migration_joint"
    assert np.asarray(payload["sampled_latlon_paths"]).shape == (3, 2, 2)
    assert np.isfinite(payload["sampled_latlon_paths"]).all()


@pytest.mark.parametrize("prior_fixes,supported", [(8, True), (4, False)])
@pytest.mark.parametrize("protocol", ["observed", "repaired"])
def test_multiscale_joint_hourly_forecast_and_support_ignore_future(tmp_path, prior_fixes, supported,protocol):
    from wildcast.migration_model import MigrationModel
    from wildcast.migration_train import load_training_data
    from wildcast import multiscale_data
    if protocol == "repaired":
        from wildcast import multiscale_repaired as multiscale_data
    hashes = reference(tmp_path / "data", multiscale="repaired" if protocol == "repaired" else True)
    training, _, _ = load_training_data(tmp_path / "data")
    assert training["horizons"].dtype == np.dtype("float64")
    assert len(training["split"]) == 3  # The poisoned reserved row is excluded.
    torch.manual_seed(4)
    model = MigrationModel(10,len(multiscale_data.STATIC_NAMES),2,multiscale_data.HORIZONS,
                           history_length=len(multiscale_data.LAGS),
                           width=8, layers=1, components=2, dropout=0.,
                           min_scale=5e-6, covariance_jitter=1e-12)
    checkpoint_path = tmp_path / "joint.pt"
    torch.save({"model_kind": "wildcast_migration_joint", "model": model.state_dict(),
                "config": model.config, "manifest": {"species": {"Known species": 1}}, **hashes}, checkpoint_path)
    dates = pd.date_range("2022-01-01", periods=prior_fixes + 4, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": dates, "lat": 30. + np.arange(len(dates)) * .001,
                          "lon": 10. + np.arange(len(dates)) * .002, "animal": "local"})
    for name in ["clean", "poison", "micros"]:
        values = frame.copy()
        if name == "poison":
            values["lat"] = values.lat.astype(object)
            values.loc[values.index > prior_fixes, "lat"] = "future is not an input"
            values.loc[values.index > prior_fixes, "animal"] = "future animal"
        source = tmp_path / (name + (".parquet" if name == "micros" else ".csv"))
        if name == "micros":
            values["timestamp"] = values.timestamp.dt.as_unit("us")
            values.to_parquet(source, index=False)
        else:
            values.to_csv(source, index=False)
        forecast(tmp_path / "data", checkpoint_path, source, tmp_path / (name + ".json"),
                 "Known species", canonical_animal="known", issue_time=dates[prior_fixes].isoformat(), samples=3)
    clean, poison = [json.loads((tmp_path / (name + ".json")).read_text()) for name in ["clean", "poison"]]
    assert clean["sampled_latlon_paths"] == poison["sampled_latlon_paths"]
    assert clean["conditioning_history"] == poison["conditioning_history"]
    micros = json.loads((tmp_path / "micros.json").read_text())
    assert clean["sampled_latlon_paths"] == micros["sampled_latlon_paths"]
    assert clean["conditioning_history_sha256"] == micros["conditioning_history_sha256"]
    assert clean["origin"] == micros["origin"]
    assert np.asarray(clean["sampled_latlon_paths"]).shape == (3, 9, 2)
    assert clean["history_support_matches_training"] is supported
    assert clean["horizons_days"][0] == 1/24
    assert clean["feature_protocol"] == multiscale_data.PROTOCOL
    assert any(row["requested_lag_days"] == 1/24 for row in clean["conditioning_history"])


def test_multiscale_consumers_reject_rounded_case_horizons(tmp_path):
    from wildcast.migration_train import load_training_data
    root = tmp_path / "data"
    reference(root, multiscale=True)
    with np.load(root / "cases.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["horizons"] = arrays["horizons"].astype("float32")
    np.savez_compressed(root / "cases.npz", **arrays)
    metadata = json.loads((root / "manifest.json").read_text())
    metadata["cases_sha256"] = sha256(root / "cases.npz")
    (root / "manifest.json").write_text(json.dumps(metadata))
    for loader in [load_cases, load_training_data]:
        with pytest.raises(ValueError, match="float64"):
            loader(root)
