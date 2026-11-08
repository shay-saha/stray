import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest
import torch

from wildcast.data import unproject
from wildcast.migration_evaluate import load_cases
from wildcast.migration_inference import forecast
from wildcast.migration_model import MigrationModel
from wildcast.migration_retrieval_train import RetrievalReweighter
from wildcast.migration_route_context import RouteContext
from test_migration_human_pressure import sources

spec = importlib.util.spec_from_file_location("wildcast_package_migration", Path(__file__).resolve().parents[1] / "scripts/package_migration.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def reference(root):
    root.mkdir()
    times = np.asarray([pd.Timestamp(t, tz="UTC").timestamp() for t in ["2020-01-01", "2020-03-01", "2021-02-01", "2022-01-01"]])
    origin = np.column_stack([np.zeros(4), np.zeros(4), times])
    horizons = np.asarray([1., 7.], np.float32)
    y = np.full((4, 2, 2), .01, np.float32)
    mask = np.ones((4, 2), bool); mask[-1] = False
    latlon = unproject(y.astype(float) * 1_000_000, 0., 0.)
    target_time = times[:, None] + horizons * 86400
    y[-1], latlon[-1], target_time[-1] = 0, 0, 0
    np.savez_compressed(root / "cases.npz", history=np.zeros((4, 25, 10), np.float32),
        history_mask=np.ones((4, 25), bool), static=np.zeros((4, 14), np.float32),
        species=np.ones(4, np.int64), metric_species=np.ones(4, np.int64), y=y, target_mask=mask,
        split=np.asarray([0, 0, 1, 3], np.int8), individual=np.asarray([10, 11, 20, 30]), study=np.ones(4, int),
        origin=origin, target_latlon=latlon, target_time=target_time, horizons=horizons)
    pd.DataFrame({"case_id": np.arange(4), "species": ["Known species"] * 4}).to_parquet(root / "origins.parquet")
    manifest = {"cases_sha256": packager.sha256(root / "cases.npz"), "origins_sha256": packager.sha256(root / "origins.parquet"),
        "species": {"Known species": 1}, "metric_species": {"Known species": 1},
        "animals": {"a": 10, "b": 11, "validation": 20, "reserved": 30}, "identity_mapping": {},
        "reserved_test_targets_sealed": True, "train_end_year": 2020, "validation_year": 2021}
    (root / "manifest.json").write_text(json.dumps(manifest))
    # A sensitive sidecar must never enter the explicit selected-asset closure.
    np.savez(root / "assessment-truth.npz", future=np.asarray([123456.]))
    return load_cases(root)[2]


def joint(path, hashes, seed=4):
    torch.manual_seed(seed)
    model = MigrationModel(10, 14, 2, [1, 7], history_length=25, width=8, layers=1, components=2, dropout=0.)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_kind": "wildcast_migration_joint", "model": model.state_dict(), "config": model.config,
                "selection_version": "synthetic-selected", **hashes}, path)
    return path


def gps(path):
    dates = pd.date_range("2022-01-01", periods=20, freq="D", tz="UTC")
    pd.DataFrame({"timestamp": dates, "lat": 30. + np.arange(20) * .01,
                  "lon": 20. + np.arange(20) * .01, "animal": "local"}).to_csv(path, index=False)


def run_offline(bundle, observations, output, cwd):
    script = '''
import runpy,socket,sys,torch
torch.set_num_threads(2)
def forbidden(*a,**k): raise RuntimeError("Network forbidden in portable inference test")
socket.socket.connect=forbidden
launcher=sys.argv[1]
sys.argv=[launcher,"--observations",sys.argv[2],"--output",sys.argv[3],"--species","Known species","--samples","6","--seed","2026","--device","cpu"]
runpy.run_path(launcher,run_name="__main__")
'''
    result = subprocess.run([sys.executable, "-c", script, str(bundle / "infer.py"), str(observations), str(output)],
                            cwd=cwd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text())


def test_joint_bundle_is_relocated_offline_equivalent_and_archive_verified(tmp_path):
    data = tmp_path / "cache"; hashes = reference(data)
    selected = joint(tmp_path / "models" / "joint.pt", hashes)
    observed = tmp_path / "observed.csv"; gps(observed)
    forecast(data, selected, observed, tmp_path / "expected.json", "Known species", samples=6, seed=2026)
    archive = tmp_path / "selected.tar.gz"
    report = packager.package(data, selected, tmp_path / "bundle", archive=archive, source_root=tmp_path)
    manifest = json.loads((tmp_path / "bundle" / "package-manifest.json").read_text())
    assert packager.sha256(data / "cases.npz") == packager.sha256(tmp_path / "bundle" / "data/reference/cases.npz")
    assert manifest["checkpoint"]["source_sha256"] == manifest["checkpoint"]["sha256"]
    assert not any(("assessment" in path and path.endswith(".npz")) or ".ssh" in path or ".git/" in path for path in manifest["files"])
    assert report["archive"]["sha256"] == packager.sha256(archive)
    with tarfile.open(archive) as tar:
        assert all(name.startswith("bundle/") and ".." not in Path(name).parts for name in tar.getnames())
    relocated = tmp_path / "relocated bundle"
    (tmp_path / "bundle").rename(relocated)
    data.rename(tmp_path / "hidden-cache"); selected.parent.rename(tmp_path / "hidden-models")
    actual = run_offline(relocated, observed, tmp_path / "actual.json", tmp_path)
    expected = json.loads((tmp_path / "expected.json").read_text())
    np.testing.assert_allclose(actual["sampled_latlon_paths"], expected["sampled_latlon_paths"], rtol=0, atol=1e-9)
    assert (archive.with_name(archive.name + ".sha256.json")).is_file()


def test_ensemble_and_context_paths_relocate_without_changing_prediction_law(tmp_path):
    data = tmp_path / "cache"; hashes = reference(data)
    models = tmp_path / "models"; models.mkdir()
    direct = joint(models / "members" / "joint.pt", hashes)
    fields = tmp_path / "fields"; fields.mkdir()
    values = np.full((4, 4, 64), 45, np.int8)
    np.savez_compressed(fields / "tile.npz", values=values, bounds=[-180., -90., 180., 90.], year=2021)
    (fields / "manifest.json").write_text(json.dumps({"records": [{"key": "tile", "year": 2021,
        "bounds": [-180., -90., 180., 90.], "sha256": packager.sha256(fields / "tile.npz")}], "license": "CC BY 4.0"}))
    context = RouteContext(fields)
    residual = RetrievalReweighter(38, 150, 2, candidates=8, hidden=8, dropout=0.,
        context={"fields_root": str(fields), "fingerprint": context.fingerprint, "mode": "full"})
    reweighted = models / "members" / "context.pt"
    torch.save({"model_kind": "wildcast_migration_retrieval", "model": residual.state_dict(), "config": residual.config, **hashes}, reweighted)
    selected = models / "ensemble.pt"
    members = [{"path": str(path), "sha256": packager.sha256(path)} for path in [direct, reweighted]]
    torch.save({"model_kind": "wildcast_migration_ensemble", "members": members, "weights": torch.tensor([.4, .6], dtype=torch.float64),
                "data": str(data), **hashes}, selected)
    observed = tmp_path / "observed.csv"; gps(observed)
    forecast(data, selected, observed, tmp_path / "expected.json", "Known species", samples=6, seed=2026)
    packager.package(data, selected, tmp_path / "bundle", source_root=tmp_path)
    catalog = json.loads((tmp_path / "bundle/package-manifest.json").read_text())
    packed = torch.load(tmp_path / "bundle/model/selected.pt", map_location="cpu", weights_only=True)
    assert packed["data"] == "data/reference" and all(not Path(m["path"]).is_absolute() for m in packed["members"])
    assert catalog["context_fields"][0]["fingerprint"] == context.fingerprint
    assert catalog["context_fields"][0]["tiles"] == 1
    moved = tmp_path / "relocated"
    (tmp_path / "bundle").rename(moved)
    data.rename(tmp_path / "hidden-cache"); models.rename(tmp_path / "hidden-models"); fields.rename(tmp_path / "hidden-fields")
    actual = run_offline(moved, observed, tmp_path / "actual.json", tmp_path)
    expected = json.loads((tmp_path / "expected.json").read_text())
    np.testing.assert_allclose(actual["sampled_latlon_paths"], expected["sampled_latlon_paths"], rtol=0, atol=1e-9)


def test_GHSL_joint_bundle_is_offline_portable_and_preserves_predictive_law(tmp_path, sources):
    from wildcast import migration_context_joint as context
    data = tmp_path / "cache"; hashes = reference(data)
    table = tmp_path / "context-table"
    metadata = context.prepare(data, sources, table)
    model = MigrationModel(10, 35, 2, [1, 7], history_length=25, width=8, layers=1, components=2, dropout=0.)
    models = tmp_path / "models"; models.mkdir()
    selected = models / "context.pt"
    torch.save({"model_kind": context.KIND, "model": model.state_dict(), "config": model.config,
                "context_features": {**metadata, "mode": "actual", "base_static_features": 14}, **hashes}, selected)
    observed = tmp_path / "observed.csv"; gps(observed)
    forecast(data, selected, observed, tmp_path / "expected.json", "Known species", samples=6, seed=2026)
    packager.package(data, selected, tmp_path / "bundle", source_root=tmp_path)
    packed = torch.load(tmp_path / "bundle/model/selected.pt", map_location="cpu", weights_only=True)
    assert packed["context_features"]["sources"].startswith("data/human-fields/")
    assert packager.tensor_digest(packed) == packager.tensor_digest(torch.load(selected, weights_only=True))
    catalog = json.loads((tmp_path / "bundle/package-manifest.json").read_text())
    assert catalog["context_fields"][0]["fingerprint"] == metadata["source_fingerprint"]
    moved = tmp_path / "relocated"; (tmp_path / "bundle").rename(moved)
    data.rename(tmp_path / "hidden-cache"); models.rename(tmp_path / "hidden-models"); sources.rename(tmp_path / "hidden-sources")
    actual = run_offline(moved, observed, tmp_path / "actual.json", tmp_path)
    expected = json.loads((tmp_path / "expected.json").read_text())
    np.testing.assert_allclose(actual["sampled_latlon_paths"], expected["sampled_latlon_paths"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("failure", ["member-hash", "member-traversal", "field-traversal", "cache-hash"])
def test_bad_hashes_and_dependency_traversal_never_publish_partial_bundle(tmp_path, failure):
    data = tmp_path / "cache"; hashes = reference(data)
    member = joint(tmp_path / "models/members/joint.pt", hashes)
    path = tmp_path / "models/ensemble.pt"
    record = {"path": str(member), "sha256": packager.sha256(member)}
    if failure == "member-hash": record["sha256"] = "0" * 64
    if failure == "member-traversal": record["path"] = "../outside.pt"
    if failure == "field-traversal":
        fields = tmp_path / "fields"; fields.mkdir()
        (fields / "manifest.json").write_text(json.dumps({"records": [{"key": "../../private-key"}]}))
        ck = torch.load(member, map_location="cpu", weights_only=True)
        ck["config"]["context"] = {"fields_root": str(fields), "fingerprint": "0" * 64}
        torch.save(ck, member); record["sha256"] = packager.sha256(member)
    torch.save({"model_kind": "wildcast_migration_ensemble", "members": [record], "weights": torch.ones(1), "data": str(data), **hashes}, path)
    if failure == "cache-hash":
        with (data / "cases.npz").open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum|traversal|hash|Unsafe"):
        packager.package(data, path, tmp_path / "bundle", source_root=tmp_path)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle-staging-*"))


def test_package_verifier_catches_corruption_and_new_only_guards(tmp_path):
    data = tmp_path / "cache"; hashes = reference(data)
    path = joint(tmp_path / "models/joint.pt", hashes)
    output = tmp_path / "bundle"
    packager.package(data, path, output, source_root=tmp_path)
    with (output / "wildcast/data.py").open("a") as stream: stream.write("\n# changed\n")
    result = subprocess.run([sys.executable, str(output / "verify_package.py")], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "checksum mismatch" in result.stderr
    with pytest.raises(FileExistsError): packager.package(data, path, output, source_root=tmp_path)
    sidecar = tmp_path / "existing.tar.gz.sha256.json"; sidecar.write_text("preserve me")
    with pytest.raises(ValueError, match="Archive must be a new"):
        packager.package(data, path, tmp_path / "new-bundle", archive=tmp_path / "existing.tar.gz", source_root=tmp_path)
    assert sidecar.read_text() == "preserve me"


def test_moveformer_bundle_includes_pinned_sources_and_licenses_and_runs_offline(tmp_path):
    pytest.importorskip("entmax")
    from wildcast.migration_moveformer import MigrationMoveFormer
    data = tmp_path / "cache"; hashes = reference(data)
    torch.manual_seed(4)
    model = MigrationMoveFormer(38, 18, 2, history_features=10, static_features=14, history_length=25,
                                candidates=8, width=8, layers=1, heads=2)
    source = tmp_path / "models/selected.pt"; source.parent.mkdir()
    torch.save({"model_kind": "wildcast_migration_moveformer", "model": model.state_dict(), "config": model.config,
                "upstream": model.upstream, **hashes}, source)
    observed = tmp_path / "observed.csv"; gps(observed)
    forecast(data, source, observed, tmp_path / "expected.json", "Known species", samples=6, seed=2026)
    packager.package(data, source, tmp_path / "bundle", source_root=tmp_path)
    assert (tmp_path / "bundle/third_party/moveformer/LICENSE").is_file()
    assert (tmp_path / "bundle/third_party/x-transformers/LICENSE").is_file()
    assert (tmp_path / "bundle/third_party/pins.json").is_file()
    assert not list((tmp_path / "bundle/third_party").rglob(".git"))
    moved = tmp_path / "relocated"; (tmp_path / "bundle").rename(moved)
    data.rename(tmp_path / "hidden-cache"); source.parent.rename(tmp_path / "hidden-models")
    actual = run_offline(moved, observed, tmp_path / "actual.json", tmp_path)
    expected = json.loads((tmp_path / "expected.json").read_text())
    np.testing.assert_allclose(actual["sampled_latlon_paths"], expected["sampled_latlon_paths"], rtol=0, atol=1e-9)


def regime_snapshot(root, mode="expert"):
    """Real frozen train-only rules and joint members; no held-out selection."""
    from wildcast import migration_regimes as regimes
    from wildcast.migration_data import HISTORY_NAMES, LAGS, STATIC_NAMES
    data = root / "cache"; reference(data)
    with np.load(data / "cases.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["history"][..., 2] = LAGS
    arrays["history"][..., 0] = np.arange(1, 5, dtype=np.float32)[:, None] / 1000
    arrays["history"][:, 0, 0] = 0
    np.savez_compressed(data / "cases.npz", **arrays)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest.update(cases_sha256=packager.sha256(data / "cases.npz"), history_names=HISTORY_NAMES,
                    static_names=STATIC_NAMES, history_lags_days=LAGS.tolist())
    (data / "manifest.json").write_text(json.dumps(manifest))
    arrays, manifest, hashes = load_cases(data)
    rules = regimes.fit_rules(arrays, manifest, hashes)
    rules_path = root / "frozen-rules.json"; rules_path.write_text(json.dumps(rules, indent=2))
    base = joint(root / "models/base.pt", hashes)
    active = joint(root / "models/active.pt", hashes, seed=17) if mode in {"expert", "calibrated"} else None
    calibration = None
    if mode == "calibrated":
        calibration = root / "finalized-scale"; calibration.mkdir()
        metadata = {"protocol": "wildcast-migration-active-covariance-final-v1", "family_selected": True,
                    "active_scale": 3., **hashes, "rules_sha256": packager.sha256(rules_path),
                    "checkpoint_sha256": packager.sha256(active), "independent_result": False,
                    "reserved_cases_scored": 0, "fit_split": "synthetic_selected_validation_HIGH"}
        (calibration / "scale.json").write_text(json.dumps(metadata, indent=2))
    regimes.assemble(data, rules_path, base, root / "models/regime", active=active,
                     active_fallback="base" if mode == "base" else "seasonal",
                     active_scale=3. if active is not None else 1., calibration=calibration)
    return data, root / "models/regime/best.pt", rules_path, rules


@pytest.mark.parametrize("mode", ["expert", "seasonal", "base", "calibrated"])
def test_regime_bundle_preserves_frozen_routing_and_relocated_offline_forecasts(tmp_path, mode):
    from wildcast import migration_regimes as regimes
    from wildcast.migration_inference import _observations, _query_view
    data, selected, rules_path, rules = regime_snapshot(tmp_path, mode)
    arrays, manifest, _ = load_cases(data)
    original = torch.load(selected, map_location="cpu", weights_only=True)
    profiles = [("moving", 40, .02, "high_past_mobility")]
    if mode == "expert":
        profiles += [("ordinary", 40, 0., "ordinary_past_mobility"),
                     ("partial", 20, .02, "missing_past_history")]
    expected, inputs = {}, {}
    for name, count, speed, expected_regime in profiles:
        observed = tmp_path / (name + ".csv")
        pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=count, tz="UTC"),
                      "lat": 30. + np.arange(count) * speed,
                      "lon": 20. + np.arange(count) * speed, "animal": "new"}).to_csv(observed, index=False)
        frame, animal, _, _ = _observations(observed, None, None)
        view, query, *_ = _query_view(arrays, manifest, frame, "Known species", None, animal)
        assert regimes.classify(view, [query], rules)["regime"].tolist() == [expected_regime]
        output = tmp_path / ("expected-" + name + ".json")
        forecast(data, selected, observed, output, "Known species", samples=6, seed=2026)
        expected[name] = json.loads(output.read_text())
        inputs[name] = observed
    packager.package(data, selected, tmp_path / "bundle", source_root=tmp_path)
    bundled = tmp_path / "bundle"
    packed = torch.load(bundled / "model/selected.pt", map_location="cpu", weights_only=True)
    catalogue = json.loads((bundled / "package-manifest.json").read_text())
    assert packed["data"] == "data/reference" and not Path(packed["rules_path"]).is_absolute()
    assert (bundled / packed["rules_path"]).read_bytes() == rules_path.read_bytes()
    assert packager.regime_policy_digest(packed) == packager.regime_policy_digest(original)
    assert catalogue["checkpoint"]["predictive_policy_sha256"] == packager.regime_policy_digest(original)
    assert packed["rules"] == original["rules"] and packed["active_scale"] == original["active_scale"]
    assert packed["active_fallback"] == original["active_fallback"]
    if original.get("calibration"):
        assert not Path(packed["calibration"]["path"]).is_absolute()
        assert (bundled / packed["calibration"]["path"]).read_bytes() == Path(original["calibration"]["path"]).read_bytes()
        assert {k: v for k, v in packed["calibration"].items() if k != "path"} == {k: v for k, v in original["calibration"].items() if k != "path"}
    for name in ("base", "active"):
        if original[name] is None:
            assert packed[name] is None
            continue
        assert not Path(packed[name]["path"]).is_absolute()
        assert packager.sha256(bundled / packed[name]["path"]) == original[name]["sha256"]
        assert {k: v for k, v in original[name].items() if k != "path"} == {k: v for k, v in packed[name].items() if k != "path"}
    moved = tmp_path / "relocated regime bundle"; bundled.rename(moved)
    data.rename(tmp_path / "hidden-cache"); selected.parents[1].rename(tmp_path / "hidden-models")
    rules_path.rename(tmp_path / "hidden-rules.json")
    if original.get("calibration"):
        Path(original["calibration"]["path"]).parent.rename(tmp_path / "hidden-finalized-scale")
    for name, observed in inputs.items():
        actual = run_offline(moved, observed, tmp_path / ("actual-" + name + ".json"), tmp_path)
        np.testing.assert_allclose(actual["sampled_latlon_paths"], expected[name]["sampled_latlon_paths"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("failure", ["rules-hash", "inline-rules", "rules-traversal", "member-hash", "member-kind", "recursive-member", "scale"])
def test_regime_corrupt_rules_members_or_policy_never_publish(tmp_path, failure):
    data, selected, rules_path, _ = regime_snapshot(tmp_path)
    ck = torch.load(selected, map_location="cpu", weights_only=True)
    if failure == "rules-hash":
        ck["rules_sha256"] = "0" * 64
    elif failure == "inline-rules":
        ck["rules"]["thresholds_km"]["1"][0] += 1
    elif failure == "rules-traversal":
        ck["rules_path"] = "../frozen-rules.json"
    elif failure == "member-hash":
        ck["base"]["sha256"] = "0" * 64
    elif failure == "member-kind":
        ck["base"]["model_kind"] = "wildcast_migration_retrieval"
    elif failure == "recursive-member":
        # Lying in the member record must not allow a nested wrapper on disk.
        ck["base"]["path"] = str(selected)
        ck["base"]["sha256"] = packager.sha256(selected)
    else:
        ck["active_scale"] = 1.7
    # A separate mutated snapshot preserves the nested-wrapper source checksum.
    invalid = selected.parent / "invalid.pt"; torch.save(ck, invalid)
    with pytest.raises(ValueError, match="checksum|equality|traversal|joint|recursive|policy"):
        packager.package(data, invalid, tmp_path / "bundle", source_root=tmp_path)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle-staging-*"))


@pytest.mark.parametrize("failure", ["file-hash", "inline-metadata", "active-member", "scale-policy", "traversal"])
def test_calibrated_regime_rejects_corrupt_or_mismatched_evidence(tmp_path, failure):
    data, selected, _, _ = regime_snapshot(tmp_path, "calibrated")
    ck = torch.load(selected, map_location="cpu", weights_only=True)
    evidence = ck["calibration"]
    if failure == "file-hash":
        evidence["sha256"] = "0" * 64
    elif failure == "inline-metadata":
        evidence["metadata"]["active_scale"] = 5.
    elif failure == "traversal":
        evidence["path"] = "../scale.json"
    else:
        metadata = evidence["metadata"]
        if failure == "active-member": metadata["checkpoint_sha256"] = "0" * 64
        else: metadata["active_scale"] = 5.
        path = Path(evidence["path"]); path.write_text(json.dumps(metadata))
        evidence["sha256"] = packager.sha256(path)
    torch.save(ck, selected)
    with pytest.raises(ValueError, match="checksum|equality|policy|provenance|traversal"):
        packager.package(data, selected, tmp_path / "bundle", source_root=tmp_path)
    assert not (tmp_path / "bundle").exists()
