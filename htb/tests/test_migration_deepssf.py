"""Author pinning, continuous-law normalization and future-free short endpoints."""
import json
import math
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.integrate import quad

from wildcast import migration_deepssf as deep
from wildcast import multiscale_data as protocol
from wildcast import multiscale_repaired as repaired_protocol
from wildcast.data import R, unproject


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def arrays(schema=protocol):
    n, h = 5, len(schema.HORIZONS)
    times = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-06-01", "2021-06-01", "2022-06-01"], utc=True).asi8 / 1e9
    origin = np.column_stack([np.zeros(n), np.zeros(n), times])
    history = np.zeros((n, len(schema.LAGS), len(schema.HISTORY_NAMES)), "float32")
    mask = np.zeros(history.shape[:2], bool); mask[:, :2] = True
    history[:, 1, 0], history[:, 1, 2] = -.0001, schema.LAGS[1]
    y = np.zeros((n, h, 2), "float32")
    y[:, :3, 0] = np.asarray([.0001, .0005, .001])
    target_mask = np.zeros((n, h), bool); target_mask[:4, :3] = True
    coords = unproject(y.astype(float) * 1_000_000, 0., 0.); coords[~target_mask] = 0
    target_time = times[:, None] + schema.HORIZONS * 86400; target_time[~target_mask] = 0
    return dict(history=history, history_mask=mask, static=np.zeros((n, len(schema.STATIC_NAMES)), "float32"),
                species=np.asarray([1, 2, 0, 1, 0], "int64"), metric_species=np.asarray([0, 1, 2, 0, 2], "int64"),
                individual=np.asarray([10, 11, 20, 10, 30], "int64"), study=np.asarray([0, 0, 1, 0, 1], "int64"),
                origin=origin, split=np.asarray([0, 0, 1, 2, 3], "int8"), y=y, target_mask=target_mask,
                target_latlon=coords, target_time=target_time, horizons=schema.HORIZONS.copy())


def small_model(a=None, **kwargs):
    a = arrays() if a is None else a
    geometry = deep.fit_geometry(a, window=17)
    return deep.DeepSSFShort(geometry, window=17, channels=2, hidden=8, embedding=2, dropout=0, **kwargs)


def write_cache(path, schema=protocol):
    """A fabricated fixture, including schema declarations rather than real repair evidence."""
    path.mkdir(); a = arrays(schema)
    np.savez_compressed(path / "cases.npz", **a)
    animals = {"train-a": 10, "train-b": 11, "validation": 20, "reserved": 30}
    manifest = dict(protocol=schema.PROTOCOL, cases_sha256=deep.sha256(path / "cases.npz"),
        horizons_days=schema.HORIZONS.tolist(), history_names=schema.HISTORY_NAMES,
        static_names=schema.STATIC_NAMES, history_lags_days=schema.LAGS.tolist(), unit_metres=schema.UNIT,
        assessment_target_matcher=schema.__name__ + ".match_targets", minimum_history_lags=2,
        minimum_history_prior_fixes=8, minimum_history_span_days=.25, minimum_observed_future_targets=1,
        reserved_test_targets_sealed=True, train_end_year=2020, validation_year=2021,
        cutoff_boundaries=["2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"],
        animals=animals, subday_clock_excluded_animals=[], subday_clock_modes={name: "declared_utc" for name in animals},
        species={"a": 1, "b": 2}, metric_species={"a": 0, "b": 1, "c": 2})
    if schema is repaired_protocol:
        proof = {"kind": "synthetic_fixture_not_real_repair", "sha256": "a" * 64}
        manifest.update(physical_repair=proof,
            parent_cache={"protocol": "wildcast-parent-physical-repair-accounted-v1",
                "physical_repair": proof, "case_accounting_sha256": "b" * 64,
                "accounted_parent_cases": len(a["split"]),
                "case_status_counts": {"retained_unaffected": len(a["split"])}},
            sources=[{"sha256": "c" * 64}], observation_source_rules=[{
                "sha256": "c" * 64, "allowed_raw_animals": list(animals), "excluded_raw_animals": []}])
    (path / "manifest.json").write_text(json.dumps(manifest))
    return a


def test_training_schema_rejects_unknown_or_incomplete_repair_and_wrong_dimensions(tmp_path):
    a = write_cache(tmp_path / "data", repaired_protocol)
    manifest = json.loads((tmp_path / "data/manifest.json").read_text())
    assert deep.training_schema(manifest, a) is repaired_protocol
    with pytest.raises(ValueError, match="known frozen"):
        deep.training_schema({**manifest, "protocol": "wildcast-multiscale-unreviewed-v99"}, a)
    with pytest.raises(ValueError, match="physical-repair"):
        deep.training_schema({k: v for k, v in manifest.items() if k != "physical_repair"}, a)
    a["history"] = a["history"][:, :-2]
    with pytest.raises(ValueError, match="history dimensions"):
        deep.training_schema(manifest, a)


def test_author_classes_are_pinned_and_mutation_is_detected_after_cache(tmp_path, monkeypatch):
    scope = deep.author_core()
    assert scope["ConvJointModel"].__module__ == "wildcast.deepssf_author"
    assert deep.provenance()["commit"] == "4eef65e021de8384b570b079bd34fa3f90d949da"
    assert deep.provenance()["official_MoveBench_reproduction"] is False
    source = tmp_path / "third_party/deepssf"
    shutil.copytree(deep.PROJECT / "third_party/deepssf", source)
    with (source / "model.py").open("a") as stream:
        stream.write("\n# changed\n")
    monkeypatch.setattr(deep, "PROJECT", tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        deep.author_core()


def test_geometry_ignores_validation_and_sealed_values_and_unknown_uses_global():
    a = arrays(); first = deep.fit_geometry(a, window=17)
    changed = {k: v.copy() for k, v in a.items()}
    changed["y"][2:] = np.nan
    changed["target_mask"][2:] = True
    assert deep.fit_geometry(changed, window=17) == first
    assert len(first["cell_metres"]) == 3 and first["fit_split"] == "train"
    assert first["training_observations"][0] == [2, 2, 2]
    model = small_model(a)
    a["species"][2] = 999
    assert model._tokens(a, [2]).tolist() == [0]
    with pytest.raises(ValueError, match="antipodal"):
        deep.fit_geometry(a, maximum_half_width=deep.MAX_RADIUS)


def test_validation_panel_is_fixed_balanced_metadata_and_never_selects_reserved():
    a = {"horizons":np.asarray(deep.SUPPORTED_HORIZONS), "target_mask":np.ones((8,3),bool),
         "split":np.array([0,3,1,4,2,1,0,2]), "species":np.ones(8,int),
         "metric_species":np.array([0,0,0,1,0,1,2,0]),
         "individual":np.array([1,1,10,11,1,2,30,1]), "y":np.zeros((8,3,2))}
    first = deep.validation_panel(a,3)
    a["y"][:] = np.nan
    np.testing.assert_array_equal(first,deep.validation_panel(a,3))
    assert len(first) == 3 and set(a["split"][first]) <= {1,2}
    assert len(np.unique(a["individual"][first])) == 3
    usable = np.flatnonzero(a["split"] < 3)
    subset = {k:v if k == "horizons" else v[usable] for k,v in a.items()}
    np.testing.assert_array_equal(usable[deep.validation_panel(subset,3)],first)


def test_unknown_dropout_trains_pooled_geometry_and_embedding_together(monkeypatch):
    a = arrays(); a["y"][0, :3, 0] = .1
    model = small_model(a)
    assert model.cell_metres[0, 0] != model.cell_metres[2, 0]
    monkeypatch.setattr(torch, "rand", lambda count, device: torch.zeros(count, device=device))
    inputs, cell, scale = model.inputs(a, [1], [0])
    assert inputs[-1].tolist() == [0]
    assert cell.tolist() == model.cell_metres[[0], [0]].tolist()
    assert scale.tolist() == model.tail_scale_metres[[0], [0]].tolist()
    model.eval()
    inputs, cell, _ = model.inputs(a, [1], [0])
    assert inputs[-1].tolist() == [2]
    assert cell.tolist() == model.cell_metres[[2], [0]].tolist()


def test_only_declared_cross_taxon_exclusion_blocks_use_within_species_weights():
    a = arrays(); a["individual"][1] = 10
    a["y"][1, :3, 0] = .01
    original_ids = a["individual"].copy()
    with pytest.raises(ValueError, match="undeclared"):
        deep.fit_geometry(a)
    first = deep.fit_geometry(a, allowed_cross_taxon_exclusion_ids={10})
    assert first["allowed_cross_taxon_exclusion_ids"] == [10]
    np.testing.assert_array_equal(a["individual"], original_ids)
    # Duplicate only one taxon within the same uncertain block. Its within-
    # species origin count changes, not that taxon's geometry fitting mass.
    a["split"][2], a["individual"][2], a["species"][2], a["metric_species"][2] = 0, 10, 2, 1
    a["y"][2], a["target_mask"][2] = a["y"][1], a["target_mask"][1]
    second = deep.fit_geometry(a, allowed_cross_taxon_exclusion_ids={10})
    assert first["cell_metres"] == second["cell_metres"]
    assert first["tail_scale_metres"] == second["tail_scale_metres"]


@pytest.mark.parametrize("scale", [100., 1_000_000., 100_000_000.])
def test_tail_density_integrates_to_one_on_exact_truncation_disc(scale):
    def radial_density(radius):
        logp = deep.tail_log_density(torch.tensor([[radius, 0.]], dtype=torch.float64), torch.tensor([scale], dtype=torch.float64))
        return 2 * math.pi * radius * math.exp(float(logp))
    boundaries = sorted(set([0., min(scale, deep.MAX_RADIUS), min(scale * 10, deep.MAX_RADIUS), deep.MAX_RADIUS]))
    mass = sum(quad(radial_density, left, right, epsabs=1e-9)[0] for left, right in zip(boundaries[:-1], boundaries[1:]))
    assert mass == pytest.approx(1, abs=2e-8)
    assert torch.isneginf(deep.tail_log_density(torch.tensor([[deep.MAX_RADIUS + 1, 0.]]), torch.tensor([scale]))).all()


def test_tail_sampling_matches_truncated_cdf_without_boundary_point_masses():
    scale = 1_000_000.
    sample = deep.sample_tail([scale], 100_000, np.random.default_rng(17))[0]
    radial = np.linalg.norm(sample, axis=-1)
    z = 1 - (1 + deep.MAX_RADIUS**2 / (deep.TAIL_DF * scale**2)) ** (-deep.TAIL_DF/2)
    for threshold in [.5 * scale, scale, 3 * scale]:
        expected = (1 - (1 + threshold**2 / (deep.TAIL_DF * scale**2)) ** (-deep.TAIL_DF/2)) / z
        assert (radial < threshold).mean() == pytest.approx(expected, abs=.006)
    assert 0 < radial.min() and radial.max() < deep.MAX_RADIUS
    assert len(np.unique(radial)) == len(radial)
    assert (sample[:, 0] > 0).mean() == pytest.approx(.5, abs=.006)


def test_cell_uniform_density_and_geographic_jacobian_are_explicit():
    logp = torch.full((2, 17, 17), -math.log(17 * 17), dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[0., 0.], [math.pi * R / 2, 0.]], dtype=torch.float64)
    cell, scale = torch.tensor([100., 100.]), torch.tensor([1000., 1000.])
    plane, inside = deep.mixture_nll(logp, y, cell, scale, geographic=False)
    geographic, _ = deep.mixture_nll(logp, y, cell, scale, geographic=True)
    assert inside.tolist() == [True, False]
    expected = .9 / (17 * 17 * 100**2) + .1 * math.exp(float(deep.tail_log_density(y[:1], scale[:1])))
    assert float(plane[0].detach()) == pytest.approx(-math.log(expected), abs=1e-10)
    assert float(geographic[0].detach()) == float(plane[0].detach())
    assert float((geographic[1] - plane[1]).detach()) == pytest.approx(math.log(2 / math.pi), abs=1e-12)
    geographic.mean().backward()
    assert torch.isfinite(logp.grad).all() and logp.grad[0].abs().sum() > 0
    assert logp.grad[1].abs().sum() == 0  # Outside outcomes train the declared fixed tail, not an invented raster label.


def test_sampling_mixture_has_continuous_cells_and_correct_tail_mass():
    a = arrays(); a["y"][:2, :3, 0] = .0015
    m = small_model(a).eval()

    class Uniform(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((len(x[0]), 17, 17, 2), device=x[0].device)

    m.core = Uniform()
    samples = m.predict_endpoints(a, [0], [1/24], 60_000, np.random.default_rng(8))[0, :, 0] * 1_000_000
    cell = float(m.cell_metres[1, 0]); scale = float(m.tail_scale_metres[1, 0])
    threshold = 17 * cell
    radial = np.linalg.norm(samples, axis=-1)
    z = 1 - (1 + deep.MAX_RADIUS**2 / (deep.TAIL_DF * scale**2)) ** (-deep.TAIL_DF / 2)
    expected = deep.TAIL_WEIGHT * (1 - (1 - (1 + threshold**2 / (deep.TAIL_DF * scale**2)) ** (-deep.TAIL_DF / 2)) / z)
    assert (radial > threshold).mean() == pytest.approx(expected, abs=.003)
    half_width = 17 * cell / 2
    central = half_width / 2
    def tail_cdf(radius):
        return (1 - (1 + radius**2 / (deep.TAIL_DF * scale**2)) ** (-deep.TAIL_DF / 2)) / z
    tail_square = 4 / math.pi * quad(lambda angle: tail_cdf(central / math.cos(angle)), 0., math.pi / 4)[0]
    expected_central = (1 - deep.TAIL_WEIGHT) * .25 + deep.TAIL_WEIGHT * tail_square
    observed_central = (np.abs(samples).max(-1) < central).mean()
    assert observed_central == pytest.approx(expected_central, abs=.006)
    # Neither raster support boundaries nor cell centres carry atomic mass.
    assert np.max(radial) < deep.MAX_RADIUS
    assert not np.any(np.isclose(samples[:, 0] / cell % 1, 0, atol=1e-12))


def test_prediction_never_reads_query_targets_and_dt_is_requested_horizon():
    a = arrays(); model = small_model(a).eval()

    class PredictorOnly(dict):
        def __getitem__(self, key):
            if key in {"y", "target_mask", "target_latlon", "target_time"}:
                raise AssertionError("Queried a future outcome")
            return super().__getitem__(key)

    safe = PredictorOnly(a)
    scalars, _ = deep.predictor_scalars(safe, [2, 3], [1/24, .5])
    assert scalars[:, 4].tolist() == [1., 12.]
    before = model.predict_endpoints(safe, [2, 3], [1/24, .5, 1], 12, np.random.default_rng(17))
    changed = PredictorOnly({k: v.copy() for k, v in a.items()})
    changed["history"][~changed["history_mask"]] = np.nan
    for name in ("y", "target_time", "target_latlon"):
        dict.__getitem__(changed, name)[:] = np.inf
    after = model.predict_endpoints(changed, [2, 3], [1/24, .5, 1], 12, np.random.default_rng(17))
    np.testing.assert_array_equal(before, after)
    assert before.shape == (2, 12, 3, 2) and np.isfinite(before).all()
    with pytest.raises(ValueError, match="no 90-day route"):
        model.predict_cases(safe, [2], 4)
    with pytest.raises(ValueError, match="strictly increasing"):
        model.predict_endpoints(safe, [2], [1., 1/24])


def test_all_observed_short_training_pairs_are_retained_including_outside():
    a = arrays(); a["y"][0, 0, 0] = .05
    model = small_model(a)
    rows, heads, columns = deep.observed_pairs(a, [0])
    assert len(rows) == 6 and set(rows) == {0, 1}
    loss, inside = deep.loss_for_pairs(model, a, rows, heads, columns)
    assert torch.isfinite(loss).all() and len(loss) == 6
    # A predeclared 1km minimum may expand on TRAIN only; force a fixed smaller support for this test.
    model.cell_metres.fill_(10.)
    loss, inside = deep.loss_for_pairs(model, a, rows, heads, columns)
    assert (~inside).any() and torch.isfinite(loss).all()
    loss.mean().backward()
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    with pytest.raises(ValueError, match="reserved"):
        deep.observed_pairs(a, [3])
    with pytest.raises(ValueError, match="reserved truth"):
        deep.loss_for_pairs(model, a, [4], [0], [0])
    with pytest.raises(ValueError, match="requested horizon"):
        deep.loss_for_pairs(model, a, [0], [0], [2])


@pytest.fixture
def ghsl(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    root = tmp_path / "fields"; root.mkdir()
    records = []
    for epoch, value in ((2000, 200_000), (2005, 800_000)):
        raster = root / f"{epoch}.tif"
        values = np.full((64, 64), value, "uint32"); values[32, 32] = 0
        with rasterio.open(raster, "w", driver="GTiff", width=64, height=64, count=1, dtype="uint32",
                           crs="ESRI:54009", transform=from_origin(-32000, 32000, 1000, 1000), nodata=4294967295) as ds:
            ds.write(values, 1)
        archive = root / f"{epoch}.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.write(raster, raster.name)
        records.append({"epoch": epoch, "path": str(archive), "sha256": deep.sha256(archive), "raster": str(raster)})
    (root / "sources.json").write_text(json.dumps({"records": records}))
    return root


def test_on_demand_GHSL_prior_vintage_missingness_and_controls(ghsl):
    context = deep.GHSLRaster(ghsl)
    def year(y):
        return pd.Timestamp(f"{y}-07-01", tz="UTC").timestamp()
    origins = np.asarray([[0., 0., year(2005)], [40., 20., year(2006)], [0., 0., year(2000)], [0., 0., year(2026)]])
    patch = context.patches(origins, [1000.] * 4, 17)
    assert patch.shape == (4, 6, 17, 17)
    assert patch[0, 0, 8, 8] == 0 and patch[0, 1, 8, 8] == 1  # observed zero
    assert patch[1, 0, 8, 8] == 0 and patch[1, 1, 8, 8] == 0  # outside map, not zero development
    assert patch[0, 0].max() == pytest.approx(.2)
    assert patch[3, 0].max() == pytest.approx(.8)
    assert patch[0, 2, 8, 8] == pytest.approx(5/25)  # 2000, not contemporaneous 2005
    assert not patch[2].any() and not patch[0, 4].any()
    assert patch[3, 4].all() and patch[3, 5].all()
    control = context.patches(origins, [1000.] * 4, 17, "availability")
    np.testing.assert_array_equal(control[:, 1:], patch[:, 1:]); assert not control[:, 0].any()
    assert not context.patches(origins, [1000.] * 4, 17, "masked").any()
    assert len(context.context.tiles) <= 8
    fingerprint = context.fingerprint
    context.close()
    # Original absolute paths disappear; unchanged metadata plus verified local
    # original/raster basenames retain the semantic fingerprint and predictions.
    relocated = ghsl.with_name("relocated-fields")
    shutil.move(ghsl, relocated)
    restored = deep.GHSLRaster(relocated, expected=fingerprint)
    np.testing.assert_array_equal(restored.patches(origins, [1000.] * 4, 17), patch)
    restored.close()


@pytest.mark.parametrize("schema", [protocol, repaired_protocol])
def test_synthetic_finite_fit_checkpoint_reload_and_horizon_rejection(tmp_path, schema):
    a = write_cache(tmp_path / "data", schema)
    result = deep.train(tmp_path / "data", tmp_path / "fit", epochs=1, batch_size=8,
                        window=17, channels=2, hidden=8, embedding=2, validation_samples=4, seed=4)
    assert result["epochs_observed"] == 2 and math.isfinite(result["best_endpoint_energy_km"])
    model, ck = deep.load_model(tmp_path / "fit/last.pt")
    assert ck["epoch"] == 1 and ck["independent_horizon_draws"] is True
    assert ck["cache_protocol"] == schema.PROTOCOL
    assert ck["feature_schema"]["history_lags_days"] == schema.LAGS.tolist()
    assert ck["feature_schema"]["static_names"] == schema.STATIC_NAMES
    if schema is repaired_protocol:
        assert {"multiscale_repaired.py", "migration_physical_repair.py", "migration_repaired_features.py"} <= ck["fitting_code_sha256"].keys()
        assert a["history"].shape[1:] == (29, 10) and a["static"].shape[1] == 22
    assert ck["training_pairs"] == 6 and ck["tail"]["weight"] == .1
    assert ck["supported_horizons_days"] == list(deep.SUPPORTED_HORIZONS)
    assert ck["metrics"]["selection_version"] == deep.SELECTION_VERSION
    assert {r["split"] for r in ck["metrics"]["split_metrics"]} == {"val_individual", "val_year"}
    initial = torch.load(tmp_path / "fit/initial.pt", weights_only=True)
    assert any(not torch.equal(initial["model"][k], ck["model"][k]) for k in ck["model"] if "weight" in k)
    before = model.predict_endpoints(a, [2, 3], [1/24, .5], 6, np.random.default_rng(1))
    shutil.copy(tmp_path / "fit/last.pt", tmp_path / "relocated.pt")
    relocated, _ = deep.load_model(tmp_path / "relocated.pt")
    np.testing.assert_array_equal(before, relocated.predict_endpoints(a, [2, 3], [1/24, .5], 6, np.random.default_rng(1)))
    with pytest.raises(ValueError, match="no 90-day route"):
        model.predict_cases(a, [2], 4)
    ck["model"]["cell_metres"] *= 2
    torch.save(ck, tmp_path / "wrong-geometry.pt")
    with pytest.raises(ValueError, match="geometry state"):
        deep.load_model(tmp_path / "wrong-geometry.pt")
    ck["author"]["commit"] = "incorrect"
    torch.save(ck, tmp_path / "wrong-source.pt")
    with pytest.raises(ValueError, match="provenance"):
        deep.load_model(tmp_path / "wrong-source.pt")
