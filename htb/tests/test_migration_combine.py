import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("migration_combine", Path(__file__).resolve().parents[1] / "scripts/migration_combine.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
combine = _module.combine


def write_report(root, learned, digest):
    root.mkdir()
    rows, joints = [], []
    for model in ("seasonal", learned):
        offset = 0 if model == "seasonal" else -1 if learned == "model-a" else -2
        for case in range(6):
            local = case % 3
            base = {"model": model, "case": case, "split": "val_individual" if case < 3 else "val_year",
                    "individual": (10 if local < 2 else 11) if case < 3 else (0 if local < 2 else 1),
                    "species": 0 if local < 2 else 1, "prediction_species": 0, "study": 0,
                    "displacement_proxy": "large_displacement_proxy"}
            joints.append({**base, "observed_horizons": 2, "joint_chordal_energy_km": 20. + case + offset})
            for h in (1., 7.):
                elapsed = h + .01
                rows.append({**base, "horizon_days": h, "actual_elapsed_days": elapsed,
                             "target_timing_error_hours": (elapsed - h) * 24,
                             "endpoint_energy_km": 10. + case + h / 10 + offset,
                             "point_error_km": 12. + case + offset, "representative_error_km": 13. + case + offset,
                             "radial_coverage50": False, "radial_coverage90": True,
                             "within_1km": False, "within_10km": False, "within_100km": True})
    frame, joint = pd.DataFrame(rows), pd.DataFrame(joints)
    frame.to_parquet(root / "per-case.parquet", index=False)
    joint.to_parquet(root / "joint-per-origin.parquet", index=False)
    timing(root, frame)
    metadata = {"manifest_sha256": "manifest-v3", "cases_sha256": "cases-v3",
                "case_ids_sha256": hashlib.sha256(np.arange(6, dtype="<i8").tobytes()).hexdigest(),
                "samples": 64, "seed": 2026, "batch_size": 64, "cases": 6, "observed_targets": 12,
                "splits": ["val_individual", "val_year"], "models": ["seasonal", learned],
                "scope": "Local migration development", "assessment": None,
                "identity_unit": "Physical exclusion groups", "notes": ["identical scoring protocol"],
                "code_sha256": {"migration_baselines.py": "baseline-v1", "migration_evaluate.py": "scorer-v1"},
                "checkpoints": [{"name": learned, "path": f"runs/{learned}/best.pt", "sha256": digest,
                                 "model_kind": "wildcast_migration_joint", "selection_version": "physical-v3",
                                 "selected_epoch": 3, "code_sha256": "model-source-v1"}]}
    (root / "metrics.json").write_text(json.dumps(metadata))
    (root / "notes.txt").write_text("retained source evidence")
    (root / "representative-routes.json").write_text(json.dumps({"routes": [{"model": "seasonal", "case": 0,
        "split": "val_individual", "representative_latlon": [[0., 0.]], "observed_future": []}]}))
    return metadata


def timing(root, frame):
    name = frame.model.iloc[0]
    table = frame.loc[frame.model == name].groupby(["split", "horizon_days"]).target_timing_error_hours.agg(
        cases="size", mean_hours="mean", minimum_hours="min", maximum_hours="max").reset_index()
    table.to_csv(root / "target-timing.csv", index=False)


def metadata_change(root, update):
    path = root / "metrics.json"; m = json.loads(path.read_text()); update(m)
    path.write_text(json.dumps(m))


def test_exact_baseline_dedup_recomputes_macro_scores_and_paired_intervals(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_report(a, "model-a", "sha-a"); write_report(b, "model-b", "sha-b")
    output = tmp_path / "combined"
    result = combine([a, b], output, bootstrap_replicates=12)
    assert result["models"] == ["seasonal", "model-a", "model-b"]
    frame = pd.read_parquet(output / "per-case.parquet")
    assert len(frame) == 36 and len(frame.loc[frame.model == "seasonal"]) == 12
    scores = pd.read_csv(output / "scores.csv").set_index(["model", "split"])
    assert scores.loc[("seasonal", "val_individual"), "endpoint_energy_km"] == pytest.approx(11.65)
    assert scores.loc[("model-b", "val_individual"), "endpoint_energy_km"] == pytest.approx(9.65)
    ci = result["paired_animal_bootstrap"]["splits"]["val_year"]["model-b"]["ci95_km"]
    np.testing.assert_allclose(ci, [-2., -2.])
    assert result["cases"] == 6 and result["observed_targets"] == 12
    assert "notes.txt" in result["source_evaluations"][0]["files_sha256"]
    assert pd.read_csv(output / "target-timing.csv").cases.sum() == 12
    assert len(json.loads((output / "representative-routes.json").read_text())["routes"]) == 1


@pytest.mark.parametrize("key,value", [("seed", 2027), ("samples", 32), ("batch_size", 32), ("cases_sha256", "other-cases"),
                                        ("manifest_sha256", "other-manifest"),
                                        ("splits", ["val_year", "val_individual"])])
def test_different_draw_or_dataset_protocol_is_rejected(tmp_path, key, value):
    a, b = tmp_path / "a", tmp_path / "b"
    write_report(a, "model-a", "sha-a"); write_report(b, "model-b", "sha-b")
    metadata_change(b, lambda m: m.update({key: value}))
    with pytest.raises(ValueError, match=key):
        combine([a, b], tmp_path / "combined", bootstrap_replicates=4)


@pytest.mark.parametrize("value", [None, 0, 64.5])
def test_unpinned_rng_batch_partition_is_rejected(tmp_path, value):
    source = tmp_path / "a"; write_report(source, "model-a", "sha-a")
    metadata_change(source, lambda m: m.update(batch_size=value))
    with pytest.raises(ValueError, match="batch_size"):
        combine([source], tmp_path / "combined", bootstrap_replicates=4)


def test_missing_calibration_cannot_silently_change_macro_denominators(tmp_path):
    source = tmp_path / "a"; write_report(source, "model-a", "sha-a")
    table = pd.read_parquet(source / "per-case.parquet")
    table["radial_coverage90"] = table.radial_coverage90.astype(float)
    table.loc[0, "radial_coverage90"] = np.nan
    table.to_parquet(source / "per-case.parquet", index=False)
    with pytest.raises(ValueError, match="complete binary"):
        combine([source], tmp_path / "combined", bootstrap_replicates=4)


def test_same_counts_but_different_target_support_is_rejected(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_report(a, "model-a", "sha-a"); write_report(b, "model-b", "sha-b")
    frame = pd.read_parquet(b / "per-case.parquet")
    selected = (frame.case == 5) & (frame.horizon_days == 7)
    frame.loc[selected, "horizon_days"] = 14.
    frame.loc[selected, "actual_elapsed_days"] = 14.01
    frame.loc[selected, "target_timing_error_hours"] = (14.01 - 14) * 24
    frame.to_parquet(b / "per-case.parquet", index=False); timing(b, frame)
    with pytest.raises(ValueError, match="support"):
        combine([a, b], tmp_path / "combined", bootstrap_replicates=4)


def test_baseline_values_and_provenance_must_be_exact(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_report(a, "model-a", "sha-a"); write_report(b, "model-b", "sha-b")
    frame = pd.read_parquet(b / "per-case.parquet")
    frame.loc[(frame.model == "seasonal") & (frame.case == 0), "endpoint_energy_km"] += 1e-10
    frame.to_parquet(b / "per-case.parquet", index=False)
    with pytest.raises(ValueError, match="baseline rows differ"):
        combine([a, b], tmp_path / "different-values", bootstrap_replicates=4)
    write_report(tmp_path / "c", "model-b", "sha-b")
    metadata_change(tmp_path / "c", lambda m: m["code_sha256"].update({"migration_baselines.py": "changed"}))
    with pytest.raises(ValueError, match="baseline provenance"):
        combine([a, tmp_path / "c"], tmp_path / "different-provenance", bootstrap_replicates=4)


def test_duplicate_learned_model_or_checkpoint_version_is_rejected(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_report(a, "model-a", "sha-a"); write_report(b, "model-a", "sha-a")
    with pytest.raises(ValueError, match="Learned model names"):
        combine([a, b], tmp_path / "duplicate", bootstrap_replicates=4)
    write_report(tmp_path / "c", "model-b", "sha-a")
    metadata_change(tmp_path / "c", lambda m: m["checkpoints"][0].update(selection_version="changed-version"))
    with pytest.raises(ValueError, match="version/provenance"):
        combine([a, tmp_path / "c"], tmp_path / "version", bootstrap_replicates=4)


def test_unfrozen_test_reports_are_rejected(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_report(a, "model-a", "sha-a"); write_report(b, "model-b", "sha-b")
    for root in (a, b):
        for name in ("per-case.parquet", "joint-per-origin.parquet"):
            frame = pd.read_parquet(root / name)
            frame["split"] = frame["split"].str.replace("val_", "test_")
            frame.to_parquet(root / name, index=False)
        frame = pd.read_parquet(root / "per-case.parquet"); timing(root, frame)
        metadata_change(root, lambda m: m.update(splits=["test_individual", "test_year"]))
        (root / "representative-routes.json").write_text(json.dumps({"routes": []}))
    with pytest.raises(ValueError, match="frozen final assessment"):
        combine([a, b], tmp_path / "forbidden", bootstrap_replicates=4)
    for root in (a, b):
        metadata_change(root, lambda m: m.update(scope="Frozen local migration final assessment",
                         assessment={"sha256": "assessment-hash", "selection_sha256": "frozen-selection-hash"}))
    result = combine([a, b], tmp_path / "frozen-combined", bootstrap_replicates=4)
    assert result["assessment"]["selection_sha256"] == "frozen-selection-hash"
    assert set(result["paired_animal_bootstrap"]["splits"]) == {"test_individual", "test_year"}
    metadata_change(b, lambda m: m["assessment"].update(selection_sha256="different-selection"))
    with pytest.raises(ValueError, match="frozen final assessment"):
        combine([a, b], tmp_path / "wrong-freeze", bootstrap_replicates=4)
