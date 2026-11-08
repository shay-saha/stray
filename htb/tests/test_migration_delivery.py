"""Promotion uses matched endpoint AND route evidence; files stay immutable."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("migration_delivery", Path(__file__).resolve().parents[1] / "scripts/migration_delivery.py")
delivery = importlib.util.module_from_spec(spec); spec.loader.exec_module(delivery)


def metrics(values):
    return {"scores": [{"model": model, "split": split, "endpoint_energy_km": score[index]}
                       for model, score in values.items() for index, split in enumerate(("val_individual", "val_year"))],
            "joint_scores": [{"model": model, "split": split, "joint_chordal_energy_km": score[index+2]}
                       for model, score in values.items() for index, split in enumerate(("val_individual", "val_year"))]}


def test_promotion_requires_complete_matched_joint_improvement():
    scores = metrics({"current-ensemble": [10., 12., 11., 13.], "human-actual": [9., 11., 10., 12.], "stationary": [0., 0., 0., 0.]})
    selected, _ = delivery.choose_model(scores, ["current-ensemble", "human-actual"])
    assert selected == "human-actual"
    # A better endpoint score cannot hide a worse joint movement distribution.
    scores = metrics({"current-ensemble": [10., 12., 11., 13.], "human-actual": [9., 11., 15., 17.]})
    assert delivery.choose_model(scores, ["current-ensemble", "human-actual"])[0] == "current-ensemble"
    scores["joint_scores"].pop()
    with pytest.raises(ValueError, match="four matched"):
        delivery.choose_model(scores, ["current-ensemble", "human-actual"])


def test_small_changes_retain_the_already_delivered_model():
    score = metrics({"current-ensemble": [10.]*4, "human-actual": [9.9]*4})
    assert delivery.choose_model(score, ["current-ensemble", "human-actual"])[0] == "current-ensemble"


def test_service_watch_is_read_only_and_filters_explicit_task_prefixes(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=3, stdout="cron RUNNING\nwildcast-migration-context-fit RUNNING\nwildcast-migration-public-fit STOPPING\nwildcast-migration-public-old EXITED\nwildcast-migration-deadline-v1 RUNNING\n", stderr="")
    monkeypatch.setattr(delivery.subprocess, "run", run)
    assert delivery.active_services(["wildcast-migration-context-", "wildcast-migration-public-"]) == [
        "wildcast-migration-context-fit", "wildcast-migration-public-fit"]
    assert calls == [["supervisorctl", "status"]]


def test_frozen_snapshot_preserves_source_weights_and_rejects_overwrite(tmp_path):
    source = tmp_path / "source" / "best.pt"; source.parent.mkdir()
    torch.save({"model_kind": "wildcast_migration_joint", "model": {"weight": torch.arange(8.)},
                "epoch": 3, "metrics": {"selection": 1.2}, "manifest_sha256": "manifest", "cases_sha256": "cases"}, source)
    output = tmp_path / "snapshot" / "best.pt"
    before = delivery.sha256(source)
    info = delivery.snapshot(source, output)
    assert delivery.sha256(output) == delivery.sha256(source) == before == info["sha256"]
    assert info["selected_epoch"] == 3 and info["original_source"] == str(source)
    with pytest.raises(FileExistsError):
        delivery.snapshot(source, output)
