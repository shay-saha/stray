"""Finish this finite migration campaign: freeze, compare and verify delivery.

This is a one-shot task-owned supervisor worker, not a recurring schedule.
It never releases reserved truth or changes source data or training services.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_info(path):
    import torch
    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    config = ck.get("training", {})
    info = {"path": str(path), "sha256": sha256(path), "kind": ck.get("model_kind"),
            "selected_epoch": ck.get("epoch"),
            "selection": ck.get("metrics", {}).get("selection"),
            "selection_version": ck.get("selection_version"),
            "context_mode": ck.get("context_features", {}).get("mode") if ck.get("context_features") else ck.get("config", {}).get("context_mode"),
            "context_only": config.get("context_only"),
            "budget": ck.get("budget", config.get("budget", {key: config[key] for key in
                        ("epochs", "batch_size", "lr", "seed", "sampling", "max_train", "patience") if key in config})),
            "manifest_sha256": ck.get("manifest_sha256"), "cases_sha256": ck.get("cases_sha256")}
    history_path = path.parent / "history.json"
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text())
            info["completed_training_epochs"] = history[-1]["epoch"]
        except (json.JSONDecodeError, IndexError, KeyError):
            info["completed_training_epochs"] = None
    return info


def snapshot(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Snapshot outputs must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        try:
            before = sha256(source)
            shutil.copyfile(source, output)
            if before != sha256(source) or before != sha256(output):
                raise ValueError("Checkpoint changed while freezing")
            info = checkpoint_info(output)
            info["original_source"] = str(source)
            history = source.parent / "history.json"
            if history.is_file():
                try:
                    info["completed_training_epochs"] = json.loads(history.read_text())[-1]["epoch"]
                except (json.JSONDecodeError, IndexError, KeyError):
                    info["completed_training_epochs"] = None
            return info
        except (OSError, ValueError, RuntimeError, EOFError):
            output.unlink(missing_ok=True)
            time.sleep(.2)
    raise RuntimeError("Could not freeze a valid checkpoint: " + str(source))


def active_services(prefixes):
    result = subprocess.run(["supervisorctl", "status"], capture_output=True, text=True, timeout=30)
    # supervisorctl returns3 when any unrelated service is stopped; parse states.
    if result.returncode not in (0, 3):
        raise RuntimeError("Cannot inspect task service states: " + result.stderr)
    rows = [line.split() for line in result.stdout.splitlines()]
    return [row[0] for row in rows if len(row) > 1 and row[0].startswith(tuple(prefixes))
            and row[1] in ("RUNNING", "STARTING", "STOPPING")]


def phase_job(path):
    config = json.loads(Path(path).read_text())
    import torch
    torch.set_num_threads(8); torch.set_float32_matmul_precision("high")
    phase = config.pop("phase")
    if phase == "endpoint":
        from wildcast.migration_endpoint_evaluate import evaluate
    elif phase == "joint":
        from wildcast.migration_evaluate import evaluate
    else:
        raise ValueError("Unrecognized delivery scoring phase")
    result = evaluate(**config)
    print(json.dumps({"cases": result["cases"], "models": result["models"]}), flush=True)


def choose_model(metrics, eligible, reference="current-ensemble"):
    """Equal split and endpoint/joint weighting, with a fixed promotion margin."""
    import math
    terms = {}
    for key, metric in (("scores", "endpoint_energy_km"), ("joint_scores", "joint_chordal_energy_km")):
        for row in metrics[key]:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError("Nonfinite matched delivery score")
            terms.setdefault(row["model"], {})[(key, row["split"])] = value
    required = {(key, split) for key in ("scores", "joint_scores") for split in ("val_individual", "val_year")}
    if any(set(values) != required for values in terms.values()) or reference not in terms:
        raise ValueError("Delivery selection requires all four matched score terms")
    scores = {key: sum(value.values()) / 4 for key, value in terms.items()}
    candidates = {key: value for key, value in scores.items() if key in eligible}
    if reference not in candidates:
        raise ValueError("Existing delivery must be among eligible snapshots")
    candidate = min(candidates, key=candidates.get)
    return (candidate if scores[candidate] < scores[reference] - .2 else reference), scores


def finish(plan):
    os.chdir(PROJECT)
    deadline_plan = json.loads(Path(plan["deadline_plan"]).read_text())
    cutoff = datetime.fromisoformat(deadline_plan["training_cutoff_utc"].replace("Z", "+00:00")).timestamp()
    deadline = datetime.fromisoformat(deadline_plan["delivery_deadline_utc"].replace("Z", "+00:00")).timestamp()
    status_path = Path(plan["status_file"])
    state = {"status": "waiting_for_finite_training", "started_utc": datetime.now(timezone.utc).isoformat(),
             "deadline_utc": deadline_plan["delivery_deadline_utc"], "source_sha256": sha256(__file__),
             "scope": "Local development; no independent test, official MoveBench rank, or frontier-superiority claim"}
    save(status_path, state)
    while time.time() < cutoff:
        services = active_services(deadline_plan["training_service_prefixes"])
        if not services:
            break
        state["active_services"] = services; save(status_path, state)
        time.sleep(min(15., max(0., cutoff - time.time())))
    # A separate task-owned cutoff worker stops remaining training processes.
    # Snapshots are checked before/after copying even when a last write is live.
    state.update(status="freezing_completed_checkpoints", active_services=active_services(deadline_plan["training_service_prefixes"]))
    save(status_path, state)
    root = Path(plan["snapshot_root"])
    snapshots, skipped = {}, []
    for label, source in plan["core"].items():
        snapshots[label] = snapshot(source, root / label / "best.pt")
    available_context = []
    for recipe in plan["context_recipes"]:
        path = Path(recipe["actual"])
        if not path.is_file():
            continue
        info = checkpoint_info(path)
        state.setdefault("context_trial_screens", []).append(info)
        if info["selection"] is not None:
            available_context.append((float(info["selection"]), recipe))
    if available_context:
        _, chosen = min(available_context, key=lambda item: item[0])
        state["context_recipe_selected_by_preexisting_16draw_screen"] = chosen
        for mode in ("actual", "availability"):
            source = Path(chosen[mode])
            if source.is_file():
                label = "human-" + mode
                snapshots[label] = snapshot(source, root / label / "best.pt")
    for label, source in plan["public"].items():
        if Path(source).is_file():
            snapshots[label] = snapshot(source, root / label / "best.pt")
        else:
            skipped.append({"label": label, "reason": "No valid checkpoint produced within finite budget"})
    state.update(status="matched_scoring", checkpoints=snapshots, skipped=skipped)
    save(status_path, state)
    core_labels = [label for label in snapshots if label in plan["core"] or label.startswith("human-")]
    common = dict(data=plan["data"], samples=64, seed=2026, batch_size=128, device="cuda", bootstrap_replicates=300,
                  reference="current-ensemble", baselines=["stationary"])
    jobs = [dict(phase="joint", output=plan["joint_report"], checkpoints=[snapshots[label]["path"] for label in core_labels], **common),
            dict(phase="endpoint", output=plan["endpoint_report"], checkpoints=[item["path"] for item in snapshots.values()],
                 horizons_days=[1/24, .5, 1.], max_origins=4096, **common)]
    processes, logs = [], []
    for gpu, job in enumerate(jobs):
        config_path = Path(plan["job_root"]) / (job["phase"] + ".json")
        save(config_path, job)
        log = (config_path.parent / (job["phase"] + ".log")).open("w"); logs.append(log)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
        process = subprocess.Popen([sys.executable, "-u", __file__, "--job", str(config_path)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        processes.append((job, process))
    while any(process.poll() is None for _, process in processes) and time.time() < deadline - 180:
        time.sleep(5.)
    for job, process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        state.setdefault("scoring_jobs", []).append({"phase": job["phase"], "returncode": process.returncode, "report": job["output"]})
    for log in logs:
        log.close()
    state["status"] = "verifying_delivery"; save(status_path, state)
    selected = "current-ensemble"
    joint_path = Path(plan["joint_report"]) / "metrics.json"
    if processes[0][1].returncode == 0 and joint_path.is_file():
        metrics = json.loads(joint_path.read_text())
        selected, scores = choose_model(metrics, snapshots)
        state["matched_joint_selection_scores_km"] = scores
        state["promotion_rule"] = "Lowest equal-split endpoint/joint development energy; require >0.2km improvement over existing delivered ensemble; no independent-test claim"
    state["selected_label"] = selected
    state["selected_checkpoint"] = snapshots[selected]
    if selected == "current-ensemble":
        bundle = Path(plan["existing_bundle"])
        result = subprocess.run([sys.executable, str(bundle / "verify_package.py")], capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError("Existing delivered package failed verification: " + result.stderr)
        state["bundle"] = {"path": str(bundle), "verified": True, "verification": result.stdout.strip(),
                           "reason": "No sufficiently improved complete matched result; existing model retained"}
    else:
        from package_migration import package
        state["bundle"] = package(plan["data"], snapshots[selected]["path"], plan["new_bundle"], archive=plan["new_archive"], source_root=PROJECT)
        bundle = Path(plan["new_bundle"])
        subprocess.run([sys.executable, str(bundle / "verify_package.py")], check=True, timeout=90)
        state["bundle"]["verified"] = True
    state.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                 deadline_met=time.time() <= deadline, endpoint_comparison=plan["endpoint_report"] if processes[1][1].returncode == 0 else plan["fallback_endpoint_report"],
                 frontier_superiority_established=False, reserved_truth_released=False)
    save(status_path, state)
    progress_path = Path("reports/migration-progress.json")
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        progress.update(updated_utc=state["completed_utc"],
            delivery_execution={"status": state["status"], "report": str(status_path), "selected": selected,
                                "deadline_met": state["deadline_met"], "bundle": state["bundle"],
                                "endpoint_comparison": state["endpoint_comparison"],
                                "frontier_superiority_established": False})
        save(progress_path, progress)
    print(json.dumps({"status": state["status"], "selected": selected, "deadline_met": state["deadline_met"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan"); parser.add_argument("--job")
    args = parser.parse_args()
    if bool(args.plan) == bool(args.job):
        parser.error("Choose a one-shot delivery plan or internal scoring job")
    if args.job:
        phase_job(args.job)
    else:
        plan = json.loads(Path(args.plan).read_text())
        try:
            finish(plan)
        except Exception as error:
            path = Path(plan["status_file"])
            state = json.loads(path.read_text()) if path.is_file() else {}
            state.update(status="failed", error=type(error).__name__ + ": " + str(error), existing_bundle_retained=plan["existing_bundle"])
            save(path, state)
            raise


if __name__ == "__main__":
    main()
