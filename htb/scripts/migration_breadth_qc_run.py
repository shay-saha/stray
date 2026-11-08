"""Run one finite, pinned breadth-QC and observation-availability job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args):
    plan = json.loads(Path(args.plan).read_text())
    acquisition = json.loads(Path(args.acquisition_finished).read_text())
    if acquisition["plan_sha256"] != sha(args.plan):
        raise ValueError("Acquisition handoff belongs to another pinned plan")
    if acquisition["sources"] != len(plan["selected_sources"]):
        raise ValueError("Acquisition handoff lacks source completion records")
    if acquisition["complete_sources"] != acquisition["sources"] and not args.allow_acquisition_gaps:
        raise ValueError("Acquisition has unresolved file gaps; inspect the handoff before QC")
    implementation = [Path(__file__), Path("wildcast/breadth_clean.py"), Path("wildcast/marine_gps_clean.py"),
                      Path("wildcast/marine_intake.py"), Path("wildcast/migration_identity.py")]
    launch = {"plan": args.plan, "plan_sha256": sha(args.plan),
              "acquisition_finished": args.acquisition_finished, "acquisition_finished_sha256": sha(args.acquisition_finished),
              "overrides": args.overrides, "overrides_sha256": sha(args.overrides),
              "code_sha256": {str(p): sha(p) for p in implementation}, "workers": args.workers,
              "raw_root": args.raw_root, "policy": args.policy, "output": args.output,
              "availability": args.availability, "train_end_year": args.train_end_year,
              "scope": "Finite source QC and actual time availability only; no forecasts, reserved labels or model fits"}
    launch_path = Path(args.state).with_name(Path(args.state).stem + "-launch.json")
    if launch_path.exists():
        if json.loads(launch_path.read_text()) != launch:
            raise ValueError("Finite QC job resume provenance differs")
    else:
        save(launch_path, launch)
    state = {"launch_sha256": sha(launch_path), "status": "running", "started_unix": time.time(), "stages": []}
    save(args.state, state)
    commands = [
        ("policy", ["policy", "--raw-root", args.raw_root, "--plan", args.plan, "--overrides", args.overrides, "--output", args.policy]),
        ("clean", ["clean", "--policy", args.policy, "--output", args.output, "--workers", str(args.workers)]),
        ("availability", ["availability", "--observations", str(Path(args.output)/"processed/observations.parquet"),
                          "--output", args.availability, "--train-end-year", str(args.train_end_year)])]
    for name, arguments in commands:
        completed = (Path(args.policy) if name == "policy" else Path(args.output)/"processed/manifest.json"
                     if name == "clean" else Path(args.availability))
        # A completed immutable stage can be reused only under this exact launch.
        if completed.exists():
            state["stages"].append({"stage": name, "status": "reused_immutable_output", "output_sha256": sha(completed)})
            save(args.state, state); continue
        record = {"stage": name, "started_unix": time.time(), "status": "running"}
        state["stages"].append(record); save(args.state, state)
        code = subprocess.run([sys.executable, "-u", "-m", "wildcast.breadth_clean", *arguments], check=False).returncode
        record.update(returncode=code, ended_unix=time.time(), status="complete" if code == 0 else "failed")
        if code:
            state["status"] = "failed"; save(args.state, state)
            raise SystemExit(code)
        record["output_sha256"] = sha(completed); save(args.state, state)
    state.update(status="complete", ended_unix=time.time()); save(args.state, state)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True); p.add_argument("--plan", required=True)
    p.add_argument("--acquisition-finished", required=True); p.add_argument("--overrides", required=True)
    p.add_argument("--policy", required=True); p.add_argument("--output", required=True)
    p.add_argument("--availability", required=True); p.add_argument("--state", required=True)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--train-end-year", type=int, default=2020)
    p.add_argument("--allow-acquisition-gaps", action="store_true")
    args = p.parse_args()
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    run(args)
