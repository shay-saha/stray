"""Run a finite experiment queue under supervisor; keep per-job evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--plan", required=True)
p.add_argument("--gpu", required=True)
p.add_argument("--max-hours", type=float, default=4.)
args = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
plan = json.loads(Path(args.plan).read_text())
state_path = Path(f"cloud/worker-{args.gpu}-{Path(args.plan).stem}.json")
state_path.parent.mkdir(exist_ok=True)
state = {"gpu": args.gpu, "started_unix": time.time(), "jobs": [], "status": "running"}

def save():
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(state_path)

for job in plan["jobs"]:
    remaining = args.max_hours * 3600 - (time.time() - state["started_unix"])
    if remaining <= 0:
        state["status"] = "time_budget_reached"
        break
    out = Path(job["output"]); out.mkdir(parents=True, exist_ok=True)
    if "python_arguments" in job:
        command = [sys.executable, "-u", *job["python_arguments"]]
    else:
        command = [sys.executable, "-u", "-m", "wildcast.cli", "--data", job["data"], job.get("task", "train"), "--output", str(out), *job["arguments"]]
    record = {"name": job["name"], "command": command, "started_unix": time.time(), "status": "running"}
    state["jobs"].append(record); save()
    with (out / "console.log").open("w") as log:
        try:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            record.update(status="complete" if done.returncode == 0 else "failed", returncode=done.returncode, ended_unix=time.time())
            if (out / "best.pt").exists():
                record["checkpoint_sha256"] = hashlib.sha256((out / "best.pt").read_bytes()).hexdigest()
        except subprocess.TimeoutExpired:
            record.update(status="timeout", ended_unix=time.time())
            state["status"] = "time_budget_reached"
            save()
            break
    save()
if state["status"] == "running":
    state["status"] = "complete"
state["ended_unix"] = time.time(); save()
