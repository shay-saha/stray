"""Enforce the user's finite migration-training window on task-owned services."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    cutoff = datetime.fromisoformat(plan["training_cutoff_utc"].replace("Z", "+00:00")).timestamp()
    deadline = datetime.fromisoformat(plan["delivery_deadline_utc"].replace("Z", "+00:00")).timestamp()
    prefixes = tuple(plan["training_service_prefixes"])
    if not cutoff < deadline or not prefixes or any(not p.startswith("wildcast-migration-") for p in prefixes):
        raise ValueError("Require a finite training cutoff and explicit migration-owned service prefixes")
    if args.show:
        print(json.dumps(plan, indent=2))
        return
    path = Path(plan["status_file"])
    state = {"status": "watching_training_cutoff", "plan": plan, "started_unix": time.time(), "stopped_services": []}
    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(path)
    save()
    while time.time() < cutoff:
        time.sleep(max(0., min(15., cutoff-time.time())))
    result = subprocess.run(["supervisorctl", "status"], capture_output=True, text=True, timeout=30)
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith(prefixes) or parts[1] not in ("RUNNING", "STARTING"):
            continue
        stop = subprocess.run(["supervisorctl", "stop", parts[0]], capture_output=True, text=True, timeout=60)
        state["stopped_services"].append({"name": parts[0], "returncode": stop.returncode,
                                          "output": stop.stdout.strip(), "error": stop.stderr.strip()})
        save()
    state.update(status="training_cutoff_reached", ended_unix=time.time(),
                 note="Checkpoint/data/log files retained. A cutoff is not successful completion of a fit or frontier claim.")
    save()
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
