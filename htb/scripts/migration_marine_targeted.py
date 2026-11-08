#!/usr/bin/env python3
"""Finite CC0 turtle/mako original intake; never convert sensors or infer truth.

Three source releases only, 100MB storage cap, 30GB free-space floor. A public
Dryad download failure remains explicit; no accounts, private endpoints, model
estimates or alternative licensed R workspaces are substituted for raw CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wildcast.sources import download, dump, get_json, session

MOVE = ("df53968d-4bcc-4dd7-9bb6-8358378836ec", "bd9f2614-11b3-48c3-8251-1aee62c8e650")
MAKO = "10.5061/dryad.2rbnzs7vm"
CAP, FLOOR = 100_000_000, 30_000_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def used(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink())


def check(root, amount=0):
    if used(root) + amount > CAP or shutil.disk_usage(root).free - amount < FLOOR:
        raise ValueError("Marine targeted quota or free-space floor exceeded")


def fetch(s, root, directory, record):
    name = record["name"]
    if Path(name).name != name or record["bytes"] > 30_000_000:
        raise ValueError("Unexpected finite marine input name or size")
    path = directory / "originals" / name
    if not path.exists():
        check(root, record["bytes"])
    checksum = record["published_checksum"]
    alg = checksum.get("algorithm", checksum.get("checkSumAlgorithm")).replace("sha-256", "sha256")
    download(s, record["url"], path, record["bytes"], (alg, checksum["value"]))
    result = {**record, "stored_as": str(path.relative_to(directory)), "sha256": digest(path),
              "status": "complete", "retrieved_utc": datetime.now(timezone.utc).isoformat()}
    if name.endswith(".csv") and record.get("role") != "ancillary_not_location_truth":
        link = directory / name
        if link.exists():
            if digest(link) != result["sha256"]:
                raise ValueError("Existing marine root link changed")
        else:
            link.symlink_to(Path("originals") / name)
        result["root_link"] = name
    return result


def acquire(root, catalogue="data/catalogue.json"):
    root = Path(root); (root / "raw").mkdir(parents=True, exist_ok=True); check(root)
    code_bytes = Path(__file__).read_bytes()
    code_hash = hashlib.sha256(code_bytes).hexdigest()
    (root / "acquisition-source.py").write_bytes(code_bytes)
    source_helpers = runpy.run_path(str(Path(__file__).with_name("migration_acquire.py")))
    records = {r["id"]: r for r in json.loads(Path(catalogue).read_text())["records"]}
    s = session(); results = []
    for identity in MOVE:
        record, directory = records[identity], root / "raw" / identity
        directory.mkdir(exist_ok=True)
        live = get_json(s, "https://datarepository.movebank.org/server/api/core/items/" + identity)
        metadata = {k: [v["value"] for v in values] for k, values in live["metadata"].items()}
        if (source_helpers["rights_status"](metadata) != "eligible" or
                not all("/publicdomain/zero/1.0" in u.lower() for u in metadata.get("dc.rights.uri", []))):
            raise ValueError("Targeted marine release is no longer explicitly CC0")
        dump(directory / "metadata" / "repository-item.json", live)
        files = source_helpers["bitstreams"](record)
        dump(directory / "metadata" / "repository-bitstreams.json", files)
        fetched = [fetch(s, root, directory, f) for f in files]
        report = {"id": identity, "title": record["title"], "url": record["url"], "metadata": metadata,
                  "files": fetched, "status": "complete", "code_sha256": code_hash,
                  "scope": "Original measurements and source reference/documentation only; preserve explicit GPS versus Doppler filenames/row sensor types. No sensor relabelling, interpolation or SSM truth.",
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        dump(directory / "source.json", report); results.append(report)
        print("READY", identity, sum(f["bytes"] for f in fetched), flush=True)
    directory = root / "raw" / "dryad-2rbnzs7vm"; directory.mkdir(exist_ok=True)
    base = "https://datadryad.org"
    dataset = get_json(s, base + "/api/v2/datasets/doi%3A10.5061%2Fdryad.2rbnzs7vm")
    version = get_json(s, base + dataset["_links"]["stash:version"]["href"])
    files = get_json(s, base + version["_links"]["stash:files"]["href"])
    rights = get_json(s, "https://api.datacite.org/dois/" + MAKO)
    if not any(v.get("rightsIdentifier") == "cc0-1.0" for v in rights["data"]["attributes"]["rightsList"]):
        raise ValueError("Mako raw release requires explicit CC0")
    for name, payload in (("dryad-dataset", dataset), ("dryad-version", version), ("dryad-files", files), ("datacite", rights)):
        dump(directory / "metadata" / (name + ".json"), payload)
    fetched = []
    for f in files["_embedded"]["stash:files"]:
        if f["path"] not in {"AtlanticArgos.csv", "PacificArgos.csv", "README.md", "Pacific_TAD.csv"}:
            continue
        identity = f["_links"]["self"]["href"].split("/")[-1]
        record = {"name": f["path"], "bytes": f["size"], "url": base + "/downloads/file_stream/" + identity,
                  "published_checksum": {"algorithm": f["digestType"], "value": f["digest"]},
                  "role": "raw_argos_observations" if f["path"] in {"AtlanticArgos.csv", "PacificArgos.csv"} else "ancillary_not_location_truth"}
        try:
            fetched.append(fetch(s, root, directory, record))
        except Exception as exc:
            fetched.append({**record, "status": "download_blocked", "error": str(exc)})
            print("BLOCKED", record["name"], str(exc), flush=True)
    report = {"id": "dryad-2rbnzs7vm", "doi": MAKO, "title": dataset["title"], "url": base + "/dataset/doi:" + MAKO,
              "metadata": {"dc.rights": ["CC0 1.0 Universal"], "dc.rights.uri": ["https://creativecommons.org/publicdomain/zero/1.0/"],
                           "sensor.name": ["Argos Doppler Shift"]}, "files": fetched,
              "status": "complete" if all(f["status"] == "complete" for f in fetched) else "public_download_blocked",
              "code_sha256": code_hash,
              "observation_schema": {"id": "source track ID; use basin/file namespace", "date": "UTC measurement timestamp", "lc": "reported Argos location class", "lon": "reported longitude", "lat": "reported latitude"},
              "excluded_alternative": "Zenodo10233093/10233092 contain CC-BY-NC R workspace/modelled SSM estimates, not byte-identical CC0 raw CSVs. Not downloaded or substituted.",
              "scope": "Raw AtlanticArgos/PacificArgos measurements only are candidate location labels; Pacific_TAD is ancillary; never use SSM/simulated tracks as truth.",
              "observed_utc": datetime.now(timezone.utc).isoformat()}
    dump(directory / "source.json", report); results.append(report)
    check(root)
    dump(root / "intake.json", {"sources": [{"id": r["id"], "status": r["status"], "source_manifest_sha256": digest(root / "raw" / r["id"] / "source.json")} for r in results],
                                 "storage_bytes": used(root), "budget_bytes": CAP, "free_bytes": shutil.disk_usage(root).free,
                                 "scope": "Separate targeted marine intake; no v3 mutation, prepared cache, training or final target release."})
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", default="data/migration-marine-targeted")
    args = p.parse_args(); acquire(args.root)
