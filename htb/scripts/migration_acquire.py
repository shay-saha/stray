"""Bounded, resumable public GPS acquisition for migration experiments.

Only source metadata determines ranking. Raw bytes and identity reference tables
are preserved; CSV chunking changes storage, never location values or sensors.
Run with --execute to acquire; the default only records the bounded source plan.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import stat
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

# Allow the documented direct invocation without relying on an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wildcast.data import ALIASES
from wildcast.sources import API, download, get_json, session

TERMS = re.compile(r"migrat\w*|\b(?:stork\w*|crane\w*|goose|geese|swallow\w*|terns?|ospreys?|pronghorn\w*|zebra\w*|salmon\w*|seal\w*|ungulates?|elk|caribou|reindeer)\b", re.I)
FOCUS = re.compile(r"\b(?:stork\w*|crane\w*|pronghorn\w*|zebra\w*|ungulates?|elk|caribou|reindeer)\b", re.I)
REPEATED = re.compile(r"annual|multi.year|20\d\d\s*[-–/]\s*20\d\d|repeat|adult|long.term", re.I)
NON_GPS = re.compile(r"(?:acceleration|accelerometer|magnetometer|orientation|gyroscope|barometer|solar.geolocator|argos|doppler|[-_ ]acc(?:[-_. ]|$))", re.I)
ALLOWED = ("/publicdomain/zero/1.0", "/licenses/by/4.0", "/licenses/by/3.0")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rights_status(meta):
    uris = meta.get("dc.rights.uri", [])
    words = " ".join(meta.get("dc.rights", [])).lower()
    if not uris or not all(any(t in u.lower() for t in ALLOWED) for u in uris):
        return "quarantined: missing or unsupported explicit licence URI"
    if re.search(r"all rights reserved|non.?commercial|no.?derivatives|by-nc|by-nd", words):
        return "quarantined: contradictory rights text"
    return "eligible"


def number(meta, name):
    try:
        return int(meta.get(name, [0])[0])
    except (ValueError, TypeError, IndexError):
        return 0


def rank_record(rec):
    m = rec["metadata"]
    text = rec["title"] + " " + " ".join(m.get("dc.subject", []))
    if not TERMS.search(text):
        return None
    sensors = m.get("sensor.name", [])
    gps = any(v.lower() == "gps" for v in sensors)
    n = number(m, "mdr.animal.count")
    count = number(m, "mdr.location.count")
    reasons = ["migration/taxon match in title or subjects"]
    score = 6 * math.log1p(n) + math.log1p(count)
    if FOCUS.search(text):
        score += 25
        reasons.append("enrich current migration groups or migratory ungulates")
    if REPEATED.search(text):
        score += 20
        reasons.append("metadata suggests repeated years/adults/annual cycle; completeness unverified")
    return {"id": rec["id"], "title": rec["title"], "url": rec["url"],
            "score": score, "reasons": reasons, "animals_metadata": n,
            "locations_metadata": count, "sensors_metadata": sensors,
            "status": rights_status(m) if gps else "gap: package does not explicitly advertise GPS",
            "rights_uris": m.get("dc.rights.uri", []), "files": []}


def safe_name(name):
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or not p.name or ":" in p.parts[0]:
        raise ValueError(f"Unsafe source member path: {name!r}")
    return p.name


def bitstreams(rec):
    s = session()
    bundles = get_json(s, f"{API}/core/items/{rec['id']}/bundles", size=100).get("_embedded", {}).get("bundles", [])
    out = []
    for bundle in bundles:
        if bundle["name"] != "ORIGINAL":
            continue
        url = bundle["_links"]["bitstreams"]["href"]
        while url:
            j = get_json(s, url, size=100)
            for b in j.get("_embedded", {}).get("bitstreams", []):
                name = safe_name(b["name"])
                out.append({"name": name, "bytes": b["sizeBytes"],
                            "url": b["_links"]["content"]["href"],
                            "published_checksum": b.get("checkSum")})
            url = j.get("_links", {}).get("next", {}).get("href")
    if len({v["name"] for v in out}) != len(out):
        raise ValueError("Duplicate bitstream basenames require manual review")
    return out


def file_role(f):
    name = f["name"].lower()
    if NON_GPS.search(name):
        return "non_gps"
    if name.endswith((".zip", ".csv.gz")):
        return "archive"
    if name.endswith(".csv"):
        if "reference" in name:
            return "identity_reference"
        return "non_gps" if NON_GPS.search(name) else "events"
    if name.endswith((".txt", ".md", ".html", ".htm")) and f["bytes"] < 2_000_000:
        return "source_documentation"
    return "unsupported"


def cache_index(roots):
    records = {}
    for root in roots:
        for path in sorted(Path(root).glob("*/source.json")):
            try:
                m = json.loads(path.read_text())
                if "metadata" not in m or rights_status(m["metadata"]) != "eligible":
                    continue
                for f in m.get("files", []):
                    candidate = path.parent / f["name"]
                    if candidate.is_file() and f.get("sha256"):
                        records.setdefault((path.parent.name, f["name"]), []).append((candidate, f))
            except (ValueError, KeyError, OSError):
                continue
    return records


def cached_file(rec_id, f, index):
    for path, manifest in index.get((rec_id, f["name"]), []):
        if path.stat().st_size != f["bytes"] or digest(path) != manifest["sha256"]:
            continue
        ck = f.get("published_checksum")
        if ck and digest(path, ck["checkSumAlgorithm"].lower()) != ck["value"].lower():
            continue
        return path
    return None


class Budget:
    def __init__(self, maximum):
        self.maximum = maximum
        self.reserved = 0
        self.lock = threading.Lock()

    def take(self, count):
        with self.lock:
            if self.reserved + count > self.maximum:
                raise ValueError("Global acquisition/storage byte budget exhausted")
            self.reserved += count

    def release(self, count):
        with self.lock:
            self.reserved -= count


def schema(header, sensors, filename):
    found = {k: next((v for v in names if v in header), None) for k, names in ALIASES.items()}
    supported = all(found[k] for k in ("timestamp", "lat", "lon", "animal", "species"))
    if not supported:
        return "custom_or_reference", found
    if "sensor-type" in header:
        return "GPS rows only: downstream cleaner must filter sensor-type == GPS", found
    if sensors == ["GPS"] or re.search(r"(?:^|[-_ ])gps(?:[._ -]|$)", filename, re.I):
        return "GPS from explicit package/file metadata; no sensor-type column", found
    return "quarantined: mixed-sensor package without row sensor labels", found


def split_csv(stream, directory, stem, budget, max_chunk_bytes, sensors, original):
    """Stream full CSV records, repeat headers, and retain original field values."""
    csv.field_size_limit(8_000_000)
    reader = csv.reader(stream)
    header = next(reader, [])
    status, aliases = schema(header, sensors, original)
    if status.startswith("quarantined") or status == "custom_or_reference":
        return {"status": status, "header": header, "aliases": aliases, "chunks": []}
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(header)
    header_text = buf.getvalue()
    chunks, handle, rows, n, charged = [], None, 0, 0, 0
    try:
        for row in reader:
            buf.seek(0)
            buf.truncate(0)
            writer.writerow(row)
            encoded = buf.getvalue().encode("utf-8")
            if handle is None or n + len(encoded) > max_chunk_bytes:
                if handle:
                    handle.close()
                    chunks[-1].update({"rows": rows, "bytes": n, "sha256": digest(directory / (chunks[-1]["name"] + ".part"))})
                name = f"{stem}__part{len(chunks)+1:04d}.csv"
                path = directory / name
                budget.take(len(header_text.encode("utf-8")))
                charged += len(header_text.encode("utf-8"))
                handle = path.with_suffix(path.suffix + ".part").open("wb")
                handle.write(header_text.encode("utf-8"))
                chunks.append({"name": name})
                rows, n = 0, len(header_text.encode("utf-8"))
            budget.take(len(encoded))
            charged += len(encoded)
            handle.write(encoded)
            rows += 1
            n += len(encoded)
        if handle:
            handle.close()
            handle = None
            chunks[-1].update({"rows": rows, "bytes": n, "sha256": digest(directory / (chunks[-1]["name"] + ".part"))})
        # Publish only complete source files. After interruption, deterministic
        # regenerated chunks may be reused only if their exact hashes match.
        for chunk in chunks:
            target = directory / chunk["name"]
            if target.exists() and digest(target) != chunk["sha256"]:
                raise ValueError(f"Existing derived chunk differs: {target}")
        for chunk in chunks:
            target = directory / chunk["name"]
            temp = directory / (chunk["name"] + ".part")
            if target.exists():
                temp.unlink()
                budget.release(chunk["bytes"])
            else:
                temp.replace(target)
    except Exception:
        if handle:
            handle.close()
            handle = None
        for chunk in chunks:
            (directory / (chunk["name"] + ".part")).unlink(missing_ok=True)
        budget.release(charged)
        raise
    finally:
        if handle:
            handle.close()
    return {"status": status, "header": header, "aliases": aliases, "chunks": chunks,
            "rows": sum(c["rows"] for c in chunks)}


def extract_csvs(path, directory, f, sensors, budget, chunk_bytes, archive_limit):
    def decode_csv(opener, stem, name):
        # Some repository CSV releases contain Windows-1252 punctuation. Retry
        # explicitly and record it; never use lossy replacement/error ignoring.
        try:
            with opener() as raw:
                result = split_csv(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), directory,
                                   stem, budget, chunk_bytes, sensors, name)
            return {"source_encoding": "utf-8-sig", **result}
        except UnicodeDecodeError:
            with opener() as raw:
                result = split_csv(io.TextIOWrapper(raw, encoding="cp1252", newline=""), directory,
                                   stem, budget, chunk_bytes, sensors, name)
            return {"source_encoding": "cp1252", "encoding_note": "UTF-8 failed; Windows-1252 decoding is explicitly inferred and recorded.", **result}
    derived = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 10_000 or sum(v.file_size for v in infos) > archive_limit:
                raise ValueError("Archive exceeds uncompressed quota or member limit")
            for member in infos:
                name = safe_name(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Archive contains a symbolic link")
                if "__MACOSX" in PurePosixPath(member.filename).parts or name.startswith("._"):
                    derived.append({"archive_member": member.filename, "status": "excluded Apple resource fork", "chunks": []})
                    continue
                if member.is_dir() or not name.lower().endswith(".csv") or NON_GPS.search(name):
                    continue
                stem = re.sub(r"[^\w.-]+", "_", Path(name).stem) + "_" + hashlib.sha256(member.filename.encode()).hexdigest()[:8]
                try:
                    info = decode_csv(lambda: archive.open(member), stem, name)
                except Exception as exc:
                    info = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "chunks": []}
                derived.append({"archive_member": member.filename, "uncompressed_bytes": member.file_size, **info})
    elif path.name.lower().endswith(".csv.gz"):
        import gzip
        # Reading is streamed; the global storage quota bounds decompression output.
        info = decode_csv(lambda: gzip.open(path, "rb"), Path(f["name"]).name[:-7], f["name"])
        derived.append(info)
    else:
        info = decode_csv(lambda: path.open("rb"), Path(f["name"]).stem, f["name"])
        derived.append(info)
    return derived


def acquire(rec, item, output, index, budget, max_file, chunk_bytes, archive_limit):
    directory = output / "raw" / rec["id"]
    directory.mkdir(parents=True, exist_ok=True)
    previous_path = directory / "source.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    completed = {v["name"]: v for v in previous.get("files", [])}
    files = []
    for f in item["files"]:
        result = dict(f)
        role = file_role(f)
        result["role"] = role
        if role in ("non_gps", "unsupported") or f["bytes"] > max_file:
            result["status"] = "excluded non-GPS/unsupported" if f["bytes"] <= max_file else "gap: larger than per-file cap"
            files.append(result)
            continue
        try:
            old = completed.get(f["name"])
            if old and old.get("status") == "complete" and all((directory / c["name"]).is_file() and digest(directory / c["name"]) == c["sha256"] for d in old.get("derived", []) for c in d["chunks"]):
                stored = directory / old["stored_as"]
                if stored.is_file() and digest(stored) == old["sha256"]:
                    files.append(old)
                    continue
            path = directory / "originals" / f["name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            cached = cached_file(rec["id"], f, index)
            if cached:
                if not path.exists():
                    path.symlink_to(cached.resolve())
                elif digest(path) != digest(cached):
                    raise ValueError("Existing source original conflicts with verified cache")
                result["reused_verified_cache"] = str(cached)
            else:
                ck = f.get("published_checksum")
                if not path.exists():
                    budget.take(f["bytes"])
                download(session(), f["url"], path, max_file,
                         (ck["checkSumAlgorithm"].lower(), ck["value"]) if ck else None)
            result.update({"stored_as": str(path.relative_to(directory)), "sha256": digest(path), "retrieved_utc": now()})
            if role in ("events", "archive"):
                result["derived"] = extract_csvs(path, directory, f, item["sensors_metadata"], budget, chunk_bytes, archive_limit)
            elif role == "identity_reference":
                link = directory / f["name"]
                if not link.exists():
                    link.symlink_to(path.relative_to(directory))
            result["status"] = "complete_with_extraction_gaps" if any(d.get("status") == "error" for d in result.get("derived", [])) else "complete"
        except Exception as exc:
            result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        files.append(result)
        # A completed source manifest exists after every file for bounded resumption.
        save(previous_path, {**rec, "source_id": rec["id"], "files": files,
                            "retrieved_utc": now(), "acquisition": item["reasons"],
                            "acquisition_complete": False,
                            "storage_note": "Originals retained under originals/; root CSVs are lossless record chunks or reference symlinks. GPS sensor filtering is mandatory."})
    cleanable = sum(len(v.get("chunks", [])) for f in files for v in f.get("derived", []))
    save(previous_path, {**rec, "source_id": rec["id"], "files": files, "retrieved_utc": now(),
                        "acquisition": item["reasons"], "acquisition_complete": True,
                        "storage_note": "Originals retained under originals/; root CSVs are lossless record chunks or reference symlinks. GPS sensor filtering is mandatory."})
    return {**item, "status": "acquired" if cleanable else "no supported GPS chunks", "files": files,
            "cleanable_csv_chunks": cleanable, "source_manifest": str(previous_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalogue", type=Path, default=Path("data/catalogue.json"))
    p.add_argument("--output", type=Path, default=Path("data/migration-expanded"))
    p.add_argument("--report", type=Path, default=Path("reports/migration-source-audit.json"))
    p.add_argument("--max-studies", type=int, default=40)
    p.add_argument("--budget-mb", type=int, default=6000)
    p.add_argument("--max-file-mb", type=int, default=512)
    p.add_argument("--chunk-mb", type=int, default=32)
    p.add_argument("--archive-max-mb", type=int, default=4000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ids", help="Optional comma-separated metadata-ranked source IDs for a finite selected tranche")
    p.add_argument("--reuse-inventory", action="store_true", help="Reuse exact saved bitstream metadata for the same catalogue hash")
    p.add_argument("--min-free-gb", type=int, default=30)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not 1 <= args.workers <= 4 or min(args.budget_mb, args.max_file_mb, args.chunk_mb, args.archive_max_mb, args.max_studies) <= 0:
        p.error("Positive quotas and 1–4 workers are required")
    free = shutil.disk_usage(args.output.parent).free
    if free < args.budget_mb * 1_000_000 + args.min_free_gb * 1_000_000_000:
        p.error(f"Acquisition budget must leave at least {args.min_free_gb} GB free")
    catalogue = json.loads(args.catalogue.read_text())
    records = {r["id"]: r for r in catalogue["records"]}
    ranked = [v for r in records.values() if (v := rank_record(r))]
    ranked.sort(key=lambda r: (-r["score"], r["id"]))
    requested = set(args.ids.split(",")) if args.ids else None
    selected = [r for r in ranked if r["status"] == "eligible" and (not requested or r["id"] in requested)][:args.max_studies]
    if requested and requested != {r["id"] for r in selected}:
        p.error("Every requested source must be present, GPS/licence-eligible, and inside max-studies")
    prior = json.loads(args.report.read_text()) if args.reuse_inventory and args.report.exists() else {}
    prior_inventory = {r["id"]: r.get("files", []) for r in prior.get("ranked_sources", [])} if prior.get("catalogue_sha256") == digest(args.catalogue) else {}
    for r in selected:
        r["files"] = prior_inventory.get(r["id"], [])
    report = {"started_utc": now(), "catalogue": str(args.catalogue), "catalogue_sha256": digest(args.catalogue),
              "catalogue_observed_utc": catalogue.get("retrieved_utc"), "disk_free_initial_bytes": free,
              "quotas": vars(args) | {"catalogue": str(args.catalogue), "output": str(args.output), "report": str(args.report)},
              "scope": "Public GPS labels only. Ranking uses metadata, never held-out errors. Metadata counts are not unique physical identities; overlap and complete seasons remain to audit.",
              "ranked_sources": ranked, "selected_ids": [r["id"] for r in selected], "results": []}
    report["access_or_observation_gaps"] = [r for r in ranked if r["status"] != "eligible"][:30]
    save(args.report, report)
    print(f"Ranked {len(ranked)} migration-related packages; selected {len(selected)} GPS/licence-eligible sources", flush=True)
    # Catalogue calls are bounded by max-studies and occur before any acquisition.
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(bitstreams, records[r["id"]]): r for r in selected if not r["files"]}
        for future in as_completed(futures):
            r = futures[future]
            try:
                r["files"] = future.result()
                print(f"Metadata {r['id']} ({sum(v['bytes'] for v in r['files'])/1e6:.1f} MB): {r['title']}", flush=True)
            except Exception as exc:
                r.update({"status": "metadata error", "error": str(exc)})
    save(args.report, report)
    if not args.execute:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    # Count existing local regular bytes (symlinks reuse external immutable bytes).
    existing = sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file() and not f.is_symlink())
    budget = Budget(args.budget_mb * 1_000_000)
    budget.take(existing)
    index = cache_index(["data/raw", "data/v2/raw", "data/migration-corpus/raw"])
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(acquire, records[r["id"]], r, args.output, index, budget,
                               args.max_file_mb*1_000_000, args.chunk_mb*1_000_000,
                               args.archive_max_mb*1_000_000): r for r in selected if r["status"] == "eligible"}
        for future in as_completed(futures):
            r = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"id": r["id"], "status": "error", "error": str(exc)}
            report["results"].append(result)
            report["reserved_storage_bytes"] = budget.reserved
            save(args.report, report)
            print(f"READY {r['id']}: {result['status']}; {result.get('cleanable_csv_chunks',0)} chunks; storage {budget.reserved/1e6:.1f} MB", flush=True)
    report["finished_utc"] = now()
    report["disk_free_final_bytes"] = shutil.disk_usage(args.output).free
    save(args.report, report)


if __name__ == "__main__":
    main()
