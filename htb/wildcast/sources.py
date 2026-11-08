"""Public archive ingestion. Never use the authenticated Movebank service."""
from __future__ import annotations

import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://datarepository.movebank.org/server/api"
JAGUAR_REVISION = "11948ed0a32248cd4ce904089e95362fd51da217"
JAGUAR_SHA256 = "f7da08430c9f2eb72f61a0d49371e795bfde7126eb250783b885d3ba2b1ee2c9"


def session():
    s = requests.Session()
    s.headers["User-Agent"] = "WildCast-HackTheBurgh/0.1 (public research dataset client)"
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return s


def get_json(s, url, **params):
    r = s.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def download(s, url, path, max_bytes, checksum=None):
    """Atomic bounded download; recheck cached bytes against published checksum."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size <= max_bytes:
        if not checksum or hashlib.new(checksum[0], path.read_bytes()).hexdigest() == checksum[1]:
            return path
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with s.get(url, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            if int(r.headers.get("Content-Length", 0)) > max_bytes:
                raise ValueError(f"Download exceeds byte budget: {url}")
            n = 0
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    n += len(chunk)
                    if n > max_bytes:
                        raise ValueError(f"Download exceeds byte budget: {url}")
                    f.write(chunk)
        if checksum and hashlib.new(checksum[0], tmp.read_bytes()).hexdigest() != checksum[1]:
            raise ValueError(f"Published checksum mismatch: {url}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def jaguar(root):
    """Full 117-animal author release; distinct from the smaller Movebank study."""
    root = Path(root) / "raw" / "jaguar-morato-2018"
    s = session()
    base = f"https://raw.githubusercontent.com/ICMBio-CENAP/jaguar_movement/{JAGUAR_REVISION}/"
    archive = download(s, base + "DataS1_v1_0.zip", root / "DataS1_v1_0.zip", 10_000_000, ("sha256", JAGUAR_SHA256))
    with zipfile.ZipFile(archive) as z:
        # Extract just the event table; preserve the archive containing source metadata.
        name = next(n for n in z.namelist() if n.endswith("jaguar_movement_data.csv"))
        (root / "events.csv").write_bytes(z.read(name))
    download(s, base + "README.md", root / "README.source.md", 100_000)
    dump(root / "source.json", {
        "source_id": "jaguar-morato-2018", "title": "Jaguar movement database (117 individuals)",
        "url": base + "DataS1_v1_0.zip", "revision": JAGUAR_REVISION,
        "citation": "Morato et al. 2018, Ecology 99:1691, doi:10.1002/ecy.2379",
        "license": "Publication states no costs, copyright, or proprietary restrictions; cite authors. Repository code GPL-2.0.",
        "license_evidence": "https://doi.org/10.1002/ecy.2379",
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "timestamp_format": "%m/%d/%y %H:%M",
        "timestamp_note": "Source timestamps treated as UTC for this prototype; verify source timezone before operational use.",
    })
    return root


def catalogue(root):
    """Paginate all public data-package metadata, including exact rights statements."""
    s = session()
    records, page = [], 0
    while True:
        j = get_json(s, API + "/discover/search/objects", query="dspace.entity.type:Datapackage", size=20, page=page)
        result = j["_embedded"]["searchResult"]
        for obj in result.get("_embedded", {}).get("objects", []):
            item = obj["_embedded"]["indexableObject"]
            meta = {k: [v["value"] for v in vals] for k, vals in item["metadata"].items()}
            records.append({"id": item["uuid"], "title": item["name"], "metadata": meta,
                            "url": f"https://datarepository.movebank.org/entities/datapackage/{item['uuid']}"})
        print(f"Catalogued {len(records)} / {result['page']['totalElements']} public data packages", flush=True)
        if page + 1 >= result["page"]["totalPages"]:
            break
        page += 1
        time.sleep(0.15)
    dump(Path(root) / "catalogue.json", {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "api": API, "records": records})
    return records


def harvest(root, query="", max_studies=20, budget_mb=512, max_file_mb=50):
    """Download public CSV events with explicit CC0/CC BY rights and checksums.

    Each item is isolated and recorded. Files with unavailable rights, restricted
    access, or other licenses remain catalogued and are not silently imported.
    """
    root = Path(root)
    records = json.loads((root / "catalogue.json").read_text())["records"]
    s, used, count, report = session(), 0, 0, []
    for rec in records:
        text = json.dumps(rec).lower()
        if query and not any(word.strip().lower() in text for word in query.split(",")):
            continue
        rights = [u.lower() for u in rec["metadata"].get("dc.rights.uri", [])]
        allowed = ["/publicdomain/zero/1.0", "/licenses/by/4.0", "/licenses/by/3.0"]
        if not rights or not all(any(x in u for x in allowed) for u in rights):
            continue
        if count >= max_studies:
            break
        try:
            bundles = get_json(s, f"{API}/core/items/{rec['id']}/bundles", size=100)["_embedded"]["bundles"]
            files = []
            for bundle in bundles:
                if bundle["name"] != "ORIGINAL":
                    continue
                url = bundle["_links"]["bitstreams"]["href"]
                while url:
                    j = get_json(s, url, size=100)
                    files.extend(j.get("_embedded", {}).get("bitstreams", []))
                    url = j.get("_links", {}).get("next", {}).get("href")
            candidates = [b for b in files if b["name"].lower().endswith(".csv") and "reference" not in b["name"].lower()]
            candidates = [b for b in candidates if b["sizeBytes"] <= max_file_mb * 1_000_000]
            if not candidates or used + sum(b["sizeBytes"] for b in candidates) > budget_mb * 1_000_000:
                continue
            dest = root / "raw" / rec["id"]
            downloaded = []
            for b in files:
                if b not in candidates and not (b["name"].lower().endswith((".txt", ".csv")) and b["sizeBytes"] < 200_000):
                    continue
                if used + b["sizeBytes"] > budget_mb * 1_000_000:
                    continue
                ck = b["checkSum"]
                path = download(s, b["_links"]["content"]["href"], dest / Path(b["name"]).name,
                                max_file_mb * 1_000_000, (ck["checkSumAlgorithm"].lower(), ck["value"]))
                used += path.stat().st_size
                downloaded.append({"name": path.name, "url": b["_links"]["content"]["href"],
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "published_checksum": ck})
            dump(dest / "source.json", {**rec, "source_id": rec["id"], "files": downloaded,
                 "retrieved_utc": datetime.now(timezone.utc).isoformat()})
            count += 1
            report.append({"id": rec["id"], "title": rec["title"], "status": "downloaded", "files": len(downloaded)})
            print(f"[{count}] {rec['title']} ({used / 1e6:.1f} MB total)", flush=True)
            time.sleep(0.2)
        except (requests.RequestException, ValueError, KeyError) as e:
            report.append({"id": rec["id"], "status": "error", "error": str(e)})
    dump(root / "harvest-report.json", {"bytes": used, "studies": count, "items": report})
    return report
