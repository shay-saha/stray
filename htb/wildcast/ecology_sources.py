"""Bounded public trait, phylogeny, topography and historical-settlement sources."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests

from .sources import dump


def download(url, path, maximum_mb=256):
    path = Path(path)
    if not path.exists():
        response = requests.get(url, stream=True, timeout=(15, 90))
        response.raise_for_status()
        if int(response.headers.get("Content-Length", 0)) > maximum_mb * 2**20:
            raise ValueError("Source exceeds the explicit download budget")
        total = 0
        with path.with_suffix(path.suffix + ".part").open("wb") as f:
            for chunk in response.iter_content(2**20):
                total += len(chunk)
                if total > maximum_mb * 2**20:
                    raise ValueError("Source exceeds the explicit download budget")
                f.write(chunk)
        path.with_suffix(path.suffix + ".part").replace(path)
    return {"path": str(path), "url": url, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verified_utc": datetime.now(timezone.utc).isoformat()}


def fetch(root, output, max_dem_tiles=96):
    from .train import load_windows
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for article, name in [(16586228, "avonet"), (3559887, "elton")]:
        meta = requests.get(f"https://api.figshare.com/v2/articles/{article}", timeout=30)
        meta.raise_for_status()
        content = meta.json()
        dump(out / (name + "-metadata.json"), content)
        for f in content["files"]:
            if f["name"] in {"ELEData.zip", "BirdFuncDat.txt", "MamFuncDat.txt", "metadata.htm"}:
                filename = "avonet-data.zip" if f["name"] == "ELEData.zip" else "elton-metadata.htm" if f["name"] == "metadata.htm" else f["name"]
                records.append({**download(f["download_url"], out / filename, 64), "license": content["license"]})
    with zipfile.ZipFile(out / "avonet-data.zip") as z:
        for name in z.namelist():
            if name.startswith(("ELEData/PhylogeneticData/", "ELEData/TraitData/AVONET1_", "ELEData/TraitData/AVONET2_")) and not name.endswith("/"):
                (out / Path(name).name).write_bytes(z.read(name))
    a, manifest = load_windows(root)
    _, first = np.unique(a["aux"][:, 6], return_index=True)
    aux = a["aux"][first]
    years = pd.to_datetime(aux[:, 2], unit="s", utc=True).year.to_numpy()
    epochs = sorted(set((5 * ((years - 1) // 5)).clip(1975, 2020).tolist()))
    for year in epochs:
        name = f"GHS_BUILT_S_E{year}_GLOBE_R2023A_54009_1000_V1_0.zip"
        url = f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{year}_GLOBE_R2023A_54009_1000/V1-0/{name}"
        record = download(url, out / name, 256)
        record.update(provider="European Commission JRC GHSL R2023A", epoch=year, role="retrospective built-up surface m2 per 1km equal-area cell")
        records.append(record)
        with zipfile.ZipFile(out / name) as z:
            for member in z.namelist():
                if member.lower().endswith(".tif"):
                    target = out / Path(member).name
                    if not target.exists():
                        with z.open(member) as source, target.open("wb") as dest:
                            import shutil
                            shutil.copyfileobj(source, dest, 2**20)
                    record["raster"] = str(target)
        print(f"Historical settlement source {year} ready", flush=True)
    records.append({**download("https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip", out / "ne_10m_land.zip", 8),
                    "license": "Natural Earth public domain", "role": "Coarse static land/ocean geography, not historical coast reconstruction"})
    tilelist = download("https://copernicus-dem-90m.s3.amazonaws.com/tileList.txt", out / "dem-tileList.txt", 4)
    available = set((out / "dem-tileList.txt").read_text().split())
    cells, counts = np.unique(np.floor(aux[:, :2]).astype(int), axis=0, return_counts=True)
    selected = []
    for i in np.argsort(-counts, kind="stable"):
        lat, lon = cells[i]
        name = f"Copernicus_DSM_COG_30_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"
        if name in available:
            selected.append((name, int(counts[i])))
        if len(selected) >= max_dem_tiles:
            break
    dem_dir = out / "dem"
    dem_dir.mkdir(exist_ok=True)

    def one(item):
        name, count = item
        url = f"https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"
        return {**download(url, dem_dir / (name + ".tif"), 32), "origins": count,
                "provider": "Copernicus DEM GLO-90 via AWS Open Data", "resolution_m": 90,
                "license": "Copernicus DEM public licence; see https://copernicus-dem-90m.s3.amazonaws.com/readme.html"}

    with ThreadPoolExecutor(max_workers=3) as pool:
        for record in pool.map(one, selected):
            records.append(record)
            print(f"Terrain tiles ready {sum(r.get('resolution_m') == 90 for r in records)}/{len(selected)}", flush=True)
    report = {"records": records, "tilelist": tilelist, "source_windows_sha256": manifest["windows_sha256"],
              "selection": "DEM tile frequency from unlabelled forecast-origin positions; no error or future target selection.",
              "temporal_rule": "Settlement epoch strictly earlier than origin calendar year. Historical reconstruction, not publication-vintage backtesting.",
              "attribution": ["Tobias et al. AVONET, Ecology Letters 2022, doi:10.1111/ele.13898",
                              "Wilman et al. EltonTraits, Ecology 2014, doi:10.1890/13-1917.1",
                              "European Commission Joint Research Centre, GHSL R2023A",
                              "Copernicus DEM, European Union/ESA; Natural Earth"]}
    dump(out / "sources.json", report)
    return {"output": str(out), "sources": len(records), "epochs": epochs}
