"""Bounded previous-year AlphaEarth route fields selected from training history.

No target coordinates, target masks, validation origins or forecast errors are
read for acquisition planning. Public COG overview bytes are fetched through a
budgeted Range reader; rasterio subsequently reads only a local sparse file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wildcast.data import distance, unproject
from wildcast.sources import dump

PREFIX = "https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/"
PRIMARY_DOC = "https://developers.google.com/earth-engine/guides/aef_on_gcs_readme"
ATTRIBUTION = "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def training_points(data, grid_degrees=.5, rollforward_year=None):
    """Unlabelled training origins/history; never deserialize future arrays."""
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    if sha256(data / "cases.npz") != manifest["cases_sha256"]:
        raise ValueError("Migration cache fingerprint differs")
    with np.load(data / "cases.npz", allow_pickle=False) as archive:
        ids = np.flatnonzero(archive["split"] == 0)
        origins = archive["origin"][ids]
        history, mask = archive["history"][ids], archive["history_mask"][ids]
        species = archive["metric_species"][ids] if "metric_species" in archive else archive["species"][ids]
        individuals = archive["individual"][ids]
        if "history_latlon" in archive:
            coords = archive["history_latlon"][ids]
        else:
            coords = unproject(history[..., :2] * 1_000_000, origins[:, None, 0], origins[:, None, 1])
        horizons = archive["horizons"]
    previous_year = pd.to_datetime(origins[:, 2], unit="s", utc=True).year.to_numpy() - 1
    years = np.broadcast_to(previous_year[:, None], mask.shape)[mask]
    points = coords[mask]
    finite = np.isfinite(points).all(-1) & (abs(points[:, 0]) <= 90) & (abs(points[:, 1]) <= 180)
    supported = (years >= 2017) & finite
    # Causal movement scale only: use observed net displacement around seven days.
    lag = np.asarray(manifest.get("history_lags_days", []))
    j = int(np.argmin(abs(lag - 7))) if len(lag) else min(8, history.shape[1] - 1)
    valid = mask[:, j] & (history[:, j, 2] > 0)
    observed = np.linalg.norm(history[valid, j, :2], axis=-1) * 1000
    buffers = {}
    for taxon in np.unique(species):
        taxon_valid = valid & (species == taxon)
        scale = np.linalg.norm(history[taxon_valid, j, :2], axis=-1) * 1000
        buffers[int(taxon)] = float(np.clip(np.quantile(scale, .9) * np.sqrt(float(max(horizons)) / 7), 100, 1000)) if len(scale) else 250.
    buffer_km = max(buffers.values(), default=250.)
    _, animal_index, counts_per_animal = np.unique(individuals, return_inverse=True, return_counts=True)
    row_weight = 1 / counts_per_animal[animal_index] / np.maximum(mask.sum(-1), 1)
    for taxon in np.unique(species):
        selected = species == taxon
        row_weight[selected] /= len(np.unique(individuals[selected]))
    point_species = np.broadcast_to(species[:, None], mask.shape)[mask]
    point_weight = np.broadcast_to(row_weight[:, None], mask.shape)[mask]
    cells = np.column_stack([point_species[supported], years[supported], np.floor(points[supported, 0] / grid_degrees),
                              np.floor(points[supported, 1] / grid_degrees)]).astype(np.int64)
    unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    groups = pd.DataFrame({"species": unique[:, 0], "year": unique[:, 1], "lat": np.clip((unique[:, 2] + .5) * grid_degrees, -89.9999, 89.9999),
                           "lon": (unique[:, 3] + .5) * grid_degrees, "observed_history_count": counts,
                           "selection_weight": np.bincount(inverse, weights=point_weight[supported]),
                           "buffer_km": [buffers[int(taxon)] for taxon in unique[:, 0]]})
    if rollforward_year is not None:
        if int(rollforward_year) != int(manifest["validation_year"]) - 1 or int(rollforward_year) < 2017:
            raise ValueError("Rollforward product year must equal the declared forecast calendar year minus one and be at least2017")
        # Calendar choice is declared by protocol. Location support still comes
        # solely from observed TRAIN history, including its older GPS records.
        forward_cells = np.column_stack([point_species[finite], np.floor(points[finite, 0] / grid_degrees),
                                          np.floor(points[finite, 1] / grid_degrees)]).astype(np.int64)
        forward, inv, count = np.unique(forward_cells, axis=0, return_inverse=True, return_counts=True)
        future_fields = pd.DataFrame({"species": forward[:, 0], "year": int(rollforward_year),
            "lat": np.clip((forward[:, 1] + .5) * grid_degrees, -89.9999, 89.9999),
            "lon": (forward[:, 2] + .5) * grid_degrees, "observed_history_count": count,
            "selection_weight": np.bincount(inv, weights=point_weight[finite]),
            "buffer_km": [buffers[int(taxon)] for taxon in forward[:, 0]]})
        groups = pd.concat([groups, future_fields], ignore_index=True)
    return groups, {"manifest_sha256": sha256(data / "manifest.json"), "cases_sha256": manifest["cases_sha256"],
                    "training_origins": len(ids), "training_history_points": len(points),
                    "eligible_prior_year_history_points": int(supported.sum()),
                    "unsupported_prior_year_history_points": int((years < 2017).sum()),
                    "selection_grid_degrees": grid_degrees, "reachable_buffer_km": buffer_km,
                    "reachable_buffer_km_by_species": buffers,
                    "rollforward_product_year": int(rollforward_year) if rollforward_year is not None else None,
                    "rollforward_rule": "Optional year=declared validation calendar minus one, locations ONLY from observed TRAIN history; query lookups still require exactpreviousyear, never substitute for older GPS dates",
                    "selection_weighting": "Equal physical training animal within species, divided across its origins and observed history points; no target/error weighting",
                    "buffer_rule": "Per-species90th percentile observed causal seven-day displacement times sqrt(maximum horizon/7), clipped to100..1000km; no future labels"}


def _unit(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def select_tiles(groups, index, buffer_km, existing_keys=(), max_new_files=32):
    """Rank reachable source tiles by unlabelled past observation support."""
    ranked = []
    existing_keys = set(existing_keys)
    for year, points in groups.groupby("year", sort=True):
        tiles = index.loc[index.year == year].reset_index(drop=True).copy()
        if not len(tiles):
            continue
        tiles["url"] = tiles.path.str.replace("gs://", "https://storage.googleapis.com/", n=1, regex=False)
        if not tiles.url.str.startswith(PREFIX).all():
            raise ValueError("Unexpected AlphaEarth source host/path")
        tiles["key"] = tiles.url.map(lambda url: hashlib.sha256(url.encode()).hexdigest()[:24])
        tiles = tiles.loc[~tiles.key.isin(existing_keys)].reset_index(drop=True)
        if not len(tiles):
            continue
        bounds = tiles[["wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]].to_numpy()
        tree = cKDTree(_unit((bounds[:, 1] + bounds[:, 3]) / 2, (bounds[:, 0] + bounds[:, 2]) / 2))
        radius = 2 * np.sin(min(np.pi, (buffer_km + 200) / 6371.0088) / 2)
        nearby = tree.query_ball_point(_unit(points.lat.to_numpy(), points.lon.to_numpy()), radius)
        score, hits = np.zeros(len(tiles)), np.zeros(len(tiles), dtype=int)
        for point, candidates in zip(points.itertuples(index=False), nearby):
            candidate = np.asarray(candidates, dtype=int)
            if not len(candidate):
                continue
            b = bounds[candidate]
            nearest_lat = np.clip(point.lat, b[:, 1], b[:, 3])
            longitude = np.clip(point.lon + np.asarray([-360., 0., 360.])[:, None], b[None, :, 0], b[None, :, 2])
            offset = distance(nearest_lat[None], longitude, point.lat, point.lon).min(0) / 1000
            local_buffer = getattr(point, "buffer_km", buffer_km)
            keep = offset <= local_buffer
            selected = candidate[keep]
            weight = getattr(point, "selection_weight", point.observed_history_count)
            score[selected] += weight * (1 + 3 * (offset[keep] < 1e-6)) * np.exp(-offset[keep] / max(50, local_buffer / 3))
            hits[selected] += 1
        for i in np.flatnonzero(score > 0):
            tile = tiles.iloc[i]
            ranked.append({"key": tile.key, "url": tile.url, "year": int(year), "bounds": bounds[i].tolist(),
                           "selection_score": float(score[i]), "selection_history_groups": int(hits[i])})
    # A fixed geographic census rule prevents one densely observed region from
    # consuming every file. No forecast error or target determines this quota.
    strata = {}
    for record in ranked:
        west, south, east, north = record["bounds"]
        region = (record["year"], int(np.floor(((south + north) / 2 + 90) / 20)),
                  int(np.floor(((west + east) / 2 + 180) / 30)))
        record["selection_region"] = list(region)
        strata.setdefault(region, []).append(record)
    for values in strata.values():
        values.sort(key=lambda record: (-record["selection_score"], record["url"]))
    order = sorted(strata, key=lambda region: (-strata[region][0]["selection_score"], region))
    selected, rank = [], 0
    while len(selected) < max_new_files and any(rank < len(strata[region]) for region in order):
        for region in order:
            if rank < len(strata[region]):
                selected.append({**strata[region][rank], "selection_region_rank": rank + 1})
                if len(selected) >= max_new_files:
                    break
        rank += 1
    return selected


class ByteBudget:
    def __init__(self, maximum_bytes):
        self.maximum_bytes, self.used_bytes = int(maximum_bytes), 0

    def check(self, requested):
        if requested < 0 or self.used_bytes + requested > self.maximum_bytes:
            raise ValueError("AlphaEarth network byte budget exhausted")


class RangeReader(io.RawIOBase):
    """Seekable read-only HTTP file with measured, capped response-body bytes."""
    def __init__(self, url, budget, session=None, block_bytes=65536):
        super().__init__()
        if not url.startswith(PREFIX):
            raise ValueError("Unexpected AlphaEarth source URL")
        self.url, self.budget = url, budget
        self.session = session or requests.Session()
        self.block_bytes, self.position = int(block_bytes), 0
        self.blocks, self.ranges, self.size, self.etag = {}, [], None, None
        self._block(0)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative range-file seek")
        self.position = int(position)
        return self.position

    def _block(self, block):
        if block in self.blocks:
            return self.blocks[block]
        start = block * self.block_bytes
        if self.size is not None and start >= self.size:
            return b""
        end = start + self.block_bytes - 1
        if self.size is not None:
            end = min(end, self.size - 1)
        self.budget.check(end - start + 1)
        with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                              stream=True, timeout=(15, 60)) as response:
            if response.status_code != 206:
                raise ValueError("Source refused bounded HTTP Range; full-file fallback is forbidden")
            matched = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not matched:
                raise ValueError("Missing or invalid source Content-Range")
            actual_start, actual_end, size = map(int, matched.groups())
            if actual_start != start or actual_end > end or actual_end != min(end, size - 1):
                raise ValueError("Unexpected source range extent")
            if self.size is not None and self.size != size:
                raise ValueError("AlphaEarth source size changed during acquisition")
            self.size = size
            etag = response.headers.get("ETag")
            if self.etag and etag and self.etag != etag:
                raise ValueError("AlphaEarth source version changed during acquisition")
            self.etag = self.etag or etag
            expected = actual_end - actual_start + 1
            body = bytearray()
            for part in response.iter_content(65536):
                if not part:
                    continue
                self.budget.check(len(part))
                self.budget.used_bytes += len(part)
                body.extend(part)
                if len(body) > expected:
                    raise ValueError("Source exceeded its declared HTTP Range")
            if len(body) != expected:
                raise ValueError("Truncated source HTTP Range")
        result = bytes(body)
        self.blocks[block] = result
        self.ranges.append({"start": start, "end": actual_end, "sha256": hashlib.sha256(result).hexdigest()})
        return result

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded source reads are forbidden")
        end = min(self.size, self.position + size)
        parts = []
        while self.position < end:
            block, offset = divmod(self.position, self.block_bytes)
            raw = self._block(block)
            take = min(end - self.position, len(raw) - offset)
            if take <= 0:
                break
            parts.append(raw[offset:offset + take])
            self.position += take
        return b"".join(parts)

    def readinto(self, destination):
        content = self.read(len(destination))
        destination[:len(content)] = content
        return len(content)


def fetch_overview(entry, output, budget):
    """Fetch selected overview plus TIFF metadata, never full-resolution pixels."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
    import tifffile
    output = Path(output)
    reader = RangeReader(entry["url"], budget)
    with tifffile.TiffFile(reader) as image:
        pages = list(image.pages)
        chosen = [page for page in pages if page.imagewidth == 64 and page.imagelength == 64]
        if len(chosen) != 1 or chosen[0].samplesperpixel != 64:
            raise ValueError("Expected the publisher's 128x64-band vector overview")
        for offset, size in zip(chosen[0].dataoffsets, chosen[0].databytecounts):
            reader.seek(int(offset))
            reader.read(int(size))
        # Force ordinary GeoTIFF/GDAL metadata values into the bounded cache.
        for page in pages:
            for tag in page.tags.values():
                _ = tag.value
    with tempfile.TemporaryDirectory(prefix="aef-sparse-", dir=output) as work:
        sparse = Path(work) / "overview.tiff"
        with sparse.open("wb") as stream:
            stream.truncate(reader.size)
            for block, content in reader.blocks.items():
                stream.seek(block * reader.block_bytes)
                stream.write(content)
        with rasterio.open(sparse) as source:
            if source.count != 64 or 128 not in source.overviews(1):
                raise ValueError("Expected AlphaEarth 64-band 128x source overview")
            raw = source.read(out_shape=(64, source.height // 128, source.width // 128), resampling=Resampling.nearest)
            valid = (raw != -128).all(0)
            vectors = np.sign(raw.astype(float)) * (raw.astype(float) / 127.5) ** 2
            norms = np.linalg.norm(vectors, axis=0)
            if valid.any() and not np.all((norms[valid] > .8) & (norms[valid] < 1.2)):
                raise ValueError("Invalid or unread overview vectors; sparse data cannot silently become zero features")
            field = np.full((64, 128, 128), -128, dtype=np.int8)
            reproject(source=raw, destination=field, src_transform=source.transform * Affine.scale(128, 128),
                      src_crs=source.crs, src_nodata=-128, dst_transform=from_bounds(*entry["bounds"], 128, 128),
                      dst_crs="EPSG:4326", dst_nodata=-128, resampling=Resampling.nearest)
    path = output / (entry["key"] + ".npz")
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, values=field.transpose(1, 2, 0), bounds=entry["bounds"], year=entry["year"])
    path.with_suffix(".tmp").replace(path)
    return {**entry, "sha256": sha256(path), "source_size_bytes": reader.size, "source_etag": reader.etag,
            "source_ranges": reader.ranges, "source_response_bytes": sum(r["end"] - r["start"] + 1 for r in reader.ranges),
            "valid_grid_fraction": float((field != -128).all(0).mean())}


def acquire(data, output, index="data/environment-alphaearth/aef_index.parquet",
            existing_fields="data/destination-fields", max_new_files=32, budget_mb=512,
            free_floor_gb=5, plan_only=False, rollforward_year=None, selection_grid_degrees=.5):
    data, output, index, existing_fields = map(Path, [data, output, index, existing_fields])
    if max_new_files < 0 or max_new_files > 96 or budget_mb <= 0 or budget_mb > 512:
        raise ValueError("This acquisition is bounded to96newsourcefiles and512MB cumulative")
    if not 0 < selection_grid_degrees <= 1:
        raise ValueError("Selection grid must be positive and at most one degree")
    if (output / "manifest.json").exists():
        raise FileExistsError("Preserve field versions: choose a new output directory")
    if not index.exists():
        raise FileNotFoundError("Use the existing verified public AlphaEarth index; automatic global acquisition is disabled")
    groups, evidence = training_points(data, selection_grid_degrees, rollforward_year)
    old = json.loads((existing_fields / "manifest.json").read_text())
    prior_bytes = int(old.get("cumulative_network_response_bytes", old.get("network_response_bytes", 0)))
    if prior_bytes + int(budget_mb * 1_000_000) > 512_000_000:
        raise ValueError("Requested response budget exceeds the remaining cumulative512MB allowance")
    selected = select_tiles(groups, pd.read_parquet(index, columns=["path", "year", "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]),
                            evidence["reachable_buffer_km"], [r["key"] for r in old["records"]], max_new_files)
    plan = {"selection": evidence, "index_sha256": sha256(index), "selected_new_sources": selected,
            "selection_diversity": "Predeclared round-robin across year/20degree-latitude/30degree-longitude source regions; past-count score within each region",
            "existing_manifest_sha256": sha256(existing_fields / "manifest.json"), "max_new_files": max_new_files,
            "prior_acquisition_response_bytes": prior_bytes,
            "maximum_network_response_bytes": int(budget_mb * 1_000_000), "minimum_free_disk_bytes": int(free_floor_gb * 1024 ** 3),
            "source_documentation": PRIMARY_DOC, "code_sha256": sha256(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "acquisition-plan.json", plan)
    if plan_only:
        return plan
    # Fail once on a missing/broken optional dependency before source attempts.
    import rasterio
    import tifffile
    if shutil.disk_usage(output).free < plan["minimum_free_disk_bytes"]:
        raise ValueError("Insufficient free disk above the declared floor")
    records = []
    for record in old["records"]:
        source = existing_fields / (record["key"] + ".npz")
        if sha256(source) != record["sha256"]:
            raise ValueError("Existing destination field checksum mismatch")
        shutil.copy2(source, output / source.name)
        records.append({**record, "reused_from": str(existing_fields)})
    budget, errors = ByteBudget(plan["maximum_network_response_bytes"]), []
    for number, entry in enumerate(selected, 1):
        if shutil.disk_usage(output).free < plan["minimum_free_disk_bytes"]:
            errors.append({"key": entry["key"], "error": "disk_floor_reached"}); break
        try:
            record = fetch_overview(entry, output, budget)
            records.append(record)
            print(json.dumps({"source": number, "new_tiles": len(records) - len(old["records"]), "response_bytes": budget.used_bytes}), flush=True)
        except Exception as error:
            errors.append({"key": entry["key"], "error": type(error).__name__, "message": str(error)[:300]})
            print(json.dumps(errors[-1]), flush=True)
            if "budget" in str(error).lower(): break
        dump(output / "acquisition-progress.json", {"records": records, "errors": errors, "network_response_bytes": budget.used_bytes})
    result = {"records": records, "license": "CC BY 4.0", "attribution": ATTRIBUTION,
        "native_resolution_metres": 10, "used_vector_overview_metres": 1280,
        "grid": "128x128WGS84cells per tile, nearest reprojection of publisher128xvectoroverview; no additional spatial detail",
        "temporal_rule": "Previous query-origin calendar year only; years before2017remain unavailable; no nearest-year substitution",
        "availability": "Retrospective imagery; historical publication vintage and operational availability are not established",
        "coverage_selection": "New tiles ranked only by unlabelled TRAIN origins and observed past-history regions within a causal-history-derived movement buffer",
        "hard_barrier_claim": False, "source_documentation": PRIMARY_DOC, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": {"rasterio": rasterio.__version__, "tifffile": tifffile.__version__},
        "plan": plan, "errors": errors, "new_tiles": len(records) - len(old["records"]),
        "network_response_bytes": budget.used_bytes, "local_field_bytes": sum((output / (r["key"] + ".npz")).stat().st_size for r in records)}
    result["cumulative_network_response_bytes"] = prior_bytes + budget.used_bytes
    dump(output / "manifest.json", result)
    return {key: result[key] for key in ["new_tiles", "network_response_bytes", "cumulative_network_response_bytes", "local_field_bytes", "errors"]}


def field_coverage(data, fields, output, comparison_fields=None):
    """Measure input coverage only, after freezing acquisition; never read targets."""
    import torch
    from wildcast.raster_context import SatelliteFields
    data = Path(data)
    metadata = json.loads((data / "manifest.json").read_text())
    if sha256(data / "cases.npz") != metadata["cases_sha256"]:
        raise ValueError("Migration cache fingerprint differs")
    with np.load(data / "cases.npz", allow_pickle=False) as archive:
        split, origin = archive["split"], archive["origin"]
        taxon = archive["metric_species"] if "metric_species" in archive else archive["species"]
        individual = archive["individual"]
    years = pd.to_datetime(origin[:, 2], unit="s", utc=True).year.to_numpy() - 1
    results = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(8, old_threads))
    try:
        for label, directory in [("expanded", fields), ("previous", comparison_fields)]:
            if directory is None:
                continue
            sampler = SatelliteFields.from_cache(directory)
            cells = {}
            for code, name in [(0, "train"), (1, "val_individual"), (2, "val_year")]:
                ids = np.flatnonzero(split == code)
                covered = 0
                flags = np.zeros(len(ids), dtype=bool)
                for start in range(0, len(ids), 4096):
                    batch = ids[start:start + 4096]
                    with torch.inference_mode():
                        values = sampler.lookup(torch.as_tensor(origin[batch, None, :2]), torch.as_tensor(years[batch]))
                    flags[start:start + len(batch)] = values[..., -1].squeeze(-1).bool().numpy()
                    covered += int(flags[start:start + len(batch)].sum())
                eligible = int((years[ids] >= 2017).sum())
                balanced = []
                for species in np.unique(taxon[ids]):
                    species_rows = taxon[ids] == species
                    animal_means = [float(flags[species_rows & (individual[ids] == animal)].mean())
                                    for animal in np.unique(individual[ids][species_rows])]
                    balanced.append(float(np.mean(animal_means)))
                cells[name] = {"origins": len(ids), "covered_origins": covered,
                               "covered_fraction_all_origins": covered / len(ids) if len(ids) else None,
                               "physical_animal_species_macro_coverage": float(np.mean(balanced)) if balanced else None,
                               "prior_year_at_least2017_origins": eligible,
                               "covered_fraction_year_eligible": covered / eligible if eligible else None}
            results[label] = {"fields": str(directory), "manifest_sha256": sha256(Path(directory) / "manifest.json"), "splits": cells}
    finally:
        torch.set_num_threads(old_threads)
    report = {"scope": "Input coverage at unlabelled development query origins; not route coverage, forecast accuracy or habitat validity",
              "cases_sha256": metadata["cases_sha256"], "data_manifest_sha256": sha256(data / "manifest.json"),
              "code_sha256": sha256(__file__), "reserved_cases_inspected": False, "results": results}
    dump(output, report)
    return report


def independent_audit(data, output):
    """Reproduce split/identity/time checks without inspecting reserved outcomes."""
    data = Path(data)
    manifest = json.loads((data / "manifest.json").read_text())
    cases_hash = sha256(data / "cases.npz")
    if cases_hash != manifest["cases_sha256"]:
        raise ValueError("Migration cases fingerprint differs")
    origins = pd.read_parquet(data / "origins.parquet")
    if sha256(data / "origins.parquet") != manifest["origins_sha256"]:
        raise ValueError("Migration origin metadata fingerprint differs")
    previous_path = Path(manifest["existing_manifest"])
    previous = json.loads(previous_path.read_text())
    if sha256(previous_path) != manifest["existing_manifest_sha256"]:
        raise ValueError("Previous project identity inventory fingerprint differs")
    seen = {manifest["identity_mapping"].get(name, name) for name in previous["animals"]}
    reserved = set(origins.loc[origins.split.str.startswith("test"), "canonical_animal"])
    with np.load(data / "cases.npz", allow_pickle=False) as archive:
        split, individual, origin = archive["split"], archive["individual"], archive["origin"][:, 2]
        train, val, test = split == 0, split == 1, split >= 3
        hm, ht = archive["history_mask"], archive["history_time"]
        tm, tt = archive["target_mask"], archive["target_time"]
        # Test truth must be absent; asserting zero storage is not scoring it.
        sealed_coordinates = not (archive["y"][test].any() or archive["target_latlon"][test].any() or tt[test].any())
        cutoff = pd.Timestamp(int(manifest["train_end_year"]) + 1, 1, 1, tz="UTC").timestamp()
        checks = {
            "individual_train_val_disjoint": not bool(set(individual[train]) & set(individual[val])),
            "individual_train_reserved_individual_disjoint": not bool(set(individual[train]) & set(individual[split == 3])),
            "reserved_identities_absent_old_project": not bool(seen & reserved),
            "all_training_origins_before_global_cutoff": bool(np.all(origin[train] < cutoff)),
            "all_training_targets_before_global_cutoff": bool(np.all(tt[train][tm[train]] < cutoff)),
            "all_observed_targets_after_origin": bool(np.all(tt[tm] > np.broadcast_to(origin[:, None], tm.shape)[tm])),
            "all_history_past": bool(np.all(ht[hm] <= np.broadcast_to(origin[:, None], hm.shape)[hm])),
            "all_reserved_masks_sealed": not bool(tm[test].any()),
            "all_reserved_coordinates_zero": bool(sealed_coordinates),
            "no_duplicate_physical_origin": not bool(origins.duplicated(["canonical_animal", "origin_time"]).any()),
            "training_prediction_species_nonzero": bool(np.all(archive["species"][train] > 0)),
            "metric_species_preserved": len(np.unique(archive["metric_species"])) == origins.species.nunique()}
        unknown_validation = int(((archive["species"] == 0) & np.isin(split, [1, 2])).sum())
    reference_path = Path(manifest["reference_audit"]) if manifest.get("reference_audit") else None
    reference_hash = sha256(reference_path) if reference_path else None
    if reference_hash != manifest.get("reference_audit_sha256"):
        raise ValueError("Reference identity audit fingerprint differs")
    report = {"scope": "Independent identity/calendar/storage integrity; no reserved predictions, scores, route summaries or outcomes",
              "data": str(data), "manifest_sha256": sha256(data / "manifest.json"), "cases_sha256": cases_hash,
              "origins_sha256": sha256(data / "origins.parquet"), "existing_manifest_sha256": sha256(previous_path),
              "reference_audit_sha256": reference_hash, "code_sha256": sha256(__file__),
              "all_checks_pass": all(checks.values()), "checks": checks, "cases": len(origins),
              "canonical_identity_groups": len(manifest["animals"]), "training_cases": int(train.sum()),
              "reserved_individual_animals": int(origins.loc[origins.split == "test_individual", "canonical_animal"].nunique()),
              "reserved_year_animals": int(origins.loc[origins.split == "test_year", "canonical_animal"].nunique()),
              "old_project_reserved_identity_overlap": len(seen & reserved),
              "validation_unknown_species_origins": unknown_validation,
              "ambiguous_identity_records_excluded": len(manifest.get("ambiguous_identity_records", [])),
              "limitation": "Passing checks relies on the conservative canonical/reference mapping; it is not proof that all biological aliases are resolved"}
    dump(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="data/migration-destination-fields")
    parser.add_argument("--index", default="data/environment-alphaearth/aef_index.parquet")
    parser.add_argument("--existing-fields", default="data/destination-fields")
    parser.add_argument("--max-new-files", type=int, default=32)
    parser.add_argument("--budget-mb", type=float, default=512)
    parser.add_argument("--free-floor-gb", type=float, default=5)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--rollforward-year", type=int)
    parser.add_argument("--selection-grid-degrees", type=float, default=.5)
    parser.add_argument("--audit-report")
    parser.add_argument("--coverage-report")
    args = vars(parser.parse_args())
    audit_output, coverage_output = args.pop("audit_report"), args.pop("coverage_report")
    if audit_output:
        result = independent_audit(args["data"], audit_output)
    elif coverage_output:
        result = field_coverage(args["data"], args["output"], coverage_output, args["existing_fields"])
    else:
        result = acquire(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
