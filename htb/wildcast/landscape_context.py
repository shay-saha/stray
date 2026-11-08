"""Terrain, historical built-up pressure and local movement-cost features.

Costs are declared heuristic spatial priors, not measured movement energetics.
Endpoints are past-use anchors, never invented breeding/wintering sites. The
land/ocean layer does not describe rivers, fences or all inland water bodies.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra

from .data import project, unproject
from .sources import dump
from .train import load_windows


def cost_graph(cost, resolution=1000.):
    height, width = cost.shape
    nodes = np.arange(height * width).reshape(height, width)
    rows, cols, weights = [], [], []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        ya, yb = slice(0, height - dy), slice(dy, height)
        xa, xb = (slice(0, width - dx), slice(dx, width)) if dx >= 0 else (slice(-dx, width), slice(0, width + dx))
        a, b = nodes[ya, xa].ravel(), nodes[yb, xb].ravel()
        w = (.5 * (cost[ya, xa] + cost[yb, xb]) * resolution * np.hypot(dx, dy)).ravel()
        rows.extend([a, b]);cols.extend([b, a]);weights.extend([w, w])
    return sparse.coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(height * width, height * width)).tocsr()


def movement_cost(land, slope, built, guild):
    # Birds can cross water; seals can haul out; terrestrial mammals may swim.
    # These are finite preferences rather than unsupported absolute prohibitions.
    terrain = np.nan_to_num(slope, nan=0.).clip(0, 3)
    human = np.nan_to_num(built, nan=0.).clip(0, 1)
    if guild == 1:
        return 1 + .5 * terrain + 2 * human
    if guild == 2:
        return 1 + 10 * land + 3 * human
    return 1 + 3 * terrain + 5 * (~land) + 10 * human


class Landscape:
    def __init__(self, sources, cache):
        import rasterio
        import shapefile
        import shapely
        from shapely.strtree import STRtree
        self.sources, self.cache = Path(sources), Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.source_info = json.loads((self.sources / "sources.json").read_text())
        self.source_hash = hashlib.sha256((self.sources / "sources.json").read_bytes()).hexdigest()
        self.human = {r["epoch"]: rasterio.open(r["raster"]) for r in self.source_info["records"] if "epoch" in r}
        self.dem = []
        for p in (self.sources / "dem").glob("*.tif"):
            ds = rasterio.open(p)
            self.dem.append(ds)
        with zipfile.ZipFile(self.sources / "ne_10m_land.zip") as z:
            parts = {p.rsplit(".", 1)[-1]: io.BytesIO(z.read(p)) for p in z.namelist() if p.endswith((".shp", ".shx", ".dbf"))}
            reader = shapefile.Reader(shp=parts["shp"], shx=parts["shx"], dbf=parts["dbf"])
            self.land_tree = STRtree([shapely.geometry.shape(s.__geo_interface__) for s in reader.shapes()])
            shapely.prepare(self.land_tree.geometries)

    def field(self, lat_cell, lon_cell, epoch):
        import rasterio
        from rasterio.transform import Affine
        from rasterio.warp import reproject, Resampling
        import shapely
        p = self.cache / f"field_{lat_cell}_{lon_cell}_{epoch}.npz"
        if p.exists():
            with np.load(p) as z:
                if str(z["source_hash"]) != self.source_hash:
                    raise ValueError("Landscape source fingerprints changed")
                return {k: z[k] for k in ["land", "elevation", "slope", "built"]}
        lat, lon = lat_cell + .5, lon_cell + .5
        transform = Affine(1000, 0, -64500, 0, -1000, 64500)
        crs = rasterio.crs.CRS.from_string(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +R=6371008.8 +units=m +no_defs")
        elevation = np.full((129, 129), np.nan, np.float32)
        for ds in self.dem:
            west, south, east, north = ds.bounds
            middle = (west + east) / 2
            delta_lon = abs((middle - lon + 180) % 360 - 180)
            if delta_lon > 2 or north < lat - 1 or south > lat + 1:
                continue
            tile = np.full_like(elevation, np.nan)
            reproject(rasterio.band(ds, 1), tile, dst_transform=transform, dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.average)
            elevation = np.where(np.isfinite(tile), tile, elevation)
        valid = np.isfinite(elevation)
        if valid.any():
            nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
            filled = elevation[tuple(nearest)]
            north, east = np.gradient(filled, 1000)
            slope = np.hypot(north, east)
            slope[~ndimage.binary_erosion(valid, structure=np.ones((3, 3)))] = np.nan
        else:
            slope = np.full_like(elevation, np.nan)
        built = np.full_like(elevation, np.nan)
        reproject(rasterio.band(self.human[epoch], 1), built, dst_transform=transform, dst_crs=crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
        built = np.where(built >= 0, np.clip(built / 1_000_000, 0, 1), np.nan)
        y, x = np.mgrid[0:129, 0:129]
        xy = np.stack([(x - 64) * 1000., (64 - y) * 1000.], -1)
        coordinates = unproject(xy, lat, lon)
        points = shapely.points(coordinates[..., 1].ravel(), coordinates[..., 0].ravel())
        matches = self.land_tree.query(points)
        inside = shapely.contains(self.land_tree.geometries[matches[1]], points[matches[0]])
        land = np.zeros(129 * 129, bool)
        land[matches[0, inside]] = True
        land = land.reshape(129, 129)
        field = {"land": land, "elevation": elevation, "slope": slope.astype(np.float32), "built": built}
        np.savez_compressed(p, **field, source_hash=self.source_hash)
        return field


def prepare(root, sources, output, taxonomy_root="data/combined-v2/environment/taxonomy", workers=6):
    a, manifest = load_windows(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from .identity import case_fingerprint
    expected = {"case_fingerprint": case_fingerprint(a),
                "source_manifest_sha256": hashlib.sha256((Path(sources) / "sources.json").read_bytes()).hexdigest()}
    cache_manifest = output / "cache-identity.json"
    if cache_manifest.exists() and json.loads(cache_manifest.read_text()) != expected:
        raise ValueError("Landscape caches belong to other cases or source versions")
    dump(cache_manifest, expected)
    source_manifest = Path(sources) / "sources.json"
    source_hash = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    human_epochs = {r["epoch"] for r in json.loads(source_manifest.read_text())["records"] if "epoch" in r}
    origins, first = np.unique(a["aux"][:, 6].astype(int), return_index=True)
    aux = a["aux"][first]
    years = pd.to_datetime(aux[:, 2], unit="s", utc=True).year.to_numpy()
    epochs = 5 * ((years - 1) // 5)
    if not set(epochs) <= human_epochs:
        raise ValueError("No past settlement epoch for some origins")
    taxa = {}
    for species, i in manifest["species"].items():
        p = Path(taxonomy_root) / (species.replace(" ", "_") + ".json")
        taxa[i] = json.loads(p.read_text())["response"] if p.exists() else {}
    guild = np.asarray([1 if taxa.get(int(s), {}).get("class") == "Aves" else
                        2 if taxa.get(int(s), {}).get("family") == "Phocidae" else 0 for s in a["meta"][first, 1]])
    groups = np.column_stack([np.floor(aux[:, :2]).astype(int), epochs])
    unique, inverse = np.unique(groups, axis=0, return_inverse=True)
    names = ["terrain_elevation_m", "terrain_slope_m_per_m", "terrain_5km_relief_m", "terrain_15km_relief_m",
             "terrain_grid_coverage", "geography_on_land", "geography_coast_distance_km",
             "human_built_fraction_1km", "human_built_fraction_5km", "human_built_fraction_15km", "human_epoch_age_years"]
    for anchor in range(1, 5):
        names += [f"past_anchor_{anchor}_log_local_cost_km", f"past_anchor_{anchor}_local_cost_ratio"]
    values = np.full((len(origins), len(names)), np.nan, np.float32)
    global _LANDSCAPE_JOB, _LANDSCAPE_WORKER
    _LANDSCAPE_JOB = (a, aux, first, years, guild, inverse, str(output))
    jobs = [(group, int(lat), int(lon), int(epoch)) for group, (lat, lon, epoch) in enumerate(unique)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        # Fork shares the read-only case arrays; every worker opens its own GDAL datasets.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork'),
                                 initializer=_landscape_worker_init, initargs=(sources, output / 'fields')) as executor:
            for done, (ids, group_values) in enumerate(executor.map(_landscape_group, jobs, chunksize=1)):
                values[ids] = group_values
                if done % 40 == 0:
                    print(f'Landscape/cost contexts {done}/{len(unique)}', flush=True)
    else:
        _landscape_worker_init(sources, output / 'fields')
        for item in jobs:
            ids, group_values = _landscape_group(item)
            values[ids] = group_values
    names = ["ecology_" + name for name in names]
    np.savez_compressed(output / "features.npz", origins=origins, names=np.asarray(names), values=values)
    report = {"origins": len(origins), "fields": len(unique), "coverage": {name: float(np.isfinite(values[:, i]).mean()) for i, name in enumerate(names)},
              "source_manifest_sha256": source_hash, "features_sha256": hashlib.sha256((output / "features.npz").read_bytes()).hexdigest(),
              "grid": "129x129 local spherical AEQD, 1km spacing, +/-64km from degree-cell center",
              "cost": "8-neighbour Dijkstra, finite heuristic terrain/ocean/built-up costs. Distances to observed-past anchors only, at 1km discretization. Out-of-grid anchors missing.",
              "limitations": ["No confirmed breeding/wintering sites, road/fence geometry, or complete inland-water barriers.",
                              "Static Copernicus DSM and Natural Earth geography are proxies; GHSL historical epochs are retrospective reconstructions, not historical publication vintages.",
                              "No hard claim that birds cannot cross water, mammals cannot swim, or seals cannot haul out."]}
    dump(output / "manifest.json", report)
    return report

_LANDSCAPE_JOB = None
_LANDSCAPE_WORKER = None

def _landscape_worker_init(sources, cache):
    global _LANDSCAPE_WORKER
    _LANDSCAPE_WORKER = Landscape(sources, cache)

def _landscape_group(item):
    group, lat_cell, lon_cell, epoch = item
    a, aux, first, years, guild, inverse, output = _LANDSCAPE_JOB
    landscape = _LANDSCAPE_WORKER
    values = np.full((len(aux), 19), np.nan, np.float32)
    cached = Path(output) / 'groups' / f'{lat_cell}_{lon_cell}_{epoch}.npz'
    if cached.exists():
        with np.load(cached) as z:
            return z['ids'], z['values']
    ids = np.flatnonzero(inverse == group)
    field = landscape.field(int(lat_cell), int(lon_cell), int(epoch))
    xy = project(aux[ids, 0], aux[ids, 1], lat_cell + .5, lon_cell + .5)
    col = np.rint(xy[:, 0] / 1000 + 64).astype(int).clip(0, 128)
    row = np.rint(64 - xy[:, 1] / 1000).astype(int).clip(0, 128)
    elevation, slope, built, land = [field[k] for k in ["elevation", "slope", "built", "land"]]
    values[ids, 0] = elevation[row, col]
    values[ids, 1] = slope[row, col]
    values[ids, 4] = np.isfinite(elevation).mean()
    values[ids, 5] = land[row, col]
    distance_to_other = np.where(land, ndimage.distance_transform_edt(land), ndimage.distance_transform_edt(~land))
    # A field without a coastline supplies a lower bound, not an exact coast distance.
    if land.any() and not land.all():
        values[ids, 6] = distance_to_other[row, col]
    values[ids, 7] = built[row, col]
    values[ids, 10] = years[ids] - epoch
    for j, (r, c) in enumerate(zip(row, col)):
        for k, radius in [(2, 2), (3, 7)]:
            patch = elevation[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1]
            if np.isfinite(patch).any():
                values[ids[j], k] = np.nanmax(patch) - np.nanmin(patch)
        for k, radius in [(8, 2), (9, 7)]:
            patch = built[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1]
            if np.isfinite(patch).any():
                values[ids[j], k] = np.nanmean(patch)
    anchor_xy = a["anchors"][first[ids], 1:5] * aux[ids, None, 8:9]
    ll = unproject(anchor_xy, aux[ids, None, 0], aux[ids, None, 1])
    target_xy = project(ll[..., 0], ll[..., 1], lat_cell + .5, lon_cell + .5)
    target_col = np.rint(target_xy[..., 0] / 1000 + 64).astype(int)
    target_row = np.rint(64 - target_xy[..., 1] / 1000).astype(int)
    supported = (target_col >= 0) & (target_col < 129) & (target_row >= 0) & (target_row < 129)
    source_nodes = row * 129 + col
    target_nodes = target_row.clip(0, 128) * 129 + target_col.clip(0, 128)
    direct = np.linalg.norm(anchor_xy, axis=-1)
    for mode in np.unique(guild[ids]):
        local = np.flatnonzero((guild[ids] == mode) & supported.any(-1))
        if not len(local):
            continue
        cost = movement_cost(land, slope, built, int(mode))
        if np.ptp(cost) < 1e-7:
            # Exact 8-neighbour shortest path on a uniform field, avoiding an
            # unnecessary graph search for open ocean or flat unbuilt fields.
            for s in local:
                delta_row = np.abs(target_row[s] - row[s])
                delta_col = np.abs(target_col[s] - col[s])
                octile = (np.maximum(delta_row, delta_col) + (np.sqrt(2) - 1) * np.minimum(delta_row, delta_col)) * 1000 * cost[0, 0]
                for anchor in np.flatnonzero(supported[s]):
                    value = max(float(octile[anchor]), float(direct[s, anchor]))
                    values[ids[s], 11 + 2 * anchor] = np.log1p(value / 1000)
                    values[ids[s], 12 + 2 * anchor] = value / max(float(direct[s, anchor]), 1000.)
            continue
        graph = cost_graph(cost)
        nodes = np.unique(source_nodes[local])
        for start in range(0, len(nodes), 16):
            chunk = nodes[start:start + 16]
            distances = dijkstra(graph, directed=False, indices=chunk)
            for k, node in enumerate(chunk):
                selected = local[source_nodes[local] == node]
                for s in selected:
                    for anchor in np.flatnonzero(supported[s]):
                        value = max(float(distances[k, target_nodes[s, anchor]]), float(direct[s, anchor]))
                        values[ids[s], 11 + 2 * anchor] = np.log1p(value / 1000)
                        values[ids[s], 12 + 2 * anchor] = value / max(float(direct[s, anchor]), 1000.)

    cached.parent.mkdir(exist_ok=True)
    np.savez_compressed(cached, ids=ids, values=values[ids])
    return ids, values[ids]

