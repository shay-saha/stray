"""Offline dated GHSL human-development covariates for migration forecasts.

No avoidance costs or biological responses are imposed. R2023A historical
layers are retrospective reconstructions using later imagery, even when the
chosen nominal epoch precedes the query. This is not an operational historical
as-of backtest and it does not identify causal effects of human development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from .data import unproject
from .migration_evaluate import load_cases
from .sources import dump

PROTOCOL = "wildcast-migration-human-pressure-v1"
EPOCHS = (2000, 2005, 2010, 2015, 2020)
WINDOWS = (1, 5, 15)
MODES = ("actual", "masked", "availability")
FEATURE_NAMES = ([f"ghsl_built_fraction_{k}km_projected_window" for k in WINDOWS] +
                 [f"ghsl_available_{k}km_projected_window" for k in WINDOWS] +
                 [f"ghsl_valid_area_fraction_{k}km_projected_window" for k in WINDOWS] +
                 ["ghsl_epoch_age_25years", "ghsl_prior_epoch_present",
                  "ghsl_release_prior_calendar_year", "ghsl_retrospective_reconstruction"])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stream_sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def prior_epoch(years, epochs=EPOCHS):
    """Latest cached nominal epoch strictly before the UTC calendar year."""
    years = np.asarray(years)
    if any(int(e) not in EPOCHS for e in epochs):
        raise ValueError("Projected or unsupported GHSL epochs cannot enter historical features")
    output = np.full(years.shape, -1, dtype="int16")
    for epoch in sorted(epochs):
        output = np.where(years > epoch, epoch, output)
    return output


def encode_lookup(lookup, mode="actual"):
    """Matched encoder: same geometry, dimensions and metadata in all arms.

    Availability control keeps coverage, availability and vintage/age channels
    while removing built fractions. Masked control removes every channel.
    No target-dependent normalization or model fitting occurs here.
    """
    if mode not in MODES:
        raise ValueError("Human-pressure mode must be actual, masked or availability")
    values = np.asarray(lookup["values"], dtype="float32")
    available = np.asarray(lookup["available"], bool)
    content = np.where(available, values, 0.) if mode == "actual" else np.zeros_like(values)
    metadata = np.stack([lookup["epoch_age_years"] / 25., lookup["epoch"] >= 0,
                         lookup["release_prior_calendar_year"], lookup["retrospective_reconstruction"]], -1)
    result = np.concatenate([content, available, lookup["coverage"], metadata], -1).astype("float32")
    result *= np.asarray(lookup["query_present"])[..., None]
    if mode == "masked":
        result.fill(0.)
    if not np.isfinite(result).all():
        raise ValueError("Human-pressure encoder returned nonfinite values")
    return result


class HumanPressure:
    """Hash-verified 1km equal-area raster lookup, bounded by an LRU tile cache."""

    def __init__(self, sources, tile_size=256, max_tiles=8, expected=None, native_cache_bytes=0):
        import rasterio
        from pyproj import Transformer
        self.root, self.tile_size, self.max_tiles = Path(sources), int(tile_size), int(max_tiles)
        if not 16 <= self.tile_size <= 512 or not 1 <= self.max_tiles <= 32:
            raise ValueError("Human-pressure raster cache exceeds declared bounds")
        if type(native_cache_bytes) is not int or not 0 <= native_cache_bytes <= 16 * 1024**3:
            raise ValueError("Native raster cache must fit its explicit0..16GiB memory budget")
        self.native_cache_bytes = native_cache_bytes
        self.native_arrays = OrderedDict()
        self.source_path = self.root / "sources.json"
        source = json.loads(self.source_path.read_text())
        self.datasets, self.records, self.tiles = {}, [], OrderedDict()
        try:
            for record in source["records"]:
                if "epoch" not in record:
                    continue
                epoch = int(record["epoch"])
                if epoch not in EPOCHS or epoch in self.datasets:
                    raise ValueError("Projected, duplicated or unsupported GHSL source epoch")
                archive, raster = self._path(record["path"]), self._path(record["raster"])
                if sha256(archive) != record["sha256"]:
                    raise ValueError("GHSL original archive checksum differs from cached source manifest")
                raster_hash = sha256(raster)
                with zipfile.ZipFile(archive) as z:
                    members = [i for i in z.infolist() if Path(i.filename).name == raster.name]
                    if len(members) != 1 or members[0].file_size != raster.stat().st_size:
                        raise ValueError("GHSL raster has no unique matching original archive member")
                    with z.open(members[0]) as stream:
                        if _stream_sha(stream) != raster_hash:
                            raise ValueError("GHSL raster bytes differ from their verified archive member")
                dataset = rasterio.open(raster)
                self.datasets[epoch] = dataset
                if (dataset.crs != rasterio.crs.CRS.from_string("ESRI:54009") or dataset.count != 1 or
                        dataset.dtypes != ("uint32",) or dataset.nodata != 4294967295 or
                        not np.allclose(tuple(dataset.transform)[:6], [1000., 0., dataset.transform.c, 0., -1000., dataset.transform.f], rtol=0, atol=1e-8)):
                    raise ValueError("GHSL expected 1km north-up Mollweide uint32 built-area schema")
                self.records.append({"epoch": epoch, "archive": str(archive.resolve()), "archive_sha256": record["sha256"],
                                     "raster": str(raster.resolve()), "raster_sha256": raster_hash,
                                     "archive_member": members[0].filename, "source_url": record.get("url"),
                                     "shape": list(dataset.shape), "transform": list(dataset.transform)[:6]})
            if not self.datasets:
                raise ValueError("No supported historical GHSL sources")
        except Exception:
            self.close()
            raise
        self.transformer = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
        self.provenance = {"source_manifest": str(self.source_path.resolve()), "source_manifest_sha256": sha256(self.source_path),
                           "records": sorted(self.records, key=lambda r: r["epoch"]),
                           "release": "GHSL R2023A", "publication_year": 2023,
                           "licence": "CC BY 4.0", "licence_url": "https://human-settlement.emergency.copernicus.eu/GHSLhowToCite.php",
                           "product_url": "https://human-settlement.emergency.copernicus.eu/ghs_buS2023.php",
                           "dataset_doi": "10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA",
                           "method_reference": "Pesaresi et al. (2024), doi:10.1080/17538947.2024.2390454",
                           "time_rule": "Largest cached epoch strictly less than each measurement calendar year; candidate routes freeze maps at issue time. No 2025/2030 projections.",
                           "vintage_rule": "R2023A historical epochs are retrospective spatial-temporal interpolations/extrapolations using later imagery. Release-prior flag is conservative calendar year >2023; this does not certify operational historical availability.",
                           "space_rule": "Containing native 1km Mollweide cell and centred 5x5/15x15 native-cell windows; projected squares, not geodesic-radius circles. Built m2 divided by valid sampled cell area; coverage reports missing area. No clamping or spatial interpolation.",
                           "meaning": "Built-up surface only; not total human pressure, a fence/road/water barrier, or a causal response/avoidance cost."}
        canonical = {k: v for k, v in self.provenance.items() if k != "source_manifest"}
        canonical["records"] = [{k: v for k, v in r.items() if k not in {"archive", "raster"}} for r in canonical["records"]]
        self.fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        if expected and expected != self.fingerprint:
            self.close(); raise ValueError("Human-pressure source fingerprint differs")

    def _path(self, raw):
        path = Path(raw)
        if path.is_absolute() or path.exists():
            return path
        candidate = self.root / path.name
        if not candidate.exists():
            raise FileNotFoundError(path)
        return candidate

    def close(self):
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()
        self.tiles.clear()
        self.native_arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _tile(self, epoch, tile_row, tile_col):
        key = (int(epoch), int(tile_row), int(tile_col))
        if key in self.tiles:
            self.tiles.move_to_end(key)
            return self.tiles[key]
        import rasterio.windows
        halo, width = max(WINDOWS) // 2, self.tile_size
        window = rasterio.windows.Window(tile_col * width - halo, tile_row * width - halo, width + 2 * halo, width + 2 * halo)
        raw = self.datasets[epoch].read(1, window=window, boundless=True, masked=True).astype("float64")
        values = raw.filled(0.)
        good = ~np.ma.getmaskarray(raw)
        if np.any((values[good] < 0) | (values[good] > 1_000_000)):
            raise ValueError("GHSL sampled built area falls outside documented square-metre units")
        fractions, available, coverage = [], [], []
        middle = (slice(halo, halo + width), slice(halo, halo + width))
        for side in WINDOWS:
            cov = uniform_filter(good.astype("float64"), size=side, mode="constant")[middle]
            area = uniform_filter(values, size=side, mode="constant")[middle]
            # Divide by valid cell area; never turn NoData into undeveloped land.
            fraction = np.divide(area, cov * 1_000_000, out=np.zeros_like(area), where=cov > 1e-10)
            # Running-sum filters can leave roundoff around exact zero/one.
            fraction = np.clip(fraction, 0., 1.)
            fractions.append(fraction); available.append(cov > 1e-10); coverage.append(np.clip(cov, 0, 1))
        result = (np.stack(fractions, -1).astype("float32"), np.stack(available, -1), np.stack(coverage, -1).astype("float32"))
        self.tiles[key] = result
        while len(self.tiles) > self.max_tiles:
            self.tiles.popitem(last=False)
        return result

    def _native_raster(self, epoch):
        """Optional lossless uint32 cache; source and sampled-value checks stay."""
        if epoch in self.native_arrays:
            self.native_arrays.move_to_end(epoch)
            return self.native_arrays[epoch]
        dataset = self.datasets[epoch]
        required = dataset.width * dataset.height * np.dtype("uint32").itemsize
        if required > self.native_cache_bytes:
            return None
        while sum(a.nbytes for a in self.native_arrays.values()) + required > self.native_cache_bytes:
            self.native_arrays.popitem(last=False)
        raw = dataset.read(1, masked=False)
        self.native_arrays[epoch] = raw
        return raw

    def _native_tile(self, epoch, tile_row, tile_col):
        """Containing-cell values without unused neighbourhood filters/halos."""
        key = ("native", int(epoch), int(tile_row), int(tile_col))
        if key in self.tiles:
            self.tiles.move_to_end(key)
            return self.tiles[key]
        import rasterio.windows
        dataset, width = self.datasets[epoch], self.tile_size
        row, col = int(tile_row) * width, int(tile_col) * width
        window = rasterio.windows.Window(col, row, min(width, dataset.width-col), min(width, dataset.height-row))
        raw = dataset.read(1, window=window, masked=True).astype("float64")
        values, good = raw.filled(0.), ~np.ma.getmaskarray(raw)
        if np.any((values[good] < 0) | (values[good] > 1_000_000)):
            raise ValueError("GHSL sampled built area falls outside documented square-metre units")
        fraction = (values / 1_000_000).astype("float32")[..., None]
        result = fraction, good[..., None], good.astype("float32")[..., None]
        self.tiles[key] = result
        while len(self.tiles) > self.max_tiles:
            self.tiles.popitem(last=False)
        return result

    def lookup(self, coordinates, measurement_time, *, native_only=False):
        """Reported WGS84 positions/time only; no identity or target inputs."""
        coordinates = np.asarray(coordinates, dtype="float64")
        if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
            raise ValueError("Expected latitude/longitude query positions")
        shape = coordinates.shape[:-1]
        times = np.asarray(measurement_time, dtype="float64")
        while times.ndim < len(shape):
            times = times[..., None]
        times = np.broadcast_to(times, shape).ravel()
        flat = coordinates.reshape(-1, 2)
        query_present = np.isfinite(flat).all(-1) & np.isfinite(times)
        query_present &= (np.abs(flat[:, 0]) <= 90) & (np.abs(flat[:, 1]) <= 180)
        years = np.zeros(len(flat), dtype="int32")
        years[query_present] = pd.to_datetime(times[query_present], unit="s", utc=True).year
        epochs = prior_epoch(years, self.datasets)
        query_present &= epochs >= 0
        channels = 1 if native_only else 3
        values, available, coverage = np.zeros((len(flat), channels), "float32"), np.zeros((len(flat), channels), bool), np.zeros((len(flat), channels), "float32")
        for epoch in np.unique(epochs[query_present]):
            ids = np.flatnonzero(query_present & (epochs == epoch))
            x, y = self.transformer.transform(flat[ids, 1], flat[ids, 0])
            dataset = self.datasets[int(epoch)]
            col = np.floor((x - dataset.transform.c) / 1000).astype("int64")
            row = np.floor((dataset.transform.f - y) / 1000).astype("int64")
            inside = (row >= 0) & (row < dataset.height) & (col >= 0) & (col < dataset.width)
            ids, row, col = ids[inside], row[inside], col[inside]
            native = self._native_raster(int(epoch)) if native_only else None
            if native is not None:
                raw = native[row, col]
                good = raw != dataset.nodata
                if np.any(raw[good] > 1_000_000):
                    raise ValueError("GHSL sampled built area falls outside documented square-metre units")
                values[ids, 0] = np.where(good, raw.astype("float64") / 1_000_000, 0).astype("float32")
                available[ids, 0] = good
                coverage[ids, 0] = good
                continue
            groups = np.column_stack([row // self.tile_size, col // self.tile_size])
            unique, inverse = np.unique(groups, axis=0, return_inverse=True)
            for group, (tr, tc) in enumerate(unique):
                local = np.flatnonzero(inverse == group)
                rr, cc = row[local] % self.tile_size, col[local] % self.tile_size
                v, m, c = (self._native_tile if native_only else self._tile)(int(epoch), int(tr), int(tc))
                values[ids[local]], available[ids[local]], coverage[ids[local]] = v[rr, cc], m[rr, cc], c[rr, cc]
        ages = np.where(epochs >= 0, years - epochs, 0).astype("float32")
        return {"values": values.reshape(*shape, channels), "available": available.reshape(*shape, channels),
                "coverage": coverage.reshape(*shape, channels), "epoch": epochs.reshape(shape),
                "epoch_age_years": ages.reshape(shape), "query_present": query_present.reshape(shape),
                "release_prior_calendar_year": ((years > 2023) & query_present).reshape(shape),
                "retrospective_reconstruction": query_present.reshape(shape)}

    def route_features(self, origins, paths, mode="actual"):
        """Predicted waypoint context, with every map frozen at forecast issue."""
        origins, paths = np.asarray(origins), np.asarray(paths)
        if paths.ndim != 4 or paths.shape[0] != len(origins) or paths.shape[-1] != 2 or not np.isfinite(paths).all():
            raise ValueError("Expected finite predicted route bank [case,candidate,horizon,2]")
        coords = unproject(paths * 1_000_000., origins[:, None, None, 0], origins[:, None, None, 1])
        return encode_lookup(self.lookup(coords, origins[:, 2]), mode)


def predictor_positions(a, manifest, ids):
    """Origin and masked actual-history positions; never access future labels."""
    ids = np.asarray(ids, dtype="int64")
    names = manifest["history_names"]
    east, north, age = [names.index(v) for v in ("east_1000km", "north_1000km", "actual_age_days")]
    origin, history, mask = a["origin"][ids], a["history"][ids], a["history_mask"][ids].astype(bool)
    selected = history[..., [east, north, age]]
    if not np.isfinite(origin).all() or not np.isfinite(selected[mask]).all() or np.any(selected[..., 2][mask] < 0):
        raise ValueError("Invalid observed origin/history metadata")
    displacement = np.where(mask[..., None], selected[..., :2], 0.)
    coords = unproject(displacement * 1_000_000., origin[:, None, 0], origin[:, None, 1])
    times = origin[:, 2:3] - np.where(mask, selected[..., 2], 0.) * 86400
    coords[~mask], times[~mask] = np.nan, np.nan
    return origin, coords, times


def prepare(data, sources, output, max_positions=3_000_000, batch_size=2048):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Human-pressure caches are immutable; choose a new directory")
    code = Path(__file__).read_bytes()
    a, manifest, hashes = load_cases(data)
    ids = np.flatnonzero(np.isin(a["split"], [0, 1, 2]))
    if batch_size < 1 or len(ids) * (1 + a["history"].shape[1]) > max_positions:
        raise ValueError("Human-pressure lookup exceeds the finite position/batch quota")
    arrays = {"case_ids": ids, "split": a["split"][ids]}
    with HumanPressure(sources) as context:
        origin_parts, history_parts = [], []
        for start in range(0, len(ids), batch_size):
            selected = ids[start:start + batch_size]
            origin, coordinates, times = predictor_positions(a, manifest, selected)
            origin_parts.append(context.lookup(origin[:, :2], origin[:, 2]))
            history_parts.append(context.lookup(coordinates, times))
            if start % (batch_size * 10) == 0:
                print(f"Human-pressure development origins {start}/{len(ids)}", flush=True)
        for prefix, parts in (("origin", origin_parts), ("history", history_parts)):
            for key in parts[0]:
                arrays[prefix + "_" + key] = np.concatenate([p[key] for p in parts])
        provenance, fingerprint = context.provenance, context.fingerprint
    for name, key in (("manifest.json", "manifest_sha256"), ("cases.npz", "cases_sha256")):
        if sha256(Path(data) / name) != hashes[key]:
            raise ValueError("Base data changed while preparing human-pressure features")
    if sha256(Path(sources) / "sources.json") != provenance["source_manifest_sha256"]:
        raise ValueError("GHSL source manifest changed while preparing features")
    for record in provenance["records"]:
        if sha256(record["raster"]) != record["raster_sha256"]:
            raise ValueError("GHSL raster changed while preparing features")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "features.npz", **arrays)
    (output / "preparation-source.py").write_bytes(code)
    summary = {"protocol": PROTOCOL, "data": str(Path(data).resolve()), **hashes,
               "source_fingerprint": fingerprint, "source_provenance": provenance,
               "features_sha256": sha256(output / "features.npz"), "code_sha256": hashlib.sha256(code).hexdigest(),
               "feature_names": FEATURE_NAMES, "modes": MODES, "development_origins": len(ids),
               "positions_quota": max_positions, "reserved_origins_prepared": 0,
               "origin_coverage": arrays["origin_available"].mean(0).tolist(),
               "history_coverage": arrays["history_available"].mean((0, 1)).tolist(),
               "origin_epoch_counts": {str(int(e)): int(np.sum(arrays["origin_epoch"] == e)) for e in np.unique(arrays["origin_epoch"])},
               "provenance": {"uses_future_targets": False, "uses_query_identity": False, "threshold_or_normalizer_fit": False,
                              "historical_product_may_use_later_imagery": True, "operational_historical_backtest": False,
                              "development_causal_effect_identified": False, "hard_avoidance_cost": False},
               "limitations": ["Built-up surface is one human-development proxy; roads, fences, dams, fishing, shipping, wind and noise remain absent.",
                   "Nominal past epochs do not remove retrospective imagery/publication-vintage limitations.",
                   "Projected square windows are not geodesic circular buffers; 1km raster resolution cannot describe individual obstacles.",
                   "Route lookup samples forecast waypoints only and freezes all maps at issue time.",
                   "Missing raster/time coverage is explicit; zero built-up surface is not zero human impact."],
               "observed_utc": datetime.now(timezone.utc).isoformat()}
    dump(output / "manifest.json", summary)
    return summary


def load_features(data, path, mode="actual"):
    """Return verified development-only origin/history encodings."""
    root = Path(path)
    metadata = json.loads((root / "manifest.json").read_text())
    a, _, hashes = load_cases(data)
    if metadata.get("protocol") != PROTOCOL or any(metadata.get(k) != v for k, v in hashes.items()):
        raise ValueError("Human-pressure cache/base provenance mismatch")
    if sha256(root / "features.npz") != metadata["features_sha256"]:
        raise ValueError("Human-pressure feature checksum mismatch")
    with HumanPressure(Path(metadata["source_provenance"]["source_manifest"]).parent, expected=metadata["source_fingerprint"]):
        pass
    with np.load(root / "features.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    expected_ids = np.flatnonzero(np.isin(a["split"], [0, 1, 2]))
    if not np.array_equal(arrays["case_ids"], expected_ids) or not np.array_equal(arrays["split"], a["split"][expected_ids]):
        raise ValueError("Human-pressure cache contains another origin set or reserved cases")
    encoded = {}
    for prefix in ("origin", "history"):
        raw = {k[len(prefix) + 1:]: v for k, v in arrays.items() if k.startswith(prefix + "_")}
        encoded[prefix] = encode_lookup(raw, mode)
    return arrays["case_ids"], encoded, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "sources", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--max-positions", type=int, default=3_000_000)
    p.add_argument("--batch-size", type=int, default=2048)
    result = prepare(**vars(p.parse_args()))
    print(json.dumps({k: result[k] for k in ("protocol", "development_origins", "reserved_origins_prepared", "origin_epoch_counts")}, indent=2))


if __name__ == "__main__":
    main()
