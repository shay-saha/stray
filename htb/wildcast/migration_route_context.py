"""Offline, previous-year AlphaEarth context along predicted migration routes.

These are landscape embedding summaries at forecast waypoints, not a terrain
barrier model or observations along the true future route. No data are fetched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import unproject


FEATURE_NAMES = ([f"aef_route_mean_{i}" for i in range(64)] +
                 [f"aef_terminal_minus_origin_{i}" for i in range(64)] +
                 ["aef_route_coverage", "aef_terminal_present", "aef_origin_present", "aef_terminal_origin_both_present"])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RouteContext:
    """Verified local WGS84 embedding tiles with strict calendar vintage rules."""
    def __init__(self, root, expected=None):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.records, self.values = [], []
        for record in manifest["records"]:
            path = self.root / (record["key"] + ".npz")
            actual = sha256(path)
            if actual != record["sha256"]:
                raise ValueError("Route context field checksum mismatch")
            with np.load(path, allow_pickle=False) as tile:
                values, bounds, year = tile["values"], tile["bounds"], int(tile["year"])
            if year != int(record["year"]) or not np.allclose(bounds, record["bounds"], rtol=0, atol=1e-10):
                raise ValueError("Route context field metadata mismatch")
            if values.ndim != 3 or values.shape[-1] != 64 or values.dtype != np.int8:
                raise ValueError("Route context requires encoded 64D int8 vector fields")
            west, south, east, north = np.asarray(bounds, float)
            if not (-180 <= west < east <= 180.000000001 and -90 <= south < north <= 90):
                raise ValueError("Unsupported or invalid WGS84 field extent")
            self.records.append({"key": record["key"], "year": year, "bounds": list(map(float, bounds)), "sha256": actual})
            self.values.append(values)
        self.provenance = {"fields_root": str(self.root), "manifest_sha256": sha256(manifest_path),
                           "records": self.records, "code_sha256": sha256(__file__),
                           "temporal_rule": "Exactly origin UTC calendar year minus one; no year substitution",
                           "grid_rule": "WGS84 north-up half-open cells; only in-bounds pixels; no coordinate or pixel clamping",
                           "decode_rule": "sign(q)*(q/127.5)^2; all 64 components must differ from nodata -128"}
        canonical = {k: v for k, v in self.provenance.items() if k != "fields_root"}
        self.fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        if expected and expected != self.fingerprint:
            raise ValueError("Route context source fingerprint mismatch")

    def lookup(self, coordinates, previous_year):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.shape[-1] != 2:
            raise ValueError("Expected latitude/longitude coordinates")
        shape = coordinates.shape[:-1]
        years = np.asarray(previous_year)
        while years.ndim < len(shape):
            years = years[..., None]
        years = np.broadcast_to(years, shape).ravel()
        flat = coordinates.reshape(-1, 2)
        output = np.zeros((len(flat), 64), np.float32)
        present = np.zeros(len(flat), bool)
        finite = np.isfinite(flat).all(-1)
        # Try all overlapping tiles, allowing a valid neighbour to fill nodata.
        for record, values in zip(self.records, self.values):
            west, south, east, north = record["bounds"]
            valid = finite & ~present & (years == record["year"])
            valid &= (flat[:, 1] >= west) & (flat[:, 1] < east) & (flat[:, 0] > south) & (flat[:, 0] <= north)
            ids = np.flatnonzero(valid)
            if not len(ids):
                continue
            height, width = values.shape[:2]
            row = np.floor((north - flat[ids, 0]) / (north - south) * height).astype(int)
            col = np.floor((flat[ids, 1] - west) / (east - west) * width).astype(int)
            inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
            ids, row, col = ids[inside], row[inside], col[inside]
            raw = values[row, col]
            good = (raw != -128).all(-1)
            decoded = raw[good].astype(np.float32)
            output[ids[good]] = np.sign(decoded) * np.square(decoded / 127.5)
            present[ids[good]] = True
        return output.reshape(*shape, 64), present.reshape(shape)

    def features(self, origin, paths, return_masks=False):
        origin, paths = np.asarray(origin), np.asarray(paths)
        if paths.ndim != 4 or paths.shape[0] != len(origin) or paths.shape[-1] != 2:
            raise ValueError("Expected predicted route bank [case,candidate,horizon,2]")
        previous_year = np.asarray([datetime.fromtimestamp(float(t), timezone.utc).year - 1 for t in origin[:, 2]])
        coordinates = unproject(paths * 1_000_000., origin[:, None, None, 0], origin[:, None, None, 1])
        vectors, present = self.lookup(coordinates, previous_year)
        start, start_present = self.lookup(origin[:, :2], previous_year)
        count = present.sum(-1)
        mean = vectors.sum(-2) / np.maximum(count[..., None], 1)
        both = present[..., -1] & start_present[:, None]
        delta = np.where(both[..., None], vectors[..., -1, :] - start[:, None, :], 0.)
        flags = np.stack((present.mean(-1), present[..., -1],
                          np.broadcast_to(start_present[:, None], present.shape[:2]), both), axis=-1)
        features = np.concatenate((mean, delta, flags), axis=-1).astype(np.float32)
        return (features, present) if return_masks else features


def prepare_context(data, bank, fields, output):
    from .migration_retrieval_train import load_bank
    a, _, hashes, b, bank_metadata = load_bank(data, bank)
    context = RouteContext(fields)
    out = Path(output)
    if (out / "manifest.json").exists():
        raise FileExistsError("Use a new path to preserve route context features")
    out.mkdir(parents=True, exist_ok=True)
    features, coverage_masks = [], []
    for start in range(0, len(b["case_ids"]), 64):
        ids = b["case_ids"][start:start + 64]
        values, masks = context.features(a["origin"][ids], b["paths"][start:start + 64], return_masks=True)
        features.append(values); coverage_masks.append(masks)
    features = np.concatenate(features)
    coverage_masks = np.concatenate(coverage_masks)
    np.savez_compressed(out / "features.npz", case_ids=b["case_ids"], features=features)
    coverage = {}
    for split in (0, 1, 2):
        selected = a["split"][b["case_ids"]] == split
        if selected.any():
            coverage[str(split)] = {"waypoint_coverage": float(features[selected, :, 128].mean()),
                                    "routes_with_any_context": float((features[selected, :, 128] > 0).mean()),
                                    "origin_coverage": float(features[selected, 0, 130].mean()),
                                    "waypoint_coverage_by_horizon": coverage_masks[selected].mean((0, 1)).tolist()}
    summary = {"protocol": "wildcast-migration-route-context-v1", **hashes,
               "bank_sha256": bank_metadata["bank_sha256"], "bank_manifest_sha256": sha256(Path(bank) / "manifest.json"),
               "features_sha256": sha256(out / "features.npz"), "context_fingerprint": context.fingerprint,
               "field_provenance": context.provenance, "feature_names": FEATURE_NAMES,
               "coverage_by_split": coverage, "reserved_test_cases_prepared": 0,
               "horizons_days": a["horizons"].tolist(),
               "limitations": ["Samples forecast waypoints, not every position between sparse times.",
                   "Partial previous-year landscape coverage; no terrain/road/fence or hard habitat constraints.",
                   "The cached overview is about 1.28 km before WGS84 reprojection, not native 10 m information.",
                   "Past valid-year checks do not establish historical publication availability.",
                   "No tiles fetched or selected from query future observations or forecast errors."]}
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_context(data_hashes, bank, bank_metadata, case_ids, path):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    for key, expected in data_hashes.items():
        if manifest.get(key) != expected:
            raise ValueError("Route context/data fingerprint mismatch")
    if manifest.get("bank_sha256") != bank_metadata["bank_sha256"] or manifest.get("bank_manifest_sha256") != sha256(Path(bank) / "manifest.json"):
        raise ValueError("Route context belongs to another candidate bank")
    if manifest.get("features_sha256") != sha256(root / "features.npz"):
        raise ValueError("Route context feature checksum mismatch")
    RouteContext(manifest["field_provenance"]["fields_root"], expected=manifest["context_fingerprint"])
    with np.load(root / "features.npz", allow_pickle=False) as source:
        ids, features = source["case_ids"], source["features"]
    if not np.array_equal(ids, case_ids) or features.ndim != 3 or features.shape[-1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("Route context feature/case schema mismatch")
    return features, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "bank", "fields", "output"):
        parser.add_argument("--" + name, required=True)
    prepare_context(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
