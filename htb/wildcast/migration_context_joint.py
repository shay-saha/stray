"""Issue-dated GHSL inputs for the corrected joint movement model.

The original frozen GPS cache is unchanged. A separate, hash-bound table adds
native built-up land at the issue position and observed one/seven-day history
positions. Historical GHSL is retrospective, not an operational as-of source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import unproject
from .migration_human_pressure import MODES, encode_lookup, sha256
from .migration_model import MigrationModel

KIND = "wildcast_migration_context_joint"
PROTOCOL = "wildcast-migration-context-joint-v1"
FIELDS = ("built_fraction_native_1km", "available", "valid_area_fraction", "epoch_age_25years",
          "prior_epoch_present", "release_prior_calendar_year", "retrospective_reconstruction")
FEATURE_NAMES = [position + "_" + field for position in ("origin", "observed_1day", "observed_7day") for field in FIELDS]


def positions(a, ids):
    """Only the current position and actually observed past predictors are read."""
    ids = np.asarray(ids, dtype=np.int64)
    origin = np.asarray(a["origin"][ids], dtype=float)
    history, mask = a["history"][ids], a["history_mask"][ids]
    coords = np.full((len(ids), 3, 2), np.nan)
    times = np.full((len(ids), 3), np.nan)
    coords[:, 0], times[:, 0] = origin[:, :2], origin[:, 2]
    for column, lag in enumerate((1., 7.), 1):
        age = history[..., 2]
        distance = np.where(mask & (age > 0), np.abs(age - lag), np.inf)
        selected = distance.argmin(-1)
        rows = np.arange(len(ids))
        present = distance[rows, selected] <= max(.25, .2 * lag)
        xy = history[rows, selected, :2] * 1_000_000
        coords[present, column] = unproject(xy[present], origin[present, 0], origin[present, 1])
        times[present, column] = origin[present, 2] - age[rows, selected][present] * 86400
    if np.any(times[np.isfinite(times)] > np.broadcast_to(origin[:, 2:3], times.shape)[np.isfinite(times)]):
        raise ValueError("Context position is later than forecast issue")
    return coords, times


def features(context, a, ids, mode="actual"):
    coords, times = positions(a, ids)
    lookup = context.lookup(coords, times, native_only=True)
    result = encode_lookup(lookup, mode)
    if result.shape != (len(ids), 3, len(FIELDS)):
        raise ValueError("Native GHSL context schema differs")
    return result.reshape(len(ids), -1)


def transform(values, mode):
    if mode not in MODES:
        raise ValueError("Choose actual/availability/masked context")
    result = np.asarray(values, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != len(FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError("Invalid context feature matrix")
    if mode == "masked":
        result.fill(0)
    elif mode == "availability":
        result[:, ::len(FIELDS)] = 0
    return result


def prepare(data, sources, output, batch_size=32768):
    from .migration_deepssf import GHSLRaster
    output, data = Path(output), Path(data)
    if output.exists():
        raise FileExistsError("Context tables are immutable; use a new output")
    if batch_size < 1:
        raise ValueError("Positive context batch size required")
    hashes = {"manifest_sha256": sha256(data / "manifest.json"), "cases_sha256": sha256(data / "cases.npz")}
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("cases_sha256", manifest.get("files_sha256", {}).get("cases.npz")) != hashes["cases_sha256"]:
        raise ValueError("Context preparation requires an intact frozen cache")
    with np.load(data / "cases.npz", allow_pickle=False) as source:
        # No target arrays, reserved histories, identities, sex or fate are read.
        split = source["split"]
        ids = np.flatnonzero(np.isin(split, [0, 1, 2]))
        a = {key: source[key][ids] for key in ("history", "history_mask", "origin")}
    raster = GHSLRaster(sources)
    try:
        values = np.empty((len(ids), len(FEATURE_NAMES)), np.float32)
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            values[start:end] = features(raster.context, a, np.arange(start, end))
            print(json.dumps({"context_origins": end, "total": len(ids)}), flush=True)
        metadata = {"protocol": PROTOCOL, **hashes, "feature_names": FEATURE_NAMES,
                    "sources": str(Path(sources).resolve()), "source_fingerprint": raster.fingerprint,
                    "source_provenance": raster.provenance, "origins": len(ids),
                    "case_ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                    "uses_future_targets": False,
                    "rule": "Native built fraction and missing/vintage flags at current and observed 1/7-day past positions; each position uses a strictly prior nominal epoch",
                    "coverage": {name: float((values[:, column] > 0).mean())
                                 for column, name in enumerate(FEATURE_NAMES) if name.endswith("_available")},
                    "code_sha256": sha256(Path(__file__))}
    finally:
        raster.close()
    if sha256(data / "cases.npz") != hashes["cases_sha256"] or sha256(data / "manifest.json") != hashes["manifest_sha256"]:
        raise ValueError("GPS cache changed during context preparation")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "features.npz", case_ids=ids, features=values)
    metadata["features_sha256"] = sha256(output / "features.npz")
    (output / "features.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def append_features(a, hashes, artifact, mode):
    root = Path(artifact)
    metadata = json.loads((root / "features.json").read_text())
    if metadata.get("protocol") != PROTOCOL or metadata.get("feature_names") != FEATURE_NAMES or metadata.get("uses_future_targets") is not False:
        raise ValueError("Unrecognized context feature protocol")
    if any(metadata.get(key) != value for key, value in hashes.items()):
        raise ValueError("Context features and GPS cache hashes differ")
    if sha256(root / "features.npz") != metadata.get("features_sha256"):
        raise ValueError("Context feature table checksum differs")
    with np.load(root / "features.npz", allow_pickle=False) as source:
        ids, values = source["case_ids"], source["features"]
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or np.any(np.diff(ids) <= 0)
            or values.shape != (len(ids), len(FEATURE_NAMES))
            or hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest() != metadata["case_ids_sha256"]):
        raise ValueError("Context table has invalid canonical origin IDs")
    positions = np.searchsorted(ids, a["case_index"])
    if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], a["case_index"]):
        raise ValueError("Context table does not cover every selected origin")
    a = {**a, "static": np.concatenate((a["static"], transform(values[positions], mode)), -1)}
    return a, {**metadata, "mode": mode, "artifact": str(root.resolve()),
               "metadata_sha256": sha256(root / "features.json"), "base_static_features": a["static"].shape[1] - len(FEATURE_NAMES)}


def initialize_expanded(model, checkpoint, hashes, device):
    """Zero new input weights: the initial predictive law equals its parent."""
    before = sha256(checkpoint)
    source = torch.load(checkpoint, map_location=device, weights_only=True)
    if sha256(checkpoint) != before or source.get("model_kind") != "wildcast_migration_joint":
        raise ValueError("Context initialization needs a frozen joint checkpoint")
    if any(source.get(key) != value for key, value in hashes.items()):
        raise ValueError("Context initialization and cache hashes differ")
    expected = {**model.config, "static_features": source["config"]["static_features"]}
    if expected != source["config"] or model.config["static_features"] != expected["static_features"] + len(FEATURE_NAMES):
        raise ValueError("Context initialization architecture differs beyond added inputs")
    state = model.state_dict()
    count = expected["static_features"]
    for key, value in source["model"].items():
        if key in ("static_mean", "static_std"):
            state[key][:count].copy_(value)
        elif key == "static_encoder.0.weight":
            state[key].zero_(); state[key][:, :count].copy_(value)
        else:
            if state[key].shape != value.shape:
                raise ValueError("Unexpected context initialization parameter shape")
            state[key].copy_(value)
    model.load_state_dict(state)
    return {"path": str(Path(checkpoint).resolve()), "sha256": before, "epoch": source.get("epoch"),
            "optimizer_state_carried": False, "expansion": "Extra context columns initialized to zero; parent prediction and training normalizers retained; new normalizers fitted on TRAIN only"}


class ContextJoint:
    def __init__(self, model, metadata):
        self.model, self.metadata, self.raster = model, metadata, None

    @torch.no_grad()
    def predict_cases(self, a, ids, samples, rng=None, baselines=None):
        mode = self.metadata["mode"]
        if mode == "masked":
            extra = np.zeros((len(ids), len(FEATURE_NAMES)), np.float32)
        else:
            if self.raster is None:
                from .migration_deepssf import GHSLRaster
                self.raster = GHSLRaster(self.metadata["sources"], expected=self.metadata["source_fingerprint"])
            extra = features(self.raster.context, a, ids, mode)
        static = np.concatenate((a["static"][ids], extra), -1)
        device = self.model.static_mean.device
        inputs = [torch.as_tensor(a[key][ids], device=device) for key in ("history", "history_mask")]
        inputs += [torch.as_tensor(static, device=device), torch.as_tensor(a["species"][ids], device=device)]
        rng = rng or np.random.default_rng(2026)
        generator = torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**63 - 1)))
        return self.model.sample_paths(*inputs, samples=samples, generator=generator).cpu().numpy()


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    if ck.get("model_kind") != KIND or not ck.get("context_features"):
        raise ValueError("Not a context-enabled joint movement checkpoint")
    model = MigrationModel(**ck["config"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return ContextJoint(model, ck["context_features"]), ck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True); parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--batch-size", type=int, default=32768)
    metadata = prepare(**vars(parser.parse_args()))
    print(json.dumps({key: metadata[key] for key in ("origins", "source_fingerprint", "coverage")}))


if __name__ == "__main__":
    main()
