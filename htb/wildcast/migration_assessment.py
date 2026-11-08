"""Explicit release of sealed migration targets after model selection is frozen.

The development cache is never rewritten. Assessment truth lives in a separate
checksummed sidecar and is overlaid into copies only after provenance checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import migration_data, migration_protocol
from .data import project

SCOPE = "wildcast-migration-final-assessment-v1"
TARGETS = ("y", "target_mask", "target_latlon", "target_time")
GUARDS = ("split", "origin", "individual", "study", "horizons")


def _target_protocol(manifest):
    schema = migration_protocol.schema_for(manifest)
    return schema, 2 if schema is migration_data else 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path, expected, label):
    path = Path(path)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Missing or invalid {label} SHA-256")
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Changed or unavailable {label}: {path}")


def _resolve(value, parent=None):
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path.resolve()
    return ((Path(parent) / path) if parent else path).resolve()


def _selection(path, hashes):
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get("frozen") is not True:
        raise ValueError("Selection must explicitly declare frozen: true")
    if payload.get("scope") != SCOPE:
        raise ValueError(f"Selection scope must be {SCOPE}")
    protocol = payload.get("model_selection_protocol")
    if not isinstance(protocol, (str, dict)) or not protocol:
        raise ValueError("Selection requires its model_selection_protocol")
    if isinstance(protocol, dict):
        if protocol.get("test_targets_used") is True:
            raise ValueError("Model selection must not use reserved test targets")
        splits = protocol.get("selection_splits")
        if splits is not None and (not splits or not set(splits) <= {"val_individual", "val_year"}):
            raise ValueError("Model selection splits must be validation splits only")
    selected_data = payload.get("data", {})
    for key in ("manifest_sha256", "cases_sha256"):
        if selected_data.get(key) != hashes[key]:
            raise ValueError(f"Frozen selection data {key} differs from the base cache")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("Frozen selection must list the chosen checkpoint paths and hashes")
    for row in checkpoints:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not row["path"]:
            raise ValueError("Invalid selected checkpoint path")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))):
            raise ValueError("Invalid selected checkpoint SHA-256")
    return payload


def _load_base(data):
    data = Path(data).resolve()
    manifest_path, cases_path, origins_path = (data / name for name in ("manifest.json", "cases.npz", "origins.parquet"))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("eligible_for_final_assessment") is False:
        raise ValueError("Derived/development cache is not eligible for final assessment")
    _target_protocol(manifest)
    _require_hash(cases_path, manifest.get("cases_sha256"), "base cases")
    _require_hash(origins_path, manifest.get("origins_sha256"), "base origins")
    with np.load(cases_path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    for key in (*TARGETS, *GUARDS):
        if key not in arrays:
            raise ValueError(f"Missing base array {key}")
    migration_protocol.schema_for(manifest, arrays)
    n, h = len(arrays["split"]), len(arrays["horizons"])
    if manifest.get("cases") != n or arrays["target_mask"].dtype != bool:
        raise ValueError("Manifest case count or base target-mask type differs")
    if arrays["split"].shape != (n,) or not np.isin(arrays["split"], [0, 1, 2, 3, 4]).all():
        raise ValueError("Invalid migration split array")
    if arrays["origin"].shape != (n, 3) or arrays["individual"].shape != (n,) or arrays["study"].shape != (n,):
        raise ValueError("Invalid migration origin/group dimensions")
    for key, shape in {"y": (n, h, 2), "target_mask": (n, h), "target_latlon": (n, h, 2), "target_time": (n, h)}.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Invalid base {key} dimensions")
    horizons = np.asarray(manifest.get("horizons_days", manifest.get("horizons", [])), float)
    # The published grid uses decimal Python floats; case storage uses float32.
    # Require its exact stored representation rather than a loose tolerance.
    if (not np.array_equal(horizons.astype(arrays["horizons"].dtype), arrays["horizons"])
            or not np.isfinite(horizons).all() or np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0)):
        raise ValueError("Manifest and case horizons differ")
    if manifest.get("reserved_test_targets_sealed") is not True:
        raise ValueError("Assessment requires an explicitly sealed development cache")
    ids = np.flatnonzero(np.isin(arrays["split"], [3, 4])).astype("int64")
    if not len(ids):
        raise ValueError("No reserved test cases in this cache")
    if arrays["target_mask"][ids].any() or any(np.any(arrays[k][ids] != 0) for k in ("y", "target_latlon", "target_time")):
        raise ValueError("Base reserved targets are not sealed zeros")
    origins = pd.read_parquet(origins_path)
    required = {"case_id", "canonical_animal", "origin_time", "split", "individual", "study"}
    if not required <= set(origins) or len(origins) != n or not np.array_equal(origins.case_id.to_numpy(), np.arange(n)):
        raise ValueError("Frozen origin case IDs do not align with base arrays")
    split_names = {"train": 0, "val_individual": 1, "val_year": 2, "test_individual": 3, "test_year": 4}
    if not np.array_equal(origins.split.map(split_names).to_numpy(), arrays["split"]):
        raise ValueError("Origin split names differ from base arrays")
    seconds = pd.DatetimeIndex(pd.to_datetime(origins.origin_time, utc=True)).asi8 / 1e9
    if not np.array_equal(seconds, arrays["origin"][:, 2]) or not np.array_equal(origins.individual.to_numpy(), arrays["individual"]):
        raise ValueError("Frozen origin timestamps/group IDs differ")
    if manifest.get("studies") and not np.array_equal(origins.study.map(manifest["studies"]).to_numpy(), arrays["study"]):
        raise ValueError("Frozen origin study IDs differ")
    if manifest.get("animals") and not np.array_equal(origins.canonical_animal.map(manifest["animals"]).to_numpy(), arrays["individual"]):
        raise ValueError("Frozen canonical animal/group IDs differ")
    hashes = {"manifest_sha256": sha256(manifest_path), "cases_sha256": sha256(cases_path),
              "origins_sha256": sha256(origins_path)}
    return arrays, manifest, origins, hashes


def _provenance(manifest, parent=None):
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Cache manifest must retain source observation evidence")
    inventory = []
    for source in sources:
        path = _resolve(source["path"], parent)
        _require_hash(path, source.get("sha256"), "source observations")
        inventory.append({"path": str(path), "declared_path": source["path"], "sha256": source["sha256"]})
    reference = None
    if manifest.get("reference_audit"):
        path = _resolve(manifest["reference_audit"], parent)
        _require_hash(path, manifest.get("reference_audit_sha256"), "reference identity audit")
        reference = {"path": str(path), "sha256": manifest["reference_audit_sha256"]}
    elif manifest.get("reference_audit_sha256"):
        raise ValueError("Reference audit hash lacks its provenance path")
    existing = None
    if manifest.get("existing_manifest") and manifest.get("existing_manifest_sha256"):
        path = _resolve(manifest["existing_manifest"], parent)
        _require_hash(path, manifest["existing_manifest_sha256"], "prior cohort manifest")
        existing = {"path": str(path), "sha256": manifest["existing_manifest_sha256"]}
    audits = []
    for audit in manifest.get("audits", []):
        path = _resolve(audit["path"], parent)
        _require_hash(path, audit.get("sha256"), "auxiliary identity audit")
        audits.append({"kind": audit.get("kind"), "path": str(path), "sha256": audit["sha256"]})
    return {"sources": inventory, "reference_audit": reference, "existing_manifest": existing, "audits": audits}


def _code_hashes(manifest):
    return {"migration_assessment.py": sha256(__file__),
            **{path.name:sha256(path) for path in migration_protocol.protocol_code_files(manifest)}}


def _observations(provenance, mapping, manifest):
    required = ["animal", "timestamp", "lat", "lon", "species"]
    schema, _ = _target_protocol(manifest)
    frames = []
    for row in provenance["sources"]:
        source = pd.read_parquet(row["path"], columns=required)
        if hasattr(schema,"select_source_rows"):
            source = schema.select_source_rows(source,row["sha256"],manifest)
        frames.append(source)
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit("ns")
    valid = frame.timestamp.notna() & frame.lat.between(-90, 90) & frame.lon.between(-180, 180)
    valid &= frame.animal.notna() & frame.species.notna()
    frame = frame.loc[valid].drop_duplicates(["animal", "timestamp", "lat", "lon"]).copy()
    if not isinstance(mapping, dict) or not set(frame.animal) <= mapping.keys():
        raise ValueError("Frozen identity mapping does not cover the source observations")
    frame["original_animal"] = frame.animal
    frame["animal"] = frame.animal.map(mapping)
    # Exact frozen mapping and tie-breaking reproduce preparation. Do not rerun
    # discovery, reference interpretation or identity rules after selection.
    return frame.sort_values(["animal", "timestamp", "original_animal"], kind="stable").drop_duplicates(["animal", "timestamp"])


def _derive(arrays, manifest, origins, provenance):
    schema, minimum_targets = _target_protocol(manifest)
    ids = np.flatnonzero(np.isin(arrays["split"], [3, 4])).astype("int64")
    h = len(arrays["horizons"])
    result = {"case_ids": ids, "y": np.zeros((len(ids), h, 2), dtype=arrays["y"].dtype),
              "target_mask": np.zeros((len(ids), h), bool),
              "target_latlon": np.zeros((len(ids), h, 2), dtype=arrays["target_latlon"].dtype),
              "target_time": np.zeros((len(ids), h), dtype=arrays["target_time"].dtype),
              "origin": arrays["origin"][ids].copy(), "individual": arrays["individual"][ids].copy(),
              "study": arrays["study"][ids].copy(), "split": arrays["split"][ids].copy(),
              "canonical_animal": origins.iloc[ids].canonical_animal.to_numpy(dtype=str)}
    observations = _observations(provenance, manifest.get("identity_mapping"), manifest)
    slots = pd.DataFrame({"case_id": ids, "slot": np.arange(len(ids)),
                          "canonical_animal": origins.iloc[ids].canonical_animal.to_numpy()})
    wanted = set(slots.canonical_animal)
    available = set(observations.animal)
    if not wanted <= available:
        raise ValueError("Frozen reserved animal is absent from the verified sources")
    unit = float(manifest.get("unit_metres", 1_000_000.))
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("Invalid frozen displacement unit")
    for animal, group in observations.loc[observations.animal.isin(wanted)].groupby("animal", sort=False):
        group = group.sort_values("timestamp", kind="stable")
        times = pd.DatetimeIndex(group.timestamp).asi8 / 1e9
        lat, lon = group.lat.to_numpy(), group.lon.to_numpy()
        for row in slots.loc[slots.canonical_animal == animal].itertuples(index=False):
            origin = arrays["origin"][row.case_id]
            index = int(np.searchsorted(times, origin[2]))
            if index >= len(times) or times[index] != origin[2] or lat[index] != origin[0] or lon[index] != origin[1]:
                raise ValueError("Reserved origin does not match its frozen canonical source observation")
            ix, mask = schema.match_targets(times, index, arrays["horizons"])
            if schema is not migration_data:
                mask &= schema.target_admission_mask(arrays["horizons"], animal, manifest,
                                                     origin_time=times[index], target_times=times[ix])
            if mask.sum() < minimum_targets:
                raise ValueError("Reserved case lacks its protocol's minimum observed future targets")
            y = project(lat[ix], lon[ix], lat[index], lon[index]) / unit
            coords = np.column_stack([lat[ix], lon[ix]])
            target_time = times[ix].copy()
            y[~mask] = 0; coords[~mask] = 0; target_time[~mask] = 0
            result["y"][row.slot] = y
            result["target_mask"][row.slot] = mask
            result["target_latlon"][row.slot] = coords
            result["target_time"][row.slot] = target_time
    return result


def release(data, selection, output):
    """Release a separate assessment artifact only for an explicitly frozen selection."""
    output, selection = Path(output), Path(selection).resolve()
    if output.exists():
        raise FileExistsError("Assessment outputs are immutable; choose a new output directory")
    arrays, manifest, origins, hashes = _load_base(data)
    selection_hash = sha256(selection)
    payload = _selection(selection, hashes)
    checkpoints = []
    for checkpoint in payload["checkpoints"]:
        path = _resolve(checkpoint["path"], selection.parent)
        _require_hash(path, checkpoint["sha256"], "selected checkpoint")
        checkpoints.append({"path": str(path), "declared_path": checkpoint["path"], "sha256": checkpoint["sha256"]})
    provenance = _provenance(manifest, Path(data).resolve())
    derived = _derive(arrays, manifest, origins, provenance)
    # Recheck selection, weights and source bytes after deriving truth, before
    # publication, so a concurrent change cannot silently enter the assessment.
    if sha256(selection) != selection_hash or json.loads(selection.read_text()) != payload:
        raise ValueError("Frozen selection changed during assessment preparation")
    for row in checkpoints:
        _require_hash(row["path"], row["sha256"], "selected checkpoint")
    if _provenance(manifest, Path(data).resolve()) != provenance:
        raise ValueError("Source provenance changed during assessment preparation")
    if sha256(Path(data) / "manifest.json") != hashes["manifest_sha256"] or sha256(Path(data) / "cases.npz") != hashes["cases_sha256"]:
        raise ValueError("Base cache changed during assessment preparation")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "assessment.npz"
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **derived)
    (output / "frozen-selection.json").write_bytes(selection.read_bytes())
    metadata = {"protocol": SCOPE, "scope": "Reserved local individual/whole-year assessment after validation selection; no official MoveBench claim.",
                "observed_utc": datetime.now(timezone.utc).isoformat(), "nonce": secrets.token_hex(16),
                "base": {"path": str(Path(data).resolve()), **hashes},
                "frozen_selection_sha256": selection_hash, "frozen_selection": payload,
                "checkpoints": checkpoints, "provenance": provenance, "code_sha256": _code_hashes(manifest),
                "assessment_sha256": sha256(archive),
                "case_ids_sha256": hashlib.sha256(derived["case_ids"].astype("<i8").tobytes()).hexdigest(),
                "cases": len(derived["case_ids"]), "target_rule": manifest.get("target_rule"),
                "rule": "Only split 3/4 truth, at nearest actual observed future timestamps. Base arrays and origins are immutable. No data/error-dependent final-case selection."}
    with (output / "assessment.json").open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return metadata


def apply_assessment(a, manifest, basehashes, assessment, selection=None):
    """Verify a released sidecar and overlay truth into new arrays, only on test rows."""
    root = Path(assessment)
    metadata_path = root if root.suffix == ".json" else root / "assessment.json"
    root = metadata_path.parent
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("protocol") != SCOPE:
        raise ValueError("Wrong assessment scope/protocol")
    for key in ("manifest_sha256", "cases_sha256"):
        if basehashes.get(key) != metadata.get("base", {}).get(key):
            raise ValueError(f"Assessment base {key} differs")
    frozen, original_manifest, origins, actual_hashes = _load_base(metadata["base"]["path"])
    if actual_hashes != {k: metadata["base"].get(k) for k in actual_hashes}:
        raise ValueError("Assessment base files differ from their release hashes")
    for key, value in original_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"Supplied manifest differs from frozen {key}")
    for key in (*TARGETS, *GUARDS):
        if key not in a or not np.array_equal(a[key], frozen[key], equal_nan=True):
            raise ValueError(f"Supplied base array differs: {key}")
    selection_path = Path(selection) if selection is not None else root / "frozen-selection.json"
    _require_hash(selection_path, metadata.get("frozen_selection_sha256"), "frozen selection")
    payload = _selection(selection_path, basehashes)
    if payload != metadata.get("frozen_selection"):
        raise ValueError("Assessment frozen-selection provenance differs")
    checkpoints = metadata.get("checkpoints", [])
    if len(checkpoints) != len(payload["checkpoints"]):
        raise ValueError("Assessment checkpoint inventory differs")
    for row, declared in zip(checkpoints, payload["checkpoints"]):
        if row.get("declared_path") != declared["path"] or row.get("sha256") != declared["sha256"]:
            raise ValueError("Assessment checkpoint selection differs")
        _require_hash(row["path"], row["sha256"], "selected checkpoint")
    if _provenance(original_manifest, Path(metadata["base"]["path"])) != metadata.get("provenance"):
        raise ValueError("Assessment source/reference provenance differs")
    if metadata.get("code_sha256") != _code_hashes(original_manifest):
        raise ValueError("Assessment derivation code changed after release")
    archive_path = root / "assessment.npz"
    _require_hash(archive_path, metadata.get("assessment_sha256"), "assessment sidecar")
    with np.load(archive_path, allow_pickle=False) as archive:
        sidecar = {k: archive[k] for k in archive.files}
    ids = sidecar.get("case_ids")
    expected = np.flatnonzero(np.isin(frozen["split"], [3, 4])).astype("int64")
    if ids is None or not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(ids, expected):
        raise ValueError("Assessment case IDs must be exactly the reserved split 3/4 origins")
    if hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest() != metadata.get("case_ids_sha256"):
        raise ValueError("Assessment case ID checksum differs")
    if metadata.get("cases") != len(ids):
        raise ValueError("Assessment case count differs")
    for key in ("origin", "individual", "study", "split"):
        if key not in sidecar or not np.array_equal(sidecar[key], frozen[key][ids]):
            raise ValueError(f"Assessment original {key} differs")
    if not np.array_equal(sidecar.get("canonical_animal"), origins.iloc[ids].canonical_animal.to_numpy(dtype=str)):
        raise ValueError("Assessment canonical origins differ")
    for key in TARGETS:
        if key not in sidecar or sidecar[key].shape != frozen[key][ids].shape:
            raise ValueError(f"Assessment {key} dimensions differ")
    mask = sidecar["target_mask"]
    schema, minimum_targets = _target_protocol(original_manifest)
    if mask.dtype != bool or np.any(mask.sum(-1) < minimum_targets):
        raise ValueError("Assessment needs its protocol's minimum observed targets per case")
    if schema is not migration_data:
        for row, animal in enumerate(sidecar["canonical_animal"]):
            admitted = schema.target_admission_mask(frozen["horizons"], str(animal), original_manifest,
                                                    origin_time=frozen["origin"][ids[row], 2],
                                                    target_times=sidecar["target_time"][row])
            if np.any(mask[row] & ~admitted):
                raise ValueError("Assessment target mask violates its frozen clock admission rule")
    for key in ("y", "target_latlon", "target_time"):
        if not np.isfinite(sidecar[key]).all() or np.any(sidecar[key][~mask] != 0):
            raise ValueError(f"Invalid or unmasked assessment {key}")
    coords, times = sidecar["target_latlon"], sidecar["target_time"]
    if np.any((np.abs(coords[..., 0]) > 90) & mask) or np.any((np.abs(coords[..., 1]) > 180) & mask):
        raise ValueError("Assessment target coordinates outside geographic bounds")
    tolerances = (schema.target_tolerances(frozen["horizons"]) if schema is not migration_data
                  else np.minimum(2., np.maximum(.125, .05 * frozen["horizons"])))
    origin_seconds = frozen["origin"][ids, 2:3]
    wanted_seconds = origin_seconds + frozen["horizons"] * migration_data.DAY
    # Mirror the matcher's seconds arithmetic exactly at inclusive boundaries;
    # subtracting converted day fractions can reject a valid 75-minute fix.
    if (np.any((times <= origin_seconds) & mask)
            or np.any((np.abs(times - wanted_seconds) > tolerances * migration_data.DAY) & mask)):
        raise ValueError("Assessment target times violate the frozen matching rule")
    projected = project(coords[..., 0], coords[..., 1], frozen["origin"][ids, 0:1], frozen["origin"][ids, 1:2]) / float(manifest.get("unit_metres", 1_000_000.))
    if not np.allclose(projected[mask], sidecar["y"][mask], rtol=1e-5, atol=1e-7):
        raise ValueError("Assessment displacements differ from its coordinates")
    copied = {key: np.array(value, copy=True) for key, value in a.items()}
    for key in TARGETS:
        copied[key][ids] = sidecar[key]
    return copied


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    metadata = release(args.data, args.selection, args.output)
    print(json.dumps({"assessment": str(Path(args.output) / "assessment.json"), "cases": metadata["cases"],
                      "frozen_selection_sha256": metadata["frozen_selection_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
