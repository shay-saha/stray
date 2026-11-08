"""Build a verified portable offline inference bundle from a selected snapshot.

Only selected checkpoints, immutable reference files, necessary local fields,
bounded provenance and source code are copied. This command never trains,
selects a model, forecasts a reserved animal or fetches network resources.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT))

from wildcast.migration_evaluate import load_cases

KINDS = {"wildcast_migration_joint", "wildcast_migration_retrieval",
         "wildcast_migration_context_joint",
         "wildcast_migration_moveformer", "wildcast_migration_moveformer_endpoint",
         "wildcast_migration_ensemble", "wildcast_migration_regime"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)


def resolve_asset(value, checkpoint, source_root):
    """Resolve declared dependencies inside explicitly trusted source roots."""
    text = str(value)
    if not text or "\\" in text or "://" in text or ".." in PurePosixPath(text).parts:
        raise ValueError("Unsafe asset path traversal or non-file reference")
    supplied = Path(text)
    roots = [Path(source_root).resolve(), Path(checkpoint).resolve().parent]
    candidates = [supplied] if supplied.is_absolute() else [roots[1] / supplied, roots[0] / supplied]
    for candidate in candidates:
        resolved = candidate.resolve()
        if not _inside(resolved, roots):
            continue
        if resolved.exists():
            return resolved
    raise ValueError("Missing asset or reference escapes trusted source roots: " + text)


def _copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise ValueError("Expected a regular selected asset: " + str(source))
    before = sha256(source)
    if expected is not None and before != expected:
        raise ValueError("Selected asset checksum mismatch: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256(source) != before or sha256(destination) != before:
        raise ValueError("Selected asset changed while copying: " + str(source))
    return before


def tensor_digest(checkpoint):
    """Device-independent hash of predictive parameter/buffer values."""
    digest = hashlib.sha256()
    values = checkpoint.get("model", {})
    if "weights" in checkpoint:
        values = {**values, "ensemble_weights": checkpoint["weights"]}
    for name, tensor in sorted(values.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Predictive checkpoint state must contain tensors")
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def regime_policy_digest(checkpoint):
    """Hash routing rules and policy independently of relocatable asset paths."""
    keys = ("rules", "active_fallback", "active_scale", "sampling_rule", "uncertainty_rule")
    policy = {key: checkpoint.get(key) for key in keys}
    if checkpoint.get("calibration"):
        policy["calibration"] = {key: value for key, value in checkpoint["calibration"].items() if key != "path"}
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


VERIFY_SOURCE = '''"""Verify the immutable package catalogue before offline inference."""
import hashlib
import json
from pathlib import Path, PurePosixPath

def verify(root=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    manifest = json.loads((root / "package-manifest.json").read_text())
    for relative, entry in manifest["files"].items():
        p = PurePosixPath(relative)
        if p.is_absolute() or ".." in p.parts or "\\\\" in relative:
            raise ValueError("Unsafe package catalogue path")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or escaping package file: " + relative)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1048576), b""):
                digest.update(chunk)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("Package file checksum mismatch: " + relative)
    return {"verified_files": len(manifest["files"]), "model_kind": manifest["model_kind"]}

if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
'''

INFER_SOURCE = '''"""Run verified local inference from any working directory."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CALLER = Path.cwd()
args = sys.argv[1:]
for i, value in enumerate(args):
    if value in ("--observations", "--output") and i + 1 < len(args):
        args[i + 1] = str((CALLER / Path(args[i + 1]).expanduser()).resolve())
    elif value.startswith("--observations=") or value.startswith("--output="):
        name, path = value.split("=", 1)
        args[i] = name + "=" + str((CALLER / Path(path).expanduser()).resolve())
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from verify_package import verify
verify(ROOT)
from wildcast.cli import main
sys.argv = [sys.argv[0], "--data", "data/reference", "migration-infer", "--checkpoint", "model/selected.pt", *args]
main()
'''


def _readme(kind, needs_moveformer):
    return f'''# WildCast offline inference bundle

This bundle contains the externally selected `{kind}` snapshot, its frozen training-reference cache, source code, and required local context assets. Packaging does not establish accuracy or select a model.

The packager supports a standalone causal regime wrapper with joint base/active members, including finalized scale evidence. Ensemble bundles currently accept direct learned members; nested ensembles and regime-wrapper members are rejected explicitly.

Inference makes no network calls. Python dependencies must already be installed, or installed from a separately supplied local wheelhouse. This bundle does not include dependency wheels. Tested versions are recorded in the included requirements files. {'The pinned MoveFormer and x-transformers Python sources and MIT licences are included; its einops/entmax dependencies are specified by the moveformer extra in pyproject.toml.' if needs_moveformer else ''}

From this extracted folder, verify every catalogued file:

```bash
python verify_package.py
```

Forecast using a new observed GPS file:

```bash
python infer.py --observations /absolute/path/observed-gps.csv \\
  --species 'Ciconia ciconia' --animal 'local-id' \\
  --issue-time '2026-11-08T12:00:00Z' --samples 64 --seed 2026 \\
  --device cpu --output /absolute/path/forecast.json
```

The equivalent project command is:

```bash
python -m wildcast.cli --data data/reference migration-infer \\
  --checkpoint model/selected.pt --observations /absolute/path/observed-gps.csv \\
  --species 'Ciconia ciconia' --samples 64 --output /absolute/path/forecast.json
```

CSV and parquet inputs require `timestamp`, `lat`, and `lon`; `animal` is optional when the file contains one individual. Use UTC-compatible timestamps. Optional `--canonical-animal` names an exact known canonical identity or recorded alias; unknown identities receive a fresh exclusion ID, and unknown species use prediction token zero. No fuzzy identity match occurs.

Observations after `--issue-time` are excluded before features are constructed. Horizons begin at the last actual retained GPS fix, which can precede the requested issue time. Horizons are fixed by the selected checkpoint/reference cache; these forecasts do not validate second-by-second animation. Forecast output contains sampled predictions and past observations, never observed future targets.

The reference files are unchanged to preserve their checksums. They contain training experiences and development labels; prospective inference strips non-training labels and coordinates before prediction. Reserved final targets must remain sealed. No raw GPS archives, assessment truth sidecars, credentials, SSH keys, optimizer-only auxiliary directories or public pretrained MoveFormer weights are bundled.

An origin predating training or checkpoint-selection evidence is identified as retrospective replay. A later forecast alone does not establish an operational backtest. Optional landscape embeddings retain exact previous-year/nodata masks and their recorded retrospective publication-vintage limitations; they are not hard barriers or measured behavioural states.

`package-manifest.json` records every bundled file's size/checksum, original and packaged checkpoint hashes, path relocations, predictive tensor digests, and source provenance. Its own hash is returned by the packaging command; an optional archive receives an external SHA-256 sidecar. Checksums establish integrity against that catalogue, not an independent digital signature.
'''


def package(data, checkpoint, output, archive=None, source_root=None):
    """Build selected assets in a staging directory and publish a new folder."""
    if Path(output).is_symlink():
        raise FileExistsError("Package output must not be an existing symlink")
    if archive and Path(archive).is_symlink():
        raise ValueError("Archive must not be an existing symlink")
    data, checkpoint, output = Path(data).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    source_root = Path(source_root or PROJECT).resolve()
    if output.exists():
        raise FileExistsError("Package output must be a new directory")
    archive_path = (Path(archive).resolve() if archive else Path(str(output) + ".tar.gz")) if archive is not None else None
    archive_sidecar = archive_path.with_name(archive_path.name + ".sha256.json") if archive_path else None
    if archive_path and (archive_path.exists() or archive_sidecar.exists() or archive_path.is_relative_to(output)):
        raise ValueError("Archive must be a new file outside the package directory")
    arrays, metadata, hashes = load_cases(data)
    reserved = arrays["split"] >= 3
    if arrays["target_mask"][reserved].any():
        raise ValueError("Reserved assessment targets must remain sealed in an inference package")
    for key in ("y", "target_latlon", "target_time"):
        value = arrays[key][reserved]
        if np.any(np.isfinite(value) & (value != 0)):
            raise ValueError("Reserved outcome coordinates/values must not be packaged")
    del arrays
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + "-staging-", dir=output.parent))
    temp_archive = None
    evidence, field_records, rewrites, checkpoints = [], [], [], {}
    needs_moveformer = False
    try:
        for name in ("cases.npz", "manifest.json", "origins.parquet"):
            path = data / name
            if not path.exists():
                raise ValueError("The frozen reference package requires " + name)
            expected = hashes["cases_sha256"] if name == "cases.npz" else hashes["manifest_sha256"] if name == "manifest.json" else metadata.get("origins_sha256")
            _copy(path, stage / "data/reference" / name, expected)

        def copy_fields(spec, owner):
            from wildcast.migration_route_context import RouteContext
            source = resolve_asset(spec["fields_root"], owner, source_root)
            source_manifest = json.loads((source / "manifest.json").read_text())
            for record in source_manifest["records"]:
                if not re.fullmatch(r"[A-Za-z0-9_-]+", record["key"]):
                    raise ValueError("Unsafe context field key path")
                if not (source / (record["key"] + ".npz")).resolve().is_relative_to(source.resolve()):
                    raise ValueError("Context field symlink escapes its selected cache")
            context = RouteContext(source, expected=spec["fingerprint"])
            relative = "data/fields/" + context.fingerprint
            if not any(record["fingerprint"] == context.fingerprint for record in field_records):
                _copy(source / "manifest.json", stage / relative / "manifest.json", context.provenance["manifest_sha256"])
                for record in context.records:
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", record["key"]):
                        raise ValueError("Unsafe context field key path")
                    name = record["key"] + ".npz"
                    _copy(source / name, stage / relative / name, record["sha256"])
                relocated = RouteContext(stage / relative, expected=context.fingerprint)
                field_records.append({"source": str(source), "path": relative, "fingerprint": relocated.fingerprint,
                                      "manifest_sha256": context.provenance["manifest_sha256"], "tiles": len(context.records)})
                for name in ("acquisition-plan.json", "acquisition-code.py", "coverage.json"):
                    if (source / name).is_file() and (source / name).stat().st_size <= 4_000_000:
                        _copy(source / name, stage / "provenance/fields" / context.fingerprint / name)
            return relative

        def copy_checkpoint(source, relative, expected=None, member=False):
            nonlocal needs_moveformer
            source = Path(source)
            original = sha256(source)
            if expected is not None and original != expected:
                raise ValueError("Selected ensemble member checksum mismatch")
            ck = torch.load(source, map_location="cpu", weights_only=True)
            kind = ck.get("model_kind")
            if kind not in KINDS or (member and kind in {"wildcast_migration_ensemble", "wildcast_migration_regime"}):
                raise ValueError("Unsupported or recursive selected checkpoint kind")
            if kind == "wildcast_migration_ensemble":
                weight = torch.as_tensor(ck.get("weights", [])).double()
                if (weight.ndim != 1 or len(weight) != len(ck.get("members", [])) or not len(weight)
                    or not torch.isfinite(weight).all() or torch.any(weight < 0)
                    or not torch.isclose(weight.sum(), torch.tensor(1., dtype=torch.float64), atol=1e-8)):
                    raise ValueError("Selected ensemble weights/members must form a finite simplex")
            elif kind == "wildcast_migration_regime":
                from wildcast.migration_regimes import SCALE_GRID
                if (ck.get("active_fallback") not in {"base", "seasonal"}
                    or ck.get("active_scale") not in SCALE_GRID
                    or (ck.get("active") is None and ck.get("active_scale") != 1.)):
                    raise ValueError("Unsupported frozen regime routing/calibration policy")
                if not isinstance(ck.get("rules"), dict):
                    raise ValueError("Selected regime lacks frozen inline rules")
            elif not isinstance(ck.get("config"), dict) or not ck.get("model"):
                raise ValueError("Selected model checkpoint lacks config or predictive state")
            for key, digest in hashes.items():
                if ck.get(key) != digest:
                    raise ValueError("Selected checkpoint/reference " + key + " mismatch")
            if original in checkpoints:
                return checkpoints[original]
            changed = False
            new = dict(ck)
            before_tensors = tensor_digest(ck)
            before_policy = regime_policy_digest(ck) if kind == "wildcast_migration_regime" else None
            if kind == "wildcast_migration_context_joint":
                from wildcast.migration_deepssf import GHSLRaster
                spec = ck["context_features"]
                new["context_features"] = dict(spec)
                if spec["mode"] != "masked":
                    fields_source = resolve_asset(spec["sources"], source, source_root)
                    fields = GHSLRaster(fields_source, expected=spec["source_fingerprint"])
                    fields_relative = "data/human-fields/" + fields.fingerprint
                    try:
                        if not any(record["fingerprint"] == fields.fingerprint for record in field_records):
                            _copy(fields_source / "sources.json", stage / fields_relative / "sources.json",
                                  fields.provenance["source_manifest_sha256"])
                            names = set()
                            for record in fields.provenance["records"]:
                                for key in ("archive", "raster"):
                                    asset = resolve_asset(record[key], source, source_root)
                                    if asset.name in names:
                                        raise ValueError("Human field archive/raster basenames collide")
                                    names.add(asset.name)
                                    _copy(asset, stage / fields_relative / asset.name, record[key + "_sha256"])
                            relocated = GHSLRaster(stage / fields_relative, expected=fields.fingerprint)
                            relocated.close()
                            field_records.append({"kind": "GHSL-native-built-up", "source": str(fields_source),
                                "path": fields_relative, "fingerprint": fields.fingerprint,
                                "manifest_sha256": fields.provenance["source_manifest_sha256"],
                                "epochs": len(fields.provenance["records"]),
                                "vintage": "Retrospective GHSL R2023A, issue-prior nominal epochs"})
                    finally:
                        fields.close()
                    new["context_features"]["sources"] = fields_relative
                    rewrites.append({"checkpoint": relative, "field": "context_features.sources",
                                     "original": spec["sources"], "packaged": fields_relative})
                    changed = True
            if kind == "wildcast_migration_ensemble":
                members = []
                for number, record in enumerate(ck["members"]):
                    resolved = resolve_asset(record["path"], source, source_root)
                    member_relative = "members/" + f"{number:03d}-" + record["sha256"][:16] + ".pt"
                    packaged = copy_checkpoint(resolved, member_relative, record["sha256"], member=True)
                    members.append({**record, "path": packaged["path"], "sha256": packaged["sha256"]})
                    rewrites.append({"checkpoint": relative, "field": f"members[{number}].path", "original": record["path"], "packaged": packaged["path"]})
                new["members"], new["data"] = members, "data/reference"
                rewrites.append({"checkpoint": relative, "field": "data", "original": ck["data"], "packaged": "data/reference"})
                changed = True
            elif kind == "wildcast_migration_regime":
                reference_root = resolve_asset(ck["data"], source, source_root)
                for key, name in (("manifest_sha256", "manifest.json"), ("cases_sha256", "cases.npz")):
                    if sha256(reference_root / name) != hashes[key] or ck["rules"].get(key) != hashes[key]:
                        raise ValueError("Selected regime rules/reference " + key + " mismatch")
                rules_source = resolve_asset(ck["rules_path"], source, source_root)
                rules_hash = sha256(rules_source)
                if rules_hash != ck.get("rules_sha256") or json.loads(rules_source.read_text()) != ck["rules"]:
                    raise ValueError("Selected regime frozen rules checksum or inline equality mismatch")
                rules_relative = "rules/" + rules_hash + ".json"
                _copy(rules_source, stage / rules_relative, rules_hash)
                new["rules_path"], new["data"] = rules_relative, "data/reference"
                for key in ("base", "active"):
                    record = ck.get(key)
                    if key == "active" and record is None:
                        continue
                    if (not isinstance(record, dict) or record.get("model_kind") != "wildcast_migration_joint"
                        or not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", "")))):
                        raise ValueError("Regime base/active members must be verified joint migration models")
                    resolved = resolve_asset(record["path"], source, source_root)
                    member_relative = "members/" + key + "-" + record["sha256"][:16] + ".pt"
                    packaged = copy_checkpoint(resolved, member_relative, record["sha256"], member=True)
                    if packaged["model_kind"] != "wildcast_migration_joint":
                        raise ValueError("Regime base/active members must be joint migration models")
                    new[key] = {**record, "path": packaged["path"], "sha256": packaged["sha256"]}
                    rewrites.append({"checkpoint": relative, "field": key + ".path", "original": record["path"], "packaged": packaged["path"]})
                for key in ("data", "rules_path"):
                    rewrites.append({"checkpoint": relative, "field": key, "original": ck[key], "packaged": new[key]})
                if ck.get("calibration"):
                    calibration = ck["calibration"]
                    calibration_source = resolve_asset(calibration["path"], source, source_root)
                    calibration_hash = sha256(calibration_source)
                    calibrated = json.loads(calibration_source.read_text())
                    if calibration_hash != calibration.get("sha256") or calibrated != calibration.get("metadata"):
                        raise ValueError("Selected regime calibration checksum or inline equality mismatch")
                    expected_calibration = {**hashes, "rules_sha256": rules_hash,
                                            "checkpoint_sha256": (ck.get("active") or {}).get("sha256")}
                    if (calibrated.get("protocol") != "wildcast-migration-active-covariance-final-v1"
                        or calibrated.get("family_selected") is not True or ck.get("active") is None
                        or calibrated.get("active_scale") != ck["active_scale"]
                        or any(calibrated.get(key) != value for key, value in expected_calibration.items())):
                        raise ValueError("Selected regime finalized calibration policy/provenance mismatch")
                    calibration_relative = "calibration/" + calibration_hash + "/scale.json"
                    _copy(calibration_source, stage / calibration_relative, calibration_hash)
                    new["calibration"] = {**calibration, "path": calibration_relative}
                    rewrites.append({"checkpoint": relative, "field": "calibration.path", "original": calibration["path"], "packaged": calibration_relative})
                changed = True
            elif ck.get("config", {}).get("context"):
                config = copy.deepcopy(ck["config"])
                original_root = config["context"]["fields_root"]
                config["context"]["fields_root"] = copy_fields(config["context"], source)
                new["config"] = config
                rewrites.append({"checkpoint": relative, "field": "config.context.fields_root", "original": original_root, "packaged": config["context"]["fields_root"]})
                changed = True
            needs_moveformer |= kind in {"wildcast_migration_moveformer", "wildcast_migration_moveformer_endpoint"}
            if kind in {"wildcast_migration_moveformer", "wildcast_migration_moveformer_endpoint"}:
                from wildcast.moveformer import provenance as author_sources
                expected_upstream = author_sources()["source_sha256"]
                if ck.get("upstream", {}).get("source_sha256") != expected_upstream:
                    raise ValueError("Selected MoveFormer source provenance mismatch")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if changed:
                torch.save(new, destination)
            else:
                _copy(source, destination, original)
            after = torch.load(destination, map_location="cpu", weights_only=True)
            if tensor_digest(after) != before_tensors or sha256(source) != original:
                raise ValueError("Predictive tensors or source snapshot changed during packaging")
            if before_policy is not None and regime_policy_digest(after) != before_policy:
                raise ValueError("Frozen regime routing rules or scalar policy changed during packaging")
            record = {"source_path": str(source), "source_sha256": original, "path": relative,
                      "sha256": sha256(destination), "model_kind": kind, "predictive_tensor_sha256": before_tensors,
                      "selection_version": ck.get("selection_version"), "paths_rewritten": changed}
            if before_policy is not None:
                record.update(predictive_policy_sha256=before_policy, rules_sha256=ck["rules_sha256"])
            evidence.append(record)
            checkpoints[original] = record
            return record

        selected = copy_checkpoint(checkpoint, "model/selected.pt")
        for path in sorted((PROJECT / "wildcast").glob("*.py")):
            if not path.resolve().is_relative_to(PROJECT):
                raise ValueError("Core source symlink escapes the project")
            _copy(path, stage / "wildcast" / path.name)
        for name in ("pyproject.toml", "requirements-validated.txt", "requirements-ecology-validated.txt", "LICENSE", "LICENSE.md", "COPYING"):
            if (PROJECT / name).is_file():
                if not (PROJECT / name).resolve().is_relative_to(PROJECT):
                    raise ValueError("Project metadata symlink escapes its source root")
                _copy(PROJECT / name, stage / name)
        if needs_moveformer:
            from wildcast.moveformer import provenance as author_sources
            upstream = author_sources()
            expected = {"third_party/moveformer/geo_transformers/models/any_horizon_forecast_transformer.py":
                            "9454f64bec902143633b981fdb64375551148b7522910550f8cd94e10b4db639",
                        "third_party/x-transformers/x_transformers/x_transformers.py":
                            "e2871f4b9e9c85ed108b25972ba378bf783e0ebd440cc79343616fc0b973274e"}
            if upstream["source_sha256"] != expected:
                raise ValueError("MoveFormer/attention sources differ from their pinned protocol")
            for repository, directory in [("moveformer", "geo_transformers"), ("x-transformers", "x_transformers")]:
                root = PROJECT / "third_party" / repository
                for path in sorted((root / directory).rglob("*.py")):
                    if not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError("Third-party source symlink escapes its pinned repository")
                    _copy(path, stage / path.relative_to(PROJECT))
                for name in ("LICENSE", "README.md"):
                    _copy(root / name, stage / "third_party" / repository / name)
            (stage / "third_party/pins.json").write_text(json.dumps(upstream, indent=2))
        _copy(Path(__file__), stage / "tools/package_migration.py")
        provenance = {"source_project": str(PROJECT), "source_checkpoint": str(checkpoint), "reference_cache": str(data),
                      "reference_sources": metadata.get("sources", []), "reference_manifest_sha256": hashes["manifest_sha256"],
                      "checkpoints": evidence, "fields": field_records, "path_rewrites": rewrites,
                      "raw_archives_included": any(record.get("kind") == "GHSL-native-built-up" for record in field_records),
                      "raw_GPS_archives_included": False, "reserved_assessment_sidecars_included": False,
                      "dependency_wheels_included": False, "copy_rule": "Explicit selected asset closure; no source-tree archive or credential directories"}
        provenance_root = stage / "provenance"
        provenance_root.mkdir(exist_ok=True)
        copied_metadata, unavailable = [], []
        metadata_bytes = 0

        def small_metadata(value, role, expected=None):
            nonlocal metadata_bytes
            try:
                path = resolve_asset(value, checkpoint, source_root)
            except ValueError:
                unavailable.append({"path": str(value), "role": role, "reason": "not available inside trusted source roots"})
                return None
            if not path.is_file() or path.suffix != ".json" or path.stat().st_size > 2_000_000:
                return None
            if len(copied_metadata) >= 256 or metadata_bytes + path.stat().st_size > 8_000_000:
                unavailable.append({"path": str(path), "role": role, "reason": "bounded metadata budget"})
                return None
            digest = sha256(path)
            relative = "provenance/source-metadata/" + digest[:20] + "-" + path.name
            _copy(path, stage / relative, expected)
            metadata_bytes += path.stat().st_size
            copied_metadata.append({"path": relative, "source_path": str(path), "sha256": digest, "role": role})
            return json.loads(path.read_text())
        # Preserve small, explicitly referenced audit metadata, not whole reports.
        for field in ("reference_audit", "existing_manifest"):
            if metadata.get(field):
                small_metadata(metadata[field], field, metadata.get(field + "_sha256"))
        source_jsons = set()
        for source in metadata.get("sources", []):
            quality_path = Path(source["path"]).parent / "quality-report.json"
            quality = small_metadata(quality_path, "cleaning provenance")
            if quality:
                for entry in quality.get("files", []):
                    raw = Path(entry.get("path", ""))
                    if "raw" in raw.parts:
                        source_jsons.add(str(raw.parent / "source.json"))
        for path in sorted(source_jsons):
            small_metadata(path, "public GPS source attribution and rights")
        provenance.update(source_metadata=copied_metadata, unavailable_optional_metadata=unavailable)
        (provenance_root / "sources.json").write_text(json.dumps(provenance, indent=2))
        (stage / "README.md").write_text(_readme(selected["model_kind"], needs_moveformer))
        (stage / "verify_package.py").write_text(VERIFY_SOURCE)
        (stage / "infer.py").write_text(INFER_SOURCE)
        files = {str(path.relative_to(stage)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(stage.rglob("*")) if path.is_file()}
        catalogue = {"format": "wildcast-offline-migration-v1", "model_kind": selected["model_kind"],
                     "checkpoint": selected, "reference": hashes, "files": files,
                     "checkpoints": evidence, "context_fields": field_records, "path_rewrites": rewrites,
                     "package_manifest_catalogued": False, "catalogue_note": "The catalogue excludes itself; its external hash is returned by the packager"}
        (stage / "package-manifest.json").write_text(json.dumps(catalogue, indent=2))
        # Verify using the exact shipped verifier before publishing anything.
        namespace = {"__name__": "package_verification", "__file__": str(stage / "verify_package.py")}
        exec(compile(VERIFY_SOURCE, str(stage / "verify_package.py"), "exec"), namespace)
        namespace["verify"](stage)
        if archive_path is not None:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix="." + archive_path.name, dir=archive_path.parent)
            os.close(descriptor)
            temp_archive = Path(temporary)
            with tarfile.open(temp_archive, "w:gz", compresslevel=3) as tar:
                for path in sorted(stage.rglob("*")):
                    if path.is_file():
                        tar.add(path, arcname=str(Path(output.name) / path.relative_to(stage)), recursive=False)
        if output.exists() or (archive_path and (archive_path.exists() or archive_sidecar.exists())):
            raise FileExistsError("A package destination appeared during assembly")
        stage.rename(output)
        summary = {"output": str(output), "model_kind": selected["model_kind"], "files": len(files),
                   "bytes": sum(record["bytes"] for record in files.values()), "package_manifest_sha256": sha256(output / "package-manifest.json")}
        if archive_path is not None:
            temp_archive.rename(archive_path)
            summary["archive"] = {"path": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": sha256(archive_path)}
            archive_sidecar.write_text(json.dumps({**summary["archive"], "package_manifest_sha256": summary["package_manifest_sha256"]}, indent=2))
        return summary
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        if temp_archive is not None and temp_archive.exists():
            temp_archive.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive", nargs="?", const="")
    parser.add_argument("--source-root")
    print(json.dumps(package(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__": main()
