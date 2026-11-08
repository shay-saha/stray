"""Physical-animal cross-fitted global mixtures of migration distributions.

An expert is chosen once per complete sampled path. Validation learns only a
small simplex of global mixture probabilities. Expert checkpoint selection also
used validation, so cross-fitting these weights is not an independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize


KIND = "wildcast_migration_ensemble"
SELECTION_VERSION = "migration-ensemble-physical-crossfit-v1"
EQUAL_SELECTION_VERSION = "migration-ensemble-equal-unfitted-v1"
EARTH_KM = 6371.0088


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_member(path, device="cpu", expected=None, hashes=None):
    path = Path(path)
    before = sha256(path)
    if expected is not None and before != expected:
        raise ValueError("Migration ensemble member checksum mismatch")
    probe = torch.load(path, map_location="cpu", weights_only=True)
    kind = probe.get("model_kind")
    if kind == "wildcast_migration_joint":
        from .migration_model import load_model as loader
    elif kind == "wildcast_migration_context_joint":
        from .migration_context_joint import load_model as loader
    elif kind == "wildcast_migration_retrieval":
        from .migration_retrieval_train import load_model as loader
    elif kind == "wildcast_migration_moveformer":
        from .migration_moveformer import load_model as loader
    elif kind == "wildcast_migration_moveformer_endpoint":
        from .migration_moveformer_endpoint import load_model as loader
    elif kind == "wildcast_migration_regime":
        from .migration_regimes import load_model as loader
    else:
        raise ValueError("Ensemble members must be joint/retrieval/MoveFormer models; recursive ensembles are not supported")
    del probe
    model, checkpoint = loader(path, device=device)
    if sha256(path) != before:
        raise ValueError("Migration ensemble member changed during loading")
    for key, value in (hashes or {}).items():
        if checkpoint.get(key) != value:
            raise ValueError("Migration ensemble member/data " + key + " mismatch")
    evidence = {"path": str(path.resolve()), "sha256": before, "model_kind": kind,
                "epoch": checkpoint.get("epoch"), "selection_version": checkpoint.get("selection_version")}
    return model, evidence


def _draw(model, a, ids, samples, rng, device, baselines=None):
    if hasattr(model, "predict_cases"):
        return model.predict_cases(a, ids, samples, rng, baselines=baselines)
    generator = torch.Generator(device=device).manual_seed(int(rng.integers(0, 2**63 - 1)))
    with torch.inference_mode():
        inputs = [torch.as_tensor(a[key][ids], device=device) for key in ("history", "history_mask", "static", "species")]
        return model.sample_paths(*inputs, samples=samples, generator=generator).detach().cpu().numpy()


class MigrationEnsemble:
    def __init__(self, members, weights, device="cpu"):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(members),) or not len(weights) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Ensemble weights must be a finite nonnegative simplex")
        if not np.isclose(weights.sum(), 1., atol=1e-8):
            raise ValueError("Ensemble probabilities must sum to one")
        self.members, self.weights, self.device = members, weights / weights.sum(), str(device)

    def predict_cases(self, a, ids, samples, rng=None, baselines=None, seed=2026):
        if samples < 1:
            raise ValueError("At least one complete route sample required")
        ids = np.asarray(ids, dtype=np.int64)
        rng = rng or np.random.default_rng(seed)
        selected = rng.choice(len(self.members), size=(len(ids), samples), p=self.weights)
        result = np.empty((len(ids), samples, len(a["horizons"]), 2), dtype=np.float32)
        for expert, member in enumerate(self.members):
            active_rows = np.flatnonzero((selected == expert).any(-1))
            if not len(active_rows):
                continue
            local_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            draws = _draw(member, a, ids[active_rows], samples, local_rng, self.device, baselines)
            for row, source in enumerate(active_rows):
                slots = selected[source] == expert
                result[source, slots] = draws[row, slots]
        return result


def physical_folds(a, manifest, folds=3, seed=1701):
    """Every canonical identity/exclusion group has exactly one stable fold."""
    if folds < 2:
        raise ValueError("At least two physical-animal folds required")
    names = {}
    exclusion = manifest.get("exclusion_groups", {})
    for name, identity in manifest.get("animals", {}).items():
        names.setdefault(int(identity), []).append(str(exclusion.get(name, name)))
    assignment, labels = {}, {}
    for identity in np.unique(a["individual"]):
        identity = int(identity)
        label = min(names[identity]) if identity in names else f"physical:{identity}"
        labels[identity] = label
        assignment[identity] = int(hashlib.sha256(f"migration-ensemble:{seed}:{label}".encode()).hexdigest()[:16], 16) % folds
    return np.asarray([assignment[int(v)] for v in a["individual"]], dtype=np.int64), labels


def macro_weights(a, ids):
    """Equal split/horizon/species/physical-animal weight on observed labels."""
    ids = np.asarray(ids)
    mask = a["target_mask"][ids]
    endpoint, joint = np.zeros(mask.shape), np.zeros(len(ids))
    species = a.get("metric_species", a["species"])[ids]
    animal, split = a["individual"][ids], a["split"][ids]
    splits = np.unique(split)
    if not len(splits):
        raise ValueError("Cannot aggregate empty calibration cases")
    for code in splits:
        origin_rows = np.flatnonzero(split == code)
        groups = np.unique(species[origin_rows])
        for group in groups:
            rows = origin_rows[species[origin_rows] == group]
            identities = np.unique(animal[rows])
            for identity in identities:
                selected = rows[animal[rows] == identity]
                joint[selected] = 1. / (len(splits) * len(groups) * len(identities) * len(selected))
        horizons = np.flatnonzero(mask[origin_rows].any(0))
        for h in horizons:
            rows = origin_rows[mask[origin_rows, h]]
            groups = np.unique(species[rows])
            for group in groups:
                block = rows[species[rows] == group]
                identities = np.unique(animal[block])
                for identity in identities:
                    selected = block[animal[block] == identity]
                    endpoint[selected, h] = 1. / (len(splits) * len(horizons) * len(groups) * len(identities) * len(selected))
    return endpoint, joint


def optimize_weights(first, pair, steps=300):
    """Solve the small proper-energy quadratic on the probability simplex."""
    first, pair = np.asarray(first, float), np.asarray(pair, float)
    count = len(first)
    if pair.shape != (count, count) or not np.isfinite(first).all() or not np.isfinite(pair).all():
        raise ValueError("Invalid ensemble score coefficients")
    pair = .5 * (pair + pair.T)
    objective = lambda w: float(w @ first - .5 * w @ pair @ w)
    jacobian = lambda w: first - pair @ w
    candidates = [np.full(count, 1 / count), *np.eye(count)]
    # Independent Monte Carlo bank estimates can slightly perturb convexity;
    # include simplex corners and start from both equal and best single member.
    starts = [candidates[0], min(candidates[1:], key=objective)]
    for initial in starts:
        result = minimize(objective, initial, jac=jacobian, method="SLSQP",
                          bounds=[(0., 1.)] * count,
                          constraints={"type": "eq", "fun": lambda w: w.sum() - 1.,
                                       "jac": lambda w: np.ones_like(w)},
                          options={"maxiter": int(steps), "ftol": 1e-10})
        if np.isfinite(result.x).all() and result.x.sum() > 0:
            value = np.maximum(result.x, 0); candidates.append(value / value.sum())
    chosen = min(candidates, key=objective)
    return chosen, objective(chosen)


def _batch_costs(paths, origin, target_latlon, target_mask, device):
    from .migration_evaluate import forecast_latlon, spherical_xyz
    batch, members, count, horizons, _ = paths.shape
    if count % 2:
        raise ValueError("Use two equally sized independent member sample banks")
    samples = count // 2
    coords = forecast_latlon(paths.reshape(batch, members * count, horizons, 2), origin)
    xyz = torch.as_tensor(spherical_xyz(coords), dtype=torch.float32, device=device).reshape(batch, members, count, horizons, 3)
    clean = np.where(target_mask[..., None], target_latlon, origin[:, None, :2])
    truth = torch.as_tensor(spherical_xyz(clean), dtype=torch.float32, device=device)
    chord = (xyz - truth[:, None, None]).norm(dim=-1)
    first = (2 * EARTH_KM * torch.asin((chord / (2 * EARTH_KM)).clamp(0, 1))).mean(2)
    a = xyz[:, :, :samples].permute(0, 3, 1, 2, 4).reshape(batch, horizons, members * samples, 3)
    b = xyz[:, :, samples:].permute(0, 3, 1, 2, 4).reshape(batch, horizons, members * samples, 3)
    distances = torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")
    distances = 2 * EARTH_KM * torch.asin((distances / (2 * EARTH_KM)).clamp(0, 1))
    pairs = distances.reshape(batch, horizons, members, samples, members, samples).mean((3, 5)).permute(0, 2, 3, 1)
    pairs = .5 * (pairs + pairs.transpose(1, 2))
    mask = torch.as_tensor(target_mask, device=device)
    factor = (mask / mask.sum(-1, keepdim=True)).sqrt()[..., None]
    embedding = (xyz * factor[:, None, None]).flatten(-2)
    observed = (truth * factor).flatten(-2)
    joint_first = (embedding - observed[:, None, None]).norm(dim=-1).mean(-1)
    ja = embedding[:, :, :samples].reshape(batch, members * samples, -1)
    jb = embedding[:, :, samples:].reshape(batch, members * samples, -1)
    joint_pair = torch.cdist(ja, jb, compute_mode="donot_use_mm_for_euclid_dist")
    joint_pair = joint_pair.reshape(batch, members, samples, members, samples).mean((2, 4))
    joint_pair = .5 * (joint_pair + joint_pair.transpose(1, 2))
    return {"endpoint_first": first.cpu().numpy(), "endpoint_pair": pairs.cpu().numpy(),
            "joint_first": joint_first.cpu().numpy(), "joint_pair": joint_pair.cpu().numpy()}


def precompute_costs(a, ids, members, member_evidence, samples=32, seed=2026, batch_size=16, device="cpu"):
    """Stream path banks; retain only O(cases * members² * horizons) costs."""
    from .migration_baselines import MigrationBaselines
    if samples < 2:
        raise ValueError("At least two samples per independent member bank required")
    fitted = MigrationBaselines(a)
    stored = {key: [] for key in ("endpoint_first", "endpoint_pair", "joint_first", "joint_pair")}
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        draws = []
        for model, evidence in zip(members, member_evidence):
            source_seed = int(hashlib.sha256(f"{seed}:{start}:{evidence['sha256']}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(source_seed)
            # Two disjoint independent banks yield unbiased expected-distance
            # estimates for both within-member and cross-member terms.
            draws.append(_draw(model, a, batch, 2 * samples, rng, device, fitted))
        costs = _batch_costs(np.stack(draws, axis=1), a["origin"][batch], a["target_latlon"][batch],
                             a["target_mask"][batch], device)
        for key, value in costs.items():
            stored[key].append(value)
    return {key: np.concatenate(value) for key, value in stored.items()}


def aggregate_coefficients(a, ids, costs, rows):
    rows = np.asarray(rows)
    endpoint, joint = macro_weights(a, ids[rows])
    first = .5 * np.einsum("nh,nmh->m", endpoint, costs["endpoint_first"][rows])
    first += .5 * np.einsum("n,nm->m", joint, costs["joint_first"][rows])
    pair = .5 * np.einsum("nh,nijh->ij", endpoint, costs["endpoint_pair"][rows])
    pair += .5 * np.einsum("n,nij->ij", joint, costs["joint_pair"][rows])
    return first, pair


def case_scores(costs, weights):
    weights = np.asarray(weights)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, costs["joint_first"].shape)
    endpoint = np.einsum("nm,nmh->nh", weights, costs["endpoint_first"])
    endpoint -= .5 * np.einsum("ni,nijh,nj->nh", weights, costs["endpoint_pair"], weights)
    joint = np.einsum("nm,nm->n", weights, costs["joint_first"])
    joint -= .5 * np.einsum("ni,nij,nj->n", weights, costs["joint_pair"], weights)
    return endpoint, joint


def score_summary(a, ids, endpoint, joint):
    result = {}
    for code, name in ((1, "val_individual"), (2, "val_year")):
        rows = np.flatnonzero(a["split"][ids] == code)
        if not len(rows):
            continue
        eweight, jweight = macro_weights(a, ids[rows])
        result[name] = {"endpoint_energy_km": float(np.sum(eweight * endpoint[rows])),
                        "joint_energy_km": float(np.sum(jweight * joint[rows])), "cases": int(len(rows))}
    return result


def fit(data, checkpoints, output, samples=32, folds=3, steps=300, seed=1701,
        batch_size=16, device="auto", bootstrap_replicates=0):
    from .migration_evaluate import load_cases
    if len(checkpoints) < 2:
        raise ValueError("At least two frozen migration members required")
    if len(set(map(str, checkpoints))) != len(checkpoints):
        raise ValueError("Duplicate ensemble checkpoint paths")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    a, manifest, hashes = load_cases(data)
    if not manifest.get("reserved_test_targets_sealed", False):
        raise ValueError("Fit ensemble weights before final test targets are released")
    ids = np.flatnonzero(np.isin(a["split"], [1, 2]) & a["target_mask"].any(-1))
    if not len(ids) or len(np.unique(a["individual"][ids])) < 2:
        raise ValueError("Calibration requires validation cases from multiple physical groups")
    assignments, labels = physical_folds(a, manifest, folds, seed)
    held_folds = assignments[ids]
    if len(np.unique(held_folds)) < 2:
        raise ValueError("All validation identities hashed into one fold; use another declared seed")
    members, evidence = [], []
    for path in checkpoints:
        model, record = load_member(path, device, hashes=hashes)
        members.append(model); evidence.append(record)
    if len({entry["sha256"] for entry in evidence}) != len(evidence):
        raise ValueError("Duplicate checkpoint contents are not distinct ensemble members")
    out = Path(output)
    if (out / "best.pt").exists():
        raise FileExistsError("Use a new output path to preserve frozen ensemble weights")
    out.mkdir(parents=True, exist_ok=True)
    costs = precompute_costs(a, ids, members, evidence, samples, seed, batch_size, device)
    np.savez_compressed(out / "validation-costs.npz", case_ids=ids, fold=held_folds, **costs)
    count = len(members)
    equal = np.full(count, 1 / count)
    crossfit_weights, crossfit_best = np.zeros((len(ids), count)), np.zeros((len(ids), count))
    fold_reports = []
    for fold in range(folds):
        held, training = np.flatnonzero(held_folds == fold), np.flatnonzero(held_folds != fold)
        if not len(held):
            fold_reports.append({"fold": fold, "status": "empty held-out fold"}); continue
        if not len(training):
            raise ValueError("No complement remains for an ensemble calibration fold")
        first, pair = aggregate_coefficients(a, ids, costs, training)
        weights, selection = optimize_weights(first, pair, steps)
        single = first - .5 * pair.diagonal()
        best_member = int(single.argmin())
        crossfit_weights[held] = weights
        crossfit_best[held, best_member] = 1.
        local_costs = {key: value[held] for key, value in costs.items()}
        endpoint, joint = case_scores(local_costs, weights)
        fold_reports.append({"fold": fold, "fit_cases": int(len(training)), "held_cases": int(len(held)),
                             "fit_groups": sorted(set(map(int, a["individual"][ids[training]]))),
                             "held_groups": sorted(set(map(int, a["individual"][ids[held]]))),
                             "weights": weights.tolist(), "fit_selection_km": selection,
                             "fit_best_member": best_member,
                             "held_scores": score_summary(a, ids[held], endpoint, joint)})
    first, pair = aggregate_coefficients(a, ids, costs, np.arange(len(ids)))
    weights, selection = optimize_weights(first, pair, steps)
    best_index = int((first - .5 * pair.diagonal()).argmin())
    all_models = {"equal_mixture": equal, "crossfit_best_member": crossfit_best,
                  "crossfit_mixture": crossfit_weights, "all_validation_mixture": weights,
                  "all_validation_best_member": np.eye(count)[best_index]}
    scores = {name: case_scores(costs, value) for name, value in all_models.items()}
    summaries = {name: score_summary(a, ids, *value) for name, value in scores.items()}
    selection_scores = {name: float(np.mean([.5 * (v["endpoint_energy_km"] + v["joint_energy_km"])
                                            for v in by_split.values()])) for name, by_split in summaries.items()}
    report = {"scope": "Local validation development. Expert checkpoints also used validation; weight cross-fitting is not an independent test.",
              "selection_version": SELECTION_VERSION, "weights": weights.tolist(), "members": evidence,
              "crossfit_folds": folds, "fold_seed": seed, "fold_reports": fold_reports,
              "canonical_group_labels": {str(k): v for k, v in labels.items()},
              "validation_cases": int(len(ids)), "validation_selection_km": selection, "scores": summaries,
              "selection_scores_km": selection_scores,
              "crossfit_delta_vs_fold_selected_best_member_km": selection_scores["crossfit_mixture"] - selection_scores["crossfit_best_member"],
              "samples_per_independent_bank": samples, "independent_banks_per_member": 2,
              "sampling_rule": "Two independent fixed banks per member, coupled across all weight alternatives. All cross-member distances retained; no location averaging.",
              "metric_rule": "Equal endpoint great-circle and joint Cartesian energy; equal split/horizon/species/physical-animal macro weights.",
              "reserved_test_cases_scored": 0, **hashes, "costs_sha256": sha256(out / "validation-costs.npz"),
              "code_sha256": sha256(__file__), "device": str(device), "batch_size": batch_size,
              "geometry_code_sha256": sha256(Path(__file__).with_name("migration_evaluate.py"))}
    report["limitations"] = ["Only global mixture weights are cross-fitted; expert checkpoint selection already used validation.",
        "Distance costs are fixed finite Monte Carlo estimates from two independent banks, not analytic model scores.",
        "Any bootstrap conditions on these member forecasts and fold-fitted weights; it does not retrain experts or calibration.",
        "The all-validation weights are for a later frozen assessment; their fitting score is not an independent result."]
    if bootstrap_replicates:
        import pandas as pd
        from .migration_evaluate import paired_animal_bootstrap
        rows = []
        species = a.get("metric_species", a["species"])
        for name in ("equal_mixture", "crossfit_best_member", "crossfit_mixture"):
            endpoint, _ = scores[name]
            for row, case in enumerate(ids):
                for h in np.flatnonzero(a["target_mask"][case]):
                    rows.append({"model": name, "case": int(case), "split": int(a["split"][case]),
                                 "species": int(species[case]), "individual": int(a["individual"][case]),
                                 "horizon_days": float(a["horizons"][h]), "endpoint_energy_km": float(endpoint[row, h])})
        report["paired_uncertainty"] = paired_animal_bootstrap(pd.DataFrame(rows), "crossfit_best_member", seed, bootstrap_replicates)
    (out / "comparison.json").write_text(json.dumps(report, indent=2))
    # A running training job must not silently replace a selected member while
    # the ensemble is being calibrated. Bind only immutable snapshots.
    for record in evidence:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Migration ensemble member changed during calibration; use frozen snapshots")
    for name, key in (("manifest.json", "manifest_sha256"), ("cases.npz", "cases_sha256")):
        if sha256(Path(data) / name) != hashes[key]:
            raise ValueError("Migration cache changed during ensemble calibration")
    checkpoint = {"model_kind": KIND, "selection_version": SELECTION_VERSION,
                  "weights": torch.as_tensor(weights, dtype=torch.float64), "members": evidence,
                  "data": str(Path(data).resolve()), "manifest": manifest, **hashes,
                  "metrics": summaries, "training": report, "code_sha256": report["code_sha256"]}
    torch.save(checkpoint, out / "best.pt")
    (out / "artifact-manifest.json").write_text(json.dumps({"checkpoint_sha256": sha256(out / "best.pt"),
        "comparison_sha256": sha256(out / "comparison.json"), "validation_costs_sha256": report["costs_sha256"]}, indent=2))
    return report


def assemble_equal(data, checkpoints, output, device="cpu", member_selection_scope=None):
    """Freeze uniform whole-route mixture weights without reading target arrays.

    Member selection can already depend on development results. Uniform weights
    avoid another fitting step; they do not make that selection independent.
    """
    root, out = Path(data).resolve(), Path(output)
    if out.exists():
        raise FileExistsError("Equal-mixture output must be a new directory")
    checkpoints = list(checkpoints)
    if len(checkpoints) < 2:
        raise ValueError("An equal mixture requires at least two frozen members")
    manifest = json.loads((root / "manifest.json").read_text())
    hashes = {"manifest_sha256": sha256(root / "manifest.json"), "cases_sha256": sha256(root / "cases.npz")}
    expected = manifest.get("cases_sha256", manifest.get("files_sha256", {}).get("cases.npz"))
    if expected != hashes["cases_sha256"]:
        raise ValueError("Equal-mixture reference cache checksum mismatch")
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    evidence = []
    for path in checkpoints:
        _, record = load_member(path, device, hashes=hashes)
        evidence.append(record)
    if len({record["sha256"] for record in evidence}) != len(evidence):
        raise ValueError("Duplicate checkpoint contents are not distinct equal-mixture members")
    for record in evidence:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Equal-mixture member changed during assembly")
    for name, key in (("manifest.json", "manifest_sha256"), ("cases.npz", "cases_sha256")):
        if sha256(root / name) != hashes[key]:
            raise ValueError("Equal-mixture reference cache changed during assembly")
    weights = np.full(len(evidence), 1. / len(evidence), dtype=np.float64)
    report = {"scope": "Development-selected frozen members; uniform weights were not fitted. Not an independent assessment.",
              "selection_version": EQUAL_SELECTION_VERSION, "members": evidence, "weights": weights.tolist(),
              "member_selection_scope": member_selection_scope or "Members were selected using local development experiments.",
              "weight_rule": "Exactly 1/member_count for each frozen member; choose one expert per complete route.",
              "weight_fitting_performed": False, "target_arrays_read": False, "validation_cases_scored": 0,
              "reserved_test_cases_scored": 0, "independent_result": False, **hashes,
              "code_sha256": sha256(__file__)}
    checkpoint = {"model_kind": KIND, "selection_version": EQUAL_SELECTION_VERSION,
                  "weights": torch.as_tensor(weights, dtype=torch.float64), "members": evidence,
                  "data": str(root), "manifest": manifest, **hashes, "training": report,
                  "code_sha256": report["code_sha256"]}
    out.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, out / "best.pt")
    (out / "assembly.json").write_text(json.dumps(report, indent=2))
    (out / "artifact-manifest.json").write_text(json.dumps({"checkpoint_sha256": sha256(out / "best.pt"),
        "assembly_sha256": sha256(out / "assembly.json")}, indent=2))
    return report


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck.get("model_kind") != KIND:
        raise ValueError("Not a migration distribution ensemble checkpoint")
    root = Path(ck["data"])
    hashes = {"manifest_sha256": sha256(root / "manifest.json"), "cases_sha256": sha256(root / "cases.npz")}
    for key, actual in hashes.items():
        if ck.get(key) != actual:
            raise ValueError("Migration ensemble/cache " + key + " mismatch")
    members = []
    for record in ck["members"]:
        model, _ = load_member(record["path"], device, expected=record["sha256"], hashes=hashes)
        members.append(model)
    return MigrationEnsemble(members, ck["weights"].numpy(), device), ck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True); parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--device", default="auto")
    parser.add_argument("--equal-weights", action="store_true", help="Assemble uniform weights; do not read targets or fit calibration")
    parser.add_argument("--member-selection-scope", help="Record how the frozen members were selected; not an independence claim")
    for name, default in (("samples", 32), ("folds", 3), ("steps", 300), ("seed", 1701),
                          ("batch-size", 16), ("bootstrap-replicates", 0)):
        parser.add_argument("--" + name, type=int, default=default)
    args = vars(parser.parse_args())
    equal = args.pop("equal_weights")
    scope = args.pop("member_selection_scope")
    if equal:
        report = assemble_equal(**{key: args[key] for key in ("data", "checkpoints", "output", "device")}, member_selection_scope=scope)
        print(json.dumps(report, indent=2))
    else:
        fit(**args)


if __name__ == "__main__":
    main()
