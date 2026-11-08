"""Training and validation for the migration-specific experimental model.

Only splits 0/1/2 are used. Reserved test targets must remain unmaterialized.
All fitted normalization and initial scales use training animals only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .migration_model import MigrationModel, VARIANTS, masked_joint_nll, mixture_energy, sample_paths

SELECTION_VERSION = "migration-physical-v2"
EARTH_KM = 6371.0088
SAMPLING = ("animal", "species_animal", "species_sqrt")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_data(data):
    root = Path(data)
    manifest_path, cases_path = root / "manifest.json", root / "cases.npz"
    manifest = json.loads(manifest_path.read_text())
    cases_hash = _hash(cases_path)
    expected = manifest.get("cases_sha256", manifest.get("files_sha256", {}).get("cases.npz"))
    if expected != cases_hash:
        raise ValueError("Migration cases hash does not match the frozen manifest")
    with np.load(cases_path, allow_pickle=False) as archive:
        # Copy split membership first; target tensors are constructed only for
        # the train/validation subset below. Test truth is never selected.
        split = archive["split"]
        if not np.isin(split, [0, 1, 2, 3, 4]).all():
            raise ValueError("unrecognized migration split")
        usable = np.flatnonzero(np.isin(split, [0, 1, 2]))
        if manifest.get("reserved_test_targets_sealed") and archive["target_mask"][split >= 3].any():
            raise ValueError("Reserved migration test target masks are not sealed")
        required = ("history", "history_mask", "static", "species", "y", "target_mask", "split", "individual", "study", "origin", "target_latlon")
        if any(name not in archive for name in required):
            raise ValueError("Physical migration validation requires origin and target_latlon arrays")
        a = {name: archive[name][usable] for name in required}
        from .migration_evaluate import load_metric_species
        grouping_input = {"species": archive["species"]}
        if "metric_species" in archive:
            grouping_input["metric_species"] = archive["metric_species"]
        grouping, _ = load_metric_species(grouping_input, manifest, root)
        a["metric_species"] = grouping[usable]
        a["horizons"] = archive["horizons"]
        a["case_index"] = usable
        if "train_end_year" in manifest and "validation_year" in manifest:
            origin = archive["origin"][usable, 2]
            target_time = archive["target_time"][usable]
            cutoff = datetime(int(manifest["train_end_year"]) + 1, 1, 1, tzinfo=timezone.utc).timestamp()
            val_end = datetime(int(manifest["validation_year"]) + 1, 1, 1, tzinfo=timezone.utc).timestamp()
            if np.any((a["split"] == 0) & (origin >= cutoff)):
                raise ValueError("Training origins cross the global year boundary")
            for code, bound in ((0, cutoff), (1, val_end), (2, val_end)):
                selected = a["split"] == code
                if np.any(target_time[selected][a["target_mask"][selected]] >= bound):
                    raise ValueError("Training/validation targets cross their global year boundary")
            if np.any((a["split"] == 2) & ((origin < cutoff) | (origin >= val_end))):
                raise ValueError("Year validation origins fall outside the held-out year")
            valid = a["target_mask"]
            if np.any(target_time[valid] <= np.broadcast_to(origin[:, None], valid.shape)[valid]):
                raise ValueError("Target observations must follow their prediction origins")
    if manifest.get("protocol", "").startswith("wildcast-multiscale-"):
        from .migration_protocol import schema_for
        schema_for(manifest, a)
    if a["history"].ndim != 3 or a["y"].shape[1:] != (len(a["horizons"]), 2):
        raise ValueError("invalid migration array schema")
    if a["history_mask"].shape != a["history"].shape[:2] or a["target_mask"].shape != a["y"].shape[:2]:
        raise ValueError("invalid observation mask shape")
    if not np.isfinite(a["history"][a["history_mask"]]).all() or not np.isfinite(a["static"]).all():
        raise ValueError("migration predictors must be finite")
    if not np.isfinite(a["y"][a["target_mask"]]).all():
        raise ValueError("observed migration targets must be finite")
    if not np.isfinite(a["origin"]).all() or not np.isfinite(a["target_latlon"][a["target_mask"]]).all():
        raise ValueError("observed migration geographic coordinates must be finite")
    if not np.any(a["split"] == 0) or not np.any(np.isin(a["split"], [1, 2])):
        raise ValueError("both training and validation migration cases are required")
    # A held-out individual cannot have another alias in training: the corpus
    # maps canonical physical identities before this check.
    train_animals = set(a["individual"][a["split"] == 0].tolist())
    if train_animals & set(a["individual"][a["split"] == 1].tolist()):
        raise ValueError("individual validation overlaps training identities")
    return a, manifest, {"manifest_sha256": _hash(manifest_path), "cases_sha256": cases_hash}


def _normalization(a, train_ix):
    history = a["history"][train_ix][a["history_mask"][train_ix]]
    static = a["static"][train_ix]
    if not len(history):
        raise ValueError("training corpus contains no observed history")
    return (history.mean(0), np.maximum(history.std(0), 1e-3),
            static.mean(0), np.maximum(static.std(0), 1e-3))


def sampling_probabilities(a, train_ix, sampling="animal", *, allowed_cross_taxon_exclusion_ids=()):
    """Training-only sampling mass, independent of future endpoint values.

    ``animal`` preserves the previous equal-physical-animal sampler exactly.
    Taxon-stratified schemes use (retained taxon, exclusion ID) strata. A shared
    risk block may span labels only when explicitly authorized by the frozen
    manifest; its global exclusion ID is never split or reassigned.
    """
    train_ix = np.asarray(train_ix, dtype=np.int64)
    if sampling not in SAMPLING or not len(train_ix) or len(np.unique(train_ix)) != len(train_ix):
        raise ValueError("Choose a valid sampling scheme and unique nonempty training cases")
    if np.any(a["split"][train_ix] != 0):
        raise ValueError("Sampling probabilities must be fitted on TRAINING cases only")
    animals, inverse, counts = np.unique(a["individual"][train_ix], return_inverse=True, return_counts=True)
    species = a.get("metric_species", a["species"])[train_ix]
    pairs, pair_inverse, pair_counts = np.unique(np.column_stack([a["individual"][train_ix], species]),
                                                axis=0, return_inverse=True, return_counts=True)
    groups, label_counts = np.unique(pairs[:, 0], return_counts=True)
    if set(groups[label_counts > 1]) - set(allowed_cross_taxon_exclusion_ids):
        raise ValueError("A physical training group has conflicting actual species labels without a declared uncertainty block")
    probability = 1. / counts[inverse]
    if sampling != "animal":
        _, species_inverse, animals_per_species = np.unique(pairs[:, 1], return_inverse=True, return_counts=True)
        probability = 1. / pair_counts[pair_inverse]
        case_species_animals = animals_per_species[species_inverse[pair_inverse]]
        probability /= case_species_animals if sampling == "species_animal" else np.sqrt(case_species_animals)
    probability /= probability.sum()
    return probability


def _sampling_evidence(a, train_ix, probability):
    species = a.get("metric_species", a["species"])[train_ix]
    return {str(int(taxon)): {"origins": int(np.sum(species == taxon)),
                             "physical_groups": int(len(np.unique(a["individual"][train_ix][species == taxon]))),
                             "probability_mass": float(probability[species == taxon].sum())}
            for taxon in np.unique(species)}


def _initialize(model, checkpoint, hashes, device):
    """Load compatible weights/buffers, retaining a fresh optimizer downstream."""
    path = Path(checkpoint)
    initial_hash = _hash(path)
    source = torch.load(path, map_location=device, weights_only=True)
    if _hash(path) != initial_hash:
        raise ValueError("Initialization checkpoint changed during loading")
    if source.get("model_kind") != "wildcast_migration_joint":
        raise ValueError("Initialization requires a joint migration model checkpoint")
    for key, value in hashes.items():
        if source.get(key) != value:
            raise ValueError("Initialization checkpoint/data " + key + " mismatch")
    if source.get("config") != model.config:
        raise ValueError("Initialization checkpoint model config/schema mismatch")
    model.load_state_dict(source["model"])
    for name in ("history_mean", "history_std", "static_mean", "static_std"):
        values = getattr(model, name)
        if not torch.isfinite(values).all() or (name.endswith("std") and not torch.all(values > 0)):
            raise ValueError("Initialization normalization buffers are invalid")
    return {"path": str(path.resolve()), "sha256": initial_hash, "epoch": source.get("epoch"),
            "selection_version": source.get("selection_version"), "optimizer_state_carried": False,
            "normalization": "Checkpoint training-only normalizers and uncertainty scales retained on the identical cache"}


def _apply_case_filter(data, a, hashes, case_filter):
    """Verify a frozen, declared past-only train/validation regime artifact."""
    path = Path(case_filter)
    if path.is_dir():
        metadata_path, masks_path = path / "filter.json", path / "filters.npz"
    elif path.suffix == ".json":
        metadata_path, masks_path = path, path.parent / "filters.npz"
    else:
        metadata_path, masks_path = path.parent / "filter.json", path
    metadata = json.loads(metadata_path.read_text())
    for key, value in hashes.items():
        if metadata.get(key) != value:
            raise ValueError("Case filter/data " + key + " mismatch")
    masks_hash = _hash(masks_path)
    if metadata.get("filters_sha256") != masks_hash:
        raise ValueError("Case filter array hash mismatch")
    provenance = metadata.get("provenance", {})
    if (provenance.get("causal") is not True or provenance.get("threshold_fit_split") != "train"
            or provenance.get("uses_future_targets") is not False or not provenance.get("rule")):
        raise ValueError("Case filter needs explicit training-only causal provenance and no future-target selection")
    with np.load(Path(data) / "cases.npz", allow_pickle=False) as source:
        split = source["split"]
    with np.load(masks_path, allow_pickle=False) as source:
        train_mask, val_mask = source["train_filter"], source["val_filter"]
    if any(mask.dtype != np.bool_ or mask.shape != split.shape for mask in (train_mask, val_mask)):
        raise ValueError("Case filters must be full-cache boolean arrays of shape [cases]")
    if np.any((train_mask | val_mask) & (split >= 3)):
        raise ValueError("Case filters must never select reserved test origins")
    if np.any(train_mask & (split != 0)) or np.any(val_mask & ~np.isin(split, [1, 2])):
        raise ValueError("Case filter masks select the wrong train/validation splits")
    original_ids = a["case_index"]
    keep = ((a["split"] == 0) & train_mask[original_ids]) | (np.isin(a["split"], [1, 2]) & val_mask[original_ids])
    selected = {name: values if name == "horizons" else values[keep] for name, values in a.items()}
    if not np.any((selected["split"] == 0) & selected["target_mask"].any(-1)) or not np.any(
            np.isin(selected["split"], [1, 2]) & selected["target_mask"].any(-1)):
        raise ValueError("Case filter must retain observed training and validation origins")
    evidence = {"path": str(path.resolve()), "filter_json_sha256": _hash(metadata_path),
                "filters_sha256": masks_hash, "metadata": metadata,
                "selected_train_origins": int(np.sum(selected["split"] == 0)),
                "selected_validation_origins": int(np.sum(np.isin(selected["split"], [1, 2]))),
                "selection_scope": "Only this declared causal validation subset is used for checkpoint selection"}
    return selected, evidence


def _tensors(a):
    return {name: torch.from_numpy(np.asarray(a[name])) for name in
            ("history", "history_mask", "static", "species", "y", "target_mask", "origin", "target_latlon")}


def _batch(values, ix, device):
    return {name: tensor[ix].to(device) for name, tensor in values.items()}


def _macro(values, species, animals):
    species_values = []
    for taxon in np.unique(species):
        by_animal = [values[(species == taxon) & (animals == animal)].mean()
                     for animal in np.unique(animals[species == taxon])]
        species_values.append(float(np.mean(by_animal)))
    return float(np.mean(species_values))


def _physical_macro(endpoint, joint, mask, species, animals):
    """Exactly match evaluator animal -> species -> horizon aggregation."""
    horizons = []
    for h in range(endpoint.shape[1]):
        selected = mask[:, h]
        if selected.any():
            horizons.append(_macro(endpoint[selected, h], species[selected], animals[selected]))
    return float(np.mean(horizons)), _macro(joint, species, animals)


def physical_validation_scores(draws, target_latlon, target_mask, origin):
    """Great-circle endpoint ES and proper spherical Cartesian joint ES, in km.

    Geometry uses float64, including near-identical sampled locations and
    dateline crossings. Whole-path samples and off-diagonal pairs are shared
    with both scores; missing observations never become zero-coordinate labels.
    """
    draws, origin = draws.double(), origin.double()
    mask = target_mask.bool()
    count = draws.shape[1]
    if count < 2:
        raise ValueError("Physical validation needs at least two path samples")
    radians = math.pi / 180
    lat0, lon0 = origin[:, None, None, 0] * radians, origin[:, None, None, 1] * radians
    x, y = (draws * 1000).unbind(-1)
    rho = torch.hypot(x, y)
    angle = rho / EARTH_KM
    ratio = torch.where(rho > 0, angle.sin() / rho.clamp_min(1e-15), torch.full_like(rho, 1 / EARTH_KM))
    lat = torch.asin((angle.cos() * lat0.sin() + y * ratio * lat0.cos()).clamp(-1, 1))
    lon = lon0 + torch.atan2(x * angle.sin(), rho * lat0.cos() * angle.cos() - y * lat0.sin() * angle.sin())
    lon = torch.where(rho > 0, lon, lon0)
    coords = torch.stack([lat, lon], -1)
    truth = torch.where(mask[..., None], target_latlon.double(), origin[:, None, :2]) * radians

    def geodesic(left, right):
        dlat = left[..., 0] - right[..., 0]
        dlon = torch.remainder(left[..., 1] - right[..., 1] + math.pi, 2 * math.pi) - math.pi
        hav = ((dlat / 2).sin().square() + left[..., 0].cos() * right[..., 0].cos() * (dlon / 2).sin().square()).clamp(0, 1)
        return 2 * EARTH_KM * torch.atan2(hav.sqrt(), (1 - hav).sqrt())

    first = geodesic(coords, truth[:, None]).mean(1)
    pair = geodesic(coords[:, :, None], coords[:, None, :])
    endpoint = first - pair.sum((1, 2)) / (2 * count * (count - 1))
    endpoint = torch.where(mask, endpoint, torch.full_like(endpoint, torch.nan))

    def xyz(c):
        la, lo = c.unbind(-1)
        return EARTH_KM * torch.stack([la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()], -1)

    weights = (mask / mask.sum(-1, keepdim=True).clamp_min(1)).sqrt()[..., None]
    # Translation avoids cancellation from large Earth-centred coordinates.
    anchor = xyz(origin[:, :2] * radians)[:, None]
    embedding = ((xyz(coords) - anchor[:, None]) * weights[:, None]).flatten(-2)
    target = ((xyz(truth) - anchor) * weights).flatten(-2)
    first = (embedding - target[:, None]).norm(dim=-1).mean(1)
    pairs = torch.cdist(embedding, embedding)
    pairs.diagonal(dim1=-2, dim2=-1).zero_()
    joint = first - pairs.sum((1, 2)) / (2 * count * (count - 1))
    return endpoint, joint


@torch.no_grad()
def validate(model, a, values, device, batch_size=256, seed=2026, repeats=8, samples=None):
    model.eval()
    result = {}
    for code, name in ((1, "val_individual"), (2, "val_year")):
        ids = np.flatnonzero((a["split"] == code) & a["target_mask"].any(-1))
        if not len(ids):
            continue
        generator = torch.Generator(device=device).manual_seed(seed + code)
        scores = {"endpoint_energy_km": [], "joint_chordal_energy_km": [], "nll_per_coordinate": []}
        for start in range(0, len(ids), batch_size):
            b = _batch(values, ids[start:start + batch_size], device)
            parameters = model(b["history"], b["history_mask"], b["static"], b["species"])
            draws = sample_paths(parameters, samples or max(2, repeats * 4), generator)
            endpoint, joint = physical_validation_scores(draws, b["target_latlon"], b["target_mask"], b["origin"])
            scores["endpoint_energy_km"].append(endpoint.cpu().numpy())
            scores["joint_chordal_energy_km"].append(joint.cpu().numpy())
            nll = masked_joint_nll(parameters, b["y"], b["target_mask"])
            scores["nll_per_coordinate"].append((nll / (2 * b["target_mask"].sum(-1))).cpu().numpy())
        groups = a.get("metric_species", a["species"])[ids]
        endpoint_by_case = np.concatenate(scores["endpoint_energy_km"])
        endpoint, joint = _physical_macro(endpoint_by_case,
                                          np.concatenate(scores["joint_chordal_energy_km"]),
                                          a["target_mask"][ids], groups, a["individual"][ids])
        result[name] = {"endpoint_energy_km": endpoint, "joint_chordal_energy_km": joint,
                        "nll_per_coordinate": _macro(np.concatenate(scores["nll_per_coordinate"]), groups, a["individual"][ids])}
        result[name].update(cases=int(len(ids)), individuals=int(len(np.unique(a["individual"][ids]))))
        result[name]["horizons"] = {}
        for h, days in enumerate(a["horizons"]):
            observed = a["target_mask"][ids, h]
            if observed.any():
                result[name]["horizons"][str(float(days))] = {
                    "horizon_hours": float(days * 24), "cases": int(observed.sum()),
                    "individuals": int(len(np.unique(a["individual"][ids][observed]))),
                    "endpoint_energy_km": _macro(endpoint_by_case[observed, h], groups[observed],
                                                 a["individual"][ids][observed])}
    selection = float(np.mean([.5 * (v["endpoint_energy_km"] + v["joint_chordal_energy_km"]) for v in result.values()]))
    return {"selection": selection, "selection_version": SELECTION_VERSION, "selection_units": "km",
            "validation_samples": samples or max(2, repeats * 4), "splits": result}


def train(data, output, epochs=40, batch_size=128, width=96, layers=2, components=6,
          lr=.0003, seed=42, device="auto", variant="goal_route", patience=8,
          nll_weight=.1, energy_weight=1., energy_repeats=2, validation_repeats=8,
          weight_decay=.01, dropout=.1, max_train=0, species_dropout=.1, validation_samples=None,
          sampling="animal", init_checkpoint=None, case_filter=None,
          min_scale=.0005, covariance_jitter=None, matmul_precision="highest",
          context_features=None, context_mode="actual", context_only=False):
    if epochs < 0 or batch_size < 1 or patience < 1:
        raise ValueError("epochs must be nonnegative; batch_size and patience must be positive")
    if nll_weight < 0 or energy_weight < 0 or nll_weight + energy_weight <= 0:
        raise ValueError("training needs nonnegative proper-score weights with a positive total")
    if matmul_precision not in ("highest", "high"):
        raise ValueError("matmul_precision must be highest or high")
    if context_only and (not context_features or not init_checkpoint):
        raise ValueError("Context-only fitting requires a feature table and parent checkpoint")
    torch.set_float32_matmul_precision(matmul_precision)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    a, manifest, hashes = load_training_data(data)
    filter_evidence = None
    if case_filter:
        a, filter_evidence = _apply_case_filter(data, a, hashes, case_filter)
    context_evidence = None
    if context_features:
        from .migration_context_joint import append_features
        a, context_evidence = append_features(a, hashes, context_features, context_mode)
    train_ix = np.flatnonzero((a["split"] == 0) & a["target_mask"].any(-1))
    rng = np.random.default_rng(seed)
    if max_train and len(train_ix) > max_train:
        train_ix = np.sort(rng.choice(train_ix, max_train, replace=False))
    if not len(train_ix):
        raise ValueError("no training cases have observed future targets")
    from .migration_protocol import cross_taxon_exclusion_ids
    probabilities = sampling_probabilities(a, train_ix, sampling,
                    allowed_cross_taxon_exclusion_ids=cross_taxon_exclusion_ids(manifest))
    species_count = (max(manifest["species"].values(), default=0) + 1 if "species" in manifest
                     else int(a["species"].max()) + 1)
    model = MigrationModel(a["history"].shape[-1], a["static"].shape[-1],
                           species_count, a["horizons"], history_length=a["history"].shape[1],
                           width=width, layers=layers, components=components, variant=variant,
                           dropout=dropout, species_dropout=species_dropout,
                           min_scale=min_scale, covariance_jitter=covariance_jitter).to(device)
    initialization = None
    if init_checkpoint:
        if Path(init_checkpoint).resolve().is_relative_to(Path(output).resolve()):
            raise ValueError("Continuation output must not overwrite its initialization checkpoint")
        if context_evidence:
            from .migration_context_joint import initialize_expanded
            model.set_normalization(*_normalization(a, train_ix))
            initialization = initialize_expanded(model, init_checkpoint, hashes, device)
        else:
            initialization = _initialize(model, init_checkpoint, hashes, device)
    else:
        model.set_normalization(*_normalization(a, train_ix))
        scales = []
        for h in range(len(a["horizons"])):
            observed = train_ix[a["target_mask"][train_ix, h]]
            scales.append(np.maximum(a["y"][observed, h].std(0), .005) if len(observed) else np.array([.1, .1]))
        model.initialize_scales(np.asarray(scales))
    values = _tensors(a)
    if context_only:
        # Learn only the added input projection. Every parent weight, buffer and
        # dropout law stays frozen; zero gradients alone would not stop AdamW
        # weight decay on the old columns of this shared parameter.
        model.requires_grad_(False)
        projection = model.static_encoder[0].weight
        projection.requires_grad_(True)
        old_columns = context_evidence["base_static_features"]
        gradient_mask = torch.ones_like(projection)
        gradient_mask[:, :old_columns] = 0
        projection.register_hook(lambda gradient: gradient * gradient_mask)
        optimizer = torch.optim.AdamW([projection], lr=lr, weight_decay=0.)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs), eta_min=lr * .1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_animals = len(np.unique(a["individual"][train_ix]))
    validation_samples = int(validation_samples if validation_samples is not None else max(2, validation_repeats * 4))
    if validation_samples < 2:
        raise ValueError("Validation requires at least two path samples")
    code_names = ["migration_model.py", "migration_train.py", "migration_data.py", "migration_evaluate.py"]
    if manifest.get("protocol", "").startswith("wildcast-multiscale-"):
        from .migration_protocol import protocol_code_files
        code_names.extend(path.name for path in protocol_code_files(manifest))
        code_names = list(dict.fromkeys(code_names))
    code_hashes = {name: _hash(Path(__file__).with_name(name)) for name in code_names}
    if context_evidence:
        for name in ("migration_context_joint.py", "migration_human_pressure.py"):
            code_hashes[name] = _hash(Path(__file__).with_name(name))
    run_config = dict(data=str(data), epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
                      patience=patience, nll_weight=nll_weight, energy_weight=energy_weight,
                      energy_repeats=energy_repeats, validation_repeats=validation_repeats,
                      weight_decay=weight_decay, max_train=max_train, device=str(device), validation_samples=validation_samples,
                      sampling=sampling, initialization=initialization, matmul_precision=matmul_precision,
                      case_filter=filter_evidence, context_features=context_evidence,
                      context_only=context_only,
                      effective_weight_decay=0. if context_only else weight_decay,
                      train_case_ids_sha256=hashlib.sha256(np.asarray(a["case_index"][train_ix], dtype="<i8").tobytes()).hexdigest(),
                      validation_case_ids_sha256=hashlib.sha256(np.asarray(a["case_index"][np.isin(a["split"], [1, 2]) & a["target_mask"].any(-1)], dtype="<i8").tobytes()).hexdigest(),
                      sampling_species=_sampling_evidence(a, train_ix, probabilities))
    scope = {"protocol": "Local migration development; reserved test labels are never selected",
             "selection": "Equal great-circle endpoint/chordal joint energy in km, equal split weight; physical-animal/species/horizon macro aggregation",
             "selection_version": SELECTION_VERSION, "selection_units": "km", "code_sha256": code_hashes,
             "training_loss": "Tangent-plane endpoint/joint energy and NLL are optimization surrogates; checkpoint selection uses physical geographic scores",
             "selection_population": "Declared causal regime subset only" if case_filter else "All eligible local validation origins",
             "target_units": "1000 km in origin tangent plane", "train_cases": int(len(train_ix)),
             "train_animals": int(train_animals), "model": model.config, **run_config, **hashes}
    (output / "config.json").write_text(json.dumps(scope, indent=2))
    history, best_score, best_epoch, stale = [], float("inf"), 0, 0
    start_time = time.monotonic()
    for epoch in range(epochs + 1):
        loss_total, batches = 0., 0
        if epoch:
            model.eval() if context_only else model.train()
            selected = rng.choice(train_ix, size=len(train_ix), replace=True, p=probabilities)
            for start in range(0, len(selected), batch_size):
                b = _batch(values, selected[start:start + batch_size], device)
                optimizer.zero_grad(set_to_none=True)
                parameters = model(b["history"], b["history_mask"], b["static"], b["species"])
                nll = masked_joint_nll(parameters, b["y"], b["target_mask"])
                nll = nll / (2 * b["target_mask"].sum(-1).clamp_min(1))
                energy = mixture_energy(parameters, b["y"], b["target_mask"], energy_repeats)
                loss = (nll_weight * nll + energy_weight * .5 *
                        (energy["endpoint_energy"] + energy["path_energy"])).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite migration training loss at epoch {epoch}")
                loss.backward()
                nnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                if not torch.isfinite(nnorm):
                    raise FloatingPointError(f"nonfinite migration gradient at epoch {epoch}")
                optimizer.step()
                loss_total += float(loss.detach())
                batches += 1
            scheduler.step()
        metrics = validate(model, a, values, device, batch_size, seed=2026, repeats=validation_repeats, samples=validation_samples)
        if not np.isfinite(metrics["selection"]):
            raise FloatingPointError(f"nonfinite migration validation score at epoch {epoch}")
        row = {"epoch": epoch, "train_loss": loss_total / batches if batches else None,
               "elapsed_seconds": time.monotonic() - start_time, **metrics}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        improved = metrics["selection"] < best_score - 1e-6
        if improved:
            best_score, best_epoch, stale = metrics["selection"], epoch, 0
        elif epoch:
            stale += 1
        checkpoint = {"model_kind": "wildcast_migration_context_joint" if context_evidence else "wildcast_migration_joint", "model": model.state_dict(),
                      "config": model.config, "epoch": epoch, "metrics": metrics,
                      "manifest": manifest, **hashes, "training": scope,
                      "case_filter": filter_evidence,
                      "context_features": context_evidence,
                      "train_case_ids_sha256": run_config["train_case_ids_sha256"],
                      "validation_case_ids_sha256": run_config["validation_case_ids_sha256"],
                      "optimizer": optimizer.state_dict(), "selection_version": SELECTION_VERSION, "code_sha256": code_hashes}
        if not epoch:
            torch.save(checkpoint, output / "initial.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        torch.save(checkpoint, output / "last.pt")
        print(json.dumps({"epoch": epoch, "train_loss": row["train_loss"],
                          "selection": metrics["selection"], "best": best_score,
                          "seconds": round(row["elapsed_seconds"], 1)}), flush=True)
        if epoch and stale >= patience:
            break
    summary = {"best_selection": best_score, "epochs_completed": history[-1]["epoch"],
               "selected_epoch": best_epoch,
               "parameters": sum(p.numel() for p in model.parameters()), "selection_version": SELECTION_VERSION,
               "selection_units": "km", "code_sha256": code_hashes, "sampling": sampling,
               "initialization": initialization, "case_filter": filter_evidence, **hashes}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="goal_route")
    for name, default in (("epochs", 40), ("batch-size", 128), ("width", 96), ("layers", 2),
                          ("components", 6), ("seed", 42), ("patience", 8), ("energy-repeats", 2),
                          ("validation-repeats", 8), ("max-train", 0)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("lr", .0003), ("nll-weight", .1), ("energy-weight", 1.),
                          ("weight-decay", .01), ("dropout", .1), ("species-dropout", .1)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--sampling", choices=SAMPLING, default="animal")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--case-filter")
    parser.add_argument("--context-features")
    parser.add_argument("--context-mode", choices=("actual", "availability", "masked"), default="actual")
    parser.add_argument("--context-only", action="store_true")
    parser.add_argument("--min-scale", type=float, default=.0005)
    parser.add_argument("--covariance-jitter", type=float, default=None)
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
