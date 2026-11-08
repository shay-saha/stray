"""Whole-source population development holdouts with alias-group exclusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .migration_data import sha256
from .migration_evaluate import load_cases
from .sources import dump


def prepare(data, output, heldout_sources):
    data, output = Path(data), Path(output)
    if (output / "manifest.json").exists(): raise FileExistsError("Preserve frozen transfer caches")
    a, manifest, hashes = load_cases(data)
    if not manifest.get("reserved_test_targets_sealed"):
        raise ValueError("Transfer development requires sealed base final targets")
    origins = pd.read_parquet(data / "origins.parquet")
    if sha256(data / "origins.parquet") != manifest["origins_sha256"]:
        raise ValueError("Origin metadata differs from the base cache")
    if origins.case_id.tolist() != list(range(len(a["split"]))): raise ValueError("Origin IDs are misaligned")
    mappings = manifest["identity_mapping"]
    held_groups = set()
    for original, canonical in mappings.items():
        if any(original.startswith(source + ":") for source in heldout_sources):
            if canonical in manifest["animals"]: held_groups.add(manifest["animals"][canonical])
    if not held_groups: raise ValueError("No observed identity groups match these source IDs")
    held = np.isin(a["individual"], list(held_groups))
    development = a["split"] < 3
    # Whole physical/exclusion groups are removed from training, including
    # aliases whose canonical source is a different release.
    a["split"][held & development] = 1
    unrelated_validation = ~held & development & (a["split"] != 0)
    a["target_mask"][unrelated_validation] = False
    a["y"][unrelated_validation] = 0
    a["target_latlon"][unrelated_validation] = 0
    a["target_time"][unrelated_validation] = 0
    if not np.any(held & development & a["target_mask"].any(-1)):
        raise ValueError("No observed development targets in the held-out population")
    def xyz(points):
        lat, lon = np.deg2rad(points[:, :2]).T
        return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    tree = cKDTree(xyz(a["origin"][a["split"] == 0]))
    scored = held & development & a["target_mask"].any(-1)
    chord, _ = tree.query(xyz(a["origin"][scored]))
    nearest_km = 2 * 6371.0088 * np.arcsin(np.clip(chord / 2, 0, 1))
    names = origins.species.to_numpy()
    train_species = {name: i + 1 for i, name in enumerate(sorted(set(names[a["split"] == 0])))}
    a["species"] = np.asarray([train_species.get(n, 0) for n in names], dtype="int64")
    origins["base_case_id"] = origins.case_id
    origins["split"] = [{0: "train", 1: "val_individual", 2: "val_year", 3: "test_individual", 4: "test_year"}[int(s)] for s in a["split"]]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "cases.npz", **a)
    origins.to_parquet(output / "origins.parquet", index=False)
    manifest.update({"species": train_species, "cases_sha256": sha256(output / "cases.npz"),
        "origins_sha256": sha256(output / "origins.parquet"), "base_cache": str(data), "base_hashes": hashes,
        "population_holdout_sources": list(heldout_sources), "population_holdout_groups": sorted(held_groups),
        "eligible_for_final_assessment": False,
        "scope": "Whole-source population DEVELOPMENT holdout; source identity is not proof of geographic separation. All corroborated/plausible alias groups are excluded from training; base final targets stay sealed.",
        "case_counts": {str(s): int((a["split"] == s).sum()) for s in np.unique(a["split"])},
        "scored_population_cases": int((held & development & a["target_mask"].any(-1)).sum()),
        "nearest_training_origin_km": {"median": float(np.median(nearest_km)), "p90": float(np.quantile(nearest_km, .9)),
                                       "fraction_over_100km": float(np.mean(nearest_km > 100))},
        "transfer_code_sha256": sha256(__file__)})
    # This derivative intentionally keeps the base final cohort sealed and is
    # used only to stress generalization, never as another final-test release.
    dump(output / "manifest.json", manifest)
    return {k: manifest[k] for k in ["population_holdout_sources", "scored_population_cases", "case_counts", "scope"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--heldout-sources", nargs="+", required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))
