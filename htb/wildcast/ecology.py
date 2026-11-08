"""Verified species traits, source-tree relatedness and causal driver expansion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .sources import dump
from .train import load_windows


def kernel_coordinates(distance, observed, dimensions=8, training=None):
    """Source branch-distance kernel coordinates, fit on training species only."""
    ids = np.flatnonzero(observed if training is None else observed & training)
    output = np.zeros((len(observed), dimensions), np.float32)
    if len(ids) < 2:
        return output
    d = distance[np.ix_(ids, ids)]
    positive = d[d > 0]
    bandwidth = np.median(positive) if len(positive) else 1.
    kernel = np.exp(-d / bandwidth)
    values, vectors = np.linalg.eigh(kernel)
    order = np.argsort(-values)
    for column, j in enumerate(order[:dimensions]):
        vector = vectors[:, j]
        # Resolve the arbitrary eigensolver sign for reproducible descriptors.
        if vector[np.argmax(abs(vector))] < 0:
            vector = -vector
        if training is None or np.array_equal(ids, np.flatnonzero(observed)):
            output[ids, column] = vector * np.sqrt(max(values[j], 0))
        elif values[j] > 1e-10:
            # Nyström extension uses source-tree distances, never new-species labels.
            query = np.flatnonzero(observed)
            output[query, column] = np.exp(-distance[np.ix_(query, ids)] / bandwidth) @ vector / np.sqrt(values[j])
    return output


def species_features(manifest, sources, taxonomy_root, training_species=None):
    from Bio import Phylo
    sources = Path(sources)
    names = [s for s, i in sorted(manifest["species"].items(), key=lambda p: p[1])]
    n = len(names) + 1
    training_mask = np.isin(np.arange(n), training_species) if training_species is not None else np.arange(n) > 0
    taxonomy = {}
    for species in names:
        p = Path(taxonomy_root) / (species.replace(" ", "_") + ".json")
        if p.exists():
            taxonomy[species] = json.loads(p.read_text())["response"]
    avonet = pd.read_excel(sources / "AVONET1_Birdlife.xlsx", sheet_name="AVONET1_BirdLife")
    avonet = avonet.drop_duplicates("Species1").set_index("Species1")
    birds = pd.read_csv(sources / "BirdFuncDat.txt", sep="\t", encoding="cp1252").set_index("Scientific")
    mammals = pd.read_csv(sources / "MamFuncDat.txt", sep="\t", encoding="cp1252").set_index("Scientific")
    features, report = {}, {}

    def put(name, species_id, value):
        features.setdefault(name, np.full(n, np.nan, np.float64))[species_id] = value

    for species in names:
        i = manifest["species"][species]
        row = {}
        if species in avonet.index:
            a = avonet.loc[species]
            for column in ["Wing.Length", "Kipps.Distance", "Hand-Wing.Index", "Tail.Length", "Tarsus.Length", "Beak.Length_Culmen", "Beak.Width", "Beak.Depth"]:
                value = pd.to_numeric(a[column], errors="coerce")
                put("avonet_" + column.replace(".", "_"), i, value)
            mass = pd.to_numeric(a["Mass"], errors="coerce")
            put("avonet_log_mass_kg", i, np.log1p(mass / 1000) if mass > 0 else np.nan)
            put("avonet_present", i, 1.)
            for column in ["Habitat", "Habitat.Density", "Trophic.Level", "Trophic.Niche", "Primary.Lifestyle", "Inference"]:
                if pd.notna(a[column]):
                    put("avonet_" + column.replace(".", "_") + "_" + str(a[column]).replace(" ", "_"), i, 1.)
            row["avonet"] = {"matched_name": species, "traits_inferred": str(a.get("Traits.inferred", "")),
                              "reference_species": str(a.get("Reference.species", ""))}
        table = birds if species in birds.index else mammals if species in mammals.index else None
        if table is not None:
            a = table.loc[species]
            if isinstance(a, pd.DataFrame):
                raise ValueError("Ambiguous EltonTraits species match")
            for column in [c for c in table.columns if c.startswith("Diet-") and c in
                           ["Diet-Inv", "Diet-Vend", "Diet-Vect", "Diet-Vfish", "Diet-Vunk", "Diet-Scav", "Diet-Fruit", "Diet-Nect", "Diet-Seed", "Diet-PlantO"]]:
                put("elton_" + column.replace("-", "_"), i, pd.to_numeric(a[column], errors="coerce") / 100)
            for column in [c for c in table.columns if c.startswith("ForStrat-") and c not in
                           ["ForStrat-Value", "ForStrat-Certainty", "ForStrat-Comment", "ForStrat-Source", "ForStrat-SpecLevel", "ForStrat-EnteredBy"]]:
                put("elton_" + column.replace("-", "_"), i, pd.to_numeric(a[column], errors="coerce") / 100)
            for column in ["Nocturnal", "Activity-Nocturnal", "Activity-Crepuscular", "Activity-Diurnal", "BodyMass-SpecLevel", "ForStrat-SpecLevel"]:
                if column in a:
                    put("elton_" + column.replace("-", "_"), i, pd.to_numeric(a[column], errors="coerce"))
            mass = pd.to_numeric(a["BodyMass-Value"], errors="coerce")
            put("elton_log_mass_kg", i, np.log1p(mass / 1000) if mass > 0 else np.nan)
            put("elton_present", i, 1.)
            if "ForStrat-Value" in a and pd.notna(a["ForStrat-Value"]):
                put("elton_foraging_code_" + str(a["ForStrat-Value"]), i, 1.)
            row["elton"] = {"matched_name": species, "mass_species_level": str(a.get("BodyMass-SpecLevel", "")),
                            "diet_certainty": str(a.get("Diet-Certainty", "")), "source": str(a.get("BodyMass-Source", ""))}
        report[species] = row
    # One-hot absence is zero only when the corresponding trait source exists.
    for key, values in features.items():
        if key.startswith("avonet_") and any(token in key for token in ["Habitat_", "Habitat_Density_", "Trophic_", "Lifestyle_", "Inference_"]):
            values[np.isnan(values) & np.isfinite(features["avonet_present"])] = 0
        if key.startswith("elton_foraging_code_"):
            values[np.isnan(values) & np.isfinite(features["elton_present"])] = 0

    tree = Phylo.read(sources / "HackettStage1_0001_1000_MCCTreeTargetHeights.nex", "nexus")
    tips = {clade.name: clade for clade in tree.get_terminals()}
    cross = pd.read_csv(sources / "BirdLife-BirdTree crosswalk.csv")
    translated = {name: g.Species3.iloc[0].replace(" ", "_") for name, g in cross.groupby("Species1") if g.Species3.nunique() == 1}
    matched = {}
    for species in names:
        target = translated.get(species, species.replace(" ", "_"))
        if target in tips:
            matched[manifest["species"][species]] = target
            report[species]["phylogeny_tip"] = target
    distance = np.full((n, n), np.nan)
    observed = np.zeros(n, bool)
    for i, a in matched.items():
        observed[i] = True
        for j, b in matched.items():
            distance[i, j] = tree.distance(tips[a], tips[b])
    coordinates = kernel_coordinates(distance, observed, training=training_mask)
    for k in range(coordinates.shape[1]):
        features[f"bird_phylogeny_kernel_{k}"] = np.where(observed, coordinates[:, k], np.nan)
    features["bird_phylogeny_present"] = observed.astype(float)
    # Taxonomic relatedness remains explicitly distinct from genetic distance.
    relatedness = np.eye(n)
    for s in names:
        for t in names:
            a, b = taxonomy.get(s, {}), taxonomy.get(t, {})
            value = sum(weight for rank, weight in [("class", .1), ("order", .2), ("family", .3), ("genus", .4)]
                        if a.get(rank) and a.get(rank) == b.get(rank))
            relatedness[manifest["species"][s], manifest["species"][t]] = value
    bird_ids = np.flatnonzero(observed)
    if len(bird_ids) > 1:
        d = distance[np.ix_(bird_ids, bird_ids)]
        scale = np.median(d[d > 0])
        relatedness[np.ix_(bird_ids, bird_ids)] = np.exp(-d / scale)
    taxonomy_distance = -np.log(np.maximum(relatedness, .01))
    proxy = kernel_coordinates(taxonomy_distance, np.arange(n) > 0, training=training_mask)
    for k in range(proxy.shape[1]):
        features[f"relatedness_mixed_kernel_{k}"] = np.where(np.arange(n) > 0, proxy[:, k], np.nan)
    metadata = {"species": report, "relatedness": relatedness.tolist(), "kernel_fit_species_ids": np.flatnonzero(training_mask).tolist(),
                "bird_phylogeny_distance": [[float(v) if np.isfinite(v) else None for v in row] for row in distance],
                "phylogeny_note": "AVONET source Hackett backbone branch distances where exact source/crosswalk matches exist; source branch-length units. Not DNA sequence distances. Mammals/missing tips retain explicit taxonomy-proxy features and a missing bird-phylogeny mask.",
                "trait_note": "Species estimates, not individual measurements. Source imputation/certainty flags retained. HWI is wing pointedness, not measured wing aspect ratio. Habitat is general habitat, not confirmed breeding grounds. No migration outcome/range-centroid labels used.",
                "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.iterdir() if p.suffix in [".xlsx", ".txt", ".nex", ".csv"]}}
    return features, metadata


def expand(source_root, output, sources, spatial_cache, taxonomy_root="data/combined-v2/environment/taxonomy"):
    from .identity import case_fingerprint
    a, manifest = load_windows(source_root)
    features, biology = species_features(manifest, sources, taxonomy_root, np.unique(a["meta"][a["meta"][:, 0] == 0, 1]))
    species = a["meta"][:, 1]
    values = {"ecology_" + key: v[species] for key, v in features.items()}
    if (Path(sources) / "sst.mnmean.nc").exists():
        from .marine import lookup
        marine_values, marine_info = lookup(sources, a["aux"])
        values.update(marine_values)
        biology["marine"] = marine_info
    if spatial_cache:
        with np.load(Path(spatial_cache) / "features.npz") as z:
            origins = z["origins"]
            positions = np.searchsorted(origins, a["aux"][:, 6].astype(int))
            if not np.array_equal(origins[positions], a["aux"][:, 6].astype(int)):
                raise ValueError("Spatial context does not cover the same origins")
            for k, name in enumerate(z["names"].tolist()):
                values[name] = z["values"][positions, k]
    # Existing lagged weather remains valid-before-origin. These changes are
    # short-window anomalies, not a climatological anomaly or future weather.
    names = manifest["driver_names"]
    def raw(name):
        i = names.index(name)
        return a["drivers"][:, i] * manifest["driver_iqr"][i] + manifest["driver_median"][i]
    for days in [1, 7]:
        p = raw(f"weather_{days}d_temperature_2m_mean_present") > .5
        p &= raw("weather_30d_temperature_2m_mean_present") > .5
        values[f"ecology_temperature_{days}d_minus_30d_C"] = np.where(p, raw(f"weather_{days}d_temperature_2m_mean") - raw("weather_30d_temperature_2m_mean"), np.nan)
    heading = a["aux"][:, 4:6]
    heading = heading / np.maximum(np.linalg.norm(heading, axis=-1, keepdims=True), 1e-8)
    for days in [1, 7]:
        present = (raw(f"weather_{days}d_wind_direction_10m_dominant_present") > .5) & (raw(f"weather_{days}d_wind_speed_10m_max_present") > .5)
        # Meteorological wind direction is where wind comes FROM. Products are
        # daily summaries; pairing max speed with dominant direction is a proxy.
        east = -raw(f"weather_{days}d_wind_sin") * raw(f"weather_{days}d_wind_speed_10m_max")
        north = -raw(f"weather_{days}d_wind_cos") * raw(f"weather_{days}d_wind_speed_10m_max")
        values[f"ecology_wind_support_proxy_{days}d"] = np.where(present, east * heading[:, 0] + north * heading[:, 1], np.nan)
        values[f"ecology_wind_cross_proxy_{days}d"] = np.where(present, east * heading[:, 1] - north * heading[:, 0], np.nan)
    new_names, columns, centers, spreads = [], [], [], []
    training = a["meta"][:, 0] == 0
    coverage = {}
    for name, value in sorted(values.items()):
        value = np.asarray(value, float)
        valid = np.isfinite(value)
        fit = value[training & valid]
        center = float(np.median(fit)) if len(fit) else 0.
        spread = max(float(np.quantile(fit, .75) - np.quantile(fit, .25)), .1) if len(fit) else 1.
        norm = np.where(valid, np.clip((value - center) / spread, -12, 12), 0).astype(np.float32)
        new_names.extend([name, name + "_available"])
        columns.extend([norm, valid.astype(np.float32)])
        centers.extend([center, 0.]);spreads.extend([spread, 1.])
        coverage[name] = float(valid.mean())
    a["drivers"] = np.column_stack([a["drivers"], *columns]).astype(np.float32)
    dest = Path(output) / "processed"
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "windows.npz", **a)
    for filename in ["observations.parquet", "quality-report.json"]:
        if (Path(source_root) / "processed" / filename).exists():
            shutil.copy2(Path(source_root) / "processed" / filename, dest / filename)
    report = {**manifest, "driver_names": names + new_names, "parent_windows_sha256": manifest["windows_sha256"],
              "windows_sha256": hashlib.sha256((dest / "windows.npz").read_bytes()).hexdigest(),
              "case_fingerprint": case_fingerprint(a), "ecology_names": new_names,
              "ecology_center": centers, "ecology_iqr": spreads, "ecology_coverage": coverage, "species_biology": biology,
              "ecology_note": "Observed-only legacy cases unchanged. New continuous features fit on training values only, missing values zero after normalization with explicit availability masks. Retrospective environmental products do not establish historical operational publication availability."}
    dump(dest / "windows.json", report)
    return {"output": str(dest), "drivers": len(report["driver_names"]), "coverage": coverage,
            "avonet_matches": sum("avonet" in r for r in biology["species"].values()),
            "elton_matches": sum("elton" in r for r in biology["species"].values()),
            "bird_phylogeny_matches": sum("phylogeny_tip" in r for r in biology["species"].values())}
