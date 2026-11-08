"""Taxonomy and deployment-time traits; never use tag mass or animal outcomes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from .sources import dump, session


def cache_taxonomy(root):
    root = Path(root)
    f = pd.read_parquet(root / "processed/observations.parquet")
    out = root / "environment/taxonomy"; out.mkdir(parents=True, exist_ok=True)
    s = session(); records = {}
    for species in sorted(f.species.unique()):
        p = out / (species.replace(" ", "_") + ".json")
        if p.exists():
            records[species] = json.loads(p.read_text())
            continue
        r = s.get("https://api.gbif.org/v1/species/match", params={"name": species, "strict": "true"}, timeout=30)
        r.raise_for_status(); j = r.json()
        record = {"query": species, "url": r.url, "response_sha256": hashlib.sha256(r.content).hexdigest(),
                  "retrieved_utc": datetime.now(timezone.utc).isoformat(), "response": j}
        dump(p, record); records[species] = record
        time.sleep(.1)
    dump(out / "manifest.json", {"provider": "GBIF species matching API", "species": list(records),
         "note": "Public taxonomy, not migration status or physiological measurements. Synonyms preserved in source response."})
    return {"species": len(records), "output": str(out)}


class TraitLookup:
    def __init__(self, source_root, feature_root, observations, train_species):
        self.records, self.taxa = {}, {}
        self.vocabulary = {}
        known = observations.drop_duplicates("animal")
        animals = known.animal.tolist()
        report = {"joined_deployments": 0, "ambiguous_identities": 0, "mass_unit_verified_sources": []}
        for directory in (Path(source_root) / "raw").glob("*"):
            manifest = directory / "source.json"
            if not manifest.exists():
                continue
            meta = json.loads(manifest.read_text())
            rights = meta.get("metadata", {}).get("dc.rights.uri", [])
            if not rights or not all(any(t in u.lower() for t in ["/publicdomain/zero/1.0", "/licenses/by/4.0", "/licenses/by/3.0"]) for u in rights):
                continue
            readmes = list(directory.glob("README*"))
            text = "\n".join(p.read_text(errors="replace") for p in readmes)
            grams = bool(re.search(r"animal mass:.{0,400}?(?:units|unit):\s*grams", text, re.I | re.S))
            if grams:
                report["mass_unit_verified_sources"].append(directory.name)
            for p in directory.glob("*reference*.csv"):
                frame = pd.read_csv(p, low_memory=False)
                if not {"animal-id", "deploy-on-date"} <= set(frame):
                    continue
                for _, row in frame.iterrows():
                    identity = str(row["animal-id"])
                    matches = [a for a in animals if a.startswith(directory.name + ":") and a.endswith(":" + identity)]
                    if len(matches) != 1:
                        report["ambiguous_identities"] += int(len(matches) > 1)
                        continue
                    deployed = pd.to_datetime(row["deploy-on-date"], utc=True, errors="coerce")
                    if pd.isna(deployed) or deployed.year < 1980:
                        continue
                    mass = pd.to_numeric(row.get("animal-mass", np.nan), errors="coerce") if grams else np.nan
                    mass = float(mass / 1000) if np.isfinite(mass) and mass > 0 else np.nan
                    self.records.setdefault(matches[0], []).append({"available_after": deployed.timestamp(),
                         "sex": str(row.get("animal-sex", "")).strip().lower(),
                         "capture_stage": str(row.get("animal-life-stage", "")).strip().lower(), "mass_kg": mass})
                    report["joined_deployments"] += 1
        for p in (Path(feature_root) / "environment/taxonomy").glob("*.json"):
            if p.name == "manifest.json":
                continue
            record = json.loads(p.read_text()); response = record["response"]
            if response.get("confidence", 0) >= 90 and response.get("matchType") == "EXACT":
                self.taxa[record["query"]] = response
        for rank in ["class", "order", "family", "genus"]:
            self.vocabulary[rank] = sorted({self.taxa[s][rank] for s in train_species if s in self.taxa and self.taxa[s].get(rank)})
        report["taxonomy_species"] = len(self.taxa)
        report["taxonomy_vocabulary"] = self.vocabulary
        report["note"] = "Mass is at capture, in explicitly verified units. Life-stage words are capture-stage labels. Unknown stages and missing traits are masked; no animal fate/reproduction outcome fields are used."
        self.report = report

    def features(self, animal, species, origin):
        choices = [r for r in self.records.get(animal, []) if r["available_after"] <= origin]
        r = max(choices, key=lambda r: r["available_after"]) if choices else {}
        sex, stage = r.get("sex", ""), r.get("capture_stage", "")
        mass = r.get("mass_kg", np.nan)
        is_subadult = "subadult" in stage or "sub-adult" in stage
        out = {"trait_sex_male": float(sex in ["m", "male"]), "trait_sex_female": float(sex in ["f", "female"]),
               "trait_sex_present": float(sex in ["m", "male", "f", "female"]),
               "trait_capture_adult": float("adult" in stage and not is_subadult),
               "trait_capture_subadult": float(is_subadult), "trait_capture_juvenile": float("juvenile" in stage or "immature" in stage),
               "trait_capture_stage_present": float(any(t in stage for t in ["adult", "juvenile", "immature"])),
               "trait_capture_log_mass_kg": float(np.log1p(mass)) if np.isfinite(mass) else 0.,
               "trait_capture_mass_present": float(np.isfinite(mass)),
               "trait_log_days_since_deployment": float(np.log1p((origin - r["available_after"]) / 86400)) if r else 0.,
               "trait_deployment_present": float(bool(r))}
        taxon = self.taxa.get(species, {})
        for rank, values in self.vocabulary.items():
            out[f"taxonomy_{rank}_present"] = float(taxon.get(rank) in values)
            for value in values:
                out[f"taxonomy_{rank}_{value}"] = float(taxon.get(rank) == value)
        return out
