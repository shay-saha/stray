"""Plot retained migration diagnostics without fitting or forecasting models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = {
    "migration-v3-goal_route-seed73": "Original goal model (73)",
    "migration-v3-goal_route-seed42": "Original goal model (42)",
    "migration-v3-balanced-species_sqrt-seed42": "Balanced goal model (42)",
    "migration-v3-regime-active-animal-seed17": "Active expert policy (17)",
}
VIEWS = [
    ("frozen-router", "high_past_mobility", "val_individual", "High past mobility\nUnseen animals"),
    ("causal-physical", "past_displacement_ge100km", "val_individual", "Past displacement ≥100 km\nUnseen animals"),
    ("knowledge-animal", "new_individual", "val_year", "New animals\nLater calendar year"),
    ("author-retrospective", "author_in_transit", "val_individual", "Author NSD in-transit\nSmall retrospective cohort"),
]


def plot(report):
    root = Path(report)
    retained = []
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.7), sharey=True)
    for ax, (view, stratum, split, title) in zip(axes, VIEWS):
        source = root / f"{view}-overall.csv"
        frame = pd.read_csv(source)
        selected = frame[(frame.stratum == stratum) & (frame.split == split) & frame.model.isin(MODELS)]
        if set(selected.model) != set(MODELS) or selected.model.duplicated().any():
            raise ValueError(f"Missing or duplicated illustrative model rows in {source}: {stratum}")
        selected = selected.set_index("model").loc[list(MODELS)].reset_index()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        retained.extend({**row, "source": source.name, "source_sha256": digest} for row in selected.to_dict("records"))
        positions = np.arange(len(selected))
        ax.hlines(positions, selected.ci95_low_km, selected.ci95_high_km, color="#277e91", linewidth=2)
        ax.scatter(selected.delta_vs_seasonal_km, positions, color="#277e91", s=30, zorder=3)
        ax.axvline(0, color="#333333", linewidth=.8)
        ax.grid(axis="x", alpha=.2)
        animals = selected.independent_animals.unique()
        if len(animals) != 1:
            raise ValueError("Illustrative models have different animal support")
        ax.set_title(title + f"\n{int(animals[0])} animals", fontsize=10)
        ax.set_xlabel("Energy difference from seasonal (km)\nLower is better", fontsize=9)
    axes[0].set_yticks(np.arange(len(MODELS)), list(MODELS.values()))
    axes[0].invert_yaxis()
    figure.suptitle("Migration development diagnostics: selected models, paired 95% animal intervals", fontsize=13)
    figure.text(.5, .015, "Exploratory validation; cohorts overlap. These selected comparisons do not establish frontier performance.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .055, 1, .92))
    figure.savefig(root / "phase-comparison.png", dpi=180)
    figure.savefig(root / "phase-comparison.svg")
    plt.close(figure)
    (root / "phase-comparison-data.json").write_text(json.dumps({"scope": "Illustrative subset of retained exploratory development scores, not model selection or an independent test", "reference": "seasonal", "rows": retained}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    plot(parser.parse_args().report)
