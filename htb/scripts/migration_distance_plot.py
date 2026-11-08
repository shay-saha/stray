"""Show physical location errors from retained, matched development forecasts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODELS = {
    "seasonal": ("Seasonal retrieval", "#777777"),
    "migration-v3-moveformer-context-masked-seed42": ("Local MoveFormer adaptation", "#c47c25"),
    "migration-v3-ensemble-equal6": ("WildCast uniform ensemble", "#197b91"),
}


def plot(evaluation, output):
    source = Path(evaluation) / "per-horizon.csv"
    metadata = json.loads((Path(evaluation) / "metrics.json").read_text())
    data = pd.read_csv(source)
    selected = data[data.model.isin(MODELS)].copy()
    if set(selected.model) != set(MODELS):
        raise ValueError("Illustrative model comparison is incomplete")
    splits = ["val_individual", "val_year"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, split in zip(axes, splits):
        for model, (label, color) in MODELS.items():
            rows = selected[(selected.model == model) & (selected.split == split)].sort_values("horizon_days")
            if len(rows) != 7 or rows.horizon_days.duplicated().any():
                raise ValueError("Expected seven distinct retained migration horizons")
            ax.plot(rows.horizon_days, rows.point_error_km, marker="o", linewidth=2, color=color, label=label)
        ax.set_xscale("log")
        days = sorted(selected.horizon_days.unique())
        ax.set_xticks(days, [str(int(d)) for d in days])
        ax.set_xlabel("Days ahead")
        ax.grid(alpha=.2)
        ax.set_title("Unseen animals" if split == "val_individual" else "Later calendar year")
    axes[0].set_ylim(0, selected.point_error_km.max() * 1.08)
    axes[0].set_ylabel("Average location error (km)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("How far predicted mean locations are from actual GPS observations")
    fig.text(.5, .015, "Matched development cases; species/animal macro averages. Illustrative selected models; final test remains sealed.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .93))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "distance-by-horizon.png", dpi=180)
    fig.savefig(output / "distance-by-horizon.svg")
    plt.close(fig)
    selected.to_csv(output / "distance-by-horizon.csv", index=False)
    evidence = {"scope": "Illustrative retained development comparison; no model fitting or final selection", "metric": "Physical-animal/species macro spherical-mean location error; distinct from distribution energy", "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "manifest_sha256": metadata["manifest_sha256"], "cases_sha256": metadata["cases_sha256"], "models": list(MODELS), "samples": metadata["samples"]}
    (output / "distance-by-horizon.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    plot(**vars(parser.parse_args()))
