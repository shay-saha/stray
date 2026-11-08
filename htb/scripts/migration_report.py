"""Create a retained scientific comparison from one frozen evaluation report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def report(evaluation, output=None):
    root = Path(evaluation)
    dest = Path(output) if output else root
    dest.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metrics.json").read_text())
    scores = pd.read_csv(root / "scores.csv")
    models = sorted(scores.model.unique())
    splits = [s for s in ["val_individual", "val_year", "test_individual", "test_year"] if s in set(scores.split)]
    paired = metadata["paired_animal_bootstrap"]
    reference = paired["reference"]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.4 * len(splits), max(4.5, len(models) * .32)), squeeze=False)
    for ax, split in zip(axes[0], splits):
        comparison = paired["splits"][split]
        names = sorted(models, key=lambda n: comparison[n]["delta_km"])
        values = [comparison[n]["delta_km"] for n in names]
        lower = [comparison[n]["ci95_km"][0] for n in names]
        upper = [comparison[n]["ci95_km"][1] for n in names]
        colors = ["#777777" if n == reference else "#187892" for n in names]
        positions = np.arange(len(names))
        ax.hlines(positions, lower, upper, color=colors, linewidth=2)
        ax.scatter(values, positions, c=colors, s=26, zorder=3)
        ax.axvline(0, color="#222222", linewidth=.8)
        ax.set_yticks(positions, [n.replace("migration-v3-", "").replace("migration-transfer-", "") for n in names])
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.2)
        ax.set_title({"val_individual": "Unseen animals: development", "val_year": "Held-out year: development",
                      "test_individual": "Reserved unseen animals", "test_year": "Reserved later years"}[split])
        ax.set_xlabel(f"Endpoint energy difference from {reference} (km)\nLower is better; paired 95% animal bootstrap interval")
    fig.tight_layout()
    fig.savefig(dest / "comparison.png", dpi=180)
    fig.savefig(dest / "comparison.svg")
    plt.close(fig)
    lines = ["# Migration forecasting comparison", "", metadata["scope"] + ".", "",
             f"{metadata['cases']:,} identical forecast origins; {metadata['observed_targets']:,} matched actual GPS targets; {metadata['samples']} sampled paths per origin.", "",
             "| Model | Split | Endpoint energy (km) | Joint energy (km) | Mean-location error (km) | 90% radial coverage |",
             "|---|---|---:|---:|---:|---:|"]
    joint = {(r["model"], r["split"]): r["joint_chordal_energy_km"] for r in metadata["joint_scores"]}
    for row in scores.sort_values(["split", "endpoint_energy_km"]).itertuples():
        lines.append(f"| {row.model} | {row.split} | {row.endpoint_energy_km:.3f} | {joint[row.model, row.split]:.3f} | {row.point_error_km:.3f} | {row.radial_coverage90:.3f} |")
    lines += ["", "Energy scores assess predictive distributions; they are not mean location errors. Endpoint distances are great-circle distances. Joint scores use a masked spherical Cartesian path embedding. Coverage is radial about the spherical sample mean, not highest-density-region coverage.", "",
              "Aggregation gives equal weight to independent animal/exclusion groups within species, then species and horizons. Intervals resample animals within species, retaining their origins and horizons; they do not establish transfer to every species or region.", "",
              "Model selection and mixture calibration use development data unless the scope explicitly identifies a frozen reserved assessment. A development win alone does not establish frontier or official MoveBench performance.", "",
              f"Frozen data: `{metadata['manifest_sha256']}` (manifest), `{metadata['cases_sha256']}` (cases).",
              f"Identical scored origin IDs: `{metadata['case_ids_sha256']}`.", "",
              "See metrics.json, per-case.parquet, per-species.csv, per-horizon.csv, joint-per-origin.parquet and target-timing.csv for complete scope and evidence."]
    (dest / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return {"output": str(dest), "models": len(models), "splits": splits}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--output")
    print(json.dumps(report(**vars(p.parse_args())), indent=2))
