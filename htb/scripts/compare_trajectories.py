"""Compare frozen local trajectory evaluations on identical audited cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--experiments", required=True, help="Map labels to {path,column} evaluation records")
p.add_argument("--reference", default="Endpoint ensemble")
p.add_argument("--output", default="reports/trajectory-comparison")
p.add_argument("--replicates", type=int, default=1000)
p.add_argument("--title", default="Connected trajectories versus the existing endpoint ensemble")
args = p.parse_args()
specs = json.loads(Path(args.experiments).read_text())
keys = ["window", "split", "study", "animal", "horizon_hours"]
rows, evidence = None, {}
for name, spec in specs.items():
    path = Path(spec["path"])
    f = pd.read_parquet(path / "per-window.parquet").sort_values(keys).reset_index(drop=True)
    if not set(f.split) <= {"val_individual", "val_temporal"}:
        raise ValueError("Validation development comparisons only")
    if rows is None:
        rows = f[keys + ["stationary_es_m"]].copy()
    else:
        pd.testing.assert_frame_equal(rows[keys], f[keys])
        np.testing.assert_allclose(rows.stationary_es_m, f.stationary_es_m, rtol=1e-6, atol=.01)
    rows[name] = f[spec["column"]]
    evidence[name] = json.loads((path / "metrics.json").read_text())
names = list(specs)
if args.reference not in names:
    raise ValueError("Reference missing")
cells = rows.groupby(["split", "study", "horizon_hours"])[["stationary_es_m", *names]].mean()
normalized = cells[names].div(cells.stationary_es_m.clip(lower=1), axis=0)
macro = normalized.groupby("split").mean()
intervals = {}
rng = np.random.default_rng(20261107)
for split, f in rows.groupby("split"):
    blocks = []
    for _, group in f.groupby("study"):
        animals, horizons = sorted(group.animal.unique()), sorted(group.horizon_hours.unique())
        ai, hi = {v: i for i, v in enumerate(animals)}, {v: i for i, v in enumerate(horizons)}
        sums = np.zeros((len(animals), len(horizons), len(names) + 1))
        counts = np.zeros((len(animals), len(horizons)))
        for (animal, horizon), g in group.groupby(["animal", "horizon_hours"]):
            sums[ai[animal], hi[horizon]] = g[["stationary_es_m", *names]].sum()
            counts[ai[animal], hi[horizon]] = len(g)
        blocks.append((sums, counts))
    deltas = []
    for _ in range(args.replicates):
        values = []
        for block in rng.integers(0, len(blocks), len(blocks)):
            sums, counts = blocks[block]
            weights = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)))
            s, n = np.tensordot(weights, sums, axes=(0, 0)), weights @ counts
            valid = n > 0
            values.extend((s[valid, 1:] / np.maximum(s[valid, :1], n[valid, None])).tolist())
        mean = np.mean(values, axis=0)
        deltas.append(mean - mean[names.index(args.reference)])
    bounds = np.quantile(deltas, [.025, .975], axis=0)
    intervals[split] = {name: {"delta_vs_reference": float(macro.loc[split, name] - macro.loc[split, args.reference]),
                               "paired_ci95": bounds[:, i].tolist()} for i, name in enumerate(names)}
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
macro.to_csv(out / "scores.csv")
normalized.to_csv(out / "study-horizon-scores.csv")
report = {"scope": "Audited local validation development; models selected on validation, not an official MoveBench or independent test result",
          "reference": args.reference, "macro_normalized_energy": macro.to_dict(orient="index"),
          "paired_uncertainty": intervals, "bootstrap": {"replicates": args.replicates, "unit": "study then animal"},
          "case_counts": rows.groupby("split").size().to_dict(),
          "case_keys_sha256": hashlib.sha256(rows[keys].to_csv(index=False).encode()).hexdigest(),
          "evidence": evidence}
(out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
for ax, split, title in zip(axes, ["val_individual", "val_temporal"], ["Unseen animals", "Later movement"]):
    scores = macro.loc[split]
    ax.bar(np.arange(len(names)), scores, color=["#24557a" if n == args.reference else "#5e9676" for n in names])
    ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right", fontsize=8)
    for i, value in enumerate(scores):
        ax.text(i, value + .015, f"{value:.3f}", ha="center", fontsize=9)
    ax.axhline(1, color="#888", linestyle=":")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Macro normalized energy score (lower is better)")
fig.suptitle(args.title)
fig.supxlabel("Same audited validation cases · development results · uncertainty in comparison.json")
fig.savefig(out / "comparison.png", dpi=180)
fig.savefig(out / "comparison.svg")
print(macro.round(5).to_string())
