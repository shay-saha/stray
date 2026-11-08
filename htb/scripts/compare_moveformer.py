"""Paired local validation comparison; never reads the untouched test splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--experiments", required=True, help="JSON mapping labels to evaluation directories")
p.add_argument("--reference", default="WildCast 296 inputs")
p.add_argument("--output", default="reports/moveformer-comparison")
p.add_argument("--replicates", type=int, default=1000)
p.add_argument("--identity-audit")
args = p.parse_args()
experiments = json.loads(Path(args.experiments).read_text())
identity_audit=json.loads(Path(args.identity_audit).read_text()) if args.identity_audit else None
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
frames = {}; evidence = {}; keys = ["window", "split", "study", "animal", "horizon_hours"]
for name, spec in experiments.items():
    path = Path(spec["path"] if isinstance(spec,dict) else spec)
    column = spec.get("column","transformer_es_m") if isinstance(spec,dict) else "transformer_es_m"
    f = pd.read_parquet(path / "per-window.parquet").sort_values(keys).reset_index(drop=True)
    if identity_audit:f=f[~f.window.isin(identity_audit["excluded_windows"])].reset_index(drop=True)
    if not set(f.split) <= {"val_temporal", "val_individual"}:
        raise ValueError("This comparison accepts validation splits only")
    if frames:
        pd.testing.assert_frame_equal(f[keys], next(iter(frames.values()))[keys])
        np.testing.assert_allclose(f.dont_move_es_m, next(iter(frames.values())).dont_move_es_m, atol=.01, rtol=1e-5)
    if column != "transformer_es_m":
        f = f.copy(); f["transformer_es_m"] = f[column]
    frames[name] = f
    evidence[name] = json.loads((path / "metrics.json").read_text())
if args.reference not in frames:
    raise ValueError("Reference experiment missing")
base = frames[args.reference]
rows = base[keys].copy()
rows["dont_move_es_m"] = base.dont_move_es_m
for name, f in frames.items():
    rows[name] = f.transformer_es_m
rows.to_parquet(out / "paired-windows.parquet", index=False)
names = list(frames)
cells = rows.groupby(["split", "study", "horizon_hours"])[["dont_move_es_m", *names]].mean()
scores = cells[names].div(cells.dont_move_es_m.clip(lower=1), axis=0)
scores.to_csv(out / "paired-study-horizon.csv")
macro = scores.groupby("split").mean()
macro.to_csv(out / "scores.csv")
intervals = {}
rng = np.random.default_rng(20261107)
for split, g in rows.groupby("split"):
    blocks = []
    for study, gg in g.groupby("study"):
        animals, horizons = sorted(gg.animal.unique()), sorted(gg.horizon_hours.unique())
        ai, hi = {v:i for i,v in enumerate(animals)}, {v:i for i,v in enumerate(horizons)}
        sums = np.zeros((len(animals), len(horizons), 1 + len(names)))
        counts = np.zeros((len(animals), len(horizons)))
        for (animal, horizon), cell in gg.groupby(["animal", "horizon_hours"]):
            sums[ai[animal], hi[horizon]] = cell[["dont_move_es_m", *names]].sum()
            counts[ai[animal], hi[horizon]] = len(cell)
        blocks.append((sums, counts))
    draws = []
    for _ in range(args.replicates):
        values = []
        for i in rng.integers(0, len(blocks), len(blocks)):
            sums, counts = blocks[i]
            w = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)))
            s = np.tensordot(w, sums, axes=(0, 0)); n = w @ counts
            valid = n > 0
            values.extend((s[valid, 1:] / np.maximum(s[valid, :1], n[valid, None])).tolist())
        means = np.mean(values, axis=0)
        draws.append(means - means[names.index(args.reference)])
    bounds = np.quantile(draws, [.025, .975], axis=0)
    intervals[split] = {n: {"delta_vs_reference": float(macro.loc[split, n] - macro.loc[split, args.reference]),
                             "paired_ci95": bounds[:, i].tolist(),
                             "fraction_study_horizon_cells_better": float((scores.loc[split,n] < scores.loc[split,args.reference]).mean())}
                       for i,n in enumerate(names)}
report = {"scope": "Local validation development comparison, not official MoveBench or fresh-test evidence",
          "reference": args.reference, "macro_normalized_energy": macro.to_dict(orient="index"),
          "paired_uncertainty": intervals, "bootstrap": {"replicates": args.replicates, "unit": "study then animal"},
          "cases": {s:len(g) for s,g in base.groupby("split")},
          "case_keys_sha256": hashlib.sha256(base[keys].to_csv(index=False).encode()).hexdigest(),
          "evidence": evidence,
          "limitations": ["Models and hyperparameters were selected on this validation data; intervals do not correct for model selection.",
                          "Author MoveFormer forward methods with a local feature adapter, not the unavailable benchmark implementation.",
                          "Neither model receives full candidate-destination environmental rasters in the matched 296-input comparison.",
                          "Training histories and parameter counts differ. Best existing ensemble includes a long-memory member."]}
protocol_path = Path("reports/moveformer-protocol.json")
if protocol_path.exists():
    report["protocol"] = json.loads(protocol_path.read_text())
if identity_audit:report["identity_audit"]=identity_audit
(out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(max(10, len(names)*1.1), 4.8), layout="constrained", sharey=True)
for ax, split in zip(axes, ["val_individual", "val_temporal"]):
    values = macro.loc[split]
    ax.bar(np.arange(len(names)), values, color=["#2d6a8a" if n.startswith("WildCast") else "#589b7b" for n in names])
    ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right", fontsize=8)
    for i,v in enumerate(values): ax.text(i,v+.012,f"{v:.3f}",ha="center",fontsize=9)
    ax.axhline(1,color="#777",linestyle=":",linewidth=1)
    ax.set_title("Unseen animals" if split=="val_individual" else "Later observations")
    ax.set_ylim(0,max(1.1,float(macro.to_numpy().max())*1.15))
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Macro normalized energy score (lower is better)")
fig.suptitle("MoveFormer author core vs WildCast — local validation")
fig.supxlabel("Validation point estimates · local input adapter · later-observation bootstrap is inconclusive", fontsize=9)
fig.savefig(out / "comparison.png", dpi=180); fig.savefig(out / "comparison.svg")
print(macro.round(5).to_string())
