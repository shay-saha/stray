"""Retain small, inspectable reports and scientific plots from the frozen run."""
from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "reports"
out.mkdir(exist_ok=True)
quality = json.loads((root / "data/processed/quality-report.json").read_text())
windows = json.loads((root / "data/processed/windows.json").read_text())
metrics = json.loads((root / "runs/final-test/metrics.json").read_text())
selection = json.loads((out / "selection.json").read_text())
assert selection["windows_sha256"] == metrics["windows_sha256"]
assert selection["checkpoints"] == metrics["checkpoint_info"]
for name in ["metrics.json", "per-study-horizon.csv"]:
    shutil.copyfile(root / "runs/final-test" / name, out / ("test-" + name))
shutil.copyfile(root / "runs/final-validation/metrics.json", out / "validation-metrics.json")
shutil.copyfile(root / "data/processed/quality-report.json", out / "quality-report.json")
obs = pd.read_parquet(root / "data/processed/observations.parquet")
species = obs.groupby("species").agg(observations=("lat", "size"), animals=("animal", "nunique"), studies=("study", "nunique"))
species.to_csv(out / "species.csv")
rows = []
for p in sorted((root / "data/raw").glob("*/source.json")):
    j = json.loads(p.read_text()); m = j.get("metadata", {})
    rows.append({"source_id": j["source_id"], "title": j.get("title", ""), "url": j.get("url", ""),
                 "license": "; ".join(m.get("dc.rights", [])) or j.get("license", ""),
                 "citation": "; ".join(m.get("dc.identifier.citation", [])) or j.get("citation", ""),
                 "retained_observations": int((obs.source == j["source_id"]).sum())})
pd.DataFrame(rows).to_csv(out / "sources.csv", index=False)

names = ["dont_move", "crw", "analogue", "transformer", "hybrid"]
labels = ["Stationary", "Random walk", "Analogues", "Transformer ×3", "Fixed hybrid"]
colors = ["#a9b2bd", "#7b8794", "#52a897", "#5074ba", "#163e67"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout="constrained")
for ax, split, title in zip(axes, ["test_individual", "test_temporal"], ["Animals excluded from training", "Later movement of training animals"]):
    cell = metrics["macro_by_split"][split]
    values = [cell[n + "_normalized_es"] for n in names]
    ax.bar(np.arange(len(names)), values, color=colors, width=.65)
    for i, value in enumerate(values):
        ax.text(i, value + .018, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(names)), labels, rotation=24, ha="right")
    ax.axhline(1, color="#a0a0a0", linestyle=":", linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(1.15, max(values) * 1.16))
    ax.set_axisbelow(True); ax.grid(axis="y", alpha=.14)
axes[0].set_ylabel("Macro normalized energy score · lower is better")
fig.suptitle("WildCast — frozen local test evaluation", fontsize=15, weight="bold")
fig.supxlabel("Local protocol · no official MoveBench comparison · unseen-animal uncertainty interval spans zero", fontsize=9)
fig.savefig(out / "test-comparison.png", dpi=180)
fig.savefig(out / "test-comparison.svg")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
for seed, color in zip([42, 17, 7], ["#5074ba", "#52a897", "#a76f45"]):
    h = pd.DataFrame(json.loads((root / f"runs/multispecies-seed{seed}/history.json").read_text()))
    ax.plot(h.epoch, h.val_macro_normalized_energy, label=f"Seed {seed}", color=color, marker=".")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro normalized energy")
ax.set_title("Three independent training seeds; checkpoint selected on validation")
ax.legend(frameon=False); ax.grid(alpha=.15)
fig.savefig(out / "training.png", dpi=160); plt.close(fig)

lines = ["# Frozen local experiment", "", "These results are not official MoveBench results and do not establish a frontier model.", "",
    f"The cleaned corpus has **{len(obs):,} observed GPS fixes, {obs.animal.nunique()} animals, {obs.species.nunique()} species and {obs.study.nunique()} study areas**. The archive catalogue contains 449 public data packages; 20 packages plus the full author jaguar release were downloaded. One package with contradictory rights was quarantined. Some tables were unsupported or contained no retained GPS records.", "",
    "The final forecast is a fixed 50/50 mixture of rotation-aligned historical analogues and a three-seed Student-t transformer ensemble. Each network has 686,778 parameters. Configuration and checkpoint hashes were frozen in `selection.json` before test evaluation. The selection criterion prioritized unseen-individual validation energy, with temporal validation required to improve on CRW.", "",
    "| Forecast | Unseen animals | Later movement |", "|---|---:|---:|"]
for name, label in zip(names, labels):
    left = metrics["macro_by_split"]["test_individual"][name + "_normalized_es"]
    right = metrics["macro_by_split"]["test_temporal"][name + "_normalized_es"]
    lines.append(f"| {label} | {left:.4f} | {right:.4f} |")
lines += ["", "Lower energy score is better. Scores are macro-averaged over study/horizon cells. The stationary forecast normalizes to 1 in this evaluated dataset. Constant-velocity scores are retained in `test-metrics.json` but omitted from the plot because of their larger scale.", "",
          "| Split | Hybrid change versus CRW | Paired 95% bootstrap interval for normalized difference | 90% radial coverage of hybrid |", "|---|---:|---|---:|"]
for split in ["test_individual", "test_temporal"]:
    m = metrics["macro_by_split"][split]
    change = 100 * (m["hybrid_normalized_es"] / m["crw_normalized_es"] - 1)
    interval = metrics["uncertainty"][split]["paired_delta_vs_crw_ci95"]["hybrid"]
    lines.append(f"| {split} | {change:+.2f}% | [{interval[0]:+.4f}, {interval[1]:+.4f}] | {m['hybrid_radial_coverage90']:.3f} |")
lines += ["", "Intervals use 300 paired hierarchical bootstrap replicates, resampling studies and then animals within study. A difference interval spanning zero does not establish a reliable advantage for the sampled study population. Radial coverage is about the sample mean, not highest-density-region coverage. Overlapping forecast windows remain clustered together by animal.", "",
          "All 1h, 12h, 24h and 7d cases use actual matched future observations; targets are never interpolated. Test windows: " + ", ".join(f"{s}={windows['splits'][s]:,}" for s in ["test_individual", "test_temporal"]) + ". Full cell counts and raw metre scores are retained in `test-per-study-horizon.csv`.", "",
          "Post-test audit: one held-out Galápagos albatross has a 7-day cell with only 13 cases and about 6.5 m average observed endpoint displacement, while the hybrid's energy score is about 9.65 km. Other animals in that study move much farther. Resampling the low-motion animal makes relative study normalization unstable and contributes to the very wide individual-test interval. This is a real failure case retained in the results. No records or scores were removed after observing it, and zero whole-track stationarity flags do not resolve this behavior-specific error.", "",
          "Missing capabilities: environmental/raster conditioning, coherent multi-step trajectory forecasts, official MoveBench assets/evaluator, prospective conservation validation, and the C++ renderer. Source timezone assumptions, incomplete identity resolution, stationarity review and biological movement checks remain research limitations. See `RESEARCH.md`.", "",
          "Reproduce the scores using the checkpoint paths recorded in `test-metrics.json`, the exact `windows_sha256`, and the `evaluate` CLI with `--samples 64 --seed 2026 --splits test_temporal test_individual`. No subsequent tuning used these test scores."]
(out / "RESULTS.md").write_text("\n".join(lines) + "\n")
print((out / "RESULTS.md").read_text())
