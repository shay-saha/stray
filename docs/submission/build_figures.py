"""Build Stray's architecture and validation figures from saved local evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAPER = "#f4efe9"
INK, MUTED = "#5b4b41", "#8b8076"
TEAL, GREEN, OCHRE, RUST = "#00897d", "#7fa844", "#dab134", "#c5753e"
GRID, RULE = "#e8e0d6", "#cfc4b7"
SERIF, SANS = "Noto Serif", "Noto Sans"
plt.rcParams.update({"font.family": SANS, "font.size": 16, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "svg.fonttype": "none", "axes.unicode_minus": False})

def evidence(metrics: Path):
    if not metrics.exists():
        return json.loads((OUT / "evidence.json").read_text())
    raw = metrics.read_bytes()
    data = json.loads(raw)
    rows = [r for r in data["scores"] if r["model"] in ("current-ensemble", "MoveFormer-short")]
    means = {m: float(np.mean([r["point_error_km"] for r in rows if r["model"] == m]))
             for m in ("current-ensemble", "MoveFormer-short")}
    result = {"source_report": str(metrics.relative_to(ROOT)),
              "source_sha256": hashlib.sha256(raw).hexdigest(),
              "scope": data["scope"], "horizons_days": data["horizons_days"],
              "forecast_origins": data["cases"], "observed_targets": data["observed_targets"],
              "samples_per_origin": data["samples"], "seed": data["seed"],
              "aggregation": "Equal physical/exclusion identity weight within species, then species and horizon; equal split weight for the combined chart.",
              "rows": rows, "mean_point_error_km": means,
              "relative_reduction_percent": 100 * (1 - means["current-ensemble"] / means["MoveFormer-short"]),
              "checkpoints": [r for r in data["checkpoints"] if any(
                  s in r["path"] for s in ("/current-ensemble/", "/MoveFormer-short/"))],
              "notes": data["notes"],
              "architecture_source": ["htb/wildcast/migration_model.py",
                                      "htb/wildcast/migration_ensemble.py",
                                      "htb/presentations/wildcast-brief/evidence.json"]}
    (OUT / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def paper(kicker, title, paragraphs, legend=True):
    from matplotlib.collections import LineCollection
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=PAPER)
    background = fig.add_axes([0, 0, 1, 1], zorder=-5)
    background.set_xlim(0, 1920); background.set_ylim(0, 1080)
    background.axis("off")
    # Fine, low-contrast drafting squares; every fifth rule is a little stronger.
    for step, color, width, alpha in [(16, GRID, .32, .65), (80, RULE, .38, .25)]:
        lines = [((x, 0), (x, 1080)) for x in range(0, 1921, step)]
        lines += [((0, y), (1920, y)) for y in range(0, 1081, step)]
        background.add_collection(LineCollection(lines, colors=color, linewidths=width, alpha=alpha))
    fig.text(.052, .94, "Stray", family=SERIF, weight="bold", fontsize=22)
    fig.text(.946, .944, kicker, fontsize=13, color=MUTED, ha="right")
    fig.text(.052, .825, title + r" $\rightarrow$", family=SERIF, weight="bold", fontsize=27)
    y = .765
    for paragraph in paragraphs:
        fig.text(.052, y, paragraph, fontsize=14, color=MUTED,
                 linespacing=1.65, va="top")
        y -= .037 * (paragraph.count("\n") + 1) + .047
    if legend:
        for i, (label, color) in enumerate([("Stray ensemble", TEAL), ("MoveFormer · local retraining", OCHRE)]):
            from matplotlib.lines import Line2D
            fig.add_artist(Line2D([.057], [.305 - i * .036], marker="o", markersize=5,
                                 color=color, linestyle="none", transform=fig.transFigure))
            fig.text(.068, .298 - i * .036, label, fontsize=12, color=MUTED)
    return fig


def footer(fig, main, detail):
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([.052, .946], [.132, .132], color=RULE, linewidth=.7,
                         transform=fig.transFigure))
    fig.text(.052, .086, main, fontsize=14, color=INK)
    fig.text(.052, .047, detail, fontsize=12, color=MUTED)


def finish(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=120, facecolor=PAPER)
    sources = OUT / "sources"
    sources.mkdir(exist_ok=True)
    svg = sources / (Path(name).stem + ".svg")
    fig.savefig(svg, facecolor=PAPER)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    assert path.stat().st_size < 5_000_000, path
    print(name, path.stat().st_size, "bytes")


def comparison(ev):
    stray = ev["mean_point_error_km"]["current-ensemble"]
    baseline = ev["mean_point_error_km"]["MoveFormer-short"]
    pct = ev["relative_reduction_percent"]
    fig = paper("Model evaluation / 01", "Prediction", [
        "How far is a predicted location\nfrom the animal’s actual position?",
        "Stray reduced the average error\non a matched short-term\nvalidation panel."])
    fig.text(.052, .425, f"{pct:.0f}%", family=SERIF, fontsize=52, color=TEAL)
    fig.text(.055, .373, "lower average location error", fontsize=14, color=MUTED)
    fig.text(.365, .797, "Average location error", family=SERIF, fontsize=27, weight="bold")
    fig.text(.365, .749, "Great-circle distance · kilometres · lower is better", fontsize=14, color=MUTED)
    ax = fig.add_axes([.49, .335, .435, .33], facecolor="none")
    for y, value, color in [(1, baseline, OCHRE), (0, stray, TEAL)]:
        ax.plot([0, value], [y, y], color=color, linewidth=2.4, solid_capstyle="round")
        ax.scatter([value], [y], s=110, color=color, zorder=4)
        ax.text(value + .7, y, f"{value:.2f}", family=SERIF, fontsize=23, va="center", color=INK)
    ax.set_yticks([1, 0], ["MoveFormer", "Stray"], family=SERIF, fontsize=20, color=INK)
    ax.set_xlim(0, 25); ax.set_ylim(-.65, 1.65)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.tick_params(axis="both", length=0, pad=14)
    ax.grid(axis="x", color=RULE, linewidth=.65, alpha=.7)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.365, .225, f"{baseline:.2f} km " + r"$\rightarrow$" + f" {stray:.2f} km", family=SERIF, fontsize=22)
    fig.text(.365, .182, "Equal average of the two split-level macro errors.", fontsize=13, color=MUTED)
    footer(fig, "1h / 12h / 24h forecasts · 4,096 origins · 8,831 observed targets",
           "Matched local development validation. Relative error reduction is 30.6%, rounded to 31%.")
    finish(fig, "07-stray-location-error.png")


def splits(ev):
    fig = paper("Model evaluation / 02", "Validation", [
        "Two views of generalisation.\nCan the model predict an animal\nit has not trained on, and a year\nit has not trained on?",
        "The ensemble has lower mean\nlocation error on both of these\nmatched validation splits."])
    names = ["val_individual", "val_year"]
    for i, (split, title, caption) in enumerate([
        (names[0], "Unseen animals", "Held-out identity groups"),
        (names[1], "Held-out year", "Later observations")]):
        b = next(r["point_error_km"] for r in ev["rows"] if r["model"] == "MoveFormer-short" and r["split"] == split)
        a = next(r["point_error_km"] for r in ev["rows"] if r["model"] == "current-ensemble" and r["split"] == split)
        left = .355 + i * .315
        ax = fig.add_axes([left, .345, .245, .36], facecolor="none")
        ax.plot([0, 1], [b, a], color=RULE, linewidth=1.5, zorder=2)
        for x, value, color in [(0, b, OCHRE), (1, a, TEAL)]:
            ax.vlines(x, 0, value, color=color, linewidth=1, alpha=.4)
            ax.scatter(x, value, s=150, color=color, zorder=3)
            ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 13), textcoords="offset points",
                        ha="center", family=SERIF, fontsize=22, color=INK)
        ax.set_ylim(0, 29); ax.set_xlim(-.22, 1.22)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_xticks([0, 1], ["MoveFormer", "Stray"], fontsize=14)
        ax.tick_params(axis="both", length=0, pad=10)
        ax.grid(axis="y", color=RULE, linewidth=.5, alpha=.65)
        for spine in ax.spines.values(): spine.set_visible(False)
        if i == 0: ax.set_ylabel("Mean location error / km", fontsize=13, labelpad=12)
        else: ax.set_yticklabels([])
        center = left + .1225
        fig.text(center, .262, title, family=SERIF, weight="bold", fontsize=21, ha="center")
        fig.text(center, .222, caption, fontsize=12, color=MUTED, ha="center")
        fig.text(center, .174, f"{100 * (1 - a / b):.1f}% lower error", family=SERIF,
                 fontsize=19, color=TEAL, ha="center")
    footer(fig, "Same forecast origins, actual target masks and horizons for both models.",
           "Local development validation · 1h / 12h / 24h · equal identity, species and horizon weighting within each split.")
    finish(fig, "08-stray-validation-splits.png")


def architecture():
    fig = paper("Model notes / 01", "Architecture", [
        "Observed movement is the input.\nPossible future routes are\nthe output.",
        "Four predictors learn different\nviews of the same history. Each\nuses a Transformer, species\nand derived time features.",
        "A global mixture selects one\npredictor for each complete\nsampled route."], legend=False)
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    def arrow(a, b, color=RULE):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                    linewidth=.9, color=color))
    def line(points, color=RULE, width=.8):
        x, y = zip(*points); ax.plot(x, y, color=color, linewidth=width)
    def label(x, y, title, body, color=INK):
        ax.text(x, y, title, family=SERIF, weight="bold", fontsize=18, color=color, va="top")
        ax.text(x, y - .052, body, fontsize=12, color=MUTED, va="top", linespacing=1.65)
    label(.32, .75, "GPS histories", "29 lag slots × 10 features\nPast fixes + missingness masks")
    label(.32, .49, "Species & time", "22 derived context features\nSpecies embedding")
    # The 29 observed lag slots are a representation of input dimensions,
    # rather than invented animal observations.
    dots = np.linspace(.322, .48, 29)
    ax.scatter(dots, np.full(29, .584), s=6, color=TEAL, linewidths=0)
    line([(.515, .727), (.515, .322)])
    line([(.49, .70), (.515, .70)])
    line([(.49, .43), (.515, .43)])
    model_y = [.755, .62, .485, .35]
    for y, title, body, color in zip(model_y,
            ["Direct · compact", "Direct · scaled", "Goal route · compact", "Goal route · scaled"],
            ["128 wide / 2 layers", "512 wide / 6 layers", "128 wide / 2 layers", "512 wide / 6 layers"],
            [TEAL, TEAL, GREEN, GREEN]):
        arrow((.515, y - .028), (.555, y - .028))
        ax.scatter([.566], [y - .027], s=33, facecolors=PAPER, edgecolors=color, linewidths=1.3)
        label(.581, y, title, body, color)
        line([(.79, y - .028), (.823, y - .028)])
    # Each predictor owns its encoders. Global weights combine distributions,
    # choosing a predictor once for a full route, not separately at each horizon.
    line([(.823, model_y[-1] - .028), (.823, model_y[0] - .028)], color=OCHRE)
    arrow((.823, .54), (.857, .54), OCHRE)
    label(.865, .6, "Joint routes", "Weighted mixture\n8 components\nper predictor", OCHRE)
    ax.text(.32, .225, "Nine forecast horizons", family=SERIF, fontsize=18, weight="bold")
    horizon_x = np.linspace(.332, .925, 9)
    line([(horizon_x[0], .18), (horizon_x[-1], .18)])
    ax.scatter(horizon_x, np.full(9, .18), s=30, color=TEAL, zorder=3)
    for x, t in zip(horizon_x, ["1h", "12h", "1d", "3d", "7d", "14d", "30d", "60d", "90d"]):
        ax.text(x, .151, t, fontsize=12, color=MUTED, ha="center")
    footer(fig, "Joint forecasts preserve dependence between future locations.",
           "Local GPU inference uses one predictor. The precomputed demo uses the four-predictor ensemble. The atlas retains real animal identities.")
    finish(fig, "06-stray-model-architecture.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "htb/reports/migration-delivery-v1-public/metrics.json")
    args = parser.parse_args()
    ev = evidence(args.metrics)
    assert ev["horizons_days"] == [1 / 24, .5, 1.]
    assert 30.5 <= ev["relative_reduction_percent"] < 31.5
    architecture(); comparison(ev); splits(ev)


if __name__ == "__main__":
    main()
