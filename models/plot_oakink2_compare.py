"""Plot the real / real+synth / synth comparison of DiffAct, FACT and MS-TCN on OakInk2.

Reads the JSON written by `eval_oakink2_subsets.py --compare` (default) or the printed
tables pasted into a text file with "Left Hand" / "Right Hand" headers (--table).

    conda activate diffact
    python plot_oakink2_compare.py                          # runs/compare_*_prim5_{left,right}.json
    python plot_oakink2_compare.py --fact_select last       # the *_factlast.json variant
    python plot_oakink2_compare.py --table results.txt

Figures (runs/figures/):
    1_overall.png      all metrics on the full real test set, per hand
    2_vs_real.png      change of each metric relative to the real-only model
    3_views.png        F1@50 per camera view (heatmap)
    4_seen_unseen.png  F1@50 on test videos whose task was / was not seen in training
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oakink2_actseg.common import HANDS, RUNS_DIR, dataset_name, prim5_dataset_name, resolve_cameras

METRICS = ("Acc", "Edit", "F1@10", "F1@25", "F1@50")
MODELS = ("DiffAct", "FACT", "MS-TCN")
SETTINGS = ("real", "real+synth", "synth")
SETTING_COLORS = {"real": "#1f77b4", "real+synth": "#ff7f0e", "synth": "#7f7f7f"}
VIEWS = ("egocentric", "allocentric_top", "allocentric_left", "allocentric_right")


def load_json(fact_select):
    base = dataset_name(resolve_cameras("all"), 15, True, "center")
    suffix = "_factlast" if fact_select == "last" else ""
    data = {}
    for hand in HANDS:
        path = RUNS_DIR / f"compare_{prim5_dataset_name(base, hand)}{suffix}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found; run eval_oakink2_subsets.py --hand {hand} --compare first")
        for label, subsets in json.loads(path.read_text()).items():
            for subset, m in subsets.items():
                if isinstance(m, dict):
                    data[(hand, label, subset)] = {k: m[k] for k in METRICS}
    return data


def load_table(path):
    data, hand = {}, None
    row = re.compile(r"^(\S+/\S+)\s+(\S+)\s+(\d+)\s+" + r"\s+".join([r"(-?[\d.]+)"] * len(METRICS)) + r"\s*$")
    for line in Path(path).read_text().splitlines():
        if line.strip().lower().startswith(("left", "right")) and "hand" in line.lower():
            hand = line.strip().split()[0].lower()
            continue
        m = row.match(line.strip())
        if m and hand:
            data[(hand, m.group(1), m.group(2))] = dict(zip(METRICS, map(float, m.groups()[3:])))
    if not data:
        raise SystemExit(f"no table rows found in {path}")
    return data


def value(data, hand, model, setting, subset, metric):
    m = data.get((hand, f"{model}/{setting}", subset))
    return np.nan if m is None else m[metric]


def bars(ax, groups, series, values, colors, ylabel, ylim=None):
    width = 0.8 / len(series)
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        v = values[i]
        rects = ax.bar(x + (i - (len(series) - 1) / 2) * width, v, width, label=s, color=colors[i])
        ax.bar_label(rects, labels=["" if np.isnan(t) else f"{t:.1f}" for t in v], fontsize=6, padding=1)
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def plot_overall(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
    for r, hand in enumerate(HANDS):
        for c, model in enumerate(MODELS):
            vals = [[value(data, hand, model, s, "test", m) for m in METRICS] for s in SETTINGS]
            bars(axes[r, c], METRICS, SETTINGS, vals, [SETTING_COLORS[s] for s in SETTINGS],
                 f"{hand} hand" if c == 0 else "", (0, 85))
            axes[r, c].set_title(f"{model} - {hand} hand")
    axes[0, 0].legend(title="training data", loc="upper right", fontsize=8)
    fig.suptitle("Real OakInk2 test set: all metrics by training data")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_vs_real(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharey=True)
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for r, hand in enumerate(HANDS):
        for c, setting in enumerate(SETTINGS[1:]):
            vals = [[value(data, hand, model, setting, "test", m) - value(data, hand, model, "real", "test", m)
                     for m in METRICS] for model in MODELS]
            bars(axes[r, c], METRICS, MODELS, vals, colors, f"{hand} hand: change (points)" if c == 0 else "")
            axes[r, c].axhline(0, color="black", lw=0.8)
            axes[r, c].set_title(f"{setting} minus real - {hand} hand")
    axes[0, 0].legend(title="model", loc="lower left", fontsize=8)
    fig.suptitle("Effect of synthetic training data relative to the real-only model")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_views(data, out, metric="F1@50"):
    rows = [(model, setting) for setting in SETTINGS for model in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True, gridspec_kw={"wspace": 0.05})
    for ax, hand in zip(axes, HANDS):
        grid = np.array([[value(data, hand, m, s, f"test_{v}", metric) for v in VIEWS] for m, s in rows])
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=60, aspect="auto")
        for i in range(grid.shape[0]):
            best = np.nanargmax(grid[i]) if not np.all(np.isnan(grid[i])) else -1
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.1f}" + (" *" if j == best else ""), ha="center", va="center",
                            fontsize=8, color="white" if grid[i, j] < 35 else "black")
        ax.set_xticks(range(len(VIEWS)), [v.replace("allocentric_", "allo_") for v in VIEWS])
        ax.set_yticks(range(len(rows)), [f"{m} / {s}" for m, s in rows])
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.set_title(f"{hand} hand ({metric}, * = best view)")
        for k in (3, 6):
            ax.axhline(k - 0.5, color="white", lw=2)
    fig.colorbar(im, ax=axes, shrink=0.8, label=metric)
    fig.suptitle(f"{metric} per camera view")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seen_unseen(data, out, metric="F1@50"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, hand in zip(axes, HANDS):
        labels, seen, unseen = [], [], []
        for setting in SETTINGS:
            for model in MODELS:
                labels.append(f"{model}\n{setting}")
                seen.append(value(data, hand, model, setting, "test_seen_task", metric))
                unseen.append(value(data, hand, model, setting, "test_unseen_task", metric))
        bars(ax, labels, ["task seen in training", "task unseen in training"], [seen, unseen],
             ["#2ca02c", "#d62728"], metric if hand == HANDS[0] else "", (0, 65))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"{hand} hand")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{metric} on test videos whose complex task was seen / unseen during training")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default=None, help="text file with the pasted comparison tables")
    parser.add_argument("--fact_select", default="best", choices=("best", "last"),
                        help="which compare JSON to read (ignored with --table)")
    parser.add_argument("--out", default=str(RUNS_DIR / "figures"))
    args = parser.parse_args()

    data = load_table(args.table) if args.table else load_json(args.fact_select)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_overall(data, out / "1_overall.png")
    plot_vs_real(data, out / "2_vs_real.png")
    plot_views(data, out / "3_views.png")
    plot_seen_unseen(data, out / "4_seen_unseen.png")
    print(f"saved 4 figures to {out}")


if __name__ == "__main__":
    main()
