"""Plot and print how the training loss (and test metrics) of DiffAct, FACT and MS-TCN evolve per epoch.

Parses the SLURM logs of the training jobs (sbatch `-o %x_%j.log`, i.e. diffact_*.log, fact_*.log,
mstcn_*.log in models/). Each log is assigned to a run from the line the training scripts print first
([diffact] results: ..., [fact] logs: ..., [mstcn] <dataset>: ...); logs of resumed jobs are merged.

    conda activate diffact
    python plot_oakink2_training.py                        # all *.log files in models/
    python plot_oakink2_training.py logs/*.log --metric Edit
    python plot_oakink2_training.py --chain                # draw real+synth after the end of the real run

What is in the logs:
    DiffAct  loss every epoch, test Acc/Edit/F1 every log_freq epochs
    FACT     loss every epoch (printed with one decimal), test metrics every 5 epochs
    MS-TCN   loss and training accuracy every epoch, no test during training

Outputs (runs/figures/training/):
    loss_curves.png    training loss per epoch: rows = models, columns = hands, one line per setting
    metric_curves.png  test metric during training (DiffAct, FACT) and MS-TCN training accuracy
    curves.csv         all parsed points
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oakink2_actseg.common import HANDS, MODELS_DIR, RUNS_DIR

MODELS = ("DiffAct", "FACT", "MS-TCN")
SETTINGS = ("real", "real+synth", "synth")
SETTING_COLORS = {"real": "#1f77b4", "real+synth": "#ff7f0e", "synth": "#7f7f7f"}

HEADERS = (
    ("DiffAct", re.compile(r"^\[diffact\] results: .*?([^/\s]+?)(?:-S\d+)?/?$")),
    ("FACT", re.compile(r"^\[fact\] logs: .*?([^/\s]+)/?$")),
    ("MS-TCN", re.compile(r"^\[mstcn\] ([^\s:]+): ")),
)
NUM = r"(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf)"
DIFFACT_LOSS = re.compile(rf"^Epoch (\d+) - Running Loss {NUM}")
DIFFACT_TEST = re.compile(rf"^Epoch (\d+) - [\w-]+?-Test-(\S+) {NUM}")
FACT_ITER = re.compile(r"^Iter(\d+), (.*)")
MSTCN_EPOCH = re.compile(rf"^\[epoch (\d+)\]: epoch loss = {NUM},\s+acc = {NUM}")
PAIR = re.compile(rf"([A-Za-z][\w@.]*):{NUM}")


def metric_key(key):
    m = re.fullmatch(r"F1@0\.(\d+)", key)
    return f"F1@{int(m.group(1))}" if m else key


def setting_of(run):
    if "realsynth" in run:
        return "real+synth"
    return "synth" if "synth" in run else "real"


def hand_of(run):
    return next((h for h in HANDS if run.endswith("_" + h)), "-")


def new_run():
    return {"loss": {}, "test": {}, "train_acc": {}, "fact_iters": {}}


def parse_log(path, runs):
    text = path.read_text(errors="replace").replace("\r", "\n")
    current, model, fact_iter, fact_testing = None, None, None, False
    for line in text.split("\n"):
        line = line.strip()
        for name, pattern in HEADERS:
            m = pattern.match(line)
            if m:
                model, current = name, runs.setdefault((name, m.group(1)), new_run())
                fact_iter, fact_testing = None, False
        if current is None:
            continue
        if model == "DiffAct":
            if m := DIFFACT_LOSS.match(line):
                current["loss"][int(m.group(1))] = float(m.group(2))
            elif m := DIFFACT_TEST.match(line):
                current["test"].setdefault(m.group(2), {})[int(m.group(1))] = float(m.group(3))
        elif model == "FACT":
            if m := FACT_ITER.match(line):
                fact_iter = int(m.group(1))
                pairs = dict(PAIR.findall(m.group(2)))
                if pairs:
                    current["fact_iters"][fact_iter] = float(pairs.get("loss", next(iter(pairs.values()))))
            elif line.startswith("TESTING"):
                fact_testing = True
            elif fact_testing and fact_iter is not None and (pairs := PAIR.findall(line)):
                for k, v in pairs:
                    current["test"].setdefault(metric_key(k), {})[fact_iter] = float(v)
                fact_testing = False
        elif model == "MS-TCN":
            if m := MSTCN_EPOCH.match(line):
                epoch = int(m.group(1))
                current["loss"][epoch] = float(m.group(2))
                current["train_acc"][epoch] = 100 * float(m.group(3))
    return current is not None


def fact_to_epochs(run):
    """FACT prints once per epoch (print_every = steps per epoch), so the iteration step gives the epoch."""
    iters = sorted(run["fact_iters"])
    if not iters:
        return
    steps = int(np.min(np.diff(iters))) if len(iters) > 1 else iters[0]
    run["loss"] = {round(i / steps): v for i, v in run["fact_iters"].items()}
    run["test"] = {k: {round(i / steps): v for i, v in d.items()} for k, d in run["test"].items()}


def series(d):
    xs = sorted(d)
    return np.array(xs, dtype=float), np.array([d[x] for x in xs], dtype=float)


def trend(values):
    n = max(2, len(values) // 10)
    if len(values) < 2 * n:
        return "too short"
    prev, last = np.mean(values[-2 * n:-n]), np.mean(values[-n:])
    rel = (last - prev) / abs(prev) if prev else 0.0
    if rel < -0.02:
        return f"still falling ({100 * rel:+.0f}%)"
    if rel > 0.02:
        return f"rising ({100 * rel:+.0f}%)"
    return f"flat ({100 * rel:+.1f}%)"


def print_summary(rows, metric):
    for model in MODELS:
        block = [r for r in rows if r["model"] == model]
        if not block:
            continue
        print(f"\n{model}")
        print(f"  {'setting':<11}{'hand':<7}{'epochs':>7}   {'loss first -> min (epoch) -> last':<40}"
              f"{'trend of last 10%':<24}{(metric if model != 'MS-TCN' else 'train Acc') + ' best (epoch) / last'}")
        for r in block:
            x, y = series(r["run"]["loss"])
            if len(x) == 0:
                loss = "no loss lines"
                tr = ""
                span = "0"
            else:
                i = int(np.nanargmin(y))
                loss = f"{y[0]:.4g} -> {y[i]:.4g} ({x[i]:.0f}) -> {y[-1]:.4g}"
                tr = trend(y)
                span = f"{x[0]:.0f}-{x[-1]:.0f}"
            d = r["run"]["train_acc"] if model == "MS-TCN" else r["run"]["test"].get(metric, {})
            mx, my = series(d)
            if len(mx):
                j = int(np.nanargmax(my))
                met = f"{my[j]:.1f} ({mx[j]:.0f}) / {my[-1]:.1f}"
            else:
                met = "-"
            print(f"  {r['setting']:<11}{r['hand']:<7}{span:>7}   {loss:<40}{tr:<24}{met}")


def chained_offset(rows, r):
    if not r.get("chain") or r["setting"] != "real+synth":
        return 0
    base = next((b for b in rows if b["model"] == r["model"] and b["hand"] == r["hand"]
                 and b["setting"] == "real" and b["run"]["loss"]), None)
    return max(base["run"]["loss"]) + 1 if base else 0


def plot_grid(rows, hands, out_path, getter, ylabels, log_y, title, mark_best=False):
    fig, axes = plt.subplots(len(MODELS), len(hands), figsize=(6.5 * len(hands), 3.4 * len(MODELS)),
                             squeeze=False)
    for i, model in enumerate(MODELS):
        for j, hand in enumerate(hands):
            ax = axes[i, j]
            drawn = False
            for r in rows:
                if r["model"] != model or r["hand"] != hand:
                    continue
                x, y = series(getter(r))
                if len(x) == 0:
                    continue
                x = x + chained_offset(rows, r)
                color = SETTING_COLORS.get(r["setting"], None)
                ax.plot(x, y, color=color, lw=1.6, marker="o" if len(x) < 40 else None, ms=3, label=r["setting"])
                if mark_best:
                    k = int(np.nanargmax(y))
                    ax.plot(x[k], y[k], marker="*", ms=13, color=color, mec="black", mew=0.6)
                    ax.annotate(f"{y[k]:.1f}", (x[k], y[k]), textcoords="offset points", xytext=(4, 5), fontsize=8)
                drawn = True
            if drawn:
                if log_y:
                    ax.set_yscale("log")
                ax.legend(fontsize=8, frameon=False)
                ax.grid(alpha=0.3, which="both")
            else:
                ax.text(0.5, 0.5, "no log found", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(f"{model} - {hand} hand", fontsize=11)
            ax.set_xlabel("epoch")
            if j == 0:
                ax.set_ylabel(ylabels[model])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", type=Path, help="default: models/*.log and models/logs/*.log")
    parser.add_argument("--metric", default="F1@50", help="test metric to follow: Acc, Edit, F1@10, F1@25, F1@50")
    parser.add_argument("--all", action="store_true", help="also include runs that are not on the prim5 labels")
    parser.add_argument("--chain", action="store_true",
                        help="shift real+synth curves to start after the real run they continue from")
    parser.add_argument("--out", type=Path, default=RUNS_DIR / "figures" / "training")
    args = parser.parse_args()

    logs = args.logs or sorted({*MODELS_DIR.glob("*.log"), *MODELS_DIR.glob("logs/*.log")})
    logs = sorted(logs, key=lambda p: p.stat().st_mtime)
    runs, skipped = {}, []
    for path in logs:
        if not parse_log(path, runs):
            skipped.append(path.name)
    print(f"read {len(logs) - len(skipped)} training logs ({len(skipped)} other logs ignored)")

    rows = []
    for (model, name), run in runs.items():
        if not args.all and "prim5" not in name:
            continue
        if model == "FACT":
            fact_to_epochs(run)
        if not run["loss"] and not run["test"]:
            continue
        rows.append({"model": model, "run_name": name, "setting": setting_of(name), "hand": hand_of(name),
                     "run": run, "chain": args.chain})
    if not rows:
        raise SystemExit("no training logs with loss lines found; pass the log files explicitly, "
                         "e.g. python plot_oakink2_training.py ~/Action_Seg/models/*.log")
    order = {s: i for i, s in enumerate(SETTINGS)}
    rows.sort(key=lambda r: (MODELS.index(r["model"]), order.get(r["setting"], 9), r["hand"], r["run_name"]))

    print_summary(rows, args.metric)

    args.out.mkdir(parents=True, exist_ok=True)
    hands = [h for h in (*HANDS, "-") if any(r["hand"] == h for r in rows)]
    plot_grid(rows, hands, args.out / "loss_curves.png", lambda r: r["run"]["loss"],
              {"DiffAct": "training loss", "FACT": "training loss", "MS-TCN": "training loss (sum of stages)"},
              True, "Training loss per epoch" + ("  (real+synth drawn after the real run)" if args.chain else ""))
    plot_grid(rows, hands, args.out / "metric_curves.png",
              lambda r: r["run"]["train_acc"] if r["model"] == "MS-TCN" else r["run"]["test"].get(args.metric, {}),
              {"DiffAct": f"test {args.metric}", "FACT": f"test {args.metric}", "MS-TCN": "training Acc (%)"},
              False, f"Test {args.metric} during training (star = best); MS-TCN only logs training accuracy",
              mark_best=True)

    with open(args.out / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "setting", "hand", "run", "series", "epoch", "value"])
        for r in rows:
            run = r["run"]
            parts = [("loss", run["loss"]), ("train_acc", run["train_acc"])]
            parts += [(f"test_{k}", d) for k, d in run["test"].items()]
            for name, d in parts:
                for x in sorted(d):
                    w.writerow([r["model"], r["setting"], r["hand"], r["run_name"], name, x, d[x]])
    print(f"\nsaved {args.out / 'loss_curves.png'}, {args.out / 'metric_curves.png'} and {args.out / 'curves.csv'}")


if __name__ == "__main__":
    main()
