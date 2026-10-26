"""Per-video / per-frame analysis of DiffAct, FACT and MS-TCN predictions on the real OakInk2 test set.

Uses the same model outputs as `eval_oakink2_subsets.py --compare` (real snapshot, real+synth, synth).

    conda activate diffact
    python analyze_oakink2_predictions.py --hand right --fact_select last
    python analyze_oakink2_predictions.py --hand left --videos "scene_01__A001++seq__...@egocentric"

Writes runs/analysis/<hand>/:
    timelines/*.png        gold vs every model for selected videos, with per-video Edit / AER / count error
                           and per-class segment counts (videos picked from worst to best Edit of --reference)
    1_confusion.png        frame-level confusion matrices (rows = gold class, row-normalised)
    2_per_class_f1.png     frame F1 and segment F1@50 per class
    3_segment_lengths.png  segment duration per class, gold vs predicted (over-segmentation shows as short tails)
    4_segment_counts.png   predicted vs gold number of segments per video
    5_boundaries.png       distance from each gold boundary to the nearest predicted boundary (CDF)
    6_per_video.png        distribution of per-video Edit and action error rate
    per_video_metrics.csv  every video x model: Acc, Edit, AER, count error, #segments
    viewer.json            gold and predicted segments plus metrics of every test video (for view_oakink2_analysis.py)
    clips/*.mp4            the test frames of the timeline videos, frame k = label k (needs imageio-ffmpeg)

Per-video metrics: Edit = MS-TCN segmental edit score; AER = Levenshtein(gold, pred segment
sequence) / #gold segments (StrokeRehab); count error = sum_c |#pred_c - #gold_c| / #gold segments.
"""
import argparse
import csv
import functools
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import eval_oakink2_subsets as E
from oakink2_actseg.common import (
    DEFAULT_DATA_ROOT,
    RUNS_DIR,
    frames_ids_path,
    frames_pack_path,
    load_json,
    load_pickle,
    select_frame_ids,
)
from oakink2_actseg.prepare import add_prepare_args, dataset_dir_from_args

CLASS_COLORS = {"stabilize": "#8b5cc4", "transport": "#d9822b", "reach": "#2f6f8f",
                "reposition": "#3f9152", "idle": "#cfd3d6"}
CLASS_ORDER = ("reach", "transport", "reposition", "stabilize", "idle")
SETTING_STYLE = {"real": "-", "real+synth": "--", "synth": ":"}
MODEL_COLORS = {"DiffAct": "#4c72b0", "FACT": "#dd8452", "MS-TCN": "#55a868"}


# ---------------------------------------------------------------- per-video metrics

def segments(labels):
    """[(label, start, end_exclusive)]"""
    out, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append((labels[s], s, i))
            s = i
    return out


def levenshtein(a, b):
    d = np.arange(len(b) + 1, dtype=float)
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[-1]


def video_metrics(gt, pred):
    g, p = [s[0] for s in segments(gt)], [s[0] for s in segments(pred)]
    lev = levenshtein(p, g)
    cg, cp = Counter(g), Counter(p)
    return {
        "Acc": 100.0 * np.mean([a == b for a, b in zip(gt, pred)]),
        "Edit": (1 - lev / max(len(p), len(g), 1)) * 100,
        "AER": lev / max(len(g), 1),
        "count_err": sum(abs(cp[c] - cg[c]) for c in set(cg) | set(cp)) / max(len(g), 1),
        "n_gold": len(g), "n_pred": len(p),
    }


def class_segment_f1(gt_all, pred_all, overlap=0.5):
    """Segment F1@overlap per class, MS-TCN style matching restricted to one class."""
    res = {}
    for c in CLASS_ORDER:
        tp = fp = fn = 0
        for gt, pred in zip(gt_all, pred_all):
            gs = [s for s in segments(gt) if s[0] == c]
            hit = [False] * len(gs)
            for _, ps, pe in (s for s in segments(pred) if s[0] == c):
                ious = [(min(pe, ge) - max(ps, gs_)) / (max(pe, ge) - min(ps, gs_)) for _, gs_, ge in gs]
                k = int(np.argmax(ious)) if ious else -1
                if k >= 0 and ious[k] >= overlap and not hit[k]:
                    tp += 1
                    hit[k] = True
                else:
                    fp += 1
            fn += hit.count(False)
        res[c] = 100.0 * 2 * tp / max(2 * tp + fp + fn, 1)
    return res


def boundary_offsets(gt, pred, fps):
    gb = [s[1] for s in segments(gt)[1:]]
    pb = np.array([s[1] for s in segments(pred)[1:]])
    if not gb:
        return []
    if len(pb) == 0:
        return [np.inf] * len(gb)
    return [np.abs(pb - b).min() / fps for b in gb]


# ---------------------------------------------------------------- loading

def load_models(args, data_dir, test_videos):
    models = {}
    for label, kind, path in E.comparison_sources(args.hand, args.real_tag):
        pred = None
        if kind == "diffact" and path is not None and (path / "prediction").is_dir():
            vids = [v for v in test_videos if (path / "prediction" / f"{v}.txt").exists()]
            pred = E.diffact_predictions(path, vids)
        elif kind == "fact" and path is not None and path.exists():
            if args.fact_select == "last":
                saves = list((path.parent / "saves").glob("*.gz"))
                path = max(saves, key=lambda p: int(p.stem)) if saves else path
            pred = E.fact_predictions(path, data_dir)
        elif kind == "mstcn" and path is not None and (path / "results").is_dir():
            pred = E.mstcn_predictions(path, test_videos)
        if pred is not None and all(v in pred for v in test_videos):
            models[label] = pred
        else:
            print(f"(skipping {label}: no complete predictions at {path})")
    if not models:
        raise SystemExit("no model outputs found")
    return models


# ---------------------------------------------------------------- figures

@functools.lru_cache(maxsize=None)
def task_targets(data_root):
    path = Path(data_root) / "program" / "task_target.json"
    return load_json(path) if path.exists() else {}


def video_title(video, data_root, fps, n_frames, hand):
    seq, _, cam = video.partition("@")
    task = task_targets(str(data_root)).get(seq.replace("++", "/"), "")
    return f"{task or seq}   {seq.split('++')[-1][:18]} | {cam or 'single view'} | {hand} hand | {n_frames / fps:.0f} s"


def plot_timeline(video, gt, preds, metrics, fps, title, out):
    rows = [("GOLD", gt)] + [(m, p[video]) for m, p in preds.items()]
    T = len(gt)
    fig = plt.figure(figsize=(14, 1.1 + 0.42 * len(rows) + 1.6))
    ax = fig.add_axes([0.13, 0.42, 0.62, 0.48])
    for r, (name, labels) in enumerate(rows):
        y = len(rows) - 1 - r
        for c, s, e in segments(labels):
            ax.barh(y, (e - s) / fps, left=s / fps, height=0.72, color=CLASS_COLORS[c], linewidth=0)
        ax.text(-0.01 * T / fps, y, name, ha="right", va="center", fontsize=8, weight="bold" if r == 0 else None)
        if r:
            m = metrics[name][video]
            ax.text(T / fps * 1.01, y, f"Edit {m['Edit']:5.1f}   AER {m['AER']:.2f}   count err {m['count_err']:.2f}",
                    ha="left", va="center", fontsize=7.5, family="monospace")
    ax.set_xlim(0, T / fps)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("seconds")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, weight="bold")
    ax.legend(handles=[Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_ORDER], ncol=5, fontsize=8,
              loc="upper left", bbox_to_anchor=(0, -0.28), frameon=False)
    # per-class segment counts: gold vs each model
    tab = fig.add_axes([0.13, 0.02, 0.84, 0.2])
    tab.axis("off")
    counts = {name: Counter(s[0] for s in segments(lab)) for name, lab in rows}
    cell = [[str(counts[name][c]) for c in CLASS_ORDER] for name, _ in rows]
    t = tab.table(cellText=cell, rowLabels=[n for n, _ in rows], colLabels=[f"#{c}" for c in CLASS_ORDER],
                  loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    t.scale(1, 0.9)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def grid_axes(models, ncols=3, size=(4.2, 3.6)):
    labels = list(models)
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    return fig, labels, axes.flat


def plot_confusion(gt_all, preds, test_videos, out):
    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    fig, labels, axes = grid_axes(preds)
    for ax, name in zip(axes, labels):
        cm = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)))
        for v in test_videos:
            for g, p in zip(gt_all[v], preds[name][v]):
                cm[idx[g], idx[p]] += 1
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(CLASS_ORDER)):
            for j in range(len(CLASS_ORDER)):
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=40, fontsize=7)
        ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER, fontsize=7)
        ax.set_xlabel("predicted", fontsize=7)
        ax.set_ylabel("gold", fontsize=7)
        ax.set_title(name, fontsize=9)
    fig.suptitle("Frame-level confusion (% of each gold class)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_per_class_f1(gt_all, preds, test_videos, out):
    gts = [gt_all[v] for v in test_videos]
    frame_f1, seg_f1 = {}, {}
    for name, p in preds.items():
        ps = [p[v] for v in test_videos]
        g, q = np.concatenate([np.array(x) for x in gts]), np.concatenate([np.array(x) for x in ps])
        frame_f1[name] = {c: 100 * 2 * ((g == c) & (q == c)).sum() / max((g == c).sum() + (q == c).sum(), 1)
                          for c in CLASS_ORDER}
        seg_f1[name] = class_segment_f1(gts, ps)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    names = list(preds)
    width = 0.8 / len(names)
    x = np.arange(len(CLASS_ORDER))
    for ax, (title, data) in zip(axes, (("frame F1", frame_f1), ("segment F1@50", seg_f1))):
        for i, n in enumerate(names):
            model, setting = n.split("/")
            ax.bar(x + (i - (len(names) - 1) / 2) * width, [data[n][c] for c in CLASS_ORDER], width,
                   color=MODEL_COLORS.get(model, "gray"), alpha={"real": 1, "real+synth": 0.65, "synth": 0.3}.get(setting, 1),
                   edgecolor="black", linewidth=0.3, label=n)
        ax.set_xticks(x, CLASS_ORDER)
        ax.set_title(f"{title} per class")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("F1")
    axes[1].legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_segment_lengths(gt_all, preds, test_videos, fps, out):
    series = {"GOLD": gt_all, **preds}
    fig, axes = plt.subplots(1, len(CLASS_ORDER), figsize=(19, 4.8), sharey=True, gridspec_kw={"wspace": 0.25})
    for ax, c in zip(axes, CLASS_ORDER):
        data = [[(e - s) / fps for v in test_videos for lab, s, e in segments(src[v]) if lab == c]
                for src in series.values()]
        bp = ax.boxplot([d if d else [np.nan] for d in data], vert=False, whis=(5, 95), showfliers=False,
                        patch_artist=True, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor(CLASS_COLORS[c])
        for k, d in enumerate(data, start=1):
            ax.text(1.0, k, f" n={len(d)}", transform=ax.get_yaxis_transform(), va="center", fontsize=6)
        ax.set_xscale("log")
        ax.set_title(c)
        ax.set_xlabel("segment duration (s, log)")
        ax.axvline(np.median(data[0]) if data[0] else 1, color="black", lw=0.8, ls="--")
    axes[0].set_yticks(range(1, len(series) + 1), list(series), fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("Segment durations per class (box 25-75 %, whiskers 5-95 %, dashed = gold median)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_segment_counts(metrics, out):
    fig, labels, axes = grid_axes(metrics, size=(4.2, 3.8))
    for ax, name in zip(axes, labels):
        g = np.array([m["n_gold"] for m in metrics[name].values()])
        p = np.array([m["n_pred"] for m in metrics[name].values()])
        hi = max(g.max(), p.max()) + 2
        ax.scatter(g, p, s=6, alpha=0.4, color=MODEL_COLORS.get(name.split("/")[0], "gray"))
        ax.plot([0, hi], [0, hi], color="black", lw=0.8)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_title(f"{name}\nmedian pred/gold = {np.median(p / np.maximum(g, 1)):.2f}", fontsize=8)
        ax.set_xlabel("# gold segments", fontsize=7)
        ax.set_ylabel("# predicted segments", fontsize=7)
    fig.suptitle("Segments per video (above the line = over-segmentation)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_boundaries(gt_all, preds, test_videos, fps, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.linspace(0, 3, 301)
    for name, p in preds.items():
        d = np.array([x for v in test_videos for x in boundary_offsets(gt_all[v], p[v], fps)])
        model, setting = name.split("/")
        ax.plot(xs, [(d <= x).mean() for x in xs], color=MODEL_COLORS.get(model, "gray"),
                ls=SETTING_STYLE.get(setting, "-"), label=f"{name} (within 0.5 s: {100 * (d <= 0.5).mean():.0f}%)")
    ax.set_xlabel("distance from a gold boundary to the nearest predicted boundary (s)")
    ax.set_ylabel("fraction of gold boundaries")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("Boundary localisation")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_per_video(metrics, out):
    names = list(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, key, title in ((axes[0], "Edit", "per-video Edit (higher is better)"),
                           (axes[1], "AER", "per-video action error rate (lower is better)")):
        data = [[m[key] for m in metrics[n].values()] for n in names]
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True)
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(MODEL_COLORS.get(n.split("/")[0], "gray"))
            patch.set_alpha({"real": 1, "real+synth": 0.65, "synth": 0.3}.get(n.split("/")[1], 1))
        ax.set_xticks(range(1, len(names) + 1), names, rotation=35, ha="right", fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- viewer export

def safe_name(video):
    return video.replace("++", "_").replace("@", "__")


def export_clip(video, data_root, fps, n_frames, out_path):
    """Test frames of one video as H.264 mp4, so that frame k shows label k."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return "imageio-ffmpeg is not installed (pip install imageio-ffmpeg)"
    import cv2

    seq, _, cam = video.partition("@")
    pack_path = frames_pack_path(data_root, cam, seq)
    if not pack_path.exists():
        return f"no frames at {pack_path}"
    ids = select_frame_ids(data_root, seq, load_json(frames_ids_path(data_root, cam, seq)), int(fps), True)
    if len(ids) != n_frames:
        return f"{len(ids)} frames vs {n_frames} labels"
    try:
        pack = load_pickle(pack_path)
    except Exception as e:
        return f"cannot read {pack_path}: {e!r}"
    jpg_of = dict(zip(pack["frame_ids"], pack["jpgs"]))
    first = cv2.imdecode(np.frombuffer(jpg_of[ids[0]], np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    writer = imageio_ffmpeg.write_frames(str(out_path), (w - w % 2, h - h % 2), fps=fps, codec="libx264",
                                         pix_fmt_in="rgb24", macro_block_size=2, quality=7,
                                         output_params=["-movflags", "+faststart"])
    writer.send(None)
    for fid in ids:
        img = cv2.imdecode(np.frombuffer(jpg_of[fid], np.uint8), cv2.IMREAD_COLOR)[: h - h % 2, : w - w % 2, ::-1]
        writer.send(np.ascontiguousarray(img).tobytes())
    writer.close()
    return None


def export_viewer(out, test_videos, gt_all, preds, metrics, data_root, fps, hand, clips):
    classes = list(CLASS_ORDER)
    idx = {c: i for i, c in enumerate(classes)}
    encode = lambda labels: [[idx[c], s, e] for c, s, e in segments(labels)]
    videos = {}
    for v in test_videos:
        videos[v] = {
            "title": video_title(v, data_root, fps, len(gt_all[v]), hand),
            "n_frames": len(gt_all[v]),
            "tracks": {"GOLD": encode(gt_all[v]), **{n: encode(p[v]) for n, p in preds.items()}},
            "metrics": {n: {k: round(float(x), 3) for k, x in metrics[n][v].items()} for n in preds},
            "clip": clips.get(v),
        }
    viewer = {"hand": hand, "fps": fps, "classes": classes, "colors": [CLASS_COLORS[c] for c in classes],
              "models": list(preds), "videos": videos}
    (out / "viewer.json").write_text(json.dumps(viewer, separators=(",", ":")))


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_prepare_args(parser)
    parser.add_argument("--data_dir", default=None, help="real prim5 dataset providing the test split and gold labels")
    parser.add_argument("--real_tag", default="real")
    parser.add_argument("--fact_select", default="best", choices=("best", "last"))
    parser.add_argument("--videos", nargs="*", default=None, help="test videos to draw timelines for")
    parser.add_argument("--n_examples", type=int, default=8, help="timelines picked from worst to best Edit")
    parser.add_argument("--reference", default="DiffAct/real", help="model whose per-video Edit picks the examples")
    parser.add_argument("--video_fps", type=float, default=15, help="frame rate of the test labels")
    parser.add_argument("--out", default=None, help="default: runs/analysis/<hand>")
    parser.add_argument("--no_clips", action="store_true", help="do not export mp4 clips of the timeline videos")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else dataset_dir_from_args(args)
    test_videos = E.read_lines(data_dir / "splits" / "test.split1.bundle")
    gt_all = {v: E.read_lines(data_dir / "groundTruth" / f"{v}.txt") for v in test_videos}
    preds = load_models(args, data_dir, test_videos)
    out = Path(args.out) if args.out else RUNS_DIR / "analysis" / (args.hand or data_dir.name)
    (out / "timelines").mkdir(parents=True, exist_ok=True)

    metrics = {n: {v: video_metrics(gt_all[v], p[v]) for v in test_videos} for n, p in preds.items()}
    with open(out / "per_video_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "model", "Acc", "Edit", "AER", "count_err", "n_gold_segments", "n_pred_segments"])
        for n, per in metrics.items():
            for v, m in per.items():
                w.writerow([v, n, f"{m['Acc']:.2f}", f"{m['Edit']:.2f}", f"{m['AER']:.3f}", f"{m['count_err']:.3f}",
                            m["n_gold"], m["n_pred"]])

    plot_confusion(gt_all, preds, test_videos, out / "1_confusion.png")
    plot_per_class_f1(gt_all, preds, test_videos, out / "2_per_class_f1.png")
    plot_segment_lengths(gt_all, preds, test_videos, args.video_fps, out / "3_segment_lengths.png")
    plot_segment_counts(metrics, out / "4_segment_counts.png")
    plot_boundaries(gt_all, preds, test_videos, args.video_fps, out / "5_boundaries.png")
    plot_per_video(metrics, out / "6_per_video.png")

    ref = args.reference if args.reference in metrics else next(iter(metrics))
    if args.videos:
        chosen = args.videos
    else:
        ranked = sorted(test_videos, key=lambda v: metrics[ref][v]["Edit"])
        picks = np.linspace(0, len(ranked) - 1, min(args.n_examples, len(ranked))).round().astype(int)
        chosen = [ranked[i] for i in dict.fromkeys(picks)]
    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    clips = {}
    for rank, v in enumerate(chosen):
        title = video_title(v, data_root, args.video_fps, len(gt_all[v]), args.hand or "")
        name = f"{rank:02d}_{ref.replace('/', '-')}_edit{metrics[ref][v]['Edit']:.0f}_{v.replace('++', '_').replace('@', '_')[:70]}.png"
        plot_timeline(v, gt_all[v], preds, metrics, args.video_fps, title, out / "timelines" / name)
        if not args.no_clips:
            (out / "clips").mkdir(exist_ok=True)
            clip = out / "clips" / f"{safe_name(v)}.mp4"
            problem = None if clip.exists() else export_clip(v, data_root, args.video_fps, len(gt_all[v]), clip)
            if problem:
                print(f"(no clip for {v}: {problem})")
            else:
                clips[v] = f"clips/{clip.name}"
    export_viewer(out, test_videos, gt_all, preds, metrics, data_root, args.video_fps, args.hand or "", clips)
    print(f"saved 6 figures, {len(chosen)} timelines ({len(clips)} with clips), per_video_metrics.csv and "
          f"viewer.json to {out}")


if __name__ == "__main__":
    main()
