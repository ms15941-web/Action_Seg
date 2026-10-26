"""Score trained DiffAct / FACT / MS-TCN models on the official OakInk2 test split and its subsets:

    test                   all official test videos
    test_seen_task         complex task also performed in the training split
    test_unseen_task       complex task never performed in the training split
    test_unseen_primitive  contains an action class that never occurs in training
    test_<camera>          one camera view (only when training on several views)

Both models are scored with the same MS-TCN metrics (oakink2_actseg/metrics.py) from their
saved predictions, so no GPU is needed:
    DiffAct: runs/diffact/<name>/prediction/*.txt  (written at its most recent evaluation epoch)
    FACT:    log/<dataset>/split1/**/best_ckpt.gz  (FACT keeps the checkpoint with the best test F1@50)
    MS-TCN:  runs/mstcn/<name>/results/*         (last epoch, as in ms-tcn)

    conda activate fact
    python eval_oakink2_subsets.py --hand right                 # five per-hand primitives
    python eval_oakink2_subsets.py --hand right --compare       # real vs real+synth vs synth
    python eval_oakink2_subsets.py --label_source program       # 70 OakInk2 primitives
"""
import argparse
import gzip
import json
import pickle
from pathlib import Path

from oakink2_actseg.common import FACT_DIR, RUNS_DIR
from oakink2_actseg.metrics import evaluate
from oakink2_actseg.prepare import add_prepare_args, dataset_dir_from_args

SUBSETS = ("test", "test_seen_task", "test_unseen_task", "test_unseen_primitive")


def read_lines(path):
    return [l for l in Path(path).read_text().split("\n") if l]


def diffact_predictions(run_dir, videos):
    pred_dir = Path(run_dir) / "prediction"
    return {v: (pred_dir / f"{v}.txt").read_text().split("\n")[1].split() for v in videos}


def mstcn_predictions(run_dir, videos):
    pred_dir = Path(run_dir) / "results"
    return {v: (pred_dir / v).read_text().split("\n")[1].split() for v in videos if (pred_dir / v).exists()}


def fact_predictions(ckpt_path, data_dir):
    from train_fact_oakink2 import import_fact_as_package

    import_fact_as_package()  # the checkpoint pickles classes from the `fact` package
    from fact.utils.utils import expand_frame_label

    index2label = {int(l.split(" ", 1)[0]): l.split(" ", 1)[1] for l in read_lines(data_dir / "mapping.txt")}
    with gzip.open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    print(f"(FACT {ckpt_path}: checkpoint with the best test F1@50 is from iteration {ckpt.iteration})")
    preds = {}
    for name, video in ckpt.videos.items():
        labels = getattr(video, "pred_label", None)
        if labels is None:
            labels = expand_frame_label(video.pred, len(video.gt_label))
        preds[name] = [index2label[int(i)] for i in labels]
    return preds


def latest(paths):
    paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def model_sources(dataset, diffact_run=None, fact_ckpt=None, mstcn_run=None):
    """(label, kind, path) for the DiffAct / FACT / MS-TCN outputs of one training dataset name."""
    diffact_run = Path(diffact_run) if diffact_run else RUNS_DIR / "diffact" / f"{dataset}-S1"
    fact_ckpt = Path(fact_ckpt) if fact_ckpt else latest(
        (FACT_DIR.parent / "log" / dataset / "split1").glob("**/best_ckpt.gz"))
    mstcn_run = Path(mstcn_run) if mstcn_run else RUNS_DIR / "mstcn" / f"{dataset}-S1"
    return [("DiffAct", "diffact", diffact_run), ("FACT", "fact", fact_ckpt), ("MS-TCN", "mstcn", mstcn_run)]


def comparison_sources(hand, real_tag):
    from oakink2_actseg.combine import combined_name

    snap = RUNS_DIR / "snapshots" / real_tag
    out = []
    for setting, kw in (
        ("real", dict(dataset="", diffact_run=snap / "diffact" / hand,
                      fact_ckpt=snap / "fact" / hand / "best_ckpt.gz", mstcn_run=snap / "mstcn" / hand)),
        ("real+synth", dict(dataset=combined_name("real+synth", hand))),
        ("synth", dict(dataset=combined_name("synth", hand))),
    ):
        out += [(f"{label}/{setting}", kind, path) for label, kind, path in model_sources(**kw)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_prepare_args(parser)
    parser.add_argument("--diffact_run", default=None, help="default: runs/diffact/<dataset>-S1")
    parser.add_argument("--fact_ckpt", default=None, help="default: newest log/<dataset>/split1/**/best_ckpt.gz")
    parser.add_argument("--data_dir", default=None,
                        help="dataset folder whose test split, subsets and ground truth to use")
    parser.add_argument("--compare", action="store_true",
                        help="score real (snapshot), real+synth and synth models of --hand on the real test set")
    parser.add_argument("--real_tag", default="real", help="snapshot holding the real-only models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else dataset_dir_from_args(args)
    name = data_dir.name
    split_dir = data_dir / "splits"
    if not all((split_dir / f"{s}.split1.bundle").exists() for s in SUBSETS):
        raise SystemExit(f"{split_dir} has no test subsets; re-run the labels stage with --split_mode official")
    per_view = sorted(p.name[: -len(".split1.bundle")] for p in split_dir.glob("test_*.split1.bundle")
                      if p.name[: -len(".split1.bundle")] not in SUBSETS)
    subsets = {s: read_lines(split_dir / f"{s}.split1.bundle") for s in (*SUBSETS, *per_view)}
    gt = {v: read_lines(data_dir / "groundTruth" / f"{v}.txt") for v in subsets["test"]}

    sources = (comparison_sources(args.hand, args.real_tag) if args.compare
               else model_sources(name, args.diffact_run, args.fact_ckpt))
    models = {}
    for label, kind, path in sources:
        pred = None
        if kind == "diffact" and path is not None and (path / "prediction").is_dir():
            pred = diffact_predictions(path, [v for v in subsets["test"] if (path / "prediction" / f"{v}.txt").exists()])
        elif kind == "fact" and path is not None and path.exists():
            pred = fact_predictions(path, data_dir)
        elif kind == "mstcn" and path is not None and (path / "results").is_dir():
            pred = mstcn_predictions(path, subsets["test"])
        missing = [v for v in subsets["test"] if pred is not None and v not in pred]
        if pred is None:
            print(f"(skipping {label}: nothing found at {path})")
        elif missing:
            print(f"(skipping {label}: {len(missing)}/{len(subsets['test'])} test videos have no prediction yet "
                  f"in {path}, the run is probably still training)")
        else:
            models[label] = (path, pred)
    if not models:
        raise SystemExit("no model outputs found")

    results = {}
    width = max(8, max(len(k) for k in models))
    header = f"{'model':{width}s} {'subset':22s} {'#vid':>5s} " + " ".join(
        f"{m:>7s}" for m in ("Acc", "Edit", "F1@10", "F1@25", "F1@50"))
    print(header)
    print("-" * len(header))
    for model, (source, pred) in models.items():
        results[model] = {"source": str(source)}
        for subset, videos in subsets.items():
            if not videos:
                print(f"{model:{width}s} {subset:22s} {0:5d}  (empty)")
                continue
            missing = [v for v in videos if v not in pred]
            if missing:
                raise SystemExit(f"{model}: {len(missing)} {subset} videos have no prediction in {source}")
            m = evaluate(gt, pred, videos)
            results[model][subset] = {"num_videos": len(videos), **m}
            print(f"{model:{width}s} {subset:22s} {len(videos):5d} " + " ".join(f"{x:7.1f}" for x in m.values()))

    out = RUNS_DIR / (f"compare_{name}.json" if args.compare else f"subset_eval_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
