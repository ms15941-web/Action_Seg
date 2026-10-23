"""Score trained DiffAct / FACT models on the official OakInk2 test split and its subsets:

    test                   all official test videos
    test_seen_task         complex task also performed in the training split
    test_unseen_task       complex task never performed in the training split
    test_unseen_primitive  contains an action class that never occurs in training

Both models are scored with the same MS-TCN metrics (oakink2_actseg/metrics.py) from their
saved predictions, so no GPU is needed:
    DiffAct: runs/diffact/<name>/prediction/*.txt  (written at its most recent evaluation epoch)
    FACT:    log/<dataset>/split1/**/best_ckpt.gz  (FACT keeps the checkpoint with the best test F1@50)

    conda activate fact
    python eval_oakink2_subsets.py
"""
import argparse
import gzip
import json
import pickle
from pathlib import Path

from oakink2_actseg.common import FACT_DIR, RUNS_DIR, dataset_name, processed_root
from oakink2_actseg.metrics import evaluate
from oakink2_actseg.prepare import add_prepare_args

SUBSETS = ("test", "test_seen_task", "test_unseen_task", "test_unseen_primitive")


def read_lines(path):
    return [l for l in Path(path).read_text().split("\n") if l]


def diffact_predictions(run_dir, videos):
    pred_dir = Path(run_dir) / "prediction"
    return {v: (pred_dir / f"{v}.txt").read_text().split("\n")[1].split() for v in videos}


def fact_predictions(ckpt_path, data_dir):
    from train_fact_oakink2 import import_fact_as_package

    import_fact_as_package()  # the checkpoint pickles classes from the `fact` package
    from fact.utils.utils import expand_frame_label

    index2label = {int(l.split(" ", 1)[0]): l.split(" ", 1)[1] for l in read_lines(data_dir / "mapping.txt")}
    with gzip.open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_prepare_args(parser)
    parser.add_argument("--diffact_run", default=None, help="default: runs/diffact/<dataset>-S1")
    parser.add_argument("--fact_ckpt", default=None, help="default: newest log/<dataset>/split1/**/best_ckpt.gz")
    args = parser.parse_args()

    name = dataset_name(args.camera, args.fps, not args.no_trim, args.crop)
    data_dir = processed_root(args.data_root) / name
    split_dir = data_dir / "splits"
    subsets = {s: read_lines(split_dir / f"{s}.split1.bundle") for s in SUBSETS if (split_dir / f"{s}.split1.bundle").exists()}
    if len(subsets) < len(SUBSETS):
        raise SystemExit(f"{split_dir} has no test subsets; re-run the labels stage with --split_mode official")
    gt = {v: read_lines(data_dir / "groundTruth" / f"{v}.txt") for v in subsets["test"]}

    models = {}
    diffact_run = Path(args.diffact_run) if args.diffact_run else RUNS_DIR / "diffact" / f"{name}-S1"
    if (diffact_run / "prediction").is_dir():
        models["DiffAct"] = (diffact_run, diffact_predictions(diffact_run, subsets["test"]))
    fact_ckpt = Path(args.fact_ckpt) if args.fact_ckpt else latest(
        (FACT_DIR.parent / "log" / name / "split1").glob("**/best_ckpt.gz"))
    if fact_ckpt is not None and fact_ckpt.exists():
        models["FACT"] = (fact_ckpt, fact_predictions(fact_ckpt, data_dir))
    if not models:
        raise SystemExit(f"no DiffAct predictions in {diffact_run} and no FACT best_ckpt.gz under log/{name}")

    results = {}
    header = f"{'model':8s} {'subset':22s} {'#vid':>5s} " + " ".join(f"{m:>7s}" for m in ("Acc", "Edit", "F1@10", "F1@25", "F1@50"))
    print(header)
    print("-" * len(header))
    for model, (source, pred) in models.items():
        results[model] = {"source": str(source)}
        for subset, videos in subsets.items():
            if not videos:
                print(f"{model:8s} {subset:22s} {0:5d}  (empty)")
                continue
            missing = [v for v in videos if v not in pred]
            if missing:
                raise SystemExit(f"{model}: {len(missing)} {subset} videos have no prediction in {source}")
            m = evaluate(gt, pred, videos)
            results[model][subset] = {"num_videos": len(videos), **m}
            print(f"{model:8s} {subset:22s} {len(videos):5d} " + " ".join(f"{x:7.1f}" for x in m.values()))

    out = RUNS_DIR / f"subset_eval_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
