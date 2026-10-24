"""Code 2: train FACT on OakInk2.

Prepares realdata/OakInk2 (labels, TV-L1 flow, I3D features; skipped when already
done), generates a FACT yaml from its Breakfast config (same statistics as
FACT/utils/gen_config.py), then runs FACT's train.py.

    conda activate fact
    python train_fact_oakink2.py --hand right --split 1             # five per-hand primitives
    python train_fact_oakink2.py --label_source program             # 70 OakInk2 primitives
"""
import argparse
import importlib.util
import math
import os
import runpy
import sys

import numpy as np

from oakink2_actseg.common import BACKGROUND, FACT_DIR, RUNS_DIR
from oakink2_actseg.labels import num_segments
from oakink2_actseg.prepare import (
    add_prepare_args,
    dataset_dir_from_args,
    prepare_from_args,
    require_split,
    require_training_labels,
)


def import_fact_as_package(name="fact"):
    """FACT uses relative imports and expects to be run as `python -m src.train`;
    register its folder as a package so that works regardless of the folder name."""
    spec = importlib.util.spec_from_file_location(
        name, FACT_DIR / "__init__.py", submodule_search_locations=[str(FACT_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def transcript_stats(data_dir, match):
    lens = []
    for gt in sorted((data_dir / "groundTruth").glob("*.txt")):
        labels = gt.read_text().split("\n")[:-1]
        lens.append(num_segments(labels) if match == "o2o" else len(set(labels)))
    return float(np.mean(lens)), int(np.max(lens))


def write_fact_config(data_dir, args):
    from yacs.config import CfgNode

    with open(FACT_DIR / "configs" / f"{args.base}.yaml") as f:
        cfg = CfgNode.load_cfg(f)

    avg_len, max_len = transcript_stats(data_dir, args.match)
    n_train = len((data_dir / "splits" / f"train.split{args.split}.bundle").read_text().split())
    steps_per_epoch = math.ceil(n_train / (args.batch_size or cfg.batch_size))

    cfg.dataset = data_dir.name
    cfg.feature_path = str(data_dir / "features")
    cfg.groundTruth_path = str(data_dir / "groundTruth")
    cfg.split_path = str(data_dir / "splits")
    cfg.map_fname = str(data_dir / "mapping.txt")
    cfg.feature_transpose = True  # features are stored (D, T) for DiffAct
    classes = [l.split(" ", 1)[1] for l in (data_dir / "mapping.txt").read_text().split("\n") if l]
    if BACKGROUND in classes:
        cfg.bg_class = classes.index(BACKGROUND)
        cfg.eval_bg = False  # exclude background from Acc/Edit/F1
    else:
        # FACT needs some bg_class; with bgw=1 and eval_bg it changes neither the loss nor the metrics.
        cfg.bg_class = 0
        cfg.Loss.bgw = 1.0
        cfg.eval_bg = True
    cfg.average_transcript_len = avg_len
    cfg.FACT.ntoken = args.ntoken or int(math.ceil(max_len * args.token_ratio))
    cfg.Loss.match = args.match
    for key in ("epoch", "batch_size", "lr", "lr_decay"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg.aux.print_every = steps_per_epoch
    # FACT crashes at the end if it never evaluated, so never exceed the total step count.
    cfg.aux.eval_every = min(steps_per_epoch * args.eval_every_epochs, steps_per_epoch * cfg.epoch)

    path = RUNS_DIR / "fact" / "configs" / f"{data_dir.name}_{args.match}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.dump())
    print(f"[fact] avg transcript {avg_len:.1f}, max {max_len}, ntoken {cfg.FACT.ntoken}")
    print(f"[fact] config: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_prepare_args(parser)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--base", default="breakfast", choices=("breakfast", "gtea", "egoprocel"))
    parser.add_argument("--match", default="o2o", choices=("o2o", "o2m"))
    parser.add_argument("--token_ratio", type=float, default=1.5)
    parser.add_argument("--ntoken", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None, help="default: from base config (150)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_decay", type=int, default=None)
    parser.add_argument("--eval_every_epochs", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0, help="index among visible GPUs")
    parser.add_argument("--wandb", default="disabled", choices=("disabled", "offline", "online"))
    args = parser.parse_args()
    require_training_labels(args)

    if args.skip_prepare:
        data_dir = dataset_dir_from_args(args)
    else:
        data_dir = prepare_from_args(args)
    require_split(data_dir, args.split)
    config_path = write_fact_config(data_dir, args)

    import wandb

    wandb_init = wandb.init
    wandb.init = lambda *a, **k: wandb_init(*a, **{**k, "mode": args.wandb})
    os.environ.setdefault("WANDB_SILENT", "true")

    package = import_fact_as_package()
    print(f"[fact] logs: {FACT_DIR.parent / 'log' / data_dir.name}")
    sys.argv = ["train.py", "--cfg", str(config_path), "--set",
                "aux.gpu", str(args.gpu), "split", f"split{args.split}"]
    runpy.run_module(f"{package}.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
