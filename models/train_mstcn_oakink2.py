"""Code 3: train MS-TCN on OakInk2 with the original ms-tcn Trainer / BatchGenerator and the
paper's settings (4 stages, 10 layers, 64 maps, Adam lr 5e-4, batch size 1, 50 epochs),
then predict the test split with the last epoch, as ms-tcn's main.py does.

    conda activate diffact
    python train_mstcn_oakink2.py --hand right --skip_prepare                     # real data
    python train_mstcn_oakink2.py --data_dir realdata/combined/oakink2_synth_prim5_right --num_epochs 8
    python train_mstcn_oakink2.py --data_dir realdata/combined/oakink2_realsynth_prim5_right \\
        --init_from runs/snapshots/real/mstcn/right/models/epoch-50.model --num_epochs 7

Outputs: runs/mstcn/<dataset>-S<split>/models/epoch-N.model and results/<video> (predictions).
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

from oakink2_actseg.common import MSTCN_DIR, RUNS_DIR
from oakink2_actseg.metrics import evaluate
from oakink2_actseg.prepare import (
    add_prepare_args,
    dataset_dir_from_args,
    prepare_from_args,
    require_split,
    require_training_labels,
)

SEED = 1538574472  # ms-tcn main.py


def read_lines(path):
    return [l for l in Path(path).read_text().split("\n") if l]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_prepare_args(parser)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--data_dir", default=None,
                        help="train on an already prepared dataset folder (e.g. from oakink2_actseg.combine)")
    parser.add_argument("--init_from", default=None, help="ms-tcn epoch-N.model to continue training from")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_stages", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_f_maps", type=int, default=64)
    parser.add_argument("--naming", default=None)
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    elif args.skip_prepare:
        require_training_labels(args)
        data_dir = dataset_dir_from_args(args)
    else:
        require_training_labels(args)
        data_dir = prepare_from_args(args)
    require_split(data_dir, args.split)

    import torch

    sys.path.insert(0, str(MSTCN_DIR))
    from batch_gen import BatchGenerator
    from model import Trainer

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = RUNS_DIR / "mstcn" / (args.naming or f"{data_dir.name}-S{args.split}")
    model_dir, results_dir = run_dir / "models", run_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ms-tcn opens groundTruth/<entry> and features/<entry before the first dot>.npy,
    # so its bundles list "<video>.txt" (our bundles list bare video names).
    bundles = {}
    for part in ("train", "test"):
        videos = read_lines(data_dir / "splits" / f"{part}.split{args.split}.bundle")
        bundles[part] = run_dir / f"{part}.split{args.split}.bundle"
        bundles[part].write_text("".join(f"{v}.txt\n" for v in videos))
    features_path, gt_path = f"{data_dir / 'features'}/", f"{data_dir / 'groundTruth'}/"

    actions_dict = {l.split()[1]: int(l.split()[0]) for l in read_lines(data_dir / "mapping.txt")}
    first = read_lines(bundles["train"])[0][: -len(".txt")]
    features_dim = int(np.load(f"{features_path}{first}.npy", mmap_mode="r").shape[0])
    trainer = Trainer(args.num_stages, args.num_layers, args.num_f_maps, features_dim, len(actions_dict))
    print(f"[mstcn] {data_dir.name}: {len(read_lines(bundles['train']))} train / {len(read_lines(bundles['test']))} "
          f"test videos, {features_dim}-d features, {len(actions_dict)} classes -> {run_dir}")

    last = model_dir / f"epoch-{args.num_epochs}.model"
    if last.exists():
        print(f"[mstcn] {last} exists, skipping training")
    else:
        if args.init_from:
            trainer.model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
            print(f"[mstcn] initialised from {args.init_from}")
        batch_gen = BatchGenerator(len(actions_dict), actions_dict, gt_path, features_path, 1)
        batch_gen.read_data(str(bundles["train"]))
        trainer.train(str(model_dir), batch_gen, num_epochs=args.num_epochs, batch_size=args.batch_size,
                      learning_rate=args.lr, device=device)

    trainer.predict(str(model_dir), str(results_dir), features_path, str(bundles["test"]), args.num_epochs,
                    actions_dict, device, 1)
    videos = [v[: -len(".txt")] for v in read_lines(bundles["test"])]
    gt = {v: read_lines(f"{gt_path}{v}.txt") for v in videos}
    pred = {v: (results_dir / v).read_text().split("\n")[1].split() for v in videos}
    m = evaluate(gt, pred, videos)
    print("[mstcn] test " + ", ".join(f"{k} {x:.1f}" for k, x in m.items()))


if __name__ == "__main__":
    main()
