"""Code 1: train DiffAct on OakInk2.

Prepares realdata/OakInk2 (labels, TV-L1 flow, I3D features; skipped when already
done), writes a DiffAct config based on its Breakfast setting, then runs DiffAct/main.py.

    conda activate diffact
    python train_diffact_oakink2.py --split 1
"""
import argparse
import ast
import json
import subprocess
import sys

import numpy as np

from oakink2_actseg.common import DIFFACT_DIR, RUNS_DIR, dataset_name, processed_root
from oakink2_actseg.prepare import add_prepare_args, prepare_from_args, require_split


def diffact_base_params(name="params_breakfast"):
    """Read a params dict literal from DiffAct/default_configs.py without executing it
    (importing that file writes configs into the current directory)."""
    tree = ast.parse((DIFFACT_DIR / "default_configs.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_prepare_args(parser)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--base", default="params_breakfast",
                        choices=("params_breakfast", "params_gtea", "params_50salads"))
    parser.add_argument("--num_epochs", type=int, default=None, help="default: from base config (1001)")
    parser.add_argument("--log_freq", type=int, default=None, help="evaluate/save every N epochs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--device", type=int, default=-1,
                        help="GPU index; -1 keeps CUDA_VISIBLE_DEVICES as set by the scheduler")
    parser.add_argument("--naming", default=None)
    args = parser.parse_args()

    if args.skip_prepare:
        data_dir = processed_root(args.data_root) / dataset_name(args.camera, args.fps, not args.no_trim, args.crop)
    else:
        data_dir = prepare_from_args(args)
    require_split(data_dir, args.split)

    first_feature = next((data_dir / "features").glob("*.npy"))
    input_dim = int(np.load(first_feature, mmap_mode="r").shape[0])

    params = diffact_base_params(args.base)
    naming = args.naming or f"{data_dir.name}-S{args.split}"
    params.update(
        naming=naming,
        root_data_dir=str(data_dir.parent),
        dataset_name=data_dir.name,
        split_id=args.split,
        sample_rate=args.sample_rate,
        result_dir=str(RUNS_DIR / "diffact"),
    )
    params["encoder_params"]["input_dim"] = input_dim
    for key in ("num_epochs", "log_freq", "batch_size", "learning_rate"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)

    config_path = RUNS_DIR / "diffact" / "configs" / f"{naming}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(params, indent=2))
    print(f"[diffact] config: {config_path}")
    print(f"[diffact] results: {RUNS_DIR / 'diffact' / naming}")

    cmd = [sys.executable, "main.py", "--config", str(config_path), "--device", str(args.device)]
    sys.exit(subprocess.call(cmd, cwd=DIFFACT_DIR))


if __name__ == "__main__":
    main()
