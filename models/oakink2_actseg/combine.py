"""Build the comparison datasets that train on synthetic data and test on real OakInk2.

    --mode synth        train: all synthetic hand videos               test: real OakInk2 test
    --mode real+synth   train: real OakInk2 train + all synthetic      test: real OakInk2 test

Validation, test and every test_* subset come from the real dataset, so all three models
(real, real+synth, synth) are scored on exactly the same videos. features/ and groundTruth/
hold per-file symlinks into the source datasets; nothing is copied.

    python -m oakink2_actseg.combine --mode synth --hand right
    python -m oakink2_actseg.combine --mode real+synth --hand right
    -> realdata/combined/oakink2_<mode>_prim5_<hand>/
"""
import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from .common import (
    DEFAULT_DATA_ROOT,
    HANDS,
    MODELS_DIR,
    dataset_name,
    prim5_dataset_name,
    processed_root,
    resolve_cameras,
)
from .labels import _write_split_files
from .synth import DEFAULT_OUT_ROOT as SYNTH_ROOT, hand_dataset_name

COMBINED_ROOT = MODELS_DIR / "realdata" / "combined"
MODES = ("synth", "real+synth")


def _bundle(path):
    return [l for l in Path(path).read_text().split("\n") if l] if Path(path).exists() else []


def _feature_dim(data_dir, video):
    return int(np.load(Path(data_dir) / "features" / f"{video}.npy", mmap_mode="r").shape[0])


def combined_name(mode, hand):
    return f"oakink2_{mode.replace('+', '')}_prim5_{hand}"


def build(mode, hand, real_dir, synth_dir, out_root=COMBINED_ROOT):
    real_dir, synth_dir = Path(real_dir).resolve(), Path(synth_dir).resolve()
    out = Path(out_root) / combined_name(mode, hand)
    mapping = (real_dir / "mapping.txt").read_text()
    if mapping != (synth_dir / "mapping.txt").read_text():
        raise ValueError("real and synthetic mapping.txt differ; both must be the five prim5 classes")

    real_splits, synth_splits = real_dir / "splits", synth_dir / "splits"
    synth_videos = sorted(set(_bundle(synth_splits / "train.split1.bundle")) | set(_bundle(synth_splits / "test.split1.bundle")))
    real_train = _bundle(real_splits / "train.split1.bundle")
    test_files = {p.name: _bundle(p) for p in sorted(real_splits.glob("*.split1.bundle"))
                  if p.name.startswith("test") or p.name.startswith("val")}
    train = synth_videos + (real_train if mode == "real+synth" else [])

    d_real = _feature_dim(real_dir, test_files["test.split1.bundle"][0])
    d_synth = _feature_dim(synth_dir, synth_videos[0])
    if d_real != d_synth:
        raise ValueError(f"feature size differs: real {d_real} vs synthetic {d_synth} (use the same --rgb_only choice)")

    if out.exists():
        shutil.rmtree(out)
    for sub in ("features", "groundTruth"):
        (out / sub).mkdir(parents=True)
    (out / "mapping.txt").write_text(mapping)
    sources = {v: synth_dir for v in synth_videos}
    sources.update({v: real_dir for v in real_train + [v for vids in test_files.values() for v in vids]})
    for v, src in sources.items():
        os.symlink(src / "features" / f"{v}.npy", out / "features" / f"{v}.npy")
        os.symlink(src / "groundTruth" / f"{v}.txt", out / "groundTruth" / f"{v}.txt")
    _write_split_files(out, f"mode={mode} real={real_dir.name} synth={synth_dir.name}",
                       {"train.split1.bundle": train, **test_files})
    print(f"[combine] {out.name}: train {len(train)} ({len(synth_videos)} synthetic"
          + (f" + {len(real_train)} real" if mode == "real+synth" else "")
          + f"), test {len(test_files['test.split1.bundle'])} real, feature dim {d_real}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--hand", required=True, choices=HANDS)
    parser.add_argument("--real_dir", default=None,
                        help="real prim5 dataset (default: realdata/OakInk2/processed/oakink2_allviews_15fps_center_prim5_<hand>)")
    parser.add_argument("--synth_dir", default=None,
                        help="synthetic dataset (default: realdata/OakInk2_synth/processed/oakink2synth_10fps_prim5_<hand>)")
    parser.add_argument("--rgb_only", action="store_true", help="use the RGB-only synthetic features")
    args = parser.parse_args()
    real = args.real_dir or processed_root(DEFAULT_DATA_ROOT) / prim5_dataset_name(
        dataset_name(resolve_cameras("all"), 15, True, "center"), args.hand)
    synth = args.synth_dir or SYNTH_ROOT / "processed" / hand_dataset_name(args.rgb_only, args.hand)
    print(build(args.mode, args.hand, real, synth))


if __name__ == "__main__":
    main()
