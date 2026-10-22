"""Build an MS-TCN style dataset folder from downloaded OakInk2 frames:

    realdata/OakInk2/processed/oakink2_<camera>_<fps>fps_<crop>/
        features/<seq>.npy   (2048, T) I3D RGB+flow
        groundTruth/<seq>.txt
        mapping.txt
        splits/{train,test}.splitK.bundle

Stages: labels (CPU, seconds) -> flow (CPU, TV-L1, slow) -> features (GPU).
Every stage skips sequences that are already done, so it is safe to re-run.

    python -m oakink2_actseg.prepare --stages labels flow      # e.g. on a CPU node
    python -m oakink2_actseg.prepare --stages features         # on a GPU node
"""
import argparse
from pathlib import Path

import numpy as np

from .common import (
    CAMERAS,
    DEFAULT_DATA_ROOT,
    dataset_name,
    default_num_workers,
    downloaded_sequences,
    flow_pack_path,
    frames_ids_path,
    frames_pack_path,
    load_json,
    processed_root,
    select_frame_ids,
)
from .labels import class_list, frame_labels, load_intervals, num_segments, write_dataset_labels, write_splits

STAGES = ("labels", "flow", "features")


def add_prepare_args(parser):
    g = parser.add_argument_group("OakInk2 data preparation")
    g.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    g.add_argument("--camera", default="egocentric", choices=CAMERAS)
    g.add_argument("--fps", type=int, default=15, help="15 matches MS-TCN; must divide 30")
    g.add_argument("--crop", default="center", choices=("center", "full"),
                   help="center: 224x224 centre crop (MS-TCN style); full: whole frame, global pooled")
    g.add_argument("--no_trim", action="store_true",
                   help="keep frames outside the annotated [seq_beg, seq_end] endpoints")
    g.add_argument("--num_folds", type=int, default=5)
    g.add_argument("--split_seed", type=int, default=0)
    g.add_argument("--flow_workers", type=int, default=default_num_workers())
    g.add_argument("--i3d_batch_size", type=int, default=16)
    return parser


def prepare(data_root, camera="egocentric", fps=15, crop="center", trim=True, num_folds=5,
            split_seed=0, flow_workers=1, i3d_batch_size=16, stages=STAGES):
    data_root = Path(data_root)
    out_dir = processed_root(data_root) / dataset_name(camera, fps, trim, crop)
    seqs = downloaded_sequences(data_root, camera)
    if not seqs:
        raise FileNotFoundError(
            f"No downloaded {camera} frames under {data_root}/frames. "
            f"Run `python -m oakink2_actseg.download --camera {camera}` first."
        )
    print(f"[prepare] {len(seqs)} downloaded sequences -> {out_dir}")

    frame_ids = {}
    for seq in seqs:
        ids = select_frame_ids(data_root, seq, load_json(frames_ids_path(data_root, camera, seq)), fps, trim)
        if len(ids) >= 2:
            frame_ids[seq] = ids

    if "labels" in stages:
        video_labels = {}
        for seq, ids in frame_ids.items():
            labels = frame_labels(ids, load_intervals(data_root, seq))
            # DiffAct's boundary targets need at least one action boundary per video.
            if num_segments(labels) >= 2:
                video_labels[seq] = labels
        dropped = len(seqs) - len(video_labels)
        if dropped:
            print(f"[prepare] skipped {dropped} sequences with <2 frames or a single segment")
        write_dataset_labels(out_dir, class_list(data_root), video_labels)
        write_splits(out_dir, list(video_labels), num_folds, split_seed)

    split_file = out_dir / "splits" / "videos.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} missing; run the labels stage first")
    videos = split_file.read_text().split("\n")[1:-1]

    if "flow" in stages:
        from .flow import compute_flows

        jobs = [
            (frames_pack_path(data_root, camera, v), frame_ids[v], flow_pack_path(data_root, camera, fps, trim, v))
            for v in videos
            if not flow_pack_path(data_root, camera, fps, trim, v).exists()
        ]
        compute_flows(jobs, flow_workers)

    if "features" in stages:
        import torch

        from .i3d_features import extract_features

        missing_flow = [v for v in videos if not flow_pack_path(data_root, camera, fps, trim, v).exists()]
        if missing_flow:
            raise FileNotFoundError(f"{len(missing_flow)} sequences have no flow yet; run the flow stage")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("[prepare] WARNING: no GPU visible, I3D extraction on CPU will be very slow")
        jobs = [
            (frames_pack_path(data_root, camera, v), flow_pack_path(data_root, camera, fps, trim, v),
             frame_ids[v], out_dir / "features" / f"{v}.npy")
            for v in videos
            if not (out_dir / "features" / f"{v}.npy").exists()
        ]
        extract_features(jobs, device, crop, i3d_batch_size)

        for v in videos:
            T = np.load(out_dir / "features" / f"{v}.npy", mmap_mode="r").shape[1]
            n = sum(1 for _ in open(out_dir / "groundTruth" / f"{v}.txt"))
            if T != n:
                raise RuntimeError(f"{v}: {T} feature frames vs {n} label frames")
    return out_dir


def prepare_from_args(args, stages=STAGES):
    return prepare(
        data_root=args.data_root, camera=args.camera, fps=args.fps, crop=args.crop,
        trim=not args.no_trim, num_folds=args.num_folds, split_seed=args.split_seed,
        flow_workers=args.flow_workers, i3d_batch_size=args.i3d_batch_size, stages=stages,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_prepare_args(parser)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    args = parser.parse_args()
    print(prepare_from_args(args, stages=args.stages))


if __name__ == "__main__":
    main()
