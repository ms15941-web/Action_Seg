"""Build MS-TCN style dataset folders from downloaded OakInk2 frames:

    realdata/OakInk2/processed/oakink2_<camera>_<fps>fps_<crop>/            base dataset
        features/<video>.npy   (2048, T) I3D RGB+flow
        groundTruth/<video>.txt          OakInk2 program_info primitives (70 classes)
        mapping.txt
        splits/{train,test}.splitK.bundle
              (official mode: split1 = official train/test, plus val and test_* subsets)
    realdata/OakInk2/processed/oakink2_<...>_prim5_{left,right}/            per-hand datasets
        features -> ../<base dataset>/features
        groundTruth/, mapping.txt, splits/   five functional primitives of one hand

Stages: labels (CPU, seconds) -> flow (CPU, TV-L1, slow) -> features (GPU).
Every stage skips sequences that are already done, so it is safe to re-run.

    python -m oakink2_actseg.prepare --stages labels flow      # e.g. on a CPU node
    python -m oakink2_actseg.prepare --stages features         # on a GPU node
"""
import os
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .common import (
    CAMERAS,
    DEFAULT_DATA_ROOT,
    DEFAULT_PRIM5_DIR,
    HANDS,
    PRIM5_CLASSES,
    dataset_name,
    default_num_workers,
    downloaded_sequences,
    flow_pack_path,
    frames_ids_path,
    frames_pack_path,
    load_json,
    prim5_dataset_name,
    processed_root,
    resolve_cameras,
    select_frame_ids,
    video_name,
)
from .labels import (
    class_list,
    frame_labels,
    load_intervals,
    load_prim5_records,
    num_segments,
    prim5_frame_labels,
    write_dataset_labels,
    write_kfold_splits,
    write_official_split,
)

STAGES = ("labels", "flow", "features")


def add_prepare_args(parser):
    g = parser.add_argument_group("OakInk2 data preparation")
    g.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    g.add_argument("--camera", default="all",
                   help="'all' (one training video per sequence and view) or comma-separated names from "
                        + ",".join(CAMERAS))
    g.add_argument("--fps", type=int, default=15, help="15 matches MS-TCN; must divide 30")
    g.add_argument("--crop", default="center", choices=("center", "full"),
                   help="center: 224x224 centre crop (MS-TCN style); full: whole frame, global pooled")
    g.add_argument("--no_trim", action="store_true",
                   help="keep frames outside the annotated [seq_beg, seq_end] endpoints")
    g.add_argument("--split_mode", default="official", choices=("official", "kfold"),
                   help="official: OakInk2 train/val/test as split1; kfold: random K-fold over sequences")
    g.add_argument("--num_folds", type=int, default=5, help="kfold mode only")
    g.add_argument("--split_seed", type=int, default=0, help="kfold mode only")
    g.add_argument("--flow_workers", type=int, default=default_num_workers())
    g.add_argument("--i3d_batch_size", type=int, default=16)
    g.add_argument("--shard", default=None, metavar="K/N",
                   help="flow/features stages only process every N-th video starting at K (0-based), "
                        "e.g. from a SLURM job array; run the labels stage once beforehand")
    g.add_argument("--label_source", default="prim5", choices=("prim5", "program"),
                   help="prim5: per-hand reach/reposition/transport/stabilize/idle from oakink2_synth; "
                        "program: OakInk2 program_info primitives (70 classes)")
    g.add_argument("--hand", default=None, choices=HANDS,
                   help="hand whose prim5 labels to train on (required for training with prim5)")
    g.add_argument("--prim5_dir", default=str(DEFAULT_PRIM5_DIR),
                   help="folder with {train,val,test}_labels.jsonl")
    return parser


def _parse_shard(shard):
    if shard is None:
        return 0, 1
    k, n = (int(x) for x in str(shard).split("/"))
    if not 0 <= k < n:
        raise ValueError(f"--shard {shard}: need 0 <= K < N")
    return k, n


def _write_splits(out_dir, data_root, split_mode, num_folds, split_seed, video_labels, split_meta):
    if split_mode == "official":
        official_dir = data_root / "official_split"
        if not (official_dir / "test.txt").exists():
            from .download import fetch_official_split

            fetch_official_split(data_root)
        write_official_split(out_dir, data_root, official_dir, video_labels, split_meta)
    else:
        write_kfold_splits(out_dir, split_meta, num_folds, split_seed)


def _write_prim5_dataset(base_dir, out_dir, data_root, videos_meta, hand, records, split_mode, num_folds,
                         split_seed):
    video_labels, far, missing = {}, 0, 0
    for v, (seq, cam, ids) in videos_meta.items():
        if seq not in records:
            missing += 1
            continue
        endpoint = load_json(data_root / "program_extension" / "endpoint" / f"{seq}.json")
        all_ids = sorted(load_json(frames_ids_path(data_root, cam, seq)))
        labels, n_far = prim5_frame_labels(all_ids, endpoint["seq_beg"], ids, records[seq], hand)
        far += n_far
        if num_segments(labels) >= 2:
            video_labels[v] = labels
    total = sum(len(l) for l in video_labels.values())
    print(f"[prepare] prim5 {hand}: {len(video_labels)} videos, {len(videos_meta) - len(video_labels) - missing} "
          f"single-segment skipped, {missing} without labels, {far}/{total} frames >0.1 s from a label step")
    write_dataset_labels(out_dir, list(PRIM5_CLASSES), video_labels)
    _write_splits(out_dir, data_root, split_mode, num_folds, split_seed, video_labels,
                  {v: videos_meta[v][:2] for v in video_labels})
    (base_dir / "features").mkdir(parents=True, exist_ok=True)
    link = out_dir / "features"
    if not link.exists():
        link.symlink_to(os.path.relpath(base_dir / "features", out_dir), target_is_directory=True)


def prepare(data_root, camera="all", fps=15, crop="center", trim=True, split_mode="official",
            num_folds=5, split_seed=0, flow_workers=1, i3d_batch_size=16, stages=STAGES, shard=None,
            label_source="prim5", hand=None, prim5_dir=DEFAULT_PRIM5_DIR):
    """Returns the dataset folder to train on: the per-hand prim5 folder when label_source is
    prim5 and a hand is given, otherwise the base folder."""
    shard_k, shard_n = _parse_shard(shard)
    if shard_n > 1 and "labels" in stages:
        raise ValueError("--shard runs in parallel jobs; run the labels stage once, then shard flow/features")
    if label_source == "prim5" and not trim:
        raise ValueError("prim5 labels only cover the annotated [seq_beg, seq_end] range; drop --no_trim")
    data_root = Path(data_root)
    cameras = resolve_cameras(camera)
    out_dir = processed_root(data_root) / dataset_name(cameras, fps, trim, crop)

    # video -> (seq, camera, selected frame ids)
    videos_meta = {}
    for cam in cameras:
        for seq in downloaded_sequences(data_root, cam):
            ids = select_frame_ids(data_root, seq, load_json(frames_ids_path(data_root, cam, seq)), fps, trim)
            if len(ids) >= 2:
                videos_meta[video_name(seq, cam, cameras)] = (seq, cam, ids)
    if not videos_meta:
        raise FileNotFoundError(
            f"No downloaded frames for {cameras} under {data_root}/frames. "
            f"Run `python -m oakink2_actseg.download --camera {camera}` first."
        )
    per_cam = Counter(cam for _, cam, _ in videos_meta.values())
    print(f"[prepare] {len(videos_meta)} videos ({dict(per_cam)}) -> {out_dir}")

    if "labels" in stages:
        intervals, video_labels = {}, {}
        for v, (seq, _, ids) in videos_meta.items():
            if seq not in intervals:
                intervals[seq] = load_intervals(data_root, seq)
            labels = frame_labels(ids, intervals[seq])
            # DiffAct's boundary targets need at least one action boundary per video.
            if num_segments(labels) >= 2:
                video_labels[v] = labels
        dropped = len(videos_meta) - len(video_labels)
        if dropped:
            print(f"[prepare] skipped {dropped} videos with a single segment")
        write_dataset_labels(out_dir, class_list(data_root), video_labels)
        _write_splits(out_dir, data_root, split_mode, num_folds, split_seed, video_labels,
                      {v: videos_meta[v][:2] for v in video_labels})
        if label_source == "prim5":
            records = load_prim5_records(prim5_dir)
            for h in ([hand] if hand else HANDS):
                _write_prim5_dataset(out_dir, out_dir.parent / prim5_dataset_name(out_dir.name, h), data_root,
                                     videos_meta, h, records, split_mode, num_folds, split_seed)

    split_file = out_dir / "splits" / "videos.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} missing; run the labels stage first")
    videos = split_file.read_text().split("\n")[1:-1][shard_k::shard_n]
    if shard_n > 1:
        print(f"[prepare] shard {shard_k}/{shard_n}: {len(videos)} videos")

    def frames_of(v):
        seq, cam, _ = videos_meta[v]
        return frames_pack_path(data_root, cam, seq)

    def flow_of(v):
        seq, cam, _ = videos_meta[v]
        return flow_pack_path(data_root, cam, fps, trim, seq)

    if "flow" in stages:
        from .flow import compute_flows

        jobs = [(frames_of(v), videos_meta[v][2], flow_of(v)) for v in videos if not flow_of(v).exists()]
        compute_flows(jobs, flow_workers)

    if "features" in stages:
        import torch

        from .i3d_features import extract_features

        missing_flow = [v for v in videos if not flow_of(v).exists()]
        if missing_flow:
            raise FileNotFoundError(f"{len(missing_flow)} videos have no flow yet; run the flow stage")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("[prepare] WARNING: no GPU visible, I3D extraction on CPU will be very slow")
        jobs = [
            (frames_of(v), flow_of(v), videos_meta[v][2], out_dir / "features" / f"{v}.npy")
            for v in videos
            if not (out_dir / "features" / f"{v}.npy").exists()
        ]
        extract_features(jobs, device, crop, i3d_batch_size)

        for v in videos:
            T = np.load(out_dir / "features" / f"{v}.npy", mmap_mode="r").shape[1]
            n = sum(1 for _ in open(out_dir / "groundTruth" / f"{v}.txt"))
            if T != n:
                raise RuntimeError(f"{v}: {T} feature frames vs {n} label frames")
    if label_source == "prim5" and hand:
        return out_dir.parent / prim5_dataset_name(out_dir.name, hand)
    return out_dir


def require_split(data_dir, split):
    if not (Path(data_dir) / "splits" / f"train.split{split}.bundle").exists():
        header = (Path(data_dir) / "splits" / "videos.txt").read_text().split("\n")[0]
        raise SystemExit(f"split{split} does not exist for {header} (official mode only has split1)")


def require_training_labels(args):
    if args.label_source == "prim5" and args.hand is None:
        raise SystemExit("prim5 labels are per hand: pass --hand left or --hand right "
                         "(or --label_source program for the 70 OakInk2 primitives)")


def dataset_dir_from_args(args):
    base = dataset_name(resolve_cameras(args.camera), args.fps, not args.no_trim, args.crop)
    if args.label_source == "prim5":
        require_training_labels(args)
        base = prim5_dataset_name(base, args.hand)
    return processed_root(args.data_root) / base


def prepare_from_args(args, stages=STAGES):
    return prepare(
        data_root=args.data_root, camera=args.camera, fps=args.fps, crop=args.crop,
        trim=not args.no_trim, split_mode=args.split_mode, num_folds=args.num_folds, split_seed=args.split_seed,
        flow_workers=args.flow_workers, i3d_batch_size=args.i3d_batch_size, stages=stages, shard=args.shard,
        label_source=args.label_source, hand=args.hand, prim5_dir=args.prim5_dir,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_prepare_args(parser)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    args = parser.parse_args()
    print(prepare_from_args(args, stages=args.stages))


if __name__ == "__main__":
    main()
