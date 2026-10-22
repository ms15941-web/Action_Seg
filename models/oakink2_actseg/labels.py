import ast
import random
from pathlib import Path

import numpy as np

from .common import BACKGROUND, IGNORED_PRIMITIVES, TRANSIENT_PRIMITIVES, all_sequences, load_json


def load_intervals(data_root, seq):
    """[(begin, end_exclusive, primitive)] in the shared 120Hz frame-id clock."""
    program_info = load_json(Path(data_root) / "program" / "program_info" / f"{seq}.json")
    intervals = []
    for key, item in program_info.items():
        lh, rh = ast.literal_eval(key)
        hands = [r for r in (lh, rh) if r is not None]
        if not hands or item["primitive"] in IGNORED_PRIMITIVES:
            continue
        intervals.append((min(r[0] for r in hands), max(r[1] for r in hands), item["primitive"]))
    return intervals


def class_list(data_root):
    names = set()
    for seq in all_sequences(data_root):
        names.update(p for _, _, p in load_intervals(data_root, seq))
    return [BACKGROUND] + sorted(names)


def frame_labels(frame_ids, intervals):
    """One label per frame. Bimanual primitives can overlap in time; transient
    primitives are painted first, then the rest in start order (later start wins)."""
    ids = np.asarray(frame_ids)
    labels = np.full(len(ids), BACKGROUND, dtype=object)
    ordered = sorted(intervals, key=lambda x: (x[2] not in TRANSIENT_PRIMITIVES, x[0]))
    for beg, end, prim in ordered:
        labels[(ids >= beg) & (ids < end)] = prim
    return labels.tolist()


def num_segments(labels):
    return 1 + sum(a != b for a, b in zip(labels[:-1], labels[1:]))


def write_dataset_labels(out_dir, classes, video_labels):
    out_dir = Path(out_dir)
    gt_dir = out_dir / "groundTruth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mapping.txt", "w") as f:
        f.write("".join(f"{i} {c}\n" for i, c in enumerate(classes)))
    for video, labels in video_labels.items():
        with open(gt_dir / f"{video}.txt", "w") as f:
            f.write("\n".join(labels) + "\n")


def write_splits(out_dir, videos, num_folds, seed):
    """K-fold splits over sequences. Rewritten only when the video set changes."""
    split_dir = Path(out_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(videos)
    index_file = split_dir / "videos.txt"
    signature = "\n".join([f"num_folds={num_folds} seed={seed}"] + videos) + "\n"
    if index_file.exists() and index_file.read_text() == signature:
        return
    if index_file.exists():
        print("WARNING: video set changed, regenerating splits (old results are not comparable).")
    if len(videos) < 2 * num_folds:
        raise ValueError(f"need at least {2 * num_folds} videos for {num_folds} folds, have {len(videos)}")
    shuffled = videos[:]
    random.Random(seed).shuffle(shuffled)
    folds = [sorted(shuffled[k::num_folds]) for k in range(num_folds)]
    for k, test in enumerate(folds, start=1):
        test_set = set(test)
        train = [v for v in videos if v not in test_set]
        (split_dir / f"train.split{k}.bundle").write_text("\n".join(train) + "\n")
        (split_dir / f"test.split{k}.bundle").write_text("\n".join(test) + "\n")
    index_file.write_text(signature)
