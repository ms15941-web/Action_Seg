import ast
import random
import re
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


def _write_split_files(out_dir, header, files):
    """splits/videos.txt = header + all split videos; files are rewritten only when it changes."""
    split_dir = Path(out_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted({v for vids in files.values() for v in vids})
    index_file = split_dir / "videos.txt"
    signature = "\n".join([header] + videos) + "\n"
    if index_file.exists() and index_file.read_text() == signature:
        return False
    if index_file.exists():
        print("WARNING: split definition or video set changed, regenerating splits "
              "(results from the old splits are not comparable).")
    for old in split_dir.glob("*.bundle"):
        old.unlink()
    for name, vids in files.items():
        (split_dir / name).write_text("".join(f"{v}\n" for v in vids))
    index_file.write_text(signature)
    return True


def _per_camera_subsets(test, video_meta):
    cameras = sorted({cam for _, cam in video_meta.values()})
    if len(cameras) < 2:
        return {}
    return {f"test_{cam}.split1.bundle": [v for v in test if video_meta[v][1] == cam] for cam in cameras}


def write_kfold_splits(out_dir, video_meta, num_folds, seed):
    """K-fold over sequences; all camera views of a sequence fall in the same fold.
    video_meta: {video: (seq, camera)}."""
    videos = sorted(video_meta)
    seqs = sorted({seq for seq, _ in video_meta.values()})
    if len(seqs) < 2 * num_folds:
        raise ValueError(f"need at least {2 * num_folds} sequences for {num_folds} folds, have {len(seqs)}")
    random.Random(seed).shuffle(seqs)
    files = {}
    for k in range(1, num_folds + 1):
        test_seqs = set(seqs[k - 1::num_folds])
        files[f"train.split{k}.bundle"] = [v for v in videos if video_meta[v][0] not in test_seqs]
        files[f"test.split{k}.bundle"] = [v for v in videos if video_meta[v][0] in test_seqs]
    _write_split_files(out_dir, f"mode=kfold num_folds={num_folds} seed={seed}", files)


def task_name(data_root, seq, _cache={}):
    """Normalised complex-task target text, e.g. 'cut the apple'."""
    if data_root not in _cache:
        _cache[data_root] = load_json(Path(data_root) / "program" / "task_target.json")
    text = _cache[data_root][seq.replace("++", "/")]
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def write_official_split(out_dir, data_root, official_dir, video_labels, video_meta):
    """Official OakInk2 split (sequence level, ~70/5/25) as split1, plus test subsets:
    test_unseen_task: test videos whose complex task never occurs in train,
    test_seen_task: the rest, test_unseen_primitive: test videos containing a class absent from train,
    test_<camera>: per-view subsets when several cameras are used.
    video_meta: {video: (seq, camera)}."""
    official = {
        s: {l.strip().replace("/", "++") for l in open(Path(official_dir) / f"{s}.txt") if l.strip()}
        for s in ("train", "val", "test")
    }
    videos = sorted(video_labels)
    train, val, test = ([v for v in videos if video_meta[v][0] in official[s]] for s in ("train", "val", "test"))
    if len(train) < 2 or len(test) < 2:
        raise ValueError(f"official split needs >=2 downloaded train and test videos, have {len(train)}/{len(test)}")
    task_of = {v: task_name(data_root, video_meta[v][0]) for v in videos}
    train_tasks = {task_of[v] for v in train}
    train_classes = {c for v in train for c in video_labels[v]}
    files = {
        "train.split1.bundle": train,
        "val.split1.bundle": val,
        "test.split1.bundle": test,
        "test_seen_task.split1.bundle": [v for v in test if task_of[v] in train_tasks],
        "test_unseen_task.split1.bundle": [v for v in test if task_of[v] not in train_tasks],
        "test_unseen_primitive.split1.bundle": [v for v in test if set(video_labels[v]) - train_classes],
        **_per_camera_subsets(test, video_meta),
    }
    if _write_split_files(out_dir, "mode=official", files):
        print("[prepare] official split: " + ", ".join(f"{k.split('.')[0]}={len(v)}" for k, v in files.items()))
