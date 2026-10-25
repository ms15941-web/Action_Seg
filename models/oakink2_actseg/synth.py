"""Preprocess the oakink2_synth synthetic videos for DiffAct / FACT.

Each synthetic video (10 fps, 320x192) has a 32 px title bar above two 160x160 hand crops,
the left hand on the left and the right hand on the right. Every labelled hand becomes one
video "<stem>@<hand>" with TV-L1 flow on its 160x160 crop and I3D features on the crop
resized to 224x224 (a 21-frame window around every frame, as for the real videos).

    realdata/OakInk2_synth/
        flow/<stem>@<hand>.pkl
        processed/oakink2synth_10fps[_rgb]/videos.json, features/<stem>@<hand>.npy
        processed/oakink2synth_10fps[_rgb]_prim5_{left,right}/
            groundTruth/, mapping.txt, splits/, features -> ../oakink2synth_10fps[_rgb]/features
    (<stem> has dots replaced by underscores, e.g. ...00049-00199_r1@left)

All synthetic videos come from OakInk2's official train split, so split1 here only holds out
about 10 % of the sequences (by seq_id, so overlapping windows never straddle the split) to
monitor training. Real OakInk2 val/test remain the place to report results.

    python -m oakink2_actseg.synth --stages labels flow        # CPU node
    python -m oakink2_actseg.synth --stages features           # GPU node
    python -m oakink2_actseg.synth --rgb_only                  # 1024-d RGB-only features, no flow
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .common import (
    DEFAULT_PRIM5_DIR,
    HANDS,
    MODELS_DIR,
    PRIM5_CLASSES,
    default_num_workers,
    load_json,
    load_pickle,
    save_json_atomic,
    save_pickle_atomic,
)
from .labels import _write_split_files, num_segments, write_dataset_labels

DEFAULT_OUT_ROOT = MODELS_DIR / "realdata" / "OakInk2_synth"
TITLE_H, CROP, I3D_SIZE = 32, 160, 224
FLOW_BOUND = 20.0
STAGES = ("labels", "flow", "features")


def base_name(rgb_only):
    return "oakink2synth_10fps" + ("_rgb" if rgb_only else "")


def hand_dataset_name(rgb_only, hand):
    return f"{base_name(rgb_only)}_prim5_{hand}"


def read_hand_frames(video_path, hand, n_frames=None):
    """(T, 160, 160, 3) uint8 BGR crops of one hand."""
    cap = cv2.VideoCapture(str(video_path))
    x0 = 0 if hand == "left" else CROP
    frames = []
    while n_frames is None or len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[TITLE_H:TITLE_H + CROP, x0:x0 + CROP])
    cap.release()
    if not frames:
        raise RuntimeError(f"could not decode {video_path}")
    return np.stack(frames)


def _frame_count(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _held_out(seq_id, percent):
    return int(hashlib.md5(seq_id.encode()).hexdigest(), 16) % 100 < percent


def video_id(stem, hand):
    # DiffAct reads video names with split('.')[0], so ids must not contain dots (e.g. ".r1").
    return f"{stem.replace('.', '_')}@{hand}"


def _migrate_dotted_names(out_root, base_dir, meta):
    """Rename flow / feature files written under the earlier dotted ids."""
    for v, m in meta.items():
        old = f"{Path(m['video']).stem}@{m['hand']}"
        if old == v:
            continue
        for folder, ext in ((Path(out_root) / "flow", ".pkl"), (base_dir / "features", ".npy")):
            src, dst = folder / f"{old}{ext}", folder / f"{v}{ext}"
            if src.exists() and not dst.exists():
                os.replace(src, dst)


# ---------------------------------------------------------------- labels

def build_labels(synth_dir, out_root, rgb_only, holdout_percent):
    synth_dir, processed = Path(synth_dir), Path(out_root) / "processed"
    base_dir = processed / base_name(rgb_only)
    rows = [json.loads(l) for l in open(synth_dir / "synthetic_labels.jsonl")]
    per_hand, meta, stats = {h: {} for h in HANDS}, {}, Counter()
    for r in tqdm(rows, desc="labels"):
        stem = Path(r["video"]).stem
        n = _frame_count(synth_dir / r["video"])
        for hand, labels in r["labels"].items():
            labels = labels[:n]
            if num_segments(labels) < 2:
                stats[f"{hand} single-segment skipped"] += 1
                continue
            v = video_id(stem, hand)
            per_hand[hand][v] = labels
            meta[v] = {"video": r["video"], "hand": hand, "seq_id": r["seq_id"], "n_frames": len(labels)}
    save_json_atomic(meta, base_dir / "videos.json")
    (base_dir / "features").mkdir(parents=True, exist_ok=True)
    for hand in HANDS:
        out_dir = processed / hand_dataset_name(rgb_only, hand)
        videos = sorted(per_hand[hand])
        write_dataset_labels(out_dir, list(PRIM5_CLASSES), per_hand[hand])
        test = [v for v in videos if _held_out(meta[v]["seq_id"], holdout_percent)]
        test_set = set(test)
        _write_split_files(out_dir, f"mode=synthetic holdout={holdout_percent}%", {
            "train.split1.bundle": [v for v in videos if v not in test_set],
            "test.split1.bundle": test,
        })
        link = out_dir / "features"
        if not link.exists():
            link.symlink_to(os.path.relpath(base_dir / "features", out_dir), target_is_directory=True)
        print(f"[synth] {hand}: {len(videos)} videos ({len(videos) - len(test)} train / {len(test)} held out)"
              f" -> {out_dir}")
    for k, v in stats.items():
        print(f"[synth] {k}: {v}")
    return base_dir


# ---------------------------------------------------------------- flow

def _encode(component):
    q = np.round((np.clip(component, -FLOW_BOUND, FLOW_BOUND) + FLOW_BOUND) * (255.0 / (2 * FLOW_BOUND)))
    ok, enc = cv2.imencode(".jpg", q.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return enc.tobytes()


def compute_clip_flow(video_path, hand, n_frames, out_path):
    """TV-L1 flow of one hand crop; flow of frame t is t -> t+1, the last frame repeats it."""
    cv2.setNumThreads(1)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in read_hand_frames(video_path, hand, n_frames)]
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    xs, ys = [], []
    for prev, cur in zip(gray[:-1], gray[1:]):
        flow = tvl1.calc(prev, cur, None)
        xs.append(_encode(flow[..., 0]))
        ys.append(_encode(flow[..., 1]))
    xs.append(xs[-1])
    ys.append(ys[-1])
    save_pickle_atomic({"n_frames": len(gray), "x": xs, "y": ys}, out_path)


# ---------------------------------------------------------------- features

def _decode_flow(pack):
    """(T, 224, 224, 2) float flow in pixels of the 224 crop, scaled to [-1, 1]."""
    out = []
    for bx, by in zip(pack["x"], pack["y"]):
        comps = []
        for b in (bx, by):
            q = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            px = q * (2 * FLOW_BOUND / 255.0) - FLOW_BOUND
            px = cv2.resize(px, (I3D_SIZE, I3D_SIZE), interpolation=cv2.INTER_LINEAR) * (I3D_SIZE / CROP)
            comps.append(np.clip(px, -FLOW_BOUND, FLOW_BOUND) / FLOW_BOUND)
        out.append(np.stack(comps, -1))
    return np.stack(out)


def _windowed(net, x, window, batch, use_amp):
    """x: (T, C, H, W) normalised; returns (T, 1024) mean-pooled Mixed_5c features."""
    import torch

    from .i3d_features import _embed

    T = x.shape[0]
    offsets = torch.arange(window, device=x.device) - window // 2
    out = []
    for start in range(0, T, batch):
        centers = torch.arange(start, min(start + batch, T), device=x.device)
        idx = (centers[:, None] + offsets[None]).clamp_(0, T - 1)
        clip = x[idx].permute(0, 2, 1, 3, 4)  # (B, C, window, H, W)
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_amp):
            out.append(_embed(net, clip).float())
    return torch.cat(out)


def extract_clip_features(models, video_path, hand, n_frames, flow_path, device, window, batch):
    import torch

    rgb = read_hand_frames(video_path, hand, n_frames)[..., ::-1]
    rgb = np.stack([cv2.resize(f, (I3D_SIZE, I3D_SIZE), interpolation=cv2.INTER_LINEAR) for f in rgb])
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        x = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        feats = [_windowed(models["rgb"], x, window, batch, use_amp)]
        if flow_path is not None:
            fl = _decode_flow(load_pickle(flow_path))[: len(rgb)]
            f = torch.from_numpy(fl).to(device).permute(0, 3, 1, 2).float()
            feats.append(_windowed(models["flow"], f, window, batch, use_amp))
        return torch.cat(feats, 1).cpu().numpy().T.astype(np.float32)  # (D, T)


# ---------------------------------------------------------------- driver

def run(synth_dir=DEFAULT_PRIM5_DIR, out_root=DEFAULT_OUT_ROOT, stages=STAGES, rgb_only=False, shard=None,
        limit=None, flow_workers=1, window=21, batch=16, holdout_percent=10):
    from .prepare import _parse_shard

    shard_k, shard_n = _parse_shard(shard)
    if shard_n > 1 and "labels" in stages:
        raise ValueError("--shard runs in parallel jobs; run the labels stage once, then shard flow/features")
    synth_dir, out_root = Path(synth_dir), Path(out_root)
    base_dir = out_root / "processed" / base_name(rgb_only)
    if "labels" in stages:
        build_labels(synth_dir, out_root, rgb_only, holdout_percent)
    if not (base_dir / "videos.json").exists():
        raise FileNotFoundError(f"{base_dir / 'videos.json'} missing; run --stages labels first")
    meta = load_json(base_dir / "videos.json")
    if "labels" in stages:
        _migrate_dotted_names(out_root, base_dir, meta)
    videos = sorted(meta)[shard_k::shard_n][:limit]
    flow_of = lambda v: out_root / "flow" / f"{v}.pkl"

    if "flow" in stages and not rgb_only:
        jobs = [v for v in videos if not flow_of(v).exists()]
        print(f"[synth] TV-L1 flow for {len(jobs)} hand videos with {flow_workers} CPU workers ...")
        with ProcessPoolExecutor(flow_workers) as ex:
            futures = [ex.submit(compute_clip_flow, synth_dir / meta[v]["video"], meta[v]["hand"],
                                 meta[v]["n_frames"], flow_of(v)) for v in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="flow"):
                fut.result()

    if "features" in stages:
        import torch

        from .i3d_features import load_i3d

        if not rgb_only:
            missing = [v for v in videos if not flow_of(v).exists()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} hand videos have no flow yet; run --stages flow")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("[synth] WARNING: no GPU visible, I3D extraction on CPU will be very slow")
        todo = [v for v in videos if not (base_dir / "features" / f"{v}.npy").exists()]
        models = load_i3d(device) if todo else None
        for v in tqdm(todo, desc="i3d"):
            m = meta[v]
            feat = extract_clip_features(models, synth_dir / m["video"], m["hand"], m["n_frames"],
                                         None if rgb_only else flow_of(v), device, window, batch)
            if feat.shape[1] != m["n_frames"]:
                raise RuntimeError(f"{v}: {feat.shape[1]} feature frames vs {m['n_frames']} labels")
            out = base_dir / "features" / f"{v}.npy"
            tmp = out.with_name(out.stem + ".tmp.npy")
            np.save(tmp, feat)
            tmp.replace(out)
    return base_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synth_dir", default=str(DEFAULT_PRIM5_DIR),
                        help="folder with synthetic_labels.jsonl and synthetic_videos/")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--rgb_only", action="store_true", help="skip optical flow; 1024-d RGB I3D features")
    parser.add_argument("--shard", default=None, metavar="K/N")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N videos (quick test)")
    parser.add_argument("--flow_workers", type=int, default=default_num_workers())
    parser.add_argument("--window", type=int, default=21, help="I3D frames around each frame (21 = MS-TCN)")
    parser.add_argument("--i3d_batch_size", type=int, default=16)
    parser.add_argument("--holdout_percent", type=int, default=10, help="sequences held out as split1 test")
    args = parser.parse_args()
    print(run(args.synth_dir, args.out_root, args.stages, args.rgb_only, args.shard, args.limit,
              args.flow_workers, args.window, args.i3d_batch_size, args.holdout_percent))


if __name__ == "__main__":
    main()
