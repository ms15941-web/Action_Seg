"""Download OakInk2 into realdata/OakInk2, keeping a single camera view.

Each sequence tarball (all 4 cameras, PNG, up to ~20GB) is streamed over HTTP and
never written to disk: only the frames of the chosen camera are kept, resized to
`--short_side` and re-encoded as JPEG into one pack file per sequence.

    python -m oakink2_actseg.download --camera egocentric
    python -m oakink2_actseg.download --num_seqs 50          # random subset for a quick test
"""
import argparse
import pickle
import random
import shutil
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from huggingface_hub import hf_hub_url
from tqdm import tqdm

from .common import (
    CAMERAS,
    DEFAULT_DATA_ROOT,
    HF_REPO_ID,
    I3D_DIR,
    all_sequences,
    frames_ids_path,
    frames_pack_path,
    load_json,
    load_pickle,
    save_json_atomic,
    save_pickle_atomic,
)

ANNOTATION_TARS = ("program.tar", "program_extension.tar")
I3D_FILES = {
    "pytorch_i3d.py": "https://raw.githubusercontent.com/piergiaj/pytorch-i3d/master/pytorch_i3d.py",
    "LICENSE.txt": "https://raw.githubusercontent.com/piergiaj/pytorch-i3d/master/LICENSE.txt",
    "models/rgb_imagenet.pt": "https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt",
    "models/flow_imagenet.pt": "https://github.com/piergiaj/pytorch-i3d/raw/master/models/flow_imagenet.pt",
}
CHUNK = 1 << 20


def _session():
    import os

    s = requests.Session()
    token = os.environ.get("HF_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _url(path):
    return hf_hub_url(HF_REPO_ID, path, repo_type="dataset")


def _download_file(session, url, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
    tmp.replace(dst)


def fetch_annotations(data_root, session):
    data_root = Path(data_root)
    for name in ANNOTATION_TARS:
        target_dir = data_root / name[: -len(".tar")]
        if target_dir.is_dir() and any(target_dir.iterdir()):
            continue
        print(f"Downloading {name} ...")
        tar_path = data_root / name
        _download_file(session, _url(name), tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(data_root)
        tar_path.unlink()


def fetch_i3d(session, i3d_dir=I3D_DIR):
    for rel, url in I3D_FILES.items():
        dst = Path(i3d_dir) / rel
        if not dst.exists():
            print(f"Downloading {url} -> {dst}")
            _download_file(session, url, dst)


def _camera_serial(session, seq, camera, cam_def_cache):
    if seq not in cam_def_cache:
        with session.get(_url(f"anno_preview/{seq}.pkl"), timeout=120) as r:
            r.raise_for_status()
            anno = pickle.loads(r.content)  # contains torch tensors, so torch must be importable
        cam_def_cache[seq] = dict(anno["cam_def"])
    name_to_serial = {v: k for k, v in cam_def_cache[seq].items()}
    if camera not in name_to_serial:
        raise KeyError(f"{seq}: camera {camera} not in {cam_def_cache[seq]}")
    return str(name_to_serial[camera])


def _resize_short_side(img, short_side):
    h, w = img.shape[:2]
    scale = short_side / min(h, w)
    if abs(scale - 1) < 1e-6:
        return img
    size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _stream_camera_frames(session, seq, serial, expected_ids, short_side, jpeg_quality):
    frames = {}
    expected = set(expected_ids)
    encode_args = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    with session.get(_url(f"data/{seq}.tar"), stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                parts = member.name.strip("/").split("/")
                if len(parts) < 2 or parts[-2] != serial:
                    continue
                stem = Path(parts[-1]).stem
                if not stem.isdigit():
                    continue
                buf = tf.extractfile(member).read()
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = _resize_short_side(img, short_side)
                ok, enc = cv2.imencode(".jpg", img, encode_args)
                assert ok
                frames[int(stem)] = enc.tobytes()
                # Frames of one camera are stored contiguously; stop once all are in.
                if expected and expected.issubset(frames.keys()):
                    break
    return frames


def download_sequence(data_root, seq, camera, short_side, jpeg_quality, retries=3):
    data_root = Path(data_root)
    if frames_ids_path(data_root, camera, seq).exists() and frames_pack_path(data_root, camera, seq).exists():
        return seq, "skip"
    session = _session()
    cam_def_path = data_root / "cam_def" / f"{seq}.json"
    cache = {seq: load_json(cam_def_path)} if cam_def_path.exists() else {}
    expected_ids = load_pickle(data_root / "program_extension" / "frame_id" / f"{seq}.pkl")["frame_id_list"]
    last_err = None
    for attempt in range(retries):
        try:
            serial = _camera_serial(session, seq, camera, cache)
            save_json_atomic(cache[seq], cam_def_path)
            frames = _stream_camera_frames(session, seq, serial, expected_ids, short_side, jpeg_quality)
            if not frames:
                raise RuntimeError(f"no frames found for camera serial {serial}")
            ids = sorted(frames)
            save_pickle_atomic(
                {"seq": seq, "camera": camera, "serial": serial, "short_side": short_side,
                 "frame_ids": ids, "jpgs": [frames[i] for i in ids]},
                frames_pack_path(data_root, camera, seq),
            )
            save_json_atomic(ids, frames_ids_path(data_root, camera, seq))
            missing = len(set(expected_ids) - set(ids))
            return seq, f"ok ({len(ids)} frames" + (f", {missing} missing)" if missing else ")")
        except Exception as e:  # network hiccups on multi-GB streams are common
            last_err = e
            time.sleep(10 * (attempt + 1))
    return seq, "FAILED: " + "".join(traceback.format_exception_only(type(last_err), last_err)).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--camera", default="egocentric", choices=CAMERAS)
    parser.add_argument("--num_seqs", type=int, default=None,
                        help="random subset size (default: all 627); increasing it later only adds sequences")
    parser.add_argument("--seq_list", default=None, help="text file with one sequence token per line")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--short_side", type=int, default=256)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=4, help="sequences downloaded in parallel")
    parser.add_argument("--skip_i3d", action="store_true", help="do not fetch pytorch-i3d code/weights")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    session = _session()
    fetch_annotations(data_root, session)
    if not args.skip_i3d:
        fetch_i3d(session)

    seqs = all_sequences(data_root)
    if args.seq_list:
        wanted = [l.strip().replace("/", "++") for l in open(args.seq_list) if l.strip()]
        seqs = [s for s in wanted if s in set(seqs)]
    elif args.num_seqs is not None:
        # Fixed shuffle + prefix, so a larger --num_seqs always contains a smaller one.
        random.Random(args.seed).shuffle(seqs)
        seqs = sorted(seqs[: args.num_seqs])
    print(f"{len(seqs)} sequences, camera={args.camera}, data_root={data_root}")
    free_gb = shutil.disk_usage(data_root).free / 1e9
    print(f"Free disk: {free_gb:.0f} GB (expect ~40 MB per sequence at short_side=256)")

    failed = []
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [
            ex.submit(download_sequence, data_root, s, args.camera, args.short_side, args.jpeg_quality)
            for s in seqs
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            seq, status = fut.result()
            if status != "skip":
                tqdm.write(f"{seq}: {status}")
            if status.startswith("FAILED"):
                failed.append(seq)
    if failed:
        print(f"{len(failed)} sequences failed; re-run the same command to retry them.")


if __name__ == "__main__":
    main()
