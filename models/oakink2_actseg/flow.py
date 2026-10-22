"""TV-L1 optical flow, stored as uint8 JPEGs ([-20, 20] px -> [0, 255]) like MS-TCN/I3D."""
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

from .common import load_pickle, save_pickle_atomic

FLOW_BOUND = 20.0


def _tvl1():
    if not hasattr(cv2, "optflow"):
        raise ImportError("cv2.optflow missing: install opencv-contrib-python-headless (not opencv-python)")
    return cv2.optflow.DualTVL1OpticalFlow_create()


def _gray(jpg):
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)


def _encode(component):
    q = np.round((np.clip(component, -FLOW_BOUND, FLOW_BOUND) + FLOW_BOUND) * (255.0 / (2 * FLOW_BOUND)))
    ok, enc = cv2.imencode(".jpg", q.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return enc.tobytes()


def compute_sequence_flow(frames_pack, frame_ids, out_path):
    """Flow of frame t is flow(t -> t+1); the last frame repeats the previous flow."""
    cv2.setNumThreads(1)
    pack = load_pickle(frames_pack)
    jpg_of = dict(zip(pack["frame_ids"], pack["jpgs"]))
    tvl1 = _tvl1()
    xs, ys = [], []
    prev = _gray(jpg_of[frame_ids[0]])
    for fid in frame_ids[1:]:
        cur = _gray(jpg_of[fid])
        flow = tvl1.calc(prev, cur, None)
        xs.append(_encode(flow[..., 0]))
        ys.append(_encode(flow[..., 1]))
        prev = cur
    if not xs:
        zero = np.zeros(prev.shape, np.float32)
        xs, ys = [_encode(zero)], [_encode(zero)]
    else:
        xs.append(xs[-1])
        ys.append(ys[-1])
    save_pickle_atomic({"frame_ids": list(frame_ids), "x": xs, "y": ys}, out_path)
    return out_path


def compute_flows(jobs, num_workers):
    """jobs: list of (frames_pack, frame_ids, out_path)."""
    if not jobs:
        return
    print(f"TV-L1 flow for {len(jobs)} sequences with {num_workers} CPU workers ...")
    with ProcessPoolExecutor(num_workers) as ex:
        futures = [ex.submit(compute_sequence_flow, *job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="flow"):
            fut.result()
