"""Per-frame I3D two-stream features (Kinetics, ImageNet-inflated), MS-TCN style:
a 21-frame window centred on every frame, average-pooled Mixed_5c output,
RGB 1024-d + flow 1024-d -> 2048-d. Saved as float32 (2048, T)."""
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .common import I3D_DIR, load_pickle

WINDOW = 21
CROP = 224


def load_i3d(device, i3d_dir=I3D_DIR):
    weights = {m: i3d_dir / "models" / f"{m}_imagenet.pt" for m in ("rgb", "flow")}
    if not (i3d_dir / "pytorch_i3d.py").exists() or not all(p.exists() for p in weights.values()):
        raise FileNotFoundError(f"pytorch-i3d not found in {i3d_dir}; run `python -m oakink2_actseg.download`.")
    sys.path.insert(0, str(i3d_dir))
    from pytorch_i3d import InceptionI3d

    models = {}
    for mode, channels in (("rgb", 3), ("flow", 2)):
        net = InceptionI3d(400, in_channels=channels)
        net.load_state_dict(torch.load(weights[mode], map_location="cpu"))
        models[mode] = net.eval().to(device)
    return models


def _embed(net, x):
    for name in net.VALID_ENDPOINTS:
        if name in net.end_points:
            x = net._modules[name](x)
    return x.mean(dim=(2, 3, 4))


def _decode(jpgs, flags):
    with ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), flags), jpgs)))


def _crop(x, crop):
    """x: (T, H, W, C)."""
    if crop == "full":
        return x
    h, w = x.shape[1:3]
    if min(h, w) < CROP:
        raise ValueError(f"frames are {h}x{w}, need short side >= {CROP} for center crop")
    top, left = (h - CROP) // 2, (w - CROP) // 2
    return x[:, top:top + CROP, left:left + CROP]


@torch.inference_mode()
def extract_sequence(models, frames_pack, flow_pack, frame_ids, device, crop="center", batch_size=16):
    pack = load_pickle(frames_pack)
    jpg_of = dict(zip(pack["frame_ids"], pack["jpgs"]))
    flow = load_pickle(flow_pack)
    if list(flow["frame_ids"]) != list(frame_ids):
        raise RuntimeError(f"{flow_pack} was computed for different frames; delete it and re-run")

    rgb = _decode([jpg_of[i] for i in frame_ids], cv2.IMREAD_COLOR)[..., ::-1]
    fxy = np.stack([_decode(flow["x"], cv2.IMREAD_GRAYSCALE), _decode(flow["y"], cv2.IMREAD_GRAYSCALE)], -1)
    rgb = torch.from_numpy(np.ascontiguousarray(_crop(rgb, crop))).to(device)
    fxy = torch.from_numpy(np.ascontiguousarray(_crop(fxy, crop))).to(device)

    T = rgb.shape[0]
    offsets = torch.arange(WINDOW, device=device) - WINDOW // 2
    use_amp = device.type == "cuda"
    feats = []
    for start in range(0, T, batch_size):
        centers = torch.arange(start, min(start + batch_size, T), device=device)
        idx = (centers[:, None] + offsets[None]).clamp_(0, T - 1)
        # uint8 [0, 255] -> [-1, 1]; (B, W, H, W, C) -> (B, C, W, H, W)
        clip_rgb = rgb[idx].permute(0, 4, 1, 2, 3).float().div_(127.5).sub_(1)
        clip_flow = fxy[idx].permute(0, 4, 1, 2, 3).float().div_(127.5).sub_(1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            f = torch.cat([_embed(models["rgb"], clip_rgb), _embed(models["flow"], clip_flow)], 1)
        feats.append(f.float().cpu())
    return torch.cat(feats).numpy().T.astype(np.float32)  # (2048, T)


def extract_features(jobs, device, crop="center", batch_size=16):
    """jobs: list of (frames_pack, flow_pack, frame_ids, out_npy)."""
    if not jobs:
        return
    models = load_i3d(device)
    for frames_pack, flow_pack, frame_ids, out_npy in tqdm(jobs, desc="i3d"):
        feat = extract_sequence(models, frames_pack, flow_pack, frame_ids, device, crop, batch_size)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_npy.with_name(out_npy.stem + ".tmp.npy")
        np.save(tmp, feat)
        tmp.replace(out_npy)
