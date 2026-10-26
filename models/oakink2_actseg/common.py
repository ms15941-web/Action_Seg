import json
import os
import pickle
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = MODELS_DIR / "realdata" / "OakInk2"
I3D_DIR = MODELS_DIR / "third_party" / "pytorch-i3d"
DIFFACT_DIR = MODELS_DIR / "DiffAct"
FACT_DIR = MODELS_DIR / "CVPR2024-FACT-main"
MSTCN_DIR = MODELS_DIR / "ms-tcn"
RUNS_DIR = MODELS_DIR / "runs"

HF_REPO_ID = "kelvin34501/OakInk-v2"
CAMERAS = ("egocentric", "allocentric_top", "allocentric_left", "allocentric_right")
COLOR_FPS = 30

BACKGROUND = "background"
# Painted first so that a manipulation primitive overlapping them wins the frame.
TRANSIENT_PRIMITIVES = ("hold", "rearrange", "swap")
IGNORED_PRIMITIVES = ("?(unk)",)

# Per-hand functional primitives (labels v4 from motion capture), see oakink2_synth/README.md.
DEFAULT_PRIM5_DIR = MODELS_DIR / "oakink2_synth"
PRIM5_CLASSES = ("idle", "reach", "reposition", "transport", "stabilize")
HANDS = ("left", "right")


def save_pickle_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def save_json_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def all_sequences(data_root):
    program_dir = Path(data_root) / "program" / "program_info"
    if not program_dir.is_dir():
        raise FileNotFoundError(
            f"{program_dir} not found. Run `python -m oakink2_actseg.download` first."
        )
    return sorted(p.stem for p in program_dir.glob("*.json"))


def frames_pack_path(data_root, camera, seq):
    return Path(data_root) / "frames" / camera / f"{seq}.pkl"


def frames_ids_path(data_root, camera, seq):
    return Path(data_root) / "frames" / camera / f"{seq}.ids.json"


def downloaded_sequences(data_root, camera):
    frame_dir = Path(data_root) / "frames" / camera
    if not frame_dir.is_dir():
        return []
    return sorted(
        p.name[: -len(".ids.json")]
        for p in frame_dir.glob("*.ids.json")
        if frames_pack_path(data_root, camera, p.name[: -len(".ids.json")]).exists()
    )


def resolve_cameras(camera):
    """'all' or a comma-separated list -> camera names in canonical order."""
    names = CAMERAS if camera == "all" else [c.strip() for c in camera.split(",") if c.strip()]
    unknown = sorted(set(names) - set(CAMERAS))
    if unknown or not names:
        raise ValueError(f"unknown camera(s) {unknown}; choose from {CAMERAS} or 'all'")
    return [c for c in CAMERAS if c in names]


def camera_tag(cameras):
    if len(cameras) == 1:
        return cameras[0]
    return "allviews" if len(cameras) == len(CAMERAS) else "+".join(cameras)


VIEW_SEP = "@"


def video_name(seq, camera, cameras):
    """One training video per (sequence, camera); single-camera datasets keep plain sequence names."""
    return seq if len(cameras) == 1 else f"{seq}{VIEW_SEP}{camera}"


def sampling_tag(camera, fps, trim):
    return f"{camera}_{fps}fps" + ("" if trim else "_untrimmed")


def dataset_name(cameras, fps, trim, crop):
    return f"oakink2_{camera_tag(cameras)}_{fps}fps" + ("" if trim else "_untrimmed") + f"_{crop}"


def prim5_dataset_name(base_name, hand):
    """Per-hand five-primitive dataset that shares the base dataset's features."""
    return f"{base_name}_prim5_{hand}"


def processed_root(data_root):
    return Path(data_root) / "processed"


def flow_pack_path(data_root, camera, fps, trim, seq):
    return Path(data_root) / "flow" / sampling_tag(camera, fps, trim) / f"{seq}.pkl"


def select_frame_ids(data_root, seq, frame_ids, fps, trim):
    """Image frame ids (30fps clock in 120Hz units) kept for one sequence."""
    if COLOR_FPS % fps != 0:
        raise ValueError(f"fps must divide {COLOR_FPS}, got {fps}")
    ids = sorted(int(i) for i in frame_ids)
    if trim:
        endpoint = load_json(Path(data_root) / "program_extension" / "endpoint" / f"{seq}.json")
        ids = [i for i in ids if endpoint["seq_beg"] <= i <= endpoint["seq_end"]]
    return ids[:: COLOR_FPS // fps]


def default_num_workers():
    for key in ("SLURM_CPUS_PER_TASK", "PBS_NUM_PPN", "NSLOTS"):
        if os.environ.get(key):
            return max(1, int(os.environ[key]))
    # os.cpu_count() reports the whole node; the affinity mask is what the job may actually use.
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)
