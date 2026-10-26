"""Freeze copies of finished DiffAct / FACT / MS-TCN runs so later runs cannot overwrite them.

    python -m oakink2_actseg.snapshot --tag real
    -> runs/snapshots/real/{diffact,fact,mstcn}/{left,right}/  plus  runs/snapshots/real/manifest.json

Re-running adds models that are not in the snapshot yet and leaves existing copies untouched
(--force re-copies them). The manifest records the checkpoint to continue training from:
DiffAct latest.pt, FACT's latest ckpts/network.iter-N.net, MS-TCN's last models/epoch-N.model.
"""
import argparse
import json
import shutil

from .common import FACT_DIR, HANDS, RUNS_DIR, dataset_name, prim5_dataset_name, resolve_cameras

MODELS = ("diffact", "fact", "mstcn")


def _newest(paths, key):
    paths = list(paths)
    return max(paths, key=key) if paths else None


def _source(model, ds):
    """(run folder, init checkpoint relative to it, extra manifest entries) of a finished run, or None."""
    if model == "diffact":
        src = RUNS_DIR / "diffact" / f"{ds}-S1"
        return (src, "latest.pt", {}) if (src / "latest.pt").exists() else None
    if model == "fact":
        best = _newest((FACT_DIR.parent / "log" / ds / "split1").glob("*/*/best_ckpt.gz"), lambda p: p.stat().st_mtime)
        if best is None:
            return None
        ckpt = _newest((best.parent / "ckpts").glob("network.iter-*.net"), lambda p: int(p.stem.split("-")[-1]))
        return best.parent, ckpt.relative_to(best.parent) if ckpt else None, {"fact_best_ckpt": "best_ckpt.gz"}
    src = RUNS_DIR / "mstcn" / f"{ds}-S1"
    ckpt = _newest((src / "models").glob("epoch-*.model"), lambda p: int(p.stem.split("-")[-1]))
    finished = ckpt is not None and (src / "results").is_dir() and any((src / "results").iterdir())
    return (src, ckpt.relative_to(src), {}) if finished else None


def snapshot(tag, datasets, models=MODELS, force=False):
    out = RUNS_DIR / "snapshots" / tag
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for hand, ds in datasets.items():
        entry = manifest.setdefault(hand, {"dataset": ds})
        for model in models:
            dst = out / model / hand
            if dst.exists() and not force:
                print(f"[snapshot] {model}/{hand} already in {out}, kept")
                continue
            found = _source(model, ds)
            if found is None:
                print(f"[snapshot] no finished {model} run for {hand} ({ds})")
                continue
            src, init, extra = found
            shutil.copytree(src, dst, dirs_exist_ok=force)
            entry[f"{model}_run"] = str(dst)
            entry[f"{model}_init"] = str(dst / init) if init else None
            entry.update({k: str(dst / v) for k, v in extra.items()})
            print(f"[snapshot] {model}/{hand}: {src} -> {dst}")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="snapshot name, e.g. real")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--camera", default="all")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--crop", default="center")
    parser.add_argument("--dataset", default=None,
                        help="base dataset name instead of the default real one; _prim5_<hand> is appended")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base = args.dataset or dataset_name(resolve_cameras(args.camera), args.fps, True, args.crop)
    snapshot(args.tag, {h: prim5_dataset_name(base, h) for h in HANDS}, args.models, args.force)


if __name__ == "__main__":
    main()
