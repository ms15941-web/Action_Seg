"""Freeze copies of finished DiffAct / FACT runs so later runs cannot overwrite them.

    python -m oakink2_actseg.snapshot --tag real
    -> runs/snapshots/real/{diffact,fact}/{left,right}/  plus  runs/snapshots/real/manifest.json

The manifest records the checkpoint to continue training from:
DiffAct latest.pt (its last saved state) and FACT's latest ckpts/network.iter-N.net.
"""
import argparse
import json
import shutil
from pathlib import Path

from .common import FACT_DIR, HANDS, RUNS_DIR, dataset_name, prim5_dataset_name, resolve_cameras


def _latest_fact_ckpt(run_dir):
    ckpts = list((run_dir / "ckpts").glob("network.iter-*.net"))
    return max(ckpts, key=lambda p: int(p.stem.split("-")[-1])) if ckpts else None


def snapshot(tag, datasets, force=False):
    out = RUNS_DIR / "snapshots" / tag
    if out.exists() and not force:
        raise SystemExit(f"{out} already exists; pick another --tag or pass --force")
    manifest = {}
    for hand, ds in datasets.items():
        entry = {"dataset": ds}
        src = RUNS_DIR / "diffact" / f"{ds}-S1"
        if (src / "latest.pt").exists():
            dst = out / "diffact" / hand
            shutil.copytree(src, dst, dirs_exist_ok=force)
            entry["diffact_run"] = str(dst)
            entry["diffact_init"] = str(dst / "latest.pt")
        else:
            print(f"[snapshot] no DiffAct run for {hand} in {src}")
        runs = sorted((FACT_DIR.parent / "log" / ds / "split1").glob("*/*/best_ckpt.gz"),
                      key=lambda p: p.stat().st_mtime)
        if runs:
            src = runs[-1].parent
            dst = out / "fact" / hand
            shutil.copytree(src, dst, dirs_exist_ok=force)
            entry["fact_run"] = str(dst)
            entry["fact_best_ckpt"] = str(dst / "best_ckpt.gz")
            latest = _latest_fact_ckpt(dst)
            entry["fact_init"] = str(latest) if latest else None
        else:
            print(f"[snapshot] no finished FACT run for {hand} under log/{ds}/split1")
        manifest[hand] = entry
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="snapshot name, e.g. real")
    parser.add_argument("--camera", default="all")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--crop", default="center")
    parser.add_argument("--dataset", default=None,
                        help="base dataset name instead of the default real one; _prim5_<hand> is appended")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base = args.dataset or dataset_name(resolve_cameras(args.camera), args.fps, True, args.crop)
    snapshot(args.tag, {h: prim5_dataset_name(base, h) for h in HANDS}, args.force)


if __name__ == "__main__":
    main()
