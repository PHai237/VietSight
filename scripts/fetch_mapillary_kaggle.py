"""Select Mapillary Vistas v2.0 images from an ATTACHED copy of the official release (no Internet needed) and lay them out
like the Supervisely mirror, so scripts/convert_supervisely.py runs unchanged.

Kaggle: Add Input -> kaggleprollc/mapillary-vistas-image-data-collection (official "Research edition v2.0" layout:
<root>/{training,validation}/images/*.jpg and <root>/{training,validation}/v2.0/polygons/*.json).
Official polygons are in annotation order (background -> camera) = the order the mirror kept -> z_order: painter.
Class names: config_v2.0.json 'readable' lower-cased (= the mirror's classTitle, verified on 7Y2EDQH4E9SPLX6Zds_0sw: same
153 objects in the same order).

  python scripts/fetch_mapillary_kaggle.py "/kaggle/input/<slug>/Mapillary Vistas" /tmp/raw/mapillary_vistas \
      --splits training validation --prefer motorcycle motorcyclist bicycle bicyclist "other rider" \
      --other-frac 0.15 --max-keep training=1700 validation=300
"""
import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sly_tar import keep_ann  # noqa: E402  (same deterministic selection rule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with config_v2.0.json and training/ validation/ (glob allowed)")
    ap.add_argument("dst")
    ap.add_argument("--splits", nargs="*", default=["training", "validation"])
    ap.add_argument("--prefer", nargs="*", default=[])
    ap.add_argument("--other-frac", type=float, default=1.0)
    ap.add_argument("--max-keep", nargs="*", default=[])
    args = ap.parse_args()
    roots = sorted(glob.glob(args.root)) or [args.root]
    root = Path(roots[0])
    cfg = json.loads((root / "config_v2.0.json").read_text(encoding="utf-8"))
    readable = {lab["name"]: lab["readable"].lower() for lab in cfg["labels"]}
    caps = {k: int(v) for k, v in (x.split("=", 1) for x in args.max_keep)}
    dst = Path(args.dst)
    prefer = set(args.prefer)
    for split in args.splits:
        polys = sorted((root / split / "v2.0" / "polygons").glob("*.json"))
        (dst / split / "ann").mkdir(parents=True, exist_ok=True)
        (dst / split / "img").mkdir(parents=True, exist_ok=True)
        kept = seen = 0
        for p in polys:
            if kept >= caps.get(split, 1 << 30):
                break
            seen += 1
            a = json.loads(p.read_text(encoding="utf-8"))
            objs = [{"classTitle": readable.get(o["label"], o["label"]), "geometryType": "polygon",
                     "points": {"exterior": o["polygon"], "interior": []}} for o in a["objects"]]
            sly = {"size": {"height": a["height"], "width": a["width"]}, "objects": objs, "tags": []}
            if not keep_ann(p.stem, sly, prefer, args.other_frac):
                continue
            img = root / split / "images" / f"{p.stem}.jpg"
            if not img.is_file():
                continue
            out_img = dst / split / "img" / img.name
            if not out_img.exists():
                try:
                    os.symlink(img, out_img)
                except OSError:
                    shutil.copyfile(img, out_img)
            (dst / split / "ann" / f"{img.name}.json").write_text(json.dumps(sly), encoding="utf-8")
            kept += 1
        print(f"{split}: {kept} kept of {seen} read ({len(polys)} available)", flush=True)
    (dst / "SOURCE.txt").write_text(f"official Mapillary Vistas v2.0 from {root} (Research Use License)\n", encoding="utf-8")


if __name__ == "__main__":
    main()
