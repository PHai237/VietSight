"""COCO instance-segmentation JSON (polygon or RLE) -> data/processed/<source>/ (YOLO-seg, 11 classes).

Works for COCO 2017 and for any COCO export (CVAT "COCO 1.0", Roboflow, Label Studio).

  python scripts/convert_coco.py --config coco2017 --ann data/raw/coco2017/instances_sel.json --images data/raw/coco2017/images
"""
import argparse
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vslib import ROOT  # noqa: E402
from vslib.geometry import rasterize  # noqa: E402
from vslib.mapping import load_source_config, role_of  # noqa: E402
from vslib.writer import SourceWriter, mask_resize, read_image  # noqa: E402


def decode_segm(seg, h0, w0):
    from pycocotools import mask as mu

    if isinstance(seg, list):
        return None  # polygons handled by caller
    if isinstance(seg.get("counts"), list):
        seg = mu.frPyObjects(seg, h0, w0)
    return mu.decode(seg).astype(bool)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/sources/<name>.yaml (name or path)")
    ap.add_argument("--ann", required=True, nargs="+", help="COCO json file(s)")
    ap.add_argument("--images", required=True, nargs="+", help="image folder(s), same order as --ann (or one for all)")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--source", help="output source name (default: config name)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=55)
    args = ap.parse_args(argv)
    cfg = load_source_config(args.config)
    source = args.source or cfg["name"]
    w = SourceWriter(source, args.out, max_side=cfg.get("max_side", 1024), min_area_640=cfg.get("min_area_640", 64.0))
    unmapped = Counter()
    req = set(cfg.get("filter", {}).get("require_any", []))
    group_re = re.compile(cfg["group_regex"]) if cfg.get("group_regex") else None
    items = []
    img_dirs = args.images if len(args.images) == len(args.ann) else args.images * len(args.ann)
    for ann_path, img_dir in zip(args.ann, img_dirs):
        data = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data["categories"]}
        by_img = {}
        for a in data["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for im in data["images"]:
            anns = by_img.get(im["id"], [])
            names = {cats[a["category_id"]] for a in anns}
            if req and not (names & req):
                continue
            items.append((Path(img_dir), im, anns, cats))
    items.sort(key=lambda t: t[1]["file_name"])
    if args.max_images and len(items) > args.max_images:
        items = random.Random(args.seed).sample(items, args.max_images)
    for img_dir, im, anns, cats in tqdm(items, desc=source):
        p = img_dir / im["file_name"]
        if not p.is_file():
            p = img_dir / Path(im["file_name"]).name
        if not p.is_file():
            w.skip("image file missing")
            continue
        img = read_image(p)
        h0, w0 = img.shape[:2]
        if (im.get("height"), im.get("width")) not in ((h0, w0), (None, None)):
            w.skip("image size differs from annotation (EXIF rotation?)")
            continue
        s = w.scale_for(w0, h0)
        W, H = max(1, round(w0 * s)), max(1, round(h0 * s))
        if s < 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        raw = []
        for a in anns:
            label = cats[a["category_id"]]
            r = role_of(cfg, label, unmapped)
            if r["role"] == "ignore":
                continue
            seg = a.get("segmentation")
            if a.get("iscrowd"):
                pol = cfg.get("crowd_policy", "blackout")
                if pol == "drop":
                    raw.append({"role": "drop", "label": f"{label} (iscrowd)", "mask": np.zeros((H, W), bool)})
                elif pol == "blackout" and seg:
                    m = mask_resize(decode_segm(seg, h0, w0), W, H) if isinstance(seg, dict) else rasterize(
                        [np.asarray(q, np.float64).reshape(-1, 2) * s for q in seg if len(q) >= 6], H, W)
                    raw.append({"role": "blackout", "label": f"{label} (iscrowd)", "mask": m})
                continue
            if not seg:
                w.drop(f"no segmentation ({label})")
                continue
            if isinstance(seg, list):
                polys = [np.asarray(q, np.float64).reshape(-1, 2) * s for q in seg if len(q) >= 6]
                m = rasterize(polys, H, W)
            else:
                m = mask_resize(decode_segm(seg, h0, w0), W, H)
            raw.append({**r, "label": label, "mask": m})
        grp = None
        if group_re:
            mt = group_re.search(im["file_name"])
            grp = mt.group(1) if mt else None
        lic = im.get("license")
        w.add(img, raw, orig=im["file_name"], group=grp, extra={"license_id": lic, "src_url": im.get("coco_url") or im.get("flickr_url")})
    s = w.close({"script": "convert_coco.py", "config": cfg["_path"], "ann": args.ann, "unmapped_labels": dict(unmapped)})
    print(json.dumps({k: s[k] for k in ("images_seen", "images_ok", "images_review", "images_skipped", "instances_ok")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
