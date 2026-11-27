"""Supervisely project (Dataset Ninja mirrors: Mapillary Vistas, BDD100K 10K, ...) -> data/processed/<source>/.

Project layout: <root>/meta.json, <root>/<split>/img/*.jpg, <root>/<split>/ann/*.jpg.json
Objects: geometryType 'bitmap' (base64 zlib PNG + origin) or 'polygon' (exterior + interior), one object = one instance.

  python scripts/convert_supervisely.py --config mapillary_vistas --root data/raw/mapillary_vistas/<project> --splits training validation
"""
import argparse
import base64
import json
import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vslib import ROOT  # noqa: E402
from vslib.geometry import rasterize  # noqa: E402
from vslib.mapping import load_source_config, role_of  # noqa: E402
from vslib.writer import SourceWriter, mask_resize, merge_shards, read_image  # noqa: E402


def decode_bitmap(data):
    raw = np.frombuffer(zlib.decompress(base64.b64decode(data)), np.uint8)
    im = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = im[:, :, 3] if im.shape[2] == 4 else im.max(axis=2)
    return im > 0


def object_mask(obj, h0, w0, s, H, W):
    g = obj.get("geometryType")
    if g == "polygon":
        pts = obj["points"]
        ext = np.asarray(pts["exterior"], np.float64) * s
        holes = [np.asarray(q, np.float64) * s for q in pts.get("interior", [])]
        return rasterize([ext], H, W, holes)
    if g == "bitmap":
        bm = decode_bitmap(obj["bitmap"]["data"])
        ox, oy = obj["bitmap"]["origin"]
        full = np.zeros((h0, w0), bool)
        hh, ww = min(bm.shape[0], h0 - oy), min(bm.shape[1], w0 - ox)
        full[oy:oy + hh, ox:ox + ww] = bm[:hh, :ww]
        return mask_resize(full, W, H)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True, help="Supervisely project folder (contains meta.json)")
    ap.add_argument("--splits", nargs="*", help="dataset folders to use (default: all with ann/)")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--source")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--blank-missing", action="store_true", help="TEST ONLY: grey image when the image file is missing")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (one shard each)")
    ap.add_argument("--shard", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    cfg = load_source_config(args.config)
    root = Path(args.root)
    source = args.source or cfg["name"]
    if args.workers > 1 and args.shard is None:
        import shutil
        import subprocess

        shutil.rmtree(Path(args.out) / source, ignore_errors=True)
        base = [sys.executable, __file__] + (argv if argv is not None else sys.argv[1:])
        procs = [subprocess.Popen(base + ["--shard", str(k)]) for k in range(args.workers)]
        if any(p.wait() for p in procs):
            raise SystemExit("a shard failed")
        s = merge_shards(Path(args.out) / source, args.workers)
        print(json.dumps({k: s[k] for k in ("images_seen", "images_ok", "images_review", "images_skipped", "instances_ok")}, ensure_ascii=False))
        return
    tag = f"_{args.shard}" if args.shard is not None else ""
    w = SourceWriter(source, args.out, max_side=cfg.get("max_side", 1024), min_area_640=cfg.get("min_area_640", 64.0),
                     clean=args.shard is None, tag=tag)
    group_re = re.compile(cfg["group_regex"]) if cfg.get("group_regex") else None
    req = set(cfg.get("filter", {}).get("require_any", []))
    split_dirs = [d for d in sorted(root.iterdir()) if (d / "ann").is_dir()]
    if args.splits:
        split_dirs = [d for d in split_dirs if d.name in args.splits]
    items = [(d, a) for d in split_dirs for a in sorted((d / "ann").glob("*.json"))]
    if args.max_images and len(items) > args.max_images:
        items = sorted(random.Random(args.seed).sample(items, args.max_images))
    if args.shard is not None:
        items = items[args.shard::args.workers]
    unmapped, empty_unlabeled = Counter(), 0
    for d, ann_path in tqdm(items, desc=f"{source}{tag}", mininterval=30):
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        objs = ann.get("objects", [])
        if not objs and cfg.get("skip_unannotated", True):
            empty_unlabeled += 1  # test split of a mirror: no labels at all, NOT background
            continue
        if req and not ({o["classTitle"] for o in objs} & req):
            continue
        img_path = d / "img" / ann_path.name[:-5]
        if not img_path.is_file():
            if not args.blank_missing:
                w.skip("image file missing")
                continue
            img = np.full((ann["size"]["height"], ann["size"]["width"], 3), 114, np.uint8)  # TEST ONLY
        else:
            img = read_image(img_path)
        h0, w0 = ann["size"]["height"], ann["size"]["width"]  # annotation coordinates are at this size
        hi, wi = img.shape[:2]
        if (hi, wi) != (h0, w0) and abs(wi / hi - w0 / h0) > 0.01:
            w.skip("image aspect differs from annotation")
            continue
        s = w.scale_for(w0, h0)
        W, H = max(1, round(w0 * s)), max(1, round(h0 * s))
        if (hi, wi) != (H, W):  # original, or pre-shrunk by fetch_sly_tar --max-side
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        raw = []
        painter = cfg.get("z_order") == "painter"
        occ = np.zeros((H, W), bool) if painter else None
        # painter: objects are layered, later ones on top -> visible = own polygon minus everything drawn after it
        seq = list(reversed(objs)) if painter else objs
        for o in seq:
            label = o["classTitle"]
            r = role_of(cfg, label, unmapped)
            if r["role"] == "ignore" and not painter:
                continue
            m = object_mask(o, h0, w0, s, H, W)
            if m is None:
                w.drop(f"unsupported geometry {o.get('geometryType')} ({label})")
                continue
            if painter:
                vis = m & ~occ
                occ |= m
                if r["role"] == "ignore":
                    continue
                m = vis
                if not m.any():
                    continue
            if r["role"] in ("drop", "blackout"):
                from vslib.geometry import area_at_640

                if area_at_640(m.sum(), H, W) < cfg.get("drop_region_min_area_640", 256):
                    continue  # tiny far-away group: scorer ignores it anyway
            raw.append({**r, "label": label, "mask": m})
        grp = None
        if group_re:
            mt = group_re.search(img_path.name)
            grp = mt.group(1) if mt else None
        w.add(img, raw, orig=f"{d.name}/{img_path.name}", group=grp, extra={"split_src": d.name})
    s = w.close({"script": "convert_supervisely.py", "config": cfg["_path"], "root": str(root), "unmapped_labels": dict(unmapped),
                 "skipped_unannotated": empty_unlabeled})
    print(json.dumps({k: s[k] for k in ("images_seen", "images_ok", "images_review", "images_skipped", "instances_ok")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
