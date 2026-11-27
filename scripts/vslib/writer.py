"""Write one converted source to data/processed/<source>/ in YOLO-seg format + provenance.

Layout:
  images/<source>__<stem>__<hash8>.jpg     accepted images (resized to max side ``max_side``)
  labels/<same>.txt                        YOLO-seg, one polygon per line, validated with the OFFICIAL notebook checks
  review/images, review/labels             AMBIGUOUS images (draft labels) -> a human fixes them in CVAT, then
                                           scripts/review_queue.py accept moves them to images/labels
  meta.jsonl                               one line per image: file, original id/path, group (video/sequence), status, flags
  summary.json                             counts per class, dropped labels and why, flags
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from . import CLASSES
from .geometry import area_at_640, largest_part_polygon, mask_to_polygon, normalize, yolo_line
from .instances import resolve
from .notebook_lib import load as load_nb


def safe_name(source, orig):
    stem = Path(str(orig)).stem
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")[:60] or "img"
    h = hashlib.sha1(str(orig).encode("utf-8")).hexdigest()[:8]
    return f"{source}__{clean}__{h}"


def read_image(path):
    """BGR uint8, EXIF orientation applied (same as the platform: PIL exif_transpose)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


class SourceWriter:
    def __init__(self, source, out_root, max_side=1024, min_area_640=64.0, jpeg_quality=92, clean=True, tag=""):
        self.source = source
        self.tag = tag  # shard suffix: meta<tag>.jsonl / summary<tag>.json (merged by merge_shards)
        self.out = Path(out_root) / source
        if clean and self.out.exists():  # converters are deterministic: always rebuild from scratch
            import shutil

            shutil.rmtree(self.out)
        for sub in ("images", "labels", "review/images", "review/labels"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.max_side, self.min_area, self.q = max_side, min_area_640, jpeg_quality
        self.meta = open(self.out / f"meta{tag}.jsonl", "w", encoding="utf-8")
        self.counts = Counter()
        self.review_counts = Counter()
        self.dropped = Counter()  # reason -> n
        self.flags = Counter()
        self.n_in = self.n_ok = self.n_review = self.n_skip = 0
        nb = load_nb()
        self._problem = nb["polygon_problem"]
        self._area = nb["area_at_imgsz"]  # the notebook's (shoelace) area at 640 scale, same rule as its report

    def scale_for(self, w, h):
        return min(1.0, self.max_side / max(w, h))

    def drop(self, reason, n=1):
        self.dropped[reason] += n

    def add(self, img_bgr, raw, orig, group=None, extra=None, force_review=False):
        """``img_bgr`` already resized with scale_for(); ``raw`` instances with masks at that size."""
        self.n_in += 1
        h, w = img_bgr.shape[:2]
        # 'blackout' regions (crowd / group of objects that can not be split into instances): paint them grey (the
        # letterbox colour 114) so the model never sees unlabeled objects as background; instances lose those pixels.
        black = [r for r in raw if r["role"] == "blackout"]
        if black:
            bm = np.zeros((h, w), bool)
            for r in black:
                bm |= r["mask"]
            img_bgr = img_bgr.copy()
            img_bgr[bm] = 114
            raw = [dict(r, mask=r["mask"] & ~bm) for r in raw if r["role"] != "blackout"]
            self.drop("group/crowd region greyed out (image kept)", len(black))
        insts, flags, st = resolve(raw, h, w, self.min_area)
        self.drop("smaller than 64 px at 640 (scorer ignores)", st.get("dropped_small", 0))
        if force_review:
            flags = flags + ["forced review"]
        lines, cls_count = [], Counter()
        for cid, m in insts:
            poly = mask_to_polygon(m)
            pn = normalize(poly, h, w) if poly is not None else None
            if pn is None or self._problem(pn, w, h):
                poly = largest_part_polygon(m)
                pn = normalize(poly, h, w) if poly is not None else None
                if pn is None or self._problem(pn, w, h):
                    self.drop(f"invalid polygon after conversion ({CLASSES[cid]})")
                    continue
                self.drop(f"multi-part mask reduced to largest part ({CLASSES[cid]}) [kept]")
            if self._area(pn, w, h, 640) < self.min_area:
                self.drop("smaller than 64 px at 640 (scorer ignores)")
                continue
            lines.append(yolo_line(cid, pn))
            cls_count[CLASSES[cid]] += 1
        name = safe_name(self.source, orig)
        status = "review" if flags else "ok"
        sub = "review/" if flags else ""
        cv2.imwrite(str(self.out / f"{sub}images" / f"{name}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        (self.out / f"{sub}labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if flags:
            self.n_review += 1
            self.review_counts.update(cls_count)
            for f in flags:
                self.flags[re.sub(r"'[^']*'", "'*'", f)] += 1
        else:
            self.n_ok += 1
            self.counts.update(cls_count)
        rec = {"file": f"{name}.jpg", "orig": str(orig), "group": group if group is not None else name, "status": status,
               "flags": flags, "w": w, "h": h, "instances": dict(cls_count)}
        if extra:
            rec.update(extra)
        self.meta.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return status

    def skip(self, reason):
        self.n_skip += 1
        self.drop(f"image skipped: {reason}")

    def close(self, info=None):
        self.meta.close()
        summary = {"source": self.source, "images_seen": self.n_in + self.n_skip, "images_ok": self.n_ok, "images_review": self.n_review,
                   "images_skipped": self.n_skip, "instances_ok": {c: self.counts.get(c, 0) for c in CLASSES},
                   "instances_review": {c: self.review_counts.get(c, 0) for c in CLASSES}, "dropped": dict(self.dropped.most_common()),
                   "review_flags": dict(self.flags.most_common()), "max_side": self.max_side, "min_area_640": self.min_area}
        if info:
            summary["converter"] = info
        (self.out / f"summary{self.tag}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
        return summary


def mask_resize(mask, w, h):
    if mask.shape[1] == w and mask.shape[0] == h:
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) >= 128


def merge_shards(out_dir, n):
    """Concatenate meta_<k>.jsonl and add up summary_<k>.json of a sharded conversion."""
    out_dir = Path(out_dir)
    total = None
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as meta:
        for k in range(n):
            mp, sp = out_dir / f"meta_{k}.jsonl", out_dir / f"summary_{k}.json"
            meta.write(mp.read_text(encoding="utf-8"))
            s = json.loads(sp.read_text(encoding="utf-8"))
            if total is None:
                total = s
            else:
                for key, v in s.items():
                    if isinstance(v, int) and key in total:
                        total[key] += v
                    elif isinstance(v, dict) and key in ("instances_ok", "instances_review", "dropped", "review_flags"):
                        for kk, vv in v.items():
                            total[key][kk] = total[key].get(kk, 0) + vv
                    elif key == "converter":
                        for kk in ("unmapped_labels",):
                            for lab, c in v.get(kk, {}).items():
                                total[key][kk][lab] = total[key][kk].get(lab, 0) + c
                        total[key]["skipped_unannotated"] = total[key].get("skipped_unannotated", 0) + v.get("skipped_unannotated", 0)
            mp.unlink()
            sp.unlink()
    total["shards"] = n
    (out_dir / "summary.json").write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
    return total
