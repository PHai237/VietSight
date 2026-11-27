"""PROPOSED (pseudo) masks for human correction. READ-ONLY on the input images/labels; writes only to --out.

Nothing here is a reviewed label: every proposal is recorded as status "pseudo_unreviewed" in proposals.jsonl, and the
package goes to CVAT for a person to fix/delete/accept through the data owner's own review flow (Codex V2 decisions with a
named reviewer). Never train on this output directly.

Backends (one or more, merged by mask IoU):
  vietsight:<.pt | .onnx | submission.zip>   11-class VietSight checkpoint (e.g. V1 e25 submission.zip -> model.onnx)
  coco:yolo26n-seg.pt                       COCO-pretrained, mapped by configs/sources/coco2017.yaml + rider pairing
  sam3:<sam3.pt>                            SAM 3 text prompts (weights are gated on Hugging Face, not bundled)

  python scripts/propose_labels.py --images D:/VN/images [--labels D:/VN/labels] --out runs/proposals/vn_b1 \
      --backend vietsight:runs/v1_colab_e25/submission.zip coco:yolo26n-seg.pt \
      --classes Sign "Traffic light" Motorcycle Motorcyclist "Other vehicle"
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from prelabel import predict_masks  # noqa: E402
from vslib import CLASS_ID, CLASSES, IMG_EXTS  # noqa: E402
from vslib.geometry import mask_to_polygon, normalize, rasterize, yolo_line  # noqa: E402
from vslib.instances import resolve  # noqa: E402
from vslib.mapping import load_source_config, role_of  # noqa: E402
from vslib.writer import mask_resize, read_image  # noqa: E402

# SAM 3 concept prompts -> roles (same roles as the source configs, so riders go through vslib.instances.resolve).
# Other vehicle prompts follow the BTC excerpt (ba gác, xích lô, trailer, construction vehicle); hand carts are NOT labelled.
SAM3_PROMPTS = {
    "traffic sign": {"role": "direct", "target": "Sign"},
    "traffic light": {"role": "direct", "target": "Traffic light"},
    "motorcycle": {"role": "vehicle2", "sub": "motorcycle"},
    "bicycle": {"role": "vehicle2", "sub": "bicycle"},
    "person": {"role": "person"},
    "car": {"role": "direct", "target": "Car"},
    "bus": {"role": "direct", "target": "Bus"},
    "truck": {"role": "direct", "target": "Truck"},
    "three-wheeled cargo motorbike": {"role": "direct", "target": "Other vehicle"},
    "cyclo": {"role": "direct", "target": "Other vehicle"},
    "trailer": {"role": "direct", "target": "Other vehicle"},
    "construction vehicle": {"role": "direct", "target": "Other vehicle"},
}


def mask_iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


class Backend:
    def __init__(self, spec, conf):
        self.kind, _, path = spec.partition(":")
        self.name, self.conf, self.tmp = spec, conf, None
        if self.kind == "sam3":
            from ultralytics.models.sam import SAM3SemanticPredictor

            self.sam = SAM3SemanticPredictor(overrides=dict(conf=conf, task="segment", mode="predict", model=path, half=False, save=False, verbose=False))
            return
        from ultralytics import YOLO

        if path.endswith(".zip"):  # a submission.zip: use its model.onnx as-is (CPU onnxruntime)
            self.tmp = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(path) as zf:
                zf.extract("model.onnx", self.tmp)
            path = str(self.tmp / "model.onnx")
        self.model = YOLO(path, task="segment")
        self.cfg = load_source_config("coco2017") if self.kind == "coco" else None
        if self.kind == "vietsight":
            names = [self.model.names[i] for i in sorted(self.model.names)] if self.model.names else None
            assert names in (None, CLASSES), f"{spec}: not an 11-class VietSight model ({names})"

    def predict(self, img):
        """-> (list of {class, mask, score}, flags)"""
        H, W = img.shape[:2]
        if self.kind == "vietsight":
            return [{"class": CLASSES[CLASS_ID[n]] if n in CLASS_ID else CLASSES[int(n)], "mask": m, "score": s}
                    for n, m, s in predict_masks(self.model, img, self.conf)], []
        if self.kind == "coco":
            raw = [{**role_of(self.cfg, n), "label": n, "mask": m, "score": s} for n, m, s in predict_masks(self.model, img, self.conf)]
        else:
            prompts = list(SAM3_PROMPTS)
            self.sam.set_image(img)  # BGR numpy, ultralytics convention
            r = self.sam(text=prompts)[0]
            raw = []
            if r.masks is not None:
                for m, c, s in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
                    raw.append({**SAM3_PROMPTS[prompts[c]], "label": prompts[c], "mask": mask_resize(m > 0.5, W, H), "score": float(s)})
        raw = [r for r in raw if r["role"] not in ("ignore", "drop", "blackout")]
        insts, flags, _ = resolve(raw, H, W)
        out = []
        for cid, m in insts:  # scores are not carried through rider pairing: best overlapping raw score
            sc = max((r["score"] for r in raw if (r["mask"] & m).sum() > 0.5 * r["mask"].sum()), default=None)
            out.append({"class": CLASSES[cid], "mask": m, "score": sc})
        return out, flags

    def close(self):
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)


def read_labels(lp, H, W):
    """Existing label file -> (kept polygon lines, list of (class, mask) occupied, list of (class, box xyxy px) bbox-only)."""
    lines, occ, boxes = [], [], []
    if lp is None or not lp.is_file():
        return lines, occ, boxes
    for ln in lp.read_text(encoding="utf-8").splitlines():
        v = ln.split()
        if len(v) == 5:  # bbox-only source: not a valid seg label -> hint only, not copied
            c, x, y, w, h = int(v[0]), *map(float, v[1:])
            boxes.append((c, ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)))
        elif len(v) >= 7:
            lines.append(ln.strip())
            occ.append((int(v[0]), rasterize([np.asarray(v[1:], float).reshape(-1, 2) * [W, H]], H, W)))
    return lines, occ, boxes


def box_iou(m, b):
    ys, xs = np.nonzero(m)
    a = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    iw, ih = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", type=Path, help="existing YOLO labels (read-only); polygons kept, 5-token boxes used as hints")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--backend", nargs="+", required=True)
    ap.add_argument("--classes", nargs="+", default=["Sign", "Traffic light", "Motorcycle", "Motorcyclist", "Other vehicle"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-merge", type=float, default=0.5)
    ap.add_argument("--all", action="store_true", help="also package images that got no proposal")
    a = ap.parse_args()
    for c in a.classes:
        assert c in CLASSES, c
    out = a.out.resolve()
    for src in filter(None, (a.images, a.labels)):
        assert not out.is_relative_to(src.resolve()) and not src.resolve().is_relative_to(out), f"--out must be outside {src}"
    if out.exists():
        raise SystemExit(f"{out} exists: choose a new --out (proposals are never overwritten)")
    (out / "images" / "train").mkdir(parents=True)
    (out / "labels" / "train").mkdir(parents=True)
    backends = [Backend(s, a.conf) for s in a.backend]
    imgs = sorted(p for p in a.images.rglob("*") if p.suffix.lower() in IMG_EXTS)
    rows, counts = [], {c: 0 for c in a.classes}
    for p in tqdm(imgs, desc="propose"):
        img = read_image(p)
        H, W = img.shape[:2]
        lp = (a.labels / p.relative_to(a.images)).with_suffix(".txt") if a.labels else None
        lines, occ, boxes = read_labels(lp, H, W)
        cands, flags = [], []
        for b in backends:
            preds, fl = b.predict(img)
            flags += [f"{b.kind}: {f}" for f in fl]
            cands += [{**x, "backend": b.kind} for x in preds if x["class"] in a.classes]
        cands.sort(key=lambda x: -(x["score"] or 0))
        kept = []
        for c in cands:
            if any(c["mask"][m].sum() > 0.3 * c["mask"].sum() for _, m in occ):  # already labelled by a human/source mask
                continue
            same = [k for k in kept if mask_iou(k["mask"], c["mask"]) > a.iou_merge]
            if same:
                same[0]["agree"].add(c["backend"])
                if same[0]["class"] != c["class"]:
                    same[0]["alt_class"].add(f"{c['backend']}:{c['class']}")
                continue
            poly = mask_to_polygon(c["mask"])
            if poly is not None:
                kept.append({**c, "poly": poly, "agree": {c["backend"]}, "alt_class": set()})
        if not kept and not a.all:
            continue
        name = p.relative_to(a.images).as_posix().replace("/", "__")
        shutil.copyfile(p, out / "images" / "train" / name)
        new = [yolo_line(CLASS_ID[k["class"]], normalize(k["poly"], H, W)) for k in kept]
        (out / "labels" / "train" / (Path(name).stem + ".txt")).write_text("\n".join(lines + new) + ("\n" if lines or new else ""), encoding="utf-8")
        props = []
        for i, k in enumerate(kept):
            hint = max(((box_iou(k["mask"], bb), CLASSES[c] if 0 <= c < len(CLASSES) else c) for c, bb in boxes), default=None, key=lambda t: t[0])
            props.append({"line": len(lines) + i, "class": k["class"], "score": None if k["score"] is None else round(k["score"], 3),
                          "backends": sorted(k["agree"]), "alt_class": sorted(k["alt_class"]),
                          "source_box": {"iou": round(float(hint[0]), 3), "class": hint[1]} if hint and hint[0] > 0.3 else None,
                          "status": "pseudo_unreviewed"})
            counts[k["class"]] += 1
        rows.append({"file": name, "source_image": str(p), "source_label": str(lp) if lp else None,
                     "source_label_sha256": hashlib.sha256(lp.read_bytes()).hexdigest() if lp and lp.is_file() else None,
                     "orig_polygon_lines": len(lines), "source_boxes": len(boxes), "proposals": props, "flags": sorted(set(flags))})
    for b in backends:
        b.close()
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "data.yaml").write_text(f"path: .\ntrain: images/train\nnames:\n{names}\n", encoding="utf-8")
    (out / "train.txt").write_text("".join(f"images/train/{r['file']}\n" for r in rows), encoding="utf-8")
    (out / "proposals.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "REVIEW_FLAGS.tsv").write_text("".join(
        f"{r['file']}\t" + " | ".join([f"line {x['line']} {x['class']} {x['backends']}" + (f" alt {x['alt_class']}" if x['alt_class'] else "") for x in r["proposals"]] + r["flags"]) + "\n"
        for r in rows), encoding="utf-8")
    (out / "README.txt").write_text(
        "PSEUDO-LABELS, NOT REVIEWED. Lines listed in proposals.jsonl were proposed by models and must be checked by a person.\n"
        f"backends: {a.backend}\nclasses: {a.classes}\nconf: {a.conf}\n"
        "CVAT: create task from images/train, then Upload annotations > YOLO Segmentation 1.0 with cvat_upload.zip.\n", encoding="utf-8")
    with zipfile.ZipFile(out / "cvat_upload.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ["data.yaml", "train.txt", *[f"labels/train/{Path(r['file']).stem}.txt" for r in rows]]:
            zf.write(out / f, f)
    print(json.dumps({"images_seen": len(imgs), "images_with_proposals": sum(1 for r in rows if r["proposals"]),
                      "proposals": counts, "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
