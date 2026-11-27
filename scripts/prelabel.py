"""AI pre-label: model predictions -> DRAFT YOLO-seg labels in data/processed/<source>/review/ (status 'review').
Drafts are NEVER used for training until a human corrected them (review_queue.py export -> CVAT -> accept).

  # new images (team photos of Vietnamese streets), with a trained VietSight checkpoint (11 classes)
  python scripts/prelabel.py new --images data/raw/vn_team/batch1 --source vn_team_b1 --model runs/v1/best.pt
  # same with the COCO-pretrained yolo26n-seg.pt (80 classes): mapped with configs/sources/coco2017.yaml + rider pairing
  python scripts/prelabel.py new --images ... --source ... --model yolo26n-seg.pt
  # fill classes a source never labelled (BDD100K has no Sign / Traffic light): only adds those classes, image -> review
  python scripts/prelabel.py fill --source bdd100k_10k --model runs/v1/best.pt --classes Sign "Traffic light"
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vslib import CLASS_ID, CLASSES, IMG_EXTS, ROOT  # noqa: E402
from vslib.geometry import mask_to_polygon, normalize, rasterize, yolo_line  # noqa: E402
from vslib.mapping import load_source_config, role_of  # noqa: E402
from vslib.writer import SourceWriter, mask_resize, read_image  # noqa: E402


def predict_masks(model, img, conf, imgsz=640):
    r = model.predict(img, imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    H, W = img.shape[:2]
    out = []
    for m, c, s in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        out.append((model.names[int(c)], mask_resize(m > 0.5, W, H), float(s)))
    return out


def cmd_new(a, model):
    competition = [model.names[i] for i in sorted(model.names)] == CLASSES
    cfg = None if competition else load_source_config("coco2017")
    w = SourceWriter(a.source, a.out, max_side=a.max_side)
    imgs = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
    for p in tqdm(imgs, desc=a.source):
        img = read_image(p)
        s = w.scale_for(img.shape[1], img.shape[0])
        if s < 1:
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        raw = []
        for name, m, sc in predict_masks(model, img, a.conf):
            if competition:  # the model already decided rider / no rider: keep its class
                raw.append({"role": "direct", "target": name, "mask": m, "label": name, "score": sc})
            else:
                r = role_of(cfg, name)
                if r["role"] not in ("ignore", "drop", "blackout"):
                    raw.append({**r, "label": name, "mask": m, "score": sc})
        grp = p.parent.name if a.group_from_parent else None
        w.add(img, raw, orig=p.relative_to(a.images).as_posix(), group=grp, extra={"prelabel_model": str(a.model)}, force_review=True)
    s = w.close({"script": "prelabel.py new", "model": str(a.model), "conf": a.conf})
    print(json.dumps({k: s[k] for k in ("images_seen", "images_review", "instances_review")}, ensure_ascii=False))


def cmd_fill(a, model):
    d = Path(a.out) / a.source
    want = set(a.classes)
    rows = [json.loads(x) for x in (d / "meta.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    added = Counter()
    for r in tqdm(rows, desc=f"fill {a.source}"):
        if r["status"] != "ok":
            continue
        ip, lp = d / "images" / r["file"], d / "labels" / (Path(r["file"]).stem + ".txt")
        img = read_image(ip)
        H, W = img.shape[:2]
        old = lp.read_text(encoding="utf-8").splitlines() if lp.is_file() else []
        occupied = np.zeros((H, W), bool)
        for line in old:
            v = line.split()
            occupied |= rasterize([np.asarray(v[1:], float).reshape(-1, 2) * [W, H]], H, W)
        new = []
        for name, m, sc in predict_masks(model, img, a.conf):
            if name not in want or (m & occupied).sum() > 0.3 * m.sum():
                continue
            poly = mask_to_polygon(m)
            if poly is not None:
                new.append(yolo_line(CLASS_ID[name], normalize(poly, H, W)))
                added[name] += 1
        if new:
            (d / "review" / "images").mkdir(parents=True, exist_ok=True)
            (d / "review" / "labels").mkdir(parents=True, exist_ok=True)
            ip.replace(d / "review" / "images" / ip.name)
            (d / "review" / "labels" / lp.name).write_text("\n".join(old + new) + "\n", encoding="utf-8")
            lp.unlink(missing_ok=True)
            r["status"] = "review"
            r["flags"] = r.get("flags", []) + [f"auto-filled {len(new)} x {'/'.join(sorted(want))} by {Path(a.model).name}: check"]
    (d / "meta.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"added drafts: {dict(added)}; images moved to review: {sum(1 for r in rows if r['status'] == 'review')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["new", "fill"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--images")
    ap.add_argument("--classes", nargs="*", default=["Sign", "Traffic light"])
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--group-from-parent", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.model)
    (cmd_new if a.cmd == "new" else cmd_fill)(a, model)


if __name__ == "__main__":
    main()
