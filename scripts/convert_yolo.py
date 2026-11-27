"""Import an existing YOLO-seg dataset (CVAT "YOLO Segmentation 1.0", Label Studio, team annotations, accepted pre-labels)
into data/processed/<source>/, remapping class ids BY NAME (tools often renumber classes).

  python scripts/convert_yolo.py --root path/to/export --names-from path/to/export/data.yaml --source vn_hanoi_m4
  python scripts/convert_yolo.py --root data/raw/sample_data --source sample_test     # pipeline test only

Labels are taken as they are (the annotator already decided rider vs. no rider). Every line is checked with the OFFICIAL
notebook parser + polygon checks; broken lines are dropped and counted, never silently fixed.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vslib import CLASSES, IMG_EXTS, ROOT  # noqa: E402
from vslib.geometry import area_at_640, rasterize, yolo_line  # noqa: E402
from vslib.notebook_lib import load as load_nb  # noqa: E402
from vslib.writer import read_image, safe_name  # noqa: E402


def load_names(path):
    if not path:
        return list(CLASSES)
    y = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = y["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return [str(n) for n in names]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--names-from", help="data.yaml / obj.names of the export (default: competition order)")
    ap.add_argument("--alias", nargs="*", default=[], help="extra name aliases old=New, e.g. motorbike_rider=Motorcyclist")
    ap.add_argument("--group-from-parent", action="store_true", help="group (video/sequence) = parent folder name")
    ap.add_argument("--status", default="ok", choices=["ok", "review"], help="review = AI pre-label drafts, not trusted yet")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args(argv)
    nb = load_nb()
    parse, problem = nb["parse_label_file"], nb["polygon_problem"]
    src_names = [n.strip() for n in (Path(args.names_from).read_text().splitlines() if args.names_from and args.names_from.endswith(".names") else load_names(args.names_from))]
    alias = dict(a.split("=", 1) for a in args.alias)
    remap = {}
    for i, n in enumerate(src_names):
        tgt = alias.get(n, n)
        if tgt not in CLASSES:
            raise SystemExit(f"class {i} '{n}' is not a competition class; give --alias {n}=<Class> (classes: {CLASSES})")
        remap[i] = CLASSES.index(tgt)
    root = Path(args.root)
    out = Path(args.out) / args.source
    import shutil

    shutil.rmtree(out, ignore_errors=True)
    sub = "review/" if args.status == "review" else ""
    (out / f"{sub}images").mkdir(parents=True)
    (out / f"{sub}labels").mkdir(parents=True)
    images = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMG_EXTS and "review" not in p.parts]
    counts, dropped = Counter(), Counter()
    with open(out / "meta.jsonl", "w", encoding="utf-8") as meta:
        for p in tqdm(images, desc=args.source):
            lab = nb["label_path_of"](p)
            rows, probs = parse(lab, len(src_names))
            dropped.update("unparseable line" for _ in probs)
            img = read_image(p)
            h0, w0 = img.shape[:2]
            s = min(1.0, args.max_side / max(h0, w0))
            if s < 1:
                img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            lines, cc = [], Counter()
            for c, pts in rows:
                if problem(pts, w, h):
                    dropped[f"invalid polygon ({problem(pts, w, h)})"] += 1
                    continue
                cid = remap[c]
                px = pts * [w, h]
                if area_at_640(rasterize([px], h, w).sum(), h, w) < 64:
                    dropped["smaller than 64 px at 640 (scorer ignores)"] += 1
                    continue
                lines.append(yolo_line(cid, pts))
                cc[CLASSES[cid]] += 1
            rel = p.relative_to(root).as_posix()
            name = safe_name(args.source, rel)
            cv2.imwrite(str(out / f"{sub}images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / f"{sub}labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts.update(cc)
            grp = p.parent.name if args.group_from_parent else name
            meta.write(json.dumps({"file": f"{name}.jpg", "orig": rel, "group": f"{args.source}:{grp}", "status": args.status, "flags": [],
                                   "w": w, "h": h, "instances": dict(cc)}, ensure_ascii=False) + "\n")
    key = "images_ok" if args.status == "ok" else "images_review"
    summary = {"source": args.source, "images_seen": len(images), key: len(images), "instances_ok" if args.status == "ok" else "instances_review":
               {c: counts.get(c, 0) for c in CLASSES}, "dropped": dict(dropped), "converter": {"script": "convert_yolo.py", "root": str(root),
               "names": src_names, "alias": alias}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
