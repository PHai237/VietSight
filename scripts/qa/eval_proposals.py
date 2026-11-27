"""Score proposal backends (propose_labels.Backend) against REFERENCE masks on a small image set. Read-only; prints/writes a report.

The reference is whatever --labels holds (e.g. Mapillary VN native-pixel drafts): it is NOT verified ground truth, so the
numbers say "agrees with the source masks", nothing more. Never use this to promote proposals to labels.

Per backend (and the greedy union that propose_labels.py would package): class-wise TP/FP/FN at mask IoU >= 0.5,
loose recall at IoU >= 0.3, mean IoU of TPs, wrong-class hits, and a class-agnostic two-wheeler group
(Motorcycle+Motorcyclist) to separate "found the bike" from "decided rider/no rider". Reference objects under 64 px^2 at
the 640 scale are 'ignore' (the scorer drops them): not counted as FN, and a prediction on them is not an FP.

  python scripts/qa/eval_proposals.py --images <dir> --labels <dir> --out runs/proposal_eval/x.json \
      --backend vietsight:runs/v1_colab_e25/submission.zip coco:yolo26n-seg.pt [sam3:sam3.pt]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propose_labels import Backend, mask_iou, read_labels  # noqa: E402
from vslib import CLASSES, IMG_EXTS  # noqa: E402
from vslib.writer import read_image  # noqa: E402

GROUPS = {"Two-wheeler (Motorcycle+Motorcyclist)": {"Motorcycle", "Motorcyclist"}}


def match(preds, refs, keep, thr):
    """preds/refs: lists of dicts with class, mask (+score, ignore). keep(cls) -> bool. Greedy by score."""
    P = sorted([p for p in preds if keep(p["class"])], key=lambda p: -(p["score"] or 0))
    R = [r for r in refs if keep(r["class"])]
    used, tp, fp, ious = set(), 0, 0, []
    for p in P:
        best, bi = 0.0, None
        for i, r in enumerate(R):
            if i not in used:
                v = mask_iou(p["mask"], r["mask"])
                if v > best:
                    best, bi = v, i
        if bi is not None and best >= thr:
            used.add(bi)
            if R[bi]["ignore"]:
                continue
            tp, ious = tp + 1, ious + [best]
        else:
            fp += 1
    fn = sum(1 for i, r in enumerate(R) if i not in used and not r["ignore"])
    return tp, fp, fn, ious


def union(per_backend, iou_merge):
    """Same greedy merge as propose_labels.py: highest score wins, later overlaps are dropped."""
    cands = sorted([x for preds in per_backend for x in preds], key=lambda x: -(x["score"] or 0))
    kept = []
    for c in cands:
        if not any(mask_iou(k["mask"], c["mask"]) > iou_merge for k in kept):
            kept.append(c)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--backend", nargs="+", required=True)
    ap.add_argument("--classes", nargs="+", default=["Sign", "Other vehicle", "Motorcycle", "Motorcyclist", "Traffic light"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-merge", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    imgs = sorted(p for p in a.images.rglob("*") if p.suffix.lower() in IMG_EXTS)
    refs, preds, secs = {}, {s: {} for s in a.backend}, {s: 0.0 for s in a.backend}
    for p in imgs:
        img = read_image(p)
        H, W = img.shape[:2]
        _, occ, _ = read_labels((a.labels / p.relative_to(a.images)).with_suffix(".txt"), H, W)
        k = (640 / max(H, W)) ** 2
        refs[p.name] = [{"class": CLASSES[c], "mask": m, "ignore": m.sum() * k < 64} for c, m in occ]
    for s in a.backend:  # one backend at a time (SAM 3 on CPU is heavy)
        b = Backend(s, a.conf)
        for p in imgs:
            t = time.time()
            preds[s][p.name] = b.predict(read_image(p))[0]
            secs[s] += time.time() - t
        b.close()
    systems = {s: preds[s] for s in a.backend}
    if len(a.backend) > 1:
        systems["union(" + " + ".join(x.split(":")[0] for x in a.backend) + ")"] = {
            n: union([preds[s][n] for s in a.backend], a.iou_merge) for n in refs}
    report = {"images": len(imgs), "conf": a.conf, "reference": str(a.labels),
              "reference_counts": {c: sum(1 for n in refs for r in refs[n] if r["class"] == c and not r["ignore"]) for c in CLASSES},
              "seconds_per_image": {s: round(secs[s] / max(1, len(imgs)), 2) for s in a.backend}, "systems": {}}
    for name, P in systems.items():
        rows = {}
        for label, keep in [*[(c, (lambda c: lambda x: x == c)(c)) for c in a.classes],
                            *[(g, (lambda g: lambda x: x in g)(m)) for g, m in GROUPS.items()]]:
            agg = [0, 0, 0, []]
            loose = 0
            for n in refs:
                tp, fp, fn, ious = match(P[n], refs[n], keep, 0.5)
                agg = [agg[0] + tp, agg[1] + fp, agg[2] + fn, agg[3] + ious]
                loose += match(P[n], refs[n], keep, 0.3)[0]
            tp, fp, fn, ious = agg
            rows[label] = {"ref": tp + fn, "pred": tp + fp, "TP": tp, "FP": fp, "FN": fn,
                           "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                           "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                           "recall@0.3": round(loose / (tp + fn), 3) if tp + fn else None,
                           "mIoU_TP": round(sum(ious) / len(ious), 3) if ious else None}
        # predictions of a focus class whose best-overlapping reference (IoU >= 0.5) has ANOTHER class
        wrong = {}
        for n in refs:
            for p in P[n]:
                if p["class"] in a.classes:
                    hit = max(((mask_iou(p["mask"], r["mask"]), r["class"]) for r in refs[n]), default=(0, None), key=lambda t: t[0])
                    if hit[0] >= 0.5 and hit[1] != p["class"]:
                        key = f"{p['class']} -> ref {hit[1]}"
                        wrong[key] = wrong.get(key, 0) + 1
        report["systems"][name] = {"per_class": rows, "wrong_class": dict(sorted(wrong.items(), key=lambda t: -t[1]))}
    txt = json.dumps(report, indent=1, ensure_ascii=False)
    print(txt)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt, encoding="utf-8")
    for name, s in report["systems"].items():
        print(f"\n## {name}\n| class | ref | pred | TP | FP | FN | recall | prec | recall@0.3 | mIoU(TP) |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for c, r in s["per_class"].items():
            print(f"| {c} | " + " | ".join(str(r[k]) for k in ["ref", "pred", "TP", "FP", "FN", "recall", "precision", "recall@0.3", "mIoU_TP"]) + " |")
        print("wrong class:", s["wrong_class"] or "-")


if __name__ == "__main__":
    main()
