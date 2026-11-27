"""Human review of converted / pre-labelled data. Nothing enters a dataset version without status 'ok'.

  python scripts/review_queue.py stats  data/processed/<source>
  python scripts/review_queue.py export data/processed/<source> --out runs/review/<source>_cvat.zip [--n 200]
        -> zip in CVAT "YOLO Segmentation 1.0" layout (data.yaml, images/train, labels/train) of the review/ images
  python scripts/review_queue.py accept data/processed/<source> --from <CVAT export dir or zip> --reviewer <name>
        -> corrected labels replace the drafts, images move from review/ to images/, meta status -> ok (+ reviewer)
  python scripts/review_queue.py reject data/processed/<source> --files a.jpg b.jpg --reason "unclear riders"
"""
import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vslib import CLASSES  # noqa: E402
from vslib.notebook_lib import load as load_nb  # noqa: E402


def read_meta(d):
    return [json.loads(x) for x in (d / "meta.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def write_meta(d, rows):
    (d / "meta.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def cmd_stats(d, _):
    rows = read_meta(d)
    st = Counter(r["status"] for r in rows)
    fl = Counter(f for r in rows if r["status"] == "review" for f in r.get("flags", []))
    print(f"{d.name}: {dict(st)}")
    for f, n in fl.most_common(15):
        print(f"  {n:5d}  {f}")


def cmd_export(d, a):
    rows = [r for r in read_meta(d) if r["status"] == "review"][: a.n or None]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("data.yaml", f"path: .\ntrain: images/train\nnames:\n{names}\n")
        zf.writestr("train.txt", "".join(f"images/train/{r['file']}\n" for r in rows))
        for r in rows:
            zf.write(d / "review" / "images" / r["file"], f"images/train/{r['file']}")
            lab = d / "review" / "labels" / (Path(r["file"]).stem + ".txt")
            zf.writestr(f"labels/train/{lab.name}", lab.read_text(encoding="utf-8") if lab.is_file() else "")
        zf.writestr("REVIEW_FLAGS.tsv", "".join(f"{r['file']}\t{' | '.join(r.get('flags', []))}\n" for r in rows))
    print(f"{len(rows)} images -> {out}  (CVAT: Create task, upload the images, then Actions > Upload annotations > YOLO Segmentation 1.0)")


def cmd_accept(d, a):
    src = Path(a.__dict__["from"])
    tmp = None
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(src) as zf:
            for m in zf.namelist():
                if ".." in Path(m).parts or Path(m).is_absolute():
                    raise SystemExit(f"unsafe path in zip: {m}")
            zf.extractall(tmp)
        src = tmp
    nb = load_nb()
    labels = {p.stem: p for p in src.rglob("*.txt") if p.parent.name != "." and p.name not in ("train.txt", "obj.names")}
    names_file = next(iter(src.rglob("data.yaml")), None)
    if names_file:
        import yaml

        nm = yaml.safe_load(names_file.read_text(encoding="utf-8"))["names"]
        nm = [nm[k] for k in sorted(nm, key=int)] if isinstance(nm, dict) else nm
        if list(nm) != CLASSES:
            raise SystemExit(f"export class order {nm} != competition order: fix the CVAT label order (or use convert_yolo.py --alias)")
    rows = read_meta(d)
    n_ok, bad = 0, []
    for r in rows:
        stem = Path(r["file"]).stem
        if r["status"] != "review" or stem not in labels:
            continue
        lab_rows, probs = nb["parse_label_file"](labels[stem], len(CLASSES))
        if probs:
            bad.append((r["file"], probs[:2]))
            continue
        shutil.move(d / "review" / "images" / r["file"], d / "images" / r["file"])
        shutil.copyfile(labels[stem], d / "labels" / f"{stem}.txt")
        (d / "review" / "labels" / f"{stem}.txt").unlink(missing_ok=True)
        r.update(status="ok", reviewed_by=a.reviewer, reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 instances=dict(Counter(CLASSES[c] for c, _ in lab_rows)))
        n_ok += 1
    write_meta(d, rows)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"accepted {n_ok} images (reviewer {a.reviewer}); {len(bad)} with label errors kept in review: {bad[:5]}")


def cmd_reject(d, a):
    rows = read_meta(d)
    files = set(a.files)
    for r in rows:
        if r["file"] in files:
            r.update(status="rejected", reject_reason=a.reason)
    write_meta(d, rows)
    print(f"rejected {len(files)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "export", "accept", "reject"])
    ap.add_argument("source_dir")
    ap.add_argument("--out", default="runs/review/review_cvat.zip")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--from", dest="from")
    ap.add_argument("--reviewer", default="unknown")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    {"stats": cmd_stats, "export": cmd_export, "accept": cmd_accept, "reject": cmd_reject}[a.cmd](Path(a.source_dir), a)


if __name__ == "__main__":
    main()
