"""Stream a Supervisely .tar (Dataset Ninja mirror) and keep only what we need - no full download/extract on disk.

Selection is deterministic (independent of archive order): an annotated image is kept when it contains one of
--prefer classes, otherwise when sha1(name) falls under --other-frac. Images of unselected/unlabeled annotations are
discarded. Use --max-ann for a quick peek (first N annotation files).

  python scripts/fetch_sly_tar.py mapillary_vistas /kaggle/tmp/mapillary --splits training validation \
      --prefer motorcycle motorcyclist bicycle bicyclist "other rider" --other-frac 0.25
  python scripts/fetch_sly_tar.py bdd100k_10k data/raw/bdd100k_10k --splits train val --other-frac 1
"""
import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests  # noqa: E402

from vslib.mapping import load_source_config  # noqa: E402


def shrink(data, max_side):
    import cv2
    import numpy as np

    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # NB: mirror images carry no EXIF rotation
    if im is None or max(im.shape[:2]) <= max_side:
        return data
    s = max_side / max(im.shape[:2])
    im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def keep_ann(name, data, prefer, frac):
    objs = data.get("objects", [])
    if not objs:
        return False
    if prefer and any(o.get("classTitle") in prefer for o in objs):
        return True
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="configs/sources/<source>.yaml with download_url (or a URL)")
    ap.add_argument("dst")
    ap.add_argument("--splits", nargs="*", help="dataset folders to keep (default all)")
    ap.add_argument("--prefer", nargs="*", default=[], help="always keep images containing these source labels")
    ap.add_argument("--other-frac", type=float, default=1.0, help="fraction of the other annotated images to keep")
    ap.add_argument("--max-ann", type=int, default=0, help="stop after N annotation files (peek)")
    ap.add_argument("--max-side", type=int, default=0, help="re-encode kept images to this max side (JPEG q95); converters "
                    "rescale polygons from the annotation size, so use the SAME value as the source config max_side")
    args = ap.parse_args()
    url = args.source if args.source.startswith("http") else load_source_config(args.source)["download_url"]
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    prefer = set(args.prefer)
    decided, pending = {}, {}  # "<split>/<img name>" -> keep?, image saved before its annotation
    n_ann = n_keep = n_img = 0
    t0, nbytes = time.time(), 0
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                parts = Path(m.name).parts
                if ".." in parts or Path(m.name).is_absolute():
                    continue
                nbytes += m.size
                if len(parts) < 3 or parts[-2] not in ("ann", "img"):
                    if len(parts) <= 2:  # meta.json, LICENSE.md, README.md
                        (dst / m.name).parent.mkdir(parents=True, exist_ok=True)
                        (dst / m.name).write_bytes(tf.extractfile(m).read())
                    continue
                split, kind, fname = parts[-3], parts[-2], parts[-1]
                if args.splits and split not in args.splits:
                    continue
                if kind == "ann":
                    data = tf.extractfile(m).read()
                    img_name = fname[:-5]
                    key = f"{split}/{img_name}"
                    n_ann += 1
                    k = keep_ann(img_name, json.loads(data), prefer, args.other_frac)
                    decided[key] = k
                    if k:
                        n_keep += 1
                        out = dst / split / "ann" / fname
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(data)
                    if key in pending:
                        p = pending.pop(key)
                        if not k:
                            p.unlink(missing_ok=True)
                    if n_ann % 500 == 0:
                        el = time.time() - t0
                        print(f"  ann {n_ann} kept {n_keep} imgs {n_img} | {nbytes / 2**30:.1f} GiB read, {nbytes / 2**20 / el:.1f} MiB/s", flush=True)
                    if args.max_ann and n_ann >= args.max_ann:
                        break
                else:
                    key = f"{split}/{fname}"
                    if decided.get(key) is False:
                        continue
                    out = dst / split / "img" / fname
                    out.parent.mkdir(parents=True, exist_ok=True)
                    data = tf.extractfile(m).read()
                    if args.max_side:
                        data = shrink(data, args.max_side)
                    out.write_bytes(data)
                    n_img += 1
                    if key not in decided:
                        pending[key] = out
    for p in pending.values():  # images whose annotation never came (e.g. stopped early)
        p.unlink(missing_ok=True)
    print(f"done: {n_ann} annotations seen, {n_keep} kept, {n_img} images written -> {dst}")


if __name__ == "__main__":
    main()
