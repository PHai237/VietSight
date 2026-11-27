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
    ap.add_argument("--max-keep", nargs="*", default=[], help="per split cap, e.g. training=1700 validation=300: the first "
                    "selected annotations (archive order = random ids) are kept; the stream STOPS once every capped split "
                    "has all its kept images (no need to read the whole archive)")
    ap.add_argument("--retries", type=int, default=4, help="reconnect after a network error (already written files are kept)")
    args = ap.parse_args()
    url = args.source if args.source.startswith("http") else load_source_config(args.source)["download_url"]
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    prefer = set(args.prefer)
    caps = {k: int(v) for k, v in (x.split("=", 1) for x in args.max_keep)}
    st = {"decided": {}, "pending": {}, "kept": {}, "written": set(), "img_seen": set(), "n_ann": 0, "n_img": 0}

    def finished():
        if not caps:
            return False
        for sp, cap in caps.items():
            kept = st["kept"].get(sp, set())
            if sp not in st["img_seen"] and len(kept) < cap:
                return False  # still reading this split's annotations
            if not kept <= st["written"]:
                return False
        return True

    t0, nbytes = time.time(), 0
    for attempt in range(args.retries + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                r.raw.decode_content = True
                with tarfile.open(fileobj=r.raw, mode="r|*") as tf:
                    for m in tf:
                        if finished():
                            break
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
                            img_name = fname[:-5]
                            key = f"{split}/{img_name}"
                            if key in st["decided"]:
                                continue  # seen before a reconnect
                            data = tf.extractfile(m).read()
                            st["n_ann"] += 1
                            kept = st["kept"].setdefault(split, set())
                            k = len(kept) < caps.get(split, 1 << 30) and keep_ann(img_name, json.loads(data), prefer, args.other_frac)
                            st["decided"][key] = k
                            if k:
                                kept.add(key)
                                out = dst / split / "ann" / fname
                                out.parent.mkdir(parents=True, exist_ok=True)
                                out.write_bytes(data)
                            if key in st["pending"]:
                                pp = st["pending"].pop(key)
                                if k:
                                    st["written"].add(key)
                                else:
                                    pp.unlink(missing_ok=True)
                            if st["n_ann"] % 500 == 0:
                                el = time.time() - t0
                                nk = sum(len(v) for v in st["kept"].values())
                                print(f"  ann {st['n_ann']} kept {nk} imgs {st['n_img']} | {nbytes / 2**30:.2f} GiB read, "
                                      f"{nbytes / 2**20 / el:.1f} MiB/s, {el / 60:.1f} min", flush=True)
                            if args.max_ann and st["n_ann"] >= args.max_ann:
                                break
                        else:
                            st["img_seen"].add(split)
                            key = f"{split}/{fname}"
                            if st["decided"].get(key) is False or key in st["written"]:
                                continue
                            if caps and split in caps and key not in st["decided"] and len(st["kept"].get(split, ())) >= caps[split]:
                                continue
                            out = dst / split / "img" / fname
                            out.parent.mkdir(parents=True, exist_ok=True)
                            data = tf.extractfile(m).read()
                            if args.max_side:
                                data = shrink(data, args.max_side)
                            out.write_bytes(data)
                            st["n_img"] += 1
                            if key in st["decided"]:
                                st["written"].add(key)
                            else:
                                st["pending"][key] = out
                            if st["n_img"] % 200 == 0:
                                el = time.time() - t0
                                print(f"  imgs {st['n_img']} | {nbytes / 2**30:.2f} GiB read, {nbytes / 2**20 / el:.1f} MiB/s, {el / 60:.1f} min", flush=True)
            break
        except (requests.RequestException, tarfile.ReadError, OSError, EOFError) as e:
            if attempt == args.retries:
                raise
            print(f"  !! network/archive error ({type(e).__name__}: {e}); reconnect {attempt + 1}/{args.retries} "
                  f"(kept files stay, decided annotations are skipped)", flush=True)
            time.sleep(10 * (attempt + 1))
    for pth in st["pending"].values():  # images whose annotation never came (e.g. stopped early)
        pth.unlink(missing_ok=True)
    nk = {sp: len(v) for sp, v in st["kept"].items()}
    missing = {sp: len(v - st["written"]) for sp, v in st["kept"].items()}
    print(f"done in {(time.time() - t0) / 60:.1f} min: {st['n_ann']} annotations seen, kept {nk}, images missing {missing}, "
          f"{nbytes / 2**30:.2f} GiB read -> {dst}")


if __name__ == "__main__":
    main()
