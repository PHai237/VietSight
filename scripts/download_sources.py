"""Download / register public sources into data/raw/<source>/ (never committed).

  python scripts/download_sources.py list
  python scripts/download_sources.py coco2017 --max-images 3000        # annotations + ONLY the selected images
  python scripts/download_sources.py ninja "BDD100K: Images 10K" --dst data/raw/bdd100k_10k   # Dataset Ninja mirror (Supervisely)
  python scripts/download_sources.py ninja "Mapillary Vistas" --dst data/raw/mapillary_vistas

Datasets that need a personal account (Mapillary official, IDD, BDD100K official, Cityscapes) can not be fetched by
script: download them by hand into data/raw/<source>/ and record it in data/sources_manifest.yaml.
"""
import argparse
import json
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vslib import ROOT  # noqa: E402
from vslib.mapping import load_source_config  # noqa: E402

RAW = ROOT / "data" / "raw"
COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def fetch(url, dst, chunk=1 << 20):
    dst = Path(dst)
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name, leave=False) as bar:
            for b in r.iter_content(chunk):
                f.write(b)
                bar.update(len(b))
    tmp.replace(dst)
    return dst


def coco2017(args):
    cfg = load_source_config("coco2017")
    out = RAW / "coco2017"
    (out / "images").mkdir(parents=True, exist_ok=True)
    z = fetch(COCO_ANN, out / "annotations_trainval2017.zip")
    sel_images, sel_anns, cats, licenses = [], [], None, None
    req = set(cfg.get("filter", {}).get("require_any", []))
    with zipfile.ZipFile(z) as zf:
        for split in ("train2017", "val2017"):
            data = json.loads(zf.read(f"annotations/instances_{split}.json"))
            cats = data["categories"]
            licenses = data["licenses"]
            names = {c["id"]: c["name"] for c in cats}
            by = {}
            for a in data["annotations"]:
                by.setdefault(a["image_id"], []).append(a)
            for im in data["images"]:
                anns = by.get(im["id"], [])
                if req and not ({names[a["category_id"]] for a in anns} & req):
                    continue
                im = dict(im, coco_split=split)
                sel_images.append(im)
                sel_anns.extend(anns)
    print(f"COCO images matching {sorted(req)}: {len(sel_images)}")
    sel_images.sort(key=lambda im: im["id"])
    if args.max_images and len(sel_images) > args.max_images:
        sel_images = sorted(random.Random(args.seed).sample(sel_images, args.max_images), key=lambda im: im["id"])
    ids = {im["id"] for im in sel_images}
    sel_anns = [a for a in sel_anns if a["image_id"] in ids]
    (out / "instances_sel.json").write_text(json.dumps({"images": sel_images, "annotations": sel_anns, "categories": cats, "licenses": licenses}), encoding="utf-8")

    def one(im):
        try:
            fetch(im["coco_url"], out / "images" / im["file_name"])
            return True
        except Exception as e:  # noqa: BLE001
            print("fail", im["file_name"], e)
            return False

    with ThreadPoolExecutor(args.workers) as ex:
        ok = sum(tqdm(ex.map(one, sel_images), total=len(sel_images), desc="coco images"))
    print(f"downloaded {ok}/{len(sel_images)} images -> {out}/images ; annotations -> {out}/instances_sel.json")


def ninja(args):
    import dataset_tools as dtools  # pip install dataset-tools

    dst = Path(args.dst or RAW / args.name.lower().replace(" ", "_").replace(":", ""))
    dst.mkdir(parents=True, exist_ok=True)
    print(f"Dataset Ninja download '{args.name}' -> {dst}")
    dtools.download(dataset=args.name, dst_dir=str(dst))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("coco2017")
    c.add_argument("--max-images", type=int, default=3000)
    c.add_argument("--seed", type=int, default=55)
    c.add_argument("--workers", type=int, default=16)
    n = sub.add_parser("ninja")
    n.add_argument("name")
    n.add_argument("--dst")
    args = ap.parse_args()
    if args.cmd == "list":
        print((ROOT / "data" / "sources_manifest.yaml").read_text(encoding="utf-8"))
    elif args.cmd == "coco2017":
        coco2017(args)
    else:
        ninja(args)


if __name__ == "__main__":
    main()
