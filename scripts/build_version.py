"""Merge processed sources into a dataset version: data/versions/<name>/{images,labels}/{train,val} + data/versions/<name>_report/.

  python scripts/build_version.py configs/versions/v1_first_gear.yaml

* Exact (sha256) and near (dhash, the notebook's dhash64) duplicates are removed ACROSS sources: val copies win, then
  sources in config order.
* VAL IS FROZEN: configs/val_frozen.txt lists '<source>/<file>' of every val image. Created on the first build (group-wise:
  every image of a video/sequence group goes to the same side), then reused by every later version, so scores are
  comparable. Val images never enter train. Regenerate only on purpose (--refreeze-val) and say so in docs/CHANGELOG.
* Then the OFFICIAL notebook report (analyze_dataset/print_report/make_previews) is run on the result.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from vslib import CLASSES, IMGSZ, MIN_GT_AREA_PX, ROOT  # noqa: E402
from vslib.notebook_lib import load as load_nb  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def label_counts(txt):
    c = Counter()
    if txt.is_file():
        for line in txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c[CLASSES[int(line.split()[0])]] += 1
    return c


def collect(src_cfg, processed):
    name = src_cfg["name"]
    d = processed / name
    if not (d / "meta.jsonl").is_file():
        raise SystemExit(f"source {name}: {d}/meta.jsonl missing -> run its converter first")
    items = []
    for line in (d / "meta.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        if m["status"] != "ok":
            continue
        img = d / "images" / m["file"]
        if not img.is_file():
            continue
        items.append({"key": f"{name}/{m['file']}", "source": name, "img": img, "lab": d / "labels" / (Path(m["file"]).stem + ".txt"),
                      "group": f"{name}:{m.get('group') or m['file']}", "orig": m.get("orig")})
    items.sort(key=lambda it: it["key"])
    if src_cfg.get("require_any"):
        req = set(src_cfg["require_any"])
        items = [it for it in items if set(label_counts(it["lab"])) & req]
    cap = src_cfg.get("max_images")
    if cap and len(items) > cap:
        # val-frozen images are always kept (they are selected later); preferred classes first, then a seeded sample
        pref = set(src_cfg.get("prefer_any", []))
        frozen = src_cfg.get("_frozen", set())
        must = [it for it in items if it["key"] in frozen]
        rest = [it for it in items if it["key"] not in frozen]
        p_items = [it for it in rest if pref and set(label_counts(it["lab"])) & pref]
        o_items = [it for it in rest if not (pref and set(label_counts(it["lab"])) & pref)]
        rng = random.Random(55)
        rng.shuffle(p_items)
        rng.shuffle(o_items)
        items = sorted(must + (p_items + o_items)[: max(0, cap - len(must))], key=lambda it: it["key"])
    return items


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--refreeze-val", action="store_true")
    ap.add_argument("--processed", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--out-root", default=str(ROOT / "data" / "versions"))
    ap.add_argument("--val-list", help="override val list path")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    nb = load_nb()
    processed = Path(args.processed)
    out = Path(args.out_root) / cfg["name"]
    val_list = Path(args.val_list or ROOT / cfg.get("val", {}).get("frozen_list", "configs/val_frozen.txt"))

    frozen_now = set()
    if val_list.is_file() and not args.refreeze_val:
        frozen_now = {x.strip() for x in val_list.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
    items = []
    for s in cfg["sources"]:
        got = collect(dict(s, _frozen=frozen_now), processed)
        print(f"  source {s['name']}: {len(got)} images")
        items += got
    by_key = {it["key"]: it for it in items}

    # ---------------- val split (frozen)
    if val_list.is_file() and not args.refreeze_val:
        frozen = [x.strip() for x in val_list.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
        missing = [k for k in frozen if k not in by_key]
        if missing:
            print(f"WARNING: {len(missing)} frozen val images are not in this build (source not included / filtered): e.g. {missing[:3]}")
        val_keys = {k for k in frozen if k in by_key}
        print(f"val: frozen list {val_list} ({len(frozen)} entries, {len(val_keys)} present)")
    else:
        vc = cfg.get("val", {})
        pool_src = set(vc.get("from_sources") or [s["name"] for s in cfg["sources"]])
        groups = defaultdict(list)
        for it in items:
            if it["source"] in pool_src:
                groups[it["group"]].append(it["key"])
        order = sorted(groups)
        random.Random(55).shuffle(order)
        target = round(vc.get("ratio", 0.12) * sum(len(v) for v in groups.values()))
        per_src_cap = vc.get("max_per_source")
        val_keys, per_src = set(), Counter()
        for g in order:
            src = g.split(":", 1)[0]
            if per_src_cap and per_src[src] >= per_src_cap.get(src, 1 << 30):
                continue
            val_keys.update(groups[g])
            per_src[src] += len(groups[g])
            if len(val_keys) >= target:
                break
        val_list.parent.mkdir(parents=True, exist_ok=True)
        val_list.write_text(f"# frozen val list created by build_version.py for {cfg['name']} (seed 55, group-wise). DO NOT EDIT casually.\n"
                            + "\n".join(sorted(val_keys)) + "\n", encoding="utf-8")
        print(f"val: NEW frozen list {val_list}: {len(val_keys)} images {dict(per_src)}")
    val_groups = {by_key[k]["group"] for k in val_keys}
    leak_group = [it["key"] for it in items if it["key"] not in val_keys and it["group"] in val_groups]
    train_items = [it for it in items if it["key"] not in val_keys and it["group"] not in val_groups]
    val_items = [by_key[k] for k in sorted(val_keys)]

    # ---------------- duplicates (val first, then config order)
    ordered = val_items + train_items
    print("hashing images ...")
    for it in tqdm(ordered, leave=False):
        it["sha"] = sha256(it["img"])
        it["dhash"] = nb["dhash64"](it["img"])
    seen, drop, dup_log = {}, set(), []
    for i, it in enumerate(ordered):
        if it["sha"] in seen:
            drop.add(i)
            dup_log.append(("exact", it["key"], ordered[seen[it["sha"]]]["key"]))
        else:
            seen[it["sha"]] = i
    ham = cfg.get("dedup", {}).get("hamming", 2)
    for i, j, d in nb["find_duplicate_pairs"]([it["dhash"] for it in ordered], ham):
        if i in drop or j in drop:
            continue
        drop.add(j)  # j > i: later in priority
        dup_log.append((f"near(d={d})", ordered[j]["key"], ordered[i]["key"]))
    n_val = len(val_items)
    val_items = [ordered[k] for k in range(n_val) if k not in drop]
    train_items = [ordered[k] for k in range(n_val, len(ordered)) if k not in drop]
    print(f"duplicates removed: {len(drop)} (exact {sum(1 for d in dup_log if d[0] == 'exact')}), group-leak removed from train: {len(leak_group)}")

    # ---------------- write
    rep_dir = out.parent / f"{cfg['name']}_report"  # outside the dataset folder: the notebook scans every image under it
    for d in (out, rep_dir):
        if d.exists():
            shutil.rmtree(d)
    manifest, stats = [], {"train": defaultdict(Counter), "val": defaultdict(Counter)}
    img_counts = {"train": Counter(), "val": Counter()}
    for split, its in (("train", train_items), ("val", val_items)):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for it in its:
            link_or_copy(it["img"], out / "images" / split / it["img"].name)
            if it["lab"].is_file():
                link_or_copy(it["lab"], out / "labels" / split / it["lab"].name)
            c = label_counts(it["lab"])
            stats[split][it["source"]].update(c)
            img_counts[split][it["source"]] += 1
            manifest.append({"split": split, "file": it["img"].name, "source": it["source"], "orig": it["orig"], "group": it["group"], "sha256": it["sha"]})
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    rep_dir.mkdir(parents=True)
    (rep_dir / "manifest.jsonl").write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n", encoding="utf-8")
    (rep_dir / "duplicates.tsv").write_text("\n".join("\t".join(map(str, d)) for d in dup_log) + "\n", encoding="utf-8")

    # ---------------- official report
    report = nb["analyze_dataset"](str(out), CLASSES, MIN_GT_AREA_PX, IMGSZ, 100, 30, ham, 10, 100)
    lines = []
    nb["print_report"](report, CLASSES, nb["CLASS_DISPLAY_VI"], printer=lines.append)
    (rep_dir / "notebook_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    nb["make_previews"](str(out), CLASSES, rep_dir / "preview_train.png", n=12, seed=55)
    nb["make_previews"](str(out), CLASSES, rep_dir / "preview_val.png", n=8, seed=55, split="val")
    fp = nb["dataset_fingerprint"](str(out))
    summary = {"version": cfg["name"], "config": str(Path(args.config)), "description": cfg.get("description"), "fingerprint_sha256": fp,
               "val_list": str(val_list), "images": {s: sum(c.values()) for s, c in img_counts.items()},
               "images_by_source": {s: dict(c) for s, c in img_counts.items()},
               "instances": {s: {c: sum(stats[s][src][c] for src in stats[s]) for c in CLASSES} for s in stats},
               "instances_by_source": {s: {src: dict(c) for src, c in stats[s].items()} for s in stats},
               "duplicates_removed": len(drop), "group_leak_removed": len(leak_group),
               "notebook_report": {k: report[k] for k in ("splits", "per_class", "tiny_total", "imbalance_ratio", "warnings", "fatal", "ok")},
               "invalid_polygons": report["invalid_polygons"][:200], "leak_exact": report["duplicates"]["leak_exact"], "leak_near": report["duplicates"]["leak_near"]}
    (rep_dir / "stats.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nVersion {cfg['name']} -> {out}\n  train {summary['images']['train']} / val {summary['images']['val']} images, fingerprint {fp[:16]}")
    if not report["ok"]:
        raise SystemExit("FATAL problems in notebook report: fix before training")


if __name__ == "__main__":
    main()
