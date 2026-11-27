"""Contact sheet of images with their YOLO-seg masks (the notebook's draw_polygons), for label review.

  python scripts/preview.py data/processed/mapillary_vistas --cls Motorcyclist Cyclist --n 16 --out runs/review/riders.png
  python scripts/preview.py data/processed/mapillary_vistas/review --n 24            # ambiguous images queue
  python scripts/preview.py data/versions/v1_first_gear --split val --n 12
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image, ImageOps  # noqa: E402

from vslib import CLASSES, CLASS_ID, IMG_EXTS, ROOT  # noqa: E402
from vslib.notebook_lib import load as load_nb  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with images/ and labels/ (or a version folder with --split)")
    ap.add_argument("--split", help="train | val (version folders)")
    ap.add_argument("--cls", nargs="*", help="only images containing one of these classes (and draw only them with --only)")
    ap.add_argument("--only", action="store_true", help="draw only the --cls classes")
    ap.add_argument("--files", nargs="*", help="explicit image file names")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--seed", type=int, default=55)
    ap.add_argument("--out", default=str(ROOT / "runs" / "review" / "preview.png"))
    args = ap.parse_args(argv)
    nb = load_nb()
    root = Path(args.root)
    img_dir, lab_dir = (root / "images" / args.split, root / "labels" / args.split) if args.split else (root / "images", root / "labels")
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    want = {CLASS_ID[c] for c in args.cls} if args.cls else None
    chosen = []
    if args.files:
        imgs = [p for p in imgs if p.name in set(args.files)]
    pool = []
    for p in imgs:
        rows, _ = nb["parse_label_file"](lab_dir / f"{p.stem}.txt" if (lab_dir / f"{p.stem}.txt").is_file() else None, len(CLASSES))
        if want and not any(c in want for c, _ in rows):
            continue
        pool.append((p, [r for r in rows if not (args.only and want) or r[0] in want]))
    random.Random(args.seed).shuffle(pool)
    chosen = pool[: args.n]
    if not chosen:
        raise SystemExit("no matching image")
    tiles = []
    for p, rows in chosen:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        im = nb["draw_polygons"](im, rows, CLASSES, fill_alpha=90, width=2)
        im.thumbnail((args.side, args.side))
        tiles.append(im)
    cols = min(args.cols, len(tiles))
    rows_n = (len(tiles) + cols - 1) // cols
    th = max(t.size[1] for t in tiles)
    sheet = Image.new("RGB", (cols * args.side, rows_n * th), (30, 30, 30))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * args.side, (k // cols) * th))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    (out.with_suffix(".txt")).write_text("\n".join(p.name for p, _ in chosen) + "\n", encoding="utf-8")
    print(f"{len(chosen)}/{len(pool)} images -> {out} (file list: {out.with_suffix('.txt')})")


if __name__ == "__main__":
    main()
