"""Independent, read-only YOLO-seg QA. No model, training or source label writes.

Outputs: audit.json, review.csv, deterministic per-class samples, original/overlay
pairs; optionally original/before/after panels against a corrected labels folder.
Instance IDs are stable FILE:LINE IDs within the supplied label revision, not
tracking IDs. Geometry checks reuse the unmodified official notebook helpers.
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

from vslib import CLASSES, IMG_EXTS
from vslib.notebook_lib import check_classes, load


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def inside(path, root):
    return path.resolve().is_relative_to(root.resolve())


def safe_output(out, protected):
    """Reject either containment direction, including symlinks, before any write."""
    for root in protected:
        if root and (inside(out, root) or inside(root, out)):
            raise ValueError(f'output overlaps input: {out} / {root}')


def read_labels(path, width, height, nb):
    rows, issues, instances = [], [], []
    prior_masks = []
    if not path.is_file():
        return rows, [{'line': None, 'reason': 'missing_label: confirm background manually'}], instances
    try:
        parsed, problems = nb['parse_label_file'](path, len(CLASSES), shown_name='labels')
    except (ValueError, OverflowError) as exc:
        return [], [{'line': None, 'reason': f'official_parser_error: {exc}'}], []
    bad_lines = set()
    for problem in problems:
        match = re.search(r'dòng (\d+)', problem)
        line_no = int(match.group(1)) if match else None
        if line_no:
            bad_lines.add(line_no)
        issues.append({'line': line_no, 'reason': problem})
    line_ids = [n for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1)
                if line.strip() and n not in bad_lines]
    if len(line_ids) != len(parsed):
        return [], issues + [{'line': None, 'reason': 'cannot recover instance line IDs'}], []
    for line_no, (cid, pts) in zip(line_ids, parsed):
        reason = nb['polygon_problem'](pts, width, height)
        if reason:
            issues.append({'line': line_no, 'reason': f'invalid_polygon: {reason}'})
        if len(pts) > 400:
            issues.append({'line': line_no, 'reason': 'over_400_points: official crossing check skipped'})
        area = nb['area_at_imgsz'](pts, width, height, 640)
        px = np.round(pts * [width, height]).astype(np.int32)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [px], 1)
        pixels = int(mask.sum())
        x0, y0 = px.min(axis=0)
        x1, y1 = px.max(axis=0)
        extent = pixels / max(1, int((x1-x0+1)*(y1-y0+1)))
        reasons = []
        short_side_640 = min(x1-x0+1, y1-y0+1) * 640 / max(width, height)
        if short_side_640 < 8:
            reasons.append('short_side_under_about_1.25_percent_image_long_side')
        if area < 64:
            reasons.append('tiny_at_640')
        if extent < .03:
            reasons.append('sparse_mask')
        if pixels > .8 * width * height:
            reasons.append('over_80_percent_image')
        if (x1-x0+1) / max(1, y1-y0+1) > 30 or (y1-y0+1) / max(1, x1-x0+1) > 30:
            reasons.append('extreme_aspect')
        if cid == 9 and (y1-y0+1) / max(1, x1-x0+1) > 5:
            reasons.append('sign_tall_check_pole')
        for item in reasons:
            issues.append({'line': line_no, 'reason': item + ': heuristic, human review required'})
        for previous_line, previous in prior_masks:
            overlap = int(np.count_nonzero(mask & previous))
            # Shared raster boundary pixels can occur even for adjacent polygons.
            if overlap > max(4, .01 * min(pixels, int(previous.sum()))):
                issues.append({'line': line_no,
                               'reason': f'overlap_with_L{previous_line}:{overlap}_pixels: inspect visible-mask ownership; raster heuristic'})
        prior_masks.append((line_no, mask))
        rows.append((cid, pts))
        instances.append({'line': line_no, 'class_id': cid, 'class_name': CLASSES[cid],
                          'area_at_640': area, 'short_side_at_640': short_side_640,
                          'raster_pixels': pixels, 'extent': extent})
    if not rows:
        issues.append({'line': None, 'reason': 'empty_or_unreadable_labels: confirm all 11 classes absent'})
    return rows, issues, instances


def panel(original, rows, instance_info, nb, caption, side):
    texts = [f"{r['class_id']} {r['class_name']} inst:L{r['line']}" for r in instance_info]
    overlay = nb['draw_polygons'](original, rows, CLASSES, texts=texts)
    overlay.thumbnail((side, side))
    tile = Image.new('RGB', (side, side + 36), '#202020')
    tile.paste(overlay, (0, 36))
    ImageDraw.Draw(tile).text((8, 9), caption, fill='white')
    return tile


def run(args):
    check_classes(CLASSES)
    nb = load()
    root, out = args.root.resolve(), args.out.resolve()
    images = root / 'images' / args.split if args.split else root / 'images'
    labels = args.labels.resolve() if args.labels else (root / 'labels' / args.split if args.split else root / 'labels')
    after = args.after_labels.resolve() if args.after_labels else None
    names_path = args.names_from.resolve() if args.names_from else root / 'data.yaml'
    class_order = 'unverified: supply --names-from or check the source mapping manually'
    if names_path.is_file():
        names = yaml.safe_load(names_path.read_text(encoding='utf-8'))['names']
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names, key=int)]
        if list(names) != CLASSES:
            raise ValueError(f'source class order does not match competition: {names_path}')
        class_order = {'file': str(names_path), 'sha256': digest(names_path), 'matches': True}
    if not images.is_dir() or not labels.is_dir():
        raise ValueError('input must contain images/ and labels/ (or supply --labels)')
    safe_output(out, [root, labels, after])
    if out.exists():
        raise ValueError('choose a new --out directory; existing evidence is never overwritten')
    files = sorted(p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not files:
        raise ValueError('no input images')
    total_images = len(files)
    if args.limit:
        files = files[:args.limit]
    records, per_class, hashes = [], {c: [] for c in CLASSES}, {}
    for path in files:
        rel = path.relative_to(images)
        lab = labels / rel.with_suffix('.txt')
        rec = {'image': rel.as_posix(), 'image_sha256': digest(path), 'label_sha256': digest(lab)}
        try:
            with Image.open(path) as im:
                original = ImageOps.exif_transpose(im).convert('RGB')
            w, h = original.size
            rows, issues, inst = read_labels(lab, w, h, nb)
            if rec['image_sha256'] in hashes:
                issues.append({'line': None, 'reason': f"exact_image_duplicate_of:{hashes[rec['image_sha256']]}"})
            hashes[rec['image_sha256']] = rel.as_posix()
            rec.update(width=w, height=h, instances=inst, issues=issues)
            for cid in {c for c, _ in rows}:
                per_class[CLASSES[cid]].append(rel.as_posix())
            if after:
                corrected = after / rel.with_suffix('.txt')
                if corrected.is_file():
                    arows, aissues, ainst = read_labels(corrected, w, h, nb)
                    rec.update(after_label_sha256=digest(corrected), after_issues=aissues)
                else:
                    rec['issues'].append({'line': None, 'reason': 'after_label_missing: no after evidence'})
        except (OSError, ValueError, OverflowError) as exc:
            rec.update(instances=[], issues=[{'line': None, 'reason': f'unreadable: {exc}'}])
        records.append(rec)
    samples = {}
    for cid, name in enumerate(CLASSES):
        pool = sorted(per_class[name])
        samples[name] = random.Random(args.seed + cid).sample(pool, min(args.per_class, len(pool)))
    chosen = set(x for v in samples.values() for x in v)
    chosen.update(r['image'] for r in records if r['issues'])
    chosen = sorted(chosen)[:args.max_previews]
    out.mkdir(parents=True)
    (out / 'evidence').mkdir()
    for rec in records:
        if rec['image'] not in chosen or 'width' not in rec:
            continue
        rel = Path(rec['image'])
        with Image.open(images / rel) as im:
            original = ImageOps.exif_transpose(im).convert('RGB')
        rows, _, inst = read_labels(labels / rel.with_suffix('.txt'), *original.size, nb)
        tiles = [panel(original, [], [], nb, 'Original', args.side),
                 panel(original, rows, inst, nb, 'Before / supplied labels', args.side)]
        if after and (after / rel.with_suffix('.txt')).is_file():
            arows, _, ainst = read_labels(after / rel.with_suffix('.txt'), *original.size, nb)
            tiles.append(panel(original, arows, ainst, nb, 'After / supplied correction', args.side))
        canvas = Image.new('RGB', (args.side * len(tiles), args.side + 36))
        for k, tile in enumerate(tiles):
            canvas.paste(tile, (k * args.side, 0))
        name = hashlib.sha256(rec['image'].encode()).hexdigest()[:16] + '.png'
        canvas.save(out / 'evidence' / name)
        rec['evidence'] = 'evidence/' + name
    count = Counter(i['class_name'] for r in records for i in r['instances'])
    summary = {'scope': str(root), 'split': args.split, 'seed': args.seed,
               'images_audited': len(records), 'limit': args.limit, 'total_images_discovered': total_images,
               'images_flagged': sum(bool(r['issues']) for r in records),
               'instances_by_class': {c: count[c] for c in CLASSES},
               'images_by_class': {c: len(per_class[c]) for c in CLASSES},
               'class_order': class_order,
               'samples_by_class': samples, 'records': records,
               'note': 'Flags are review candidates, not confirmed semantic errors. No label edits or approvals performed.'}
    (out / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out / 'review.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image','instance_id','reason','evidence','decision','reviewer','note'])
        writer.writeheader()
        for rec in records:
            review_items = rec['issues'] or [{'line': None, 'reason': 'manual_completeness_check: geometry pass does not confirm all objects labelled'}]
            for issue in review_items:
                writer.writerow({'image': rec['image'], 'instance_id': f"L{issue['line']}" if issue['line'] else '',
                                 'reason': issue['reason'], 'evidence': rec.get('evidence',''),
                                 'decision': 'pending', 'reviewer': '', 'note': ''})
    print(json.dumps({k: summary[k] for k in ['images_audited','images_flagged','instances_by_class']}, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--split', choices=['train','val'])
    ap.add_argument('--labels', type=Path)
    ap.add_argument('--after-labels', type=Path)
    ap.add_argument('--names-from', type=Path, help='source data.yaml with names; must match competition order')
    ap.add_argument('--seed', type=int, default=55)
    ap.add_argument('--per-class', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0, help='0 = all; otherwise first N sorted images')
    ap.add_argument('--max-previews', type=int, default=40)
    ap.add_argument('--side', type=int, default=640)
    args = ap.parse_args()
    if min(args.limit, args.per_class, args.max_previews) < 0 or not 64 <= args.side <= 2048:
        ap.error('counts must be nonnegative; side must be 64..2048')
    try:
        run(args)
    except (ValueError, OSError) as exc:
        ap.exit(2, str(exc) + '\n')


if __name__ == '__main__':
    main()
