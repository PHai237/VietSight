"""Mask <-> polygon helpers. All conversions go through a raster mask at the STORED image resolution, so every
source format (COCO polygon/RLE, Supervisely bitmap/polygon, instance PNG, panoptic PNG) ends in the same code path.

One YOLO-seg line = one polygon. Instances split in several parts (occlusion by a pole, a rider...) are joined into
ONE ring with zero-width bridges, so the scorer (pycocotools frPyObjects) rasterises all parts as one instance.
"""
import cv2
import numpy as np

from . import IMGSZ

MAX_POINTS = 300  # official self-intersection check only runs for <= 400 points; keep well below


def rasterize(polys, h, w, holes=()):
    """Polygons in PIXEL coords ((n, 2) arrays) -> bool mask (h, w). ``holes`` are erased afterwards."""
    m = np.zeros((h, w), np.uint8)
    pts = [np.round(np.asarray(p, np.float64)).astype(np.int32).reshape(-1, 1, 2) for p in polys if len(p) >= 3]
    if pts:
        cv2.fillPoly(m, pts, 1)
    hp = [np.round(np.asarray(p, np.float64)).astype(np.int32).reshape(-1, 1, 2) for p in holes if len(p) >= 3]
    if hp:
        cv2.fillPoly(m, hp, 0)
    return m.astype(bool)


def area_at_640(pixels, h, w):
    return float(pixels) * (IMGSZ / max(h, w)) ** 2


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _simplify(cnt, max_points):
    eps = 0.6
    out = cv2.approxPolyDP(cnt, eps, True)
    while len(out) > max_points:
        eps *= 1.5
        out = cv2.approxPolyDP(cnt, eps, True)
    return out.reshape(-1, 2).astype(np.float64)


def _bridge(parts):
    """Join several rings into one ring: walk P to its point nearest to C, go once around C, come back."""
    parts = sorted(parts, key=lambda p: -cv2.contourArea(p.astype(np.float32)))
    poly = parts[0]
    rest = parts[1:]
    while rest:
        best = None
        for k, c in enumerate(rest):
            d = ((poly[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if best is None or d[i, j] < best[0]:
                best = (d[i, j], k, i, j)
        _, k, i, j = best
        c = np.roll(rest.pop(k), -j, axis=0)
        poly = np.concatenate([poly[: i + 1], c, c[:1], poly[i:]], axis=0)
    return poly


def mask_to_polygon(mask, min_part_frac=0.02, max_points=MAX_POINTS):
    """bool mask -> (n, 2) float PIXEL polygon (one ring, parts bridged) or None. Parts smaller than
    ``min_part_frac`` of the largest part (and < 4 px) are dropped (antialias crumbs)."""
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if len(c) >= 3]
    if not cnts:
        return None
    areas = [cv2.contourArea(c) for c in cnts]
    big = max(areas)
    keep = [c for c, a in zip(cnts, areas) if a >= max(4.0, min_part_frac * big)] or [cnts[int(np.argmax(areas))]]
    per = max(8, max_points // len(keep))
    parts = [_simplify(c, per) for c in keep]
    parts = [p for p in parts if len(p) >= 3]
    if not parts:
        return None
    return _bridge(parts) if len(parts) > 1 else parts[0]


def largest_part_polygon(mask, max_points=MAX_POINTS):
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    p = _simplify(c, max_points)
    return p if len(p) >= 3 else None


def normalize(poly_px, h, w):
    p = np.asarray(poly_px, np.float64).copy()
    p[:, 0] = np.clip((p[:, 0] + 0.5) / w, 0.0, 1.0)  # +0.5: pixel centres, matches fillPoly/frPyObjects closely
    p[:, 1] = np.clip((p[:, 1] + 0.5) / h, 0.0, 1.0)
    return p


def yolo_line(cid, poly_norm):
    return f"{cid} " + " ".join(f"{v:.6f}" for v in np.asarray(poly_norm).reshape(-1))
