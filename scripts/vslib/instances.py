"""Turn the raw instances of ONE image (any source) into competition instances.

Each raw instance: dict(role=..., mask=bool HxW, label=source label name, sub=..., target=...)
  role 'direct'   -> target class name (Car, Bus, Truck, Other vehicle, Pedestrian, Sign, Traffic light)
  role 'vehicle2' -> two-wheeler, sub = 'motorcycle' | 'bicycle'
  role 'rider'    -> human KNOWN to be riding (Mapillary/BDD/IDD/Cityscapes 'rider'); sub = 'motorcycle'|'bicycle'|None
  role 'person'   -> human, may be a pedestrian or a rider (COCO 'person')
  role 'ignore'   -> dropped, image kept
  role 'drop'     -> group/crowd region of a class we label: image can not be labelled completely -> image excluded

Rule (official sample_data.zip, docs/LABEL_GUIDELINE.md): Motorcyclist / Cyclist = ONE mask = vehicle UNION its rider(s);
Motorcycle / Bicycle = two-wheeler with nobody on it. A rider whose vehicle can not be found, or a person/vehicle pair we
can not decide, makes the image AMBIGUOUS: it is not written to the auto dataset but sent to the review queue.
"""
import cv2
import numpy as np

from . import CLASS_ID
from .geometry import area_at_640, bbox

VEH_CLASS = {"motorcycle": ("Motorcycle", "Motorcyclist"), "bicycle": ("Bicycle", "Cyclist")}


def _bbox_inter(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _touch(hmask, vmask, k):
    if k > 0:
        hmask = cv2.dilate(hmask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)
    return bool((hmask & vmask).any())


def _ride_score(hb, vb, hmask, vmask, k):
    """How much human box ``hb`` looks like it sits on vehicle box ``vb`` (0 = not at all)."""
    inter = _bbox_inter(hb, vb)
    if inter == 0:
        return 0.0, 0.0
    h_area = max(1, (hb[2] - hb[0]) * (hb[3] - hb[1]))
    ratio = inter / h_area  # share of the human box over the vehicle
    hcx = 0.5 * (hb[0] + hb[2])
    vw, vh = vb[2] - vb[0], vb[3] - vb[1]
    centred = vb[0] - 0.15 * vw <= hcx <= vb[2] + 0.15 * vw
    # feet / lower body inside the vehicle's vertical extent (a pedestrian standing behind a bike has feet below it or far above)
    feet_ok = vb[1] + 0.2 * vh <= hb[3] <= vb[3] + 0.15 * vh
    touch = _touch(hmask, vmask, k)
    strong = ratio if (centred and feet_ok and touch) else 0.0
    return strong, ratio


def resolve(raw, h, w, min_area_640=64.0, strong_thr=0.25, weak_thr=0.05):
    """-> (instances [(class_id, mask)], flags [str], stats dict). ``flags`` non-empty = image needs human review."""
    flags, out = [], []
    k = max(2, int(round(0.006 * max(h, w))))
    if any(r["role"] == "drop" for r in raw):
        labels = sorted({r["label"] for r in raw if r["role"] == "drop"})
        return [], [f"group/crowd region: {', '.join(labels)}"], {}
    for r in raw:
        r["bb"] = bbox(r["mask"])
    raw = [r for r in raw if r["bb"] is not None and r["role"] != "ignore"]
    vehicles = [r for r in raw if r["role"] == "vehicle2"]
    humans = [r for r in raw if r["role"] in ("rider", "person")]
    owner = {}  # vehicle index -> [human]
    for hu in humans:
        cands = [(vi, *_ride_score(hu["bb"], v["bb"], hu["mask"], v["mask"], k)) for vi, v in enumerate(vehicles)
                 if hu["role"] != "rider" or hu.get("sub") in (None, v["sub"])]
        strong = [c for c in cands if c[1] >= (0.10 if hu["role"] == "rider" else strong_thr)]
        if hu["role"] == "rider" and not strong:  # rider: any touching vehicle of the right type is fine
            strong = [c for c in cands if c[2] > 0 and _touch(hu["mask"], vehicles[c[0]]["mask"], k)]
        if strong:
            vi = max(strong, key=lambda c: (c[1], c[2]))[0]
            owner.setdefault(vi, []).append(hu)
            hu["assigned"] = True
        elif hu["role"] == "rider":
            flags.append(f"rider '{hu['label']}' without a visible vehicle")
        else:
            weak = [c for c in cands if c[2] >= weak_thr]
            if weak:
                hu["weak"] = [c[0] for c in weak]
            out.append(("Pedestrian", hu["mask"], hu))
    # a pedestrian weakly overlapping a FREE two-wheeler: can not tell walking-the-bike / mounting / behind -> review
    for name, _, hu in out:
        for vi in hu.get("weak", []):
            if vi not in owner:
                flags.append(f"person '{hu['label']}' partly overlaps a riderless {vehicles[vi]['sub']}")
    final = [(n, m) for n, m, _ in out]
    for vi, v in enumerate(vehicles):
        free_cls, ridden_cls = VEH_CLASS[v["sub"]]
        if vi in owner:
            m = v["mask"].copy()
            for hu in owner[vi]:
                m |= hu["mask"]
            final.append((ridden_cls, m))
        else:
            final.append((free_cls, v["mask"]))
    for r in raw:
        if r["role"] == "direct":
            final.append((r["target"], r["mask"]))
    insts, small = [], 0
    for name, m in final:
        if area_at_640(m.sum(), h, w) < min_area_640:
            small += 1
            continue
        insts.append((CLASS_ID[name], m))
    return insts, flags, {"dropped_small": small}
