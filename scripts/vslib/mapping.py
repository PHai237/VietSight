"""Source config (configs/sources/<name>.yaml) -> label roles.

mapping:
  car: Car                                   # direct class
  person: {role: person}                     # maybe rider, maybe pedestrian (decided geometrically)
  rider: {role: rider}                       # known rider, vehicle found geometrically
  human--rider--motorcyclist: {role: rider, sub: motorcycle}
  motorcycle: {role: vehicle2, sub: motorcycle}
  person-group: blackout                     # region we can not split -> painted grey (114), image kept
  crowd-x: drop                              # same, but the whole image is excluded
  boat: ignore                               # not a competition object -> instance removed, image kept
Any label not listed follows ``unmapped`` (default ignore) and is reported in summary.json.
"""
from pathlib import Path

import yaml

from . import CLASSES, ROOT


def load_source_config(name_or_path):
    p = Path(name_or_path)
    if not p.suffix:
        p = ROOT / "configs" / "sources" / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_path"] = str(p)
    cfg["roles"] = {str(k): _role(k, v) for k, v in (cfg.get("mapping") or {}).items()}
    cfg.setdefault("unmapped", "ignore")
    return cfg


def _role(label, v):
    if isinstance(v, str):
        if v in ("ignore", "drop", "blackout"):
            return {"role": v}
        if v not in CLASSES:
            raise ValueError(f"mapping {label!r} -> {v!r}: not a competition class {CLASSES}")
        if v in ("Motorcycle", "Motorcyclist", "Bicycle", "Cyclist"):
            raise ValueError(f"mapping {label!r} -> {v!r}: two-wheelers must use role vehicle2 (rider decided per image)")
        return {"role": "direct", "target": v}
    v = dict(v)
    if v.get("role") not in ("direct", "vehicle2", "rider", "person", "ignore", "drop", "blackout"):
        raise ValueError(f"mapping {label!r}: bad role {v}")
    if v["role"] == "vehicle2" and v.get("sub") not in ("motorcycle", "bicycle"):
        raise ValueError(f"mapping {label!r}: vehicle2 needs sub motorcycle|bicycle")
    return v


def role_of(cfg, label, unmapped_counter=None):
    r = cfg["roles"].get(str(label))
    if r is None:
        if unmapped_counter is not None:
            unmapped_counter[str(label)] += 1
        return {"role": cfg["unmapped"]}
    return r
