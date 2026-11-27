"""VietSight data pipeline library (shared by every script in scripts/)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['Car', 'Bus', 'Truck', 'Other vehicle', 'Motorcycle', 'Motorcyclist', 'Bicycle', 'Cyclist', 'Pedestrian', 'Sign', 'Traffic light']
CLASS_ID = {n: i for i, n in enumerate(CLASSES)}
IMGSZ = 640
MIN_GT_AREA_PX = 64.0  # scorer ignores GT masks smaller than this at 640 scale
NOTEBOOK = ROOT / "notebooks" / "VietSight_TEAM55.ipynb"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
