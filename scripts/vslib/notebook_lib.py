"""Load the OFFICIAL helper code from notebooks/VietSight_TEAM55.ipynb instead of copying it.

The notebook's library cells are "definitions only (no side effects)", so we exec the constants cell and the
library cells in one namespace. That gives us exactly the organisers' validator: parse_label_file, polygon_problem,
area_at_imgsz, analyze_dataset, print_report, make_previews, dhash64, find_duplicate_pairs, dataset_fingerprint,
load_selfcheck (scorer), make_submission ...  Never edit those cells; if the notebook is updated, this follows.
"""
import json
from functools import lru_cache

from . import NOTEBOOK

TITLES = ("#@title Hằng số", "#@title Thư viện 1/4", "#@title Thư viện 2/4", "#@title Thư viện 3/4", "#@title Thư viện 4/4")


@lru_cache(maxsize=None)
def load(upto=3):
    """Namespace (dict) with the notebook definitions. upto=3 -> constants + libs 1..2 (data checks only: numpy+PIL);
    upto=5 -> also the scoring/export libs (need ultralytics/onnx/pycocotools at call time)."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for t in TITLES:
            if src.startswith(t):
                sources[t] = src
    missing = [t for t in TITLES[:upto] if t not in sources]
    if missing:
        raise RuntimeError(f"Notebook {NOTEBOOK} is missing cells {missing}: was it replaced by another version?")
    ns = {"__name__": "vietsight_notebook"}
    for t in TITLES[:upto]:
        exec(compile(sources[t], f"<{NOTEBOOK.name}:{t}>", "exec"), ns)  # noqa: S102 - official, local notebook
    return ns


def check_classes(classes):
    nb_classes = load()["CLASSES"]
    if list(classes) != list(nb_classes):
        raise RuntimeError(f"Class list drift! pipeline={classes} notebook={nb_classes}")
