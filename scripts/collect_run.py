"""Bring a Kaggle training run home and log it.

  python scripts/collect_run.py <downloaded vietsight_outputs/<version> folder> --name v1_e60 [--note "..."]

Copies everything into runs/<name>/ (best.pt / submission.zip / model.onnx stay local: gitignored) and appends one row to
runs/RESULTS.md + runs/results.jsonl: dataset fingerprint, val mask mAP50-95, AP50, per-class AP, train args, zip sha256.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vslib import CLASSES, ROOT  # noqa: E402

SHORT = ["Car", "Bus", "Truck", "Oth", "Moto", "Motoist", "Bike", "Cyclist", "Ped", "Sign", "TL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--public-lb", type=float, help="public leaderboard score once submitted")
    a = ap.parse_args()
    src, dst = Path(a.src), ROOT / "runs" / a.name
    if dst.exists():
        raise SystemExit(f"{dst} exists: choose another --name")
    shutil.copytree(src, dst)
    rep = json.loads((dst / "run_report.json").read_text(encoding="utf-8"))
    ve = rep.get("val_estimate", {})
    pc = ve.get("per_class_ap", {}) or {}
    row = {"name": a.name, "date": datetime.now().isoformat(timespec="minutes"), "dataset_sha256": rep["dataset"]["sha256"],
           "counts": rep["dataset"]["counts"], "map50_95": ve.get("map50_95"), "map50": ve.get("map50"), "per_class_ap": pc,
           "epochs": rep["params"].get("EPOCHS"), "batch": rep["params"].get("BATCH"), "zip_sha256": rep["submission"]["zip_sha256"],
           "public_lb": a.public_lb, "note": a.note}
    with open(ROOT / "runs" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    md = ROOT / "runs" / "RESULTS.md"
    if not md.is_file():
        md.write_text("# VietSight runs (val = frozen configs/val_frozen.txt, scored by the notebook's embedded grader)\n\n"
                      "| run | date | data sha | train/val img | ep | mAP50-95 | mAP50 | " + " | ".join(SHORT) + " | public LB | note |\n"
                      "|" + "---|" * (9 + len(SHORT)) + "\n", encoding="utf-8")

    def f(x):
        return "-" if x is None else f"{x:.3f}"

    cnt = rep["dataset"]["counts"]
    line = (f"| {a.name} | {row['date'][:16]} | {row['dataset_sha256'][:8]} | {cnt['train']['images']}/{cnt['val']['images']} | {row['epochs']} | "
            f"{f(row['map50_95'])} | {f(row['map50'])} | " + " | ".join(f(pc.get(c)) for c in CLASSES) + f" | {a.public_lb if a.public_lb is not None else '-'} | {a.note} |\n")
    with open(md, "a", encoding="utf-8") as fh:
        fh.write(line)
    print(line)


if __name__ == "__main__":
    main()
