"""Generate notebooks/colab/VietSight_TEAM55_colab.ipynb from the OFFICIAL notebook.

Changes vs the official notebook (asserted below, everything else byte-identical):
  * parameter cell: DRIVE_OUTPUT_DIR set to Google Drive, EPOCHS
  * + one setup cell after it: Drive mount, Kaggle auth from Colab Secrets (KAGGLE_USERNAME / KAGGLE_KEY), download the
    V1 dataset built on Kaggle (prep kernel output), check fingerprint + frozen val, CUDA check
  * training cell (Step 3, "bạn được sửa tự do"): save_period, runs written to Drive, auto-resume from Drive last.pt

  python scripts/make_colab_notebook.py --epochs 25
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_kaggle_notebooks import code, md  # noqa: E402
from vslib import NOTEBOOK, ROOT  # noqa: E402

SETUP = r'''
# [VietSight] Colab setup: Drive, Kaggle (Colab Secrets), dataset V1 from the Kaggle prep kernel, integrity checks, GPU
import os, glob, hashlib, subprocess, sys
from pathlib import Path
from google.colab import drive, userdata
drive.mount("/content/drive")
Path(DRIVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Colab: key icon (Secrets) -> add KAGGLE_USERNAME and KAGGLE_KEY, enable "Notebook access". Never written to disk/notebook.
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")

PREP_KERNEL = "__PREP_KERNEL__"
V1_DIR = Path("/content/vietsight_data/versions/__VERSION__")
if not (V1_DIR / "data.yaml").is_file():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
    r = subprocess.run(["kaggle", "kernels", "output", PREP_KERNEL, "-p", "/content/vietsight_data",
                        "--file-pattern", r"versions/__VERSION__/.*"], capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-800:])
    assert (V1_DIR / "data.yaml").is_file(), "Không tải được dataset từ Kaggle: kiểm tra Secrets KAGGLE_USERNAME/KAGGLE_KEY"
n_img = {s: len(list((V1_DIR / "images" / s).glob("*.jpg"))) for s in ("train", "val")}
print("ảnh:", n_img)
assert n_img == __COUNTS__, f"số ảnh khác bản build {__COUNTS__}"
fp = dataset_fingerprint(V1_DIR)  # official helper (Thư viện 2/4)
assert fp == "__FINGERPRINT__", f"fingerprint dataset khác bản build đã QA: {fp}"
val_names = "\n".join(sorted(p.name for p in (V1_DIR / "images" / "val").glob("*.jpg")))
assert hashlib.sha256(val_names.encode()).hexdigest() == "__VAL_SHA__", "tập val khác configs/val_frozen.txt"
print("fingerprint OK, frozen val OK:", fp[:16])
DATASET_PATH = str(V1_DIR)

import torch
assert torch.cuda.is_available(), "Không có GPU: Runtime > Change runtime type > T4 GPU"
print("CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))
subprocess.run("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv", shell=True)
'''

TRAIN = r'''from ultralytics import YOLO

# [VietSight] Step 3 (free to edit per the official notebook): same args + checkpoints every SAVE_PERIOD epochs written
# straight to Google Drive, and automatic resume from Drive last.pt after a Colab disconnect.
SAVE_PERIOD = __SAVE_PERIOD__
RUN_NAME = "__RUN_NAME__"
TRAIN_ARGS = dict(
    data=DATASET["data_yaml"],
    imgsz=TRAIN_IMGSZ,      # giữ 640 (xem ô tham số)
    epochs=EPOCHS,
    batch=BATCH,
    seed=SEED,              # seed của đội
    deterministic=True,
    device=DEVICE,
    workers=WORKERS,
    patience=PATIENCE,
    cache=CACHE,
    plots=PLOTS,
    project=str(Path(DRIVE_OUTPUT_DIR) / "runs"),
    name=RUN_NAME,
    exist_ok=True,
    save_period=SAVE_PERIOD,
)
if TRAIN_ARGS["imgsz"] != IMGSZ:
    print(f"CẢNH BÁO: imgsz huấn luyện = {TRAIN_ARGS['imgsz']} khác {IMGSZ}. Hệ thống chấm chạy mô hình ở {IMGSZ}, điểm có thể thấp hơn.")
n_train, n_val = DATASET["counts"]["train"]["images"], DATASET["counts"]["val"]["images"]
print(f"Ước lượng thô (chưa đo): {n_train} ảnh train x {EPOCHS} epoch, cỡ {MODEL_SIZE} -> khoảng {fmt_minutes(estimate_train_minutes(n_train, n_val, EPOCHS, MODEL_SIZE))} trên T4.")

seed_everything(SEED)
RUN_DIR = Path(TRAIN_ARGS["project"]) / RUN_NAME
LAST = RUN_DIR / "weights" / "last.pt"
done_epochs = len((RUN_DIR / "results.csv").read_text().splitlines()) - 1 if (RUN_DIR / "results.csv").is_file() else 0
if LAST.is_file() and done_epochs < EPOCHS:
    print(f"RESUME từ {LAST} (đã xong {done_epochs}/{EPOCHS} epoch)")
    model = YOLO(str(LAST))
    try:
        model.train(resume=True)
    except (AssertionError, KeyError) as e:  # finished early (patience): optimizer stripped, nothing to resume
        print("Không resume được, dùng best.pt:", e)
        model = YOLO(str(RUN_DIR / "weights" / "best.pt"))
elif LAST.is_file():
    print(f"Run {RUN_NAME} đã train xong {done_epochs} epoch: dùng lại best.pt trên Drive")
    model = YOLO(str(RUN_DIR / "weights" / "best.pt"))
else:
__FRESH_OR_CONTINUE__
BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"
trained_names = [model.names[i] for i in sorted(model.names)]
if trained_names != list(CLASSES):
    raise NotebookError(f"Tên lớp của mô hình {trained_names} khác danh sách lớp của cuộc thi {list(CLASSES)}. Hãy huấn luyện bằng data.yaml do Bước 1 tạo.", "TRAIN")
TRAIN_INFO = {"best_weights": str(BEST_WEIGHTS), "best_sha256": sha256_file(BEST_WEIGHTS), "best_tensors_sha256": weights_fingerprint(BEST_WEIGHTS),
              "train_args": {k: v for k, v in TRAIN_ARGS.items() if k != "project"}, "device": DEVICE, "gpu": GPU_NAME}
print("Trọng số tốt nhất:", BEST_WEIGHTS, "| checkpoint định kỳ:", sorted(p.name for p in (RUN_DIR / "weights").glob("*.pt")))
'''

CONTINUE = r'''
# [VietSight] CONTINUE the e25 run to EPOCHS total. The e25 folder on Drive is only READ, never written.
# Periodic checkpoints (epochN.pt) keep the optimizer; last.pt/best.pt were stripped at the end of e25 -> resume from the
# latest epochN.pt with train_args edited (epochs=EPOCHS, save_dir = new folder). Same data, val, seed, hyper-parameters.
# Note: the LR schedule is recomputed for EPOCHS from that epoch on (not bit-identical to a fresh 50-epoch run).
import shutil, torch
SRC_RUN = Path("__SRC_RUN_DIR__")
ckpts = sorted(SRC_RUN.glob("weights/epoch*.pt"), key=lambda q: int(q.stem[5:]))
assert ckpts, f"không thấy checkpoint định kỳ trong {SRC_RUN}/weights"
src = ckpts[-1]
ck = torch.load(src, map_location="cpu", weights_only=False)
assert ck.get("optimizer") is not None, f"{src} không có optimizer"
done_src = int(ck["epoch"]) + 1
(RUN_DIR / "weights").mkdir(parents=True, exist_ok=True)
ck["train_args"].update(epochs=EPOCHS, project=str(RUN_DIR.parent), name=RUN_NAME, save_dir=str(RUN_DIR), save_period=SAVE_PERIOD,
                        data=DATASET["data_yaml"])
START = RUN_DIR / "weights" / f"start_from_e25_{src.stem}.pt"
torch.save(ck, START)
rows = (SRC_RUN / "results.csv").read_text().splitlines()
(RUN_DIR / "results.csv").write_text("\n".join(rows[: done_src + 1]) + "\n")  # keep epochs 1..done_src for the curves
print(f"Tiếp tục từ {src} ({done_src} epoch đã train) -> {EPOCHS} epoch, lưu ở {RUN_DIR}")
model = YOLO(str(START))
model.train(resume=True, data=DATASET["data_yaml"])
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--save-period", type=int, default=5)
    ap.add_argument("--version", default="vietsight_v1")
    ap.add_argument("--prep-kernel", default="mangashounen/vietsight-prep-vietsight-v1")
    ap.add_argument("--continue-from", help="run name on Drive to continue, e.g. vietsight_v1_e25 (its folder is only read)")
    a = ap.parse_args()
    stats = json.loads((ROOT / "data" / "versions" / f"{a.version}_stats.json").read_text(encoding="utf-8"))
    frozen = [x.strip() for x in (ROOT / "configs" / "val_frozen.txt").read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    val_sha = hashlib.sha256("\n".join(sorted(k.split("/", 1)[1] for k in frozen)).encode()).hexdigest()
    run_name = f"{a.version}_e{a.epochs}"
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    out = copy.deepcopy(nb)
    cells = out["cells"]
    ip = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith("#@title Tham số của đội")][0]
    it = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith("from ultralytics import YOLO\n\nTRAIN_ARGS")][0]
    src = "".join(cells[ip]["source"])
    for old, new in [('EPOCHS = 60  #@param {type:"integer"}', f'EPOCHS = {a.epochs}  #@param {{type:"integer"}}'),
                     ('DRIVE_OUTPUT_DIR = ""', f'DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/VietSight/{run_name}"')]:
        assert old in src, old
        src = src.replace(old, new)
    cells[ip]["source"] = src.splitlines(True)
    if a.continue_from:
        src_dir = f"/content/drive/MyDrive/VietSight/{a.continue_from}/runs/{a.continue_from}"
        body = "\n".join("    " + x if x else x for x in CONTINUE.strip("\n").replace("__SRC_RUN_DIR__", src_dir).splitlines())
    else:
        body = "    model = YOLO(MODEL_WEIGHTS)  # .pt: pretrained (tự tải)\n    model.train(**TRAIN_ARGS)"
    cells[it]["source"] = (TRAIN.replace("__SAVE_PERIOD__", str(a.save_period)).replace("__RUN_NAME__", run_name)
                           .replace("__FRESH_OR_CONTINUE__", body)).splitlines(True)
    setup = (SETUP.replace("__PREP_KERNEL__", a.prep_kernel).replace("__VERSION__", a.version).replace("__FINGERPRINT__", stats["fingerprint_sha256"])
             .replace("__VAL_SHA__", val_sha).replace("__COUNTS__", repr(stats["images"])))
    cells.insert(ip + 1, code(setup))
    cells.insert(ip + 1, md("## [VietSight] Colab: Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY` → tải dataset V1 từ Kaggle, kiểm tra, GPU. Chạy lại toàn bộ sau khi mất kết nối: tự resume từ Drive."))
    cells.insert(0, md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PHai237/VietSight/blob/main/notebooks/colab/VietSight_TEAM55_colab.ipynb)\n\n"
                       f"VietSight TEAM 55 — {run_name}: Runtime → T4 GPU → Run all. Kết quả: Google Drive `MyDrive/VietSight/{run_name}/`."))
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    # guard: official cells unchanged except parameter + training cells
    added = 3
    for i, oc in enumerate(nb["cells"]):
        j = i + 1 + (2 if i > ip else 0)
        if i in (ip, it):
            continue
        assert "".join(oc["source"]) == "".join(cells[j]["source"]), f"official cell {i} changed"
    assert len(cells) == len(nb["cells"]) + added
    out["metadata"]["accelerator"] = "GPU"
    out["metadata"].setdefault("colab", {})["gpuType"] = "T4"
    dst = ROOT / "notebooks" / "colab" / ("VietSight_TEAM55_colab.ipynb" if not a.continue_from else f"VietSight_TEAM55_colab_e{a.epochs}.ipynb")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(dst, f"val_sha {val_sha[:12]} fp {stats['fingerprint_sha256'][:12]}")


if __name__ == "__main__":
    main()
