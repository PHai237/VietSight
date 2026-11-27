"""Generate the two Kaggle notebooks from the repo (re-run after every code/config change, then upload them to Kaggle).

  python scripts/make_kaggle_notebooks.py --version-config configs/versions/v1_first_gear.yaml --epochs 60

1. notebooks/kaggle/prep/VietSight_prep_kaggle.ipynb   (CPU, Internet ON) the repo code is EMBEDDED (zip, sha256 printed), it
   streams the public archives, converts them and builds data/versions/<name>; output = the dataset folder + reports.
2. notebooks/kaggle/train/VietSight_TEAM55_kaggle.ipynb (GPU, Internet ON) = the OFFICIAL notebook, only the parameter cell
   is changed (DATASET_PATH auto-detected under /kaggle/input, EPOCHS/BATCH) + one last cell that gathers the results.
   The validator / export / submission cells are byte-identical to notebooks/VietSight_TEAM55.ipynb (checked here).
"""
import argparse
import base64
import copy
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml  # noqa: E402

from vslib import NOTEBOOK, ROOT  # noqa: E402

OUT = ROOT / "notebooks" / "kaggle"


def code_bundle():
    buf = io.BytesIO()
    files = sorted([p for p in (ROOT / "scripts").rglob("*.py") if "__pycache__" not in p.parts]
                   + [p for p in (ROOT / "configs").rglob("*") if p.is_file()] + [NOTEBOOK, ROOT / "requirements.txt"])
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zi = zipfile.ZipInfo(p.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))  # reproducible zip
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, p.read_bytes())
    data = buf.getvalue()
    return base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest(), len(files)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}


def prep_notebook(vcfg_path, other_frac, workers):
    b64, sha, n = code_bundle()
    vcfg = yaml.safe_load((ROOT / vcfg_path).read_text(encoding="utf-8"))
    name = vcfg["name"]
    cells = [
        md(f"""
# VietSight — chuẩn bị dữ liệu trên Kaggle ({name})
Kaggle: **Accelerator: None (CPU)**, **Internet: On**. Run all. Thời gian: tải ~21 GB stream Mapillary + chuyển đổi (~30–60 phút).
Kết quả (Output của notebook): `versions/{name}/` (images/labels train+val, `data.yaml`) + `versions/{name}_report/` + `val_frozen.txt`.
Sau đó ở notebook huấn luyện: *Add Input → Notebook Output → notebook này*.
Code repo được nhúng (sha256 `{sha[:16]}…`, {n} tệp) — sinh bởi `scripts/make_kaggle_notebooks.py`, không sửa tay.
"""),
        code(f"""
VERSION_CONFIG = "{Path(vcfg_path).as_posix()}"
MAPILLARY_OTHER_FRAC = {other_frac}   # tỉ lệ ảnh KHÔNG có xe 2 bánh/người lái được giữ (ảnh có thì luôn giữ)
WORKERS = {workers}
RAW = "/tmp/raw"            # dữ liệu gốc: ổ tạm, không vào Output
PROCESSED = "/tmp/processed"
OUT_ROOT = "/kaggle/working/versions"
FETCH = True                # False = dùng dữ liệu đã có trong RAW (chạy lại nhanh)
MAX_ANN = 0                 # >0: chỉ đọc N annotation đầu (thử nhanh)
"""),
        code(f"""
import base64, hashlib, io, os, sys, zipfile, subprocess
BUNDLE = "{b64}"
raw = base64.b64decode(BUNDLE)
assert hashlib.sha256(raw).hexdigest() == "{sha}", "bundle hỏng"
REPO = os.path.abspath("VietSight")  # Kaggle: /kaggle/working/VietSight
zipfile.ZipFile(io.BytesIO(raw)).extractall(REPO)
os.chdir(REPO)
print("code bundle {sha[:16]} ->", REPO)
"""),
        code("""
!"{sys.executable}" -m pip install -q opencv-python-headless pycocotools pyyaml tqdm requests 2>&1 | tail -2
"""),
        code("""
if FETCH:
    !"{sys.executable}" scripts/fetch_sly_tar.py mapillary_vistas {RAW}/mapillary_vistas --splits training validation --prefer motorcycle motorcyclist bicycle bicyclist "other rider" "other vehicle" "wheeled slow" --other-frac {MAPILLARY_OTHER_FRAC} --max-ann {MAX_ANN} --max-side 1024
!df -h /tmp /kaggle/working || true
"""),
        code("""
!"{sys.executable}" scripts/convert_supervisely.py --config mapillary_vistas --root {RAW}/mapillary_vistas --workers {WORKERS} --out {PROCESSED}
"""),
        code("""
# Nguồn đã xử lý sẵn được attach (vd. dữ liệu VN của nhóm, đóng gói bằng scripts/package_kaggle.py source <tên>)
import glob, shutil, os
for meta in glob.glob("/kaggle/input/**/processed/*/meta.jsonl", recursive=True):
    src = os.path.dirname(meta)
    dst = os.path.join(PROCESSED, os.path.basename(src))
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
        print("nguồn đã xử lý:", src, "->", dst)
"""),
        code("""
!"{sys.executable}" scripts/build_version.py {VERSION_CONFIG} --processed {PROCESSED} --out-root {OUT_ROOT}
"""),
        code(f"""
import shutil
try:
    from IPython.display import Image, display
except ImportError:  # plain python (local test)
    Image, display = (lambda filename: filename), print
shutil.copyfile("configs/val_frozen.txt", f"{{OUT_ROOT}}/val_frozen.txt")
for p in ("preview_train.png", "preview_val.png"):
    display(Image(filename=f"{{OUT_ROOT}}/{name}_report/" + p))
print(open(f"{{OUT_ROOT}}/{name}_report/notebook_report.txt", encoding="utf-8").read())
print("TẢI VỀ và commit: versions/val_frozen.txt -> configs/val_frozen.txt, versions/{name}_report/stats.json -> data/versions/")
"""),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


PARAM_ANCHOR = "# Kaggle: /kaggle/input/<ten-dataset>. Máy riêng: đường dẫn trên máy.\n"
AUTODETECT = '''# [VietSight] Kaggle: để trống -> tự tìm thư mục có data.yaml + images/train + images/val trong /kaggle/input
import glob as _glob, os as _os
if not DATASET_PATH and _os.path.isdir("/kaggle/input"):
    _c = sorted(_os.path.dirname(p) for p in _glob.glob("/kaggle/input/**/data.yaml", recursive=True)
                if _os.path.isdir(_os.path.join(_os.path.dirname(p), "images", "train")))
    print("Ứng viên DATASET_PATH:", _c)
    DATASET_PATH = _c[0] if _c else ""
'''


def train_notebook(epochs, batch, version_name):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    out = copy.deepcopy(nb)
    idx = [i for i, c in enumerate(out["cells"]) if c["cell_type"] == "code" and "".join(c["source"]).startswith("#@title Tham số của đội")]
    if len(idx) != 1:
        raise SystemExit("parameter cell not found in the official notebook")
    cell = out["cells"][idx[0]]
    src = "".join(cell["source"])
    assert "TEAM_ID = 55  #@param" in src, "TEAM_ID must already be 55 in the official notebook"
    assert PARAM_ANCHOR in src and 'EPOCHS = 60  #@param {type:"integer"}' in src and 'BATCH = 16  #@param {type:"integer"}' in src
    src = src.replace(PARAM_ANCHOR, PARAM_ANCHOR + AUTODETECT)
    src = src.replace('EPOCHS = 60  #@param {type:"integer"}', f'EPOCHS = {epochs}  #@param {{type:"integer"}}')
    src = src.replace('BATCH = 16  #@param {type:"integer"}', f'BATCH = {batch}  #@param {{type:"integer"}}')
    cell["source"] = src.splitlines(True)
    cell["outputs"], cell["execution_count"] = [], None
    out["cells"].append(md("## [VietSight] Gom kết quả để tải về (best.pt, submission.zip, run_report.json, biểu đồ, báo cáo val)"))
    out["cells"].append(code(f'''
import shutil, json
from pathlib import Path
_dst = Path(WORK).resolve().parent / "vietsight_outputs" / "{version_name}"  # Kaggle: /kaggle/working/vietsight_outputs
_dst.mkdir(parents=True, exist_ok=True)
_run = Path(WORK) / "runs" / "train"
for _p in [Path(BEST_WEIGHTS), SUBMISSION["zip"], MANIFEST_PATH, Path(DATASET["out_dir"]) / "source_map.json"] + sorted(_run.glob("*.png")) + sorted(_run.glob("*.csv")) + sorted(_run.glob("*.jpg"))[:6] + sorted((Path(WORK) / "report").rglob("*.png")):
    if Path(_p).is_file():
        shutil.copy2(_p, _dst / Path(_p).name)
(_dst / "val_result.json").write_text(json.dumps(VAL_RESULT, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print("Đã gom vào", _dst, sorted(p.name for p in _dst.iterdir()))
'''))
    for c in out["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    # guard: every official cell except the parameter cell is unchanged
    for i, (a, b) in enumerate(zip(nb["cells"], out["cells"])):
        if i != idx[0]:
            assert "".join(a["source"]) == "".join(b["source"]), f"official cell {i} changed!"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version-config", default="configs/versions/v1_first_gear.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--other-frac", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kaggle-user", default="YOUR_KAGGLE_USERNAME")
    args = ap.parse_args()
    vname = yaml.safe_load((ROOT / args.version_config).read_text(encoding="utf-8"))["name"]
    u = args.kaggle_user
    prep_slug, train_slug = f"vietsight-prep-{vname.replace('_', '-')}", f"vietsight-train-{vname.replace('_', '-')}"
    (OUT / "prep").mkdir(parents=True, exist_ok=True)
    (OUT / "train").mkdir(parents=True, exist_ok=True)
    p1 = OUT / "prep" / "VietSight_prep_kaggle.ipynb"
    p1.write_text(json.dumps(prep_notebook(args.version_config, args.other_frac, args.workers), ensure_ascii=False, indent=1), encoding="utf-8")
    p2 = OUT / "train" / "VietSight_TEAM55_kaggle.ipynb"
    p2.write_text(json.dumps(train_notebook(args.epochs, args.batch, vname), ensure_ascii=False, indent=1), encoding="utf-8")
    # `kaggle kernels push -p notebooks/kaggle/prep`, then `-p notebooks/kaggle/train` once prep finished. Private, Internet on.
    common = {"language": "python", "kernel_type": "notebook", "is_private": True, "enable_internet": True,
              "dataset_sources": [], "competition_sources": []}
    (OUT / "prep" / "kernel-metadata.json").write_text(json.dumps(dict(common, id=f"{u}/{prep_slug}", title=prep_slug, code_file=p1.name,
                                                                          enable_gpu=False, kernel_sources=[]), indent=1), encoding="utf-8")
    (OUT / "train" / "kernel-metadata.json").write_text(json.dumps(dict(common, id=f"{u}/{train_slug}", title=train_slug, code_file=p2.name,
                                                                           enable_gpu=True, kernel_sources=[f"{u}/{prep_slug}"]), indent=1), encoding="utf-8")
    print(f"{p1} ({p1.stat().st_size / 1024:.0f} KiB)\n{p2} ({p2.stat().st_size / 1024:.0f} KiB)\nkernels: {u}/{prep_slug} -> {u}/{train_slug}")


if __name__ == "__main__":
    main()
