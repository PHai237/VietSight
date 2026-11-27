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


def prep_notebook(vcfg_path, other_frac, workers, max_train=1700, max_val=300):
    b64, sha, n = code_bundle()
    vcfg = yaml.safe_load((ROOT / vcfg_path).read_text(encoding="utf-8"))
    name = vcfg["name"]
    cells = [
        md(f"""
# VietSight — chuẩn bị dữ liệu trên Kaggle ({name})
Kaggle: **Accelerator: None (CPU)**, **Internet: On**. Run all. Tải stream Mapillary, dừng khi đủ số ảnh (MAX_KEEP), chuyển đổi, build. Lỗi mạng: tự nối lại; chạy lại với FETCH=False để dùng dữ liệu đã tải.
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
MAX_KEEP = "training={max_train} validation={max_val}"   # số ảnh giữ mỗi split; tải DỪNG khi đủ (không đọc hết 21 GB)
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
import subprocess, sys


def sh(cmd, check=True):
    # run a shell command, stream its output into the notebook, STOP the notebook when it fails
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0 and check:
        raise RuntimeError(f"FAILED ({p.returncode}): {cmd}")
"""),
        code("""
sh(f'"{sys.executable}" -m pip install -q opencv-python-headless pycocotools pyyaml tqdm requests', check=False)  # offline: Kaggle image already has them
import cv2, yaml, tqdm, numpy, PIL; print('deps ok', cv2.__version__)
"""),
        code("""
import glob
ATTACHED = sorted(glob.glob("/kaggle/input/*/Mapillary Vistas")) + sorted(glob.glob("/kaggle/input/*/*/Mapillary Vistas"))
print("Mapillary attached:", ATTACHED)
if FETCH and ATTACHED:  # official v2.0 attached via Add Input: no Internet needed
    sh(f'"{sys.executable}" scripts/fetch_mapillary_kaggle.py "{ATTACHED[0]}" {RAW}/mapillary_vistas --splits training validation --prefer motorcycle motorcyclist bicycle bicyclist "other rider" "other vehicle" --other-frac {MAPILLARY_OTHER_FRAC} --max-keep {MAX_KEEP}')
elif FETCH:
    sh("df -h /tmp /kaggle/working", check=False)
    sh(f'"{sys.executable}" scripts/fetch_sly_tar.py mapillary_vistas {RAW}/mapillary_vistas --splits training validation --prefer motorcycle motorcyclist bicycle bicyclist "other rider" "other vehicle" --other-frac {MAPILLARY_OTHER_FRAC} --max-ann {MAX_ANN} --max-keep {MAX_KEEP} --max-side 1024')
sh("df -h /tmp /kaggle/working; du -sh " + RAW, check=False)
"""),
        code("""
sh(f'"{sys.executable}" scripts/convert_supervisely.py --config mapillary_vistas --root {RAW}/mapillary_vistas --workers {WORKERS} --out {PROCESSED}')
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
sh(f'"{sys.executable}" scripts/build_version.py {VERSION_CONFIG} --processed {PROCESSED} --out-root {OUT_ROOT}')
"""),
        code(f"""
import shutil
try:
    from IPython.display import Image, display
except ImportError:  # plain python (local test)
    Image, display = (lambda filename: filename), print
shutil.copyfile("configs/val_frozen.txt", f"{{OUT_ROOT}}/val_frozen.txt")
import subprocess, sys
subprocess.run([sys.executable, "scripts/preview.py", f"{{OUT_ROOT}}/{name}", "--split", "train", "--cls", "Motorcyclist", "Cyclist", "Motorcycle", "Bicycle",
                "--n", "12", "--cols", "3", "--side", "800", "--out", f"{{OUT_ROOT}}/{name}_report/preview_riders.png"], check=True)
for p in ("preview_train.png", "preview_val.png", "preview_riders.png"):
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
# [VietSight] Kaggle without Internet: pinned wheels + yolo26n-seg.pt + Arial.ttf from the attached dataset
# mangashounen/vietsight-offline-wheels. pip reads PIP_NO_INDEX/PIP_FIND_LINKS, so the official install step is unchanged.
_wh = sorted({_os.path.dirname(p) for p in _glob.glob("/kaggle/input/**/ultralytics-*.whl", recursive=True)})
if _wh:
    import shutil as _sh
    _os.environ["PIP_NO_INDEX"], _os.environ["PIP_FIND_LINKS"] = "1", _wh[0]
    if not _os.path.exists("yolo26n-seg.pt"):
        _sh.copyfile(_wh[0] + "/yolo26n-seg.pt", "yolo26n-seg.pt")
    for _d in (_os.path.expanduser("~/.config/Ultralytics"), "/tmp/Ultralytics"):
        _os.makedirs(_d, exist_ok=True)
        _sh.copyfile(_wh[0] + "/Arial.ttf", _d + "/Arial.ttf")
    print("Cài offline từ", _wh[0])
'''


def train_notebook(epochs, batch, version_name, allow_cpu=False):
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
    if allow_cpu:
        assert "REQUIRE_GPU = True" in src
        src = src.replace("REQUIRE_GPU = True", "REQUIRE_GPU = False  # [VietSight] tài khoản chưa có GPU: cho phép CPU (GPU vẫn được dùng nếu có)")
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
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--other-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kaggle-user", default="YOUR_KAGGLE_USERNAME")
    ap.add_argument("--max-train", type=int, default=1700)
    ap.add_argument("--allow-cpu", action="store_true", help="REQUIRE_GPU=False (account without GPU)")
    ap.add_argument("--train-suffix", default="", help="extra slug suffix for parallel training kernels, e.g. e5")
    ap.add_argument("--only-train", action="store_true", help="write only the train notebook (to notebooks/kaggle/train<suffix>/)")
    ap.add_argument("--max-val", type=int, default=300)
    args = ap.parse_args()
    vname = yaml.safe_load((ROOT / args.version_config).read_text(encoding="utf-8"))["name"]
    u = args.kaggle_user
    prep_slug, train_slug = f"vietsight-prep-{vname.replace('_', '-')}", f"vietsight-train-{vname.replace('_', '-')}" + (f"-{args.train_suffix}" if args.train_suffix else "")
    (OUT / "prep").mkdir(parents=True, exist_ok=True)
    tdir = OUT / (f"train_{args.train_suffix}" if args.train_suffix else "train")
    tdir.mkdir(parents=True, exist_ok=True)
    p1 = OUT / "prep" / "VietSight_prep_kaggle.ipynb"
    if not args.only_train:
        p1.write_text(json.dumps(prep_notebook(args.version_config, args.other_frac, args.workers, args.max_train, args.max_val), ensure_ascii=False, indent=1), encoding="utf-8")
    p2 = tdir / "VietSight_TEAM55_kaggle.ipynb"
    p2.write_text(json.dumps(train_notebook(args.epochs, args.batch, vname, args.allow_cpu), ensure_ascii=False, indent=1), encoding="utf-8")
    # `kaggle kernels push -p notebooks/kaggle/prep`, then `-p notebooks/kaggle/train` once prep finished. Private, Internet on.
    common = {"language": "python", "kernel_type": "notebook", "is_private": True, "enable_internet": True,
              "dataset_sources": [], "competition_sources": []}
    mapillary_ds = ["kaggleprollc/mapillary-vistas-image-data-collection"]  # official v2.0 layout, attached read-only
    if not args.only_train:
      (OUT / "prep" / "kernel-metadata.json").write_text(json.dumps(dict(common, id=f"{u}/{prep_slug}", title=prep_slug, code_file=p1.name,
                                                                          enable_gpu=False, kernel_sources=[], dataset_sources=mapillary_ds), indent=1), encoding="utf-8")
    (tdir / "kernel-metadata.json").write_text(json.dumps(dict(common, id=f"{u}/{train_slug}", title=train_slug, code_file=p2.name,
                                                               enable_gpu=True, kernel_sources=[f"{u}/{prep_slug}"],
                                                               dataset_sources=[f"{u}/vietsight-offline-wheels"]), indent=1), encoding="utf-8")
    print(f"{p1} ({p1.stat().st_size / 1024:.0f} KiB)\n{p2} ({p2.stat().st_size / 1024:.0f} KiB)\nkernels: {u}/{prep_slug} -> {u}/{train_slug}")


if __name__ == "__main__":
    main()
