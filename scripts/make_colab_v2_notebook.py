"""Generate the V2 Colab notebook (fine-tune V1 e25 best.pt on the Vietnam-only V2 dataset handed over by Codex).

Changes vs the OFFICIAL notebook (asserted below, every other cell byte-identical):
  * parameter cell: EPOCHS, DATASET_PATH (fixed local folder filled by the setup cell -> re-running the parameter cell can
    no longer leave it empty and trigger the upload prompt), DRIVE_OUTPUT_DIR (one Drive folder per run)
  * + setup cell: Drive/Secrets, handoff gate (status READY), dataset download with progress, image/label completeness,
    fingerprint + VN-val checks, Mapillary V1 val (separate), e25 best.pt (sha-checked) as init weights, GPU check
  * training cell (Step 3, free to edit): fresh fine-tune from e25 best.pt (new optimizer + LR schedule), checkpoints on
    Drive, resume=True only for THIS run after a disconnect and only with the same dataset/init/epochs
  * + last cell: e25 vs this run on VN val and Mapillary val with the official platform inference + scorer

  python scripts/make_colab_v2_notebook.py --epochs 12                       # handoff pending: stops before training
  python scripts/make_colab_v2_notebook.py --epochs 12 --handoff configs/handoff/v2.json
  # fallback experiment: low LR, frozen backbone (docs/V2_SMALL_FINETUNE.md)
  python scripts/make_colab_v2_notebook.py --epochs 12 --lr0 0.0001 --freeze 10 --handoff configs/handoff/v2.json
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

DRIVE = "/content/drive/MyDrive/VietSight"
LOCAL_DATASET = "/content/vietsight_data/v2_current"

SETUP = r'''
# [VietSight V2] setup: Drive + Secrets, handoff gate, V2 dataset (progress + checks), Mapillary val, e25 init, GPU
import os, json, time, shutil, zipfile, hashlib, subprocess, threading, collections
from pathlib import Path

# ---- HANDOFF from Codex (docs/V2_DATA_HANDOFF.md). Training starts only when status == "READY". ----
# source: "drive:/content/drive/MyDrive/....zip" | "drive:/content/drive/MyDrive/<folder>" | "kaggle-dataset:<owner>/<slug>"
#         | "kaggle-kernel:<owner>/<kernel>:<file regex>"
HANDOFF = __HANDOFF__
HANDOFF.update(globals().get("HANDOFF_OVERRIDE", {}))
# baseline/init checkpoint (V1 e25, or V3 final002 e50 best.pt for --init v3best); its Drive copy is only read
BASE_TAG = "__BASE_TAG__"
E25 = {"best": "__BASE_BEST__", "best_sha256": "__E25_BEST_SHA__",
       "zip": "__BASE_ZIP__", "zip_sha256": "__E25_ZIP_SHA__"}
E25.update(globals().get("E25_OVERRIDE", {}))
MAPILLARY_VAL = {"source": "kaggle-kernel:__PREP_KERNEL__:versions/vietsight_v1/(images|labels)/val/.*",
                 "dir": "/content/vietsight_data/mapillary_val", "n": __V1_NVAL__, "val_sha256": "__V1_VAL_SHA__"}
MAPILLARY_VAL.update(globals().get("MAPILLARY_VAL_OVERRIDE", {}))
REQUIRE_CUDA = globals().get("REQUIRE_CUDA", True)
# Emergency export (GPU quota / time running out): set to a checkpoint of THIS run on Drive, e.g. "last.pt" or
# "epoch25.pt" (= after 26 epochs), then Run all (CPU runtime is fine). No training: the official export + self-check
# run on that checkpoint, results go to DRIVE_OUTPUT_DIR/export_<name>/ (the run folder itself is not overwritten).
EXPORT_ONLY_CKPT = ""  #@param {type:"string"}
if EXPORT_ONLY_CKPT:
    REQUIRE_GPU = False  # read by the official environment cell below
    print("*** CHẾ ĐỘ CHỈ EXPORT:", EXPORT_ONLY_CKPT, "— không train ***")

if Path(DRIVE_OUTPUT_DIR).name in ("vietsight_v1_e25", "vietsight_v1_e50", "vietsight_mixed_final002_v3_pretrained_e50_EXPERIMENT"):
    raise NotebookError("DRIVE_OUTPUT_DIR trỏ vào run đã nộp: đặt thư mục mới cho run này.", "PARAMS")
if HANDOFF.get("status") != "READY" or not HANDOFF.get("fingerprint_sha256"):
    raise NotebookError(f"Dataset V2 chưa bàn giao ở trạng thái READY (status={HANDOFF.get('status')!r}). "
                        "Chờ docs/V2_DATA_HANDOFF.md của Codex rồi điền HANDOFF; không huấn luyện.", "DATA")
if IN_COLAB:
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):  # Colab Secrets (key icon); never written to the notebook
        try:
            os.environ[k] = userdata.get(k)
        except Exception:
            print(f"Secret {k} chưa có (chỉ cần khi tải từ Kaggle)")
Path(DRIVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
if not HANDOFF.get("submission_eligible"):
    (Path(DRIVE_OUTPUT_DIR) / "NOT_FOR_SUBMISSION.txt").write_text(
        "BẢN TRAIN THỬ NGHIỆM: quyền sử dụng nguồn dữ liệu V2 chưa được xác nhận (rights_confirmed != true).\n"
        "Không nộp submission.zip trong thư mục này lên cuộc thi.\n", encoding="utf-8")
    print("*** BẢN THỬ NGHIỆM — quyền nguồn chưa xác nhận: KHÔNG nộp submission.zip của run này ***")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _du(d):
    fs = [f for f in Path(d).rglob("*") if f.is_file()]
    return len(fs), sum(f.stat().st_size for f in fs) / 1e6


def _run_progress(cmd, watch):
    """Run a download command, print files/MB arrived every 10 s (the Kaggle CLI prints nothing useful per file)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = collections.deque(maxlen=12)
    th = threading.Thread(target=lambda: [tail.append(x.rstrip()) for x in p.stdout], daemon=True)
    th.start()
    t0 = time.time()
    while True:
        try:
            p.wait(timeout=10)
            break
        except subprocess.TimeoutExpired:
            n, mb = _du(watch)
            print(f"  tải... {time.time() - t0:4.0f}s  {n} tệp  {mb:.1f} MB", flush=True)
    th.join(5)
    n, mb = _du(watch)
    print(f"  xong {time.time() - t0:.0f}s  {n} tệp  {mb:.1f} MB")
    if p.returncode:
        raise NotebookError("Tải thất bại (Secrets KAGGLE_USERNAME/KAGGLE_KEY? quyền truy cập?):\n" + "\n".join(tail), "DATA")


def _copy_progress(src, dst):
    size, done, last = Path(src).stat().st_size, 0, 0.0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for b in iter(lambda: fi.read(8 << 20), b""):
            fo.write(b)
            done += len(b)
            if time.time() - last > 5 or done == size:
                print(f"  chép {done / 1e6:.0f}/{size / 1e6:.0f} MB", flush=True)
                last = time.time()


def _extract_progress(zp, dst):
    with zipfile.ZipFile(zp) as zf:
        members = zf.infolist()
        for i, m in enumerate(members, 1):
            if not (Path(dst) / m.filename).resolve().is_relative_to(Path(dst).resolve()):
                raise NotebookError(f"zip có đường dẫn không an toàn: {m.filename}", "DATA")
            zf.extract(m, dst)
            if i % 500 == 0 or i == len(members):
                print(f"  giải nén {i}/{len(members)}", flush=True)


def fetch(spec, stage):
    """Bring a dataset source into the empty folder `stage` (progress printed). Returns nothing; caller locates root."""
    shutil.rmtree(stage, ignore_errors=True)
    Path(stage).mkdir(parents=True)
    kind, _, rest = spec.partition(":")
    if kind in ("drive", "dir"):
        src = Path(rest)
        if src.is_dir():
            files = [f for f in src.rglob("*") if f.is_file()]
            for i, f in enumerate(files, 1):
                (Path(stage) / f.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, Path(stage) / f.relative_to(src))
                if i % 500 == 0 or i == len(files):
                    print(f"  chép {i}/{len(files)} tệp", flush=True)
        elif src.suffix.lower() == ".zip":
            _copy_progress(src, Path(stage) / "_v2.zip")
            _extract_progress(Path(stage) / "_v2.zip", stage)
            (Path(stage) / "_v2.zip").unlink()
        else:
            raise NotebookError(f"Không thấy nguồn {src} (Drive đã mount? đường dẫn đúng chữ hoa/thường?)", "DATA")
    elif kind == "kaggle-dataset":
        _run_progress(["kaggle", "datasets", "download", rest, "-p", str(stage)], stage)
        for zp in Path(stage).glob("*.zip"):
            _extract_progress(zp, stage)
            zp.unlink()
    elif kind == "kaggle-kernel":
        kernel, _, pattern = rest.partition(":")
        _run_progress(["kaggle", "kernels", "output", kernel, "-p", str(stage)] + (["--file-pattern", pattern] if pattern else []), stage)
    else:
        raise NotebookError(f"Nguồn không hỗ trợ: {spec}", "DATA")


def find_root(stage, splits):
    """The folder holding images/<split> and labels/<split> (exactly one)."""
    roots = sorted({p.parent for p in Path(stage).rglob("images") if p.is_dir() and all((p / s).is_dir() for s in splits)})
    if len(roots) != 1:
        raise NotebookError(f"Cần đúng một thư mục chứa images/{{{','.join(splits)}}} trong nguồn, thấy {[str(r) for r in roots]}", "DATA")
    return roots[0]


def check_split(root, split):
    """Every image has a label file and vice versa; label lines = class 0..10 + >=3 normalized points. Returns #images."""
    imgs = {p.stem for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS}
    labs = {p.stem for p in (root / "labels" / split).glob("*.txt")}
    if imgs - labs or labs - imgs:
        raise NotebookError(f"{split}: {len(imgs - labs)} ảnh thiếu nhãn (vd {sorted(imgs - labs)[:3]}), "
                            f"{len(labs - imgs)} nhãn thiếu ảnh (vd {sorted(labs - imgs)[:3]})", "DATA")
    bad = []
    for t in (root / "labels" / split).glob("*.txt"):
        for ln in t.read_text(encoding="utf-8").splitlines():
            v = ln.split()
            if not v:
                continue
            try:
                ok = 0 <= int(v[0]) < len(CLASSES) and len(v) >= 7 and len(v) % 2 == 1 and all(0 <= float(x) <= 1 for x in v[1:])
            except ValueError:
                ok = False
            if not ok:
                bad.append(t.name)
                break
    if bad:
        raise NotebookError(f"{split}: {len(bad)} tệp nhãn sai định dạng YOLO-seg (vd {bad[:3]})", "DATA")
    return len(imgs)


def names_sha(d):
    return hashlib.sha256("\n".join(sorted(p.name for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS)).encode()).hexdigest()


# ---- 1. V2 dataset -> DATASET_PATH (the fixed folder named in the parameter cell) ----
target, marker = Path(DATASET_PATH), Path(DATASET_PATH) / ".vietsight_ok"
# V4: same image/label payload (fingerprint) as V3, told apart only by the sampling list -> both in the cache key
DATA_KEY = HANDOFF["fingerprint_sha256"] + (":" + HANDOFF["sampling_sha256"] if HANDOFF.get("sampling_sha256") else "")
if marker.is_file() and marker.read_text().strip() == DATA_KEY and (not HANDOFF.get("training_yaml") or (target / HANDOFF["training_yaml"]).is_file()):
    print("Dataset V2 đã có sẵn ở", target)
else:
    print(f"Tải dataset V2 {HANDOFF.get('version')} từ {HANDOFF['source']}")
    stage = target.parent / "_stage_v2"
    fetch(HANDOFF["source"], stage)
    root = find_root(stage, ("train", "val"))
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(root), str(target))
    shutil.rmtree(stage, ignore_errors=True)
counts = {s: check_split(target, s) for s in ("train", "val")}
print("ảnh/nhãn đầy đủ:", counts)
if counts != HANDOFF["images"]:
    raise NotebookError(f"Số ảnh {counts} khác bản bàn giao {HANDOFF['images']}", "DATA")
fp = dataset_fingerprint(target)  # official helper (Thư viện 2/4)
if fp != HANDOFF["fingerprint_sha256"]:
    raise NotebookError(f"Fingerprint {fp} khác bản bàn giao {HANDOFF['fingerprint_sha256']}", "DATA")
# Separately scored validation sets: name -> (image paths, labels dir). Two handoff schemas:
#   validation_sets: {name: {path: "images/<split>", images, names_sha256}}  (V3: images/val + images/val_v1, own folders)
#   val_subsets:     {name: {match: <regex on file name>, images, names_sha256}} (parts of one images/val folder)
VAL_SETS = {}
_imgs = lambda split: sorted((p for p in (target / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS), key=lambda p: p.name)  # str order on every OS
_names_sha = lambda paths: hashlib.sha256("\n".join(p.name for p in paths).encode()).hexdigest()
if HANDOFF.get("validation_sets"):
    for name, spec in HANDOFF["validation_sets"].items():
        split = Path(spec["path"]).name
        if not spec["path"].startswith("images/") or split == "train":
            raise NotebookError(f"validation_sets '{name}': path {spec['path']!r} phải là images/<split val>", "DATA")
        n = check_split(target, split)  # every image has a label, YOLO-seg lines valid
        sel = _imgs(split)
        if n != spec["images"] or _names_sha(sel) != spec["names_sha256"]:
            raise NotebookError(f"val '{name}' ({spec['path']}): {n} ảnh / sha {_names_sha(sel)[:16]} khác bàn giao "
                                f"({spec['images']} / {spec['names_sha256'][:16]})", "DATA")
        VAL_SETS[name] = (sel, target / "labels" / split)
    if not any(Path(s["path"]).name == "val" for s in HANDOFF["validation_sets"].values()):
        raise NotebookError("validation_sets phải có tập images/val (tập val dùng khi train)", "DATA")
elif HANDOFF.get("val_subsets"):
    import re
    val_imgs = _imgs("val")
    for name, spec in HANDOFF["val_subsets"].items():
        sel = [p for p in val_imgs if re.search(spec["match"], p.name)]
        if len(sel) != spec["images"] or _names_sha(sel) != spec["names_sha256"]:
            raise NotebookError(f"val '{name}': {len(sel)} ảnh / sha {_names_sha(sel)[:16]} khác bàn giao ({spec['images']} / {spec['names_sha256'][:16]})", "DATA")
        VAL_SETS[name] = (sel, target / "labels" / "val")
    used = [p for s, _ in VAL_SETS.values() for p in s]
    if len(used) != len(set(used)) or set(used) != set(val_imgs):
        raise NotebookError("val_subsets phải chia images/val thành các phần rời nhau, phủ đủ mọi ảnh val", "DATA")
elif names_sha(target / "images" / "val") != HANDOFF["vn_val_sha256"]:
    raise NotebookError("Tập val Việt Nam khác danh sách đã khóa (vn_val_sha256)", "DATA")
if VAL_SETS:
    train_names = {p.name for p in (target / "images" / "train").iterdir()}
    leak = train_names & {p.name for s, _ in VAL_SETS.values() for p in s}
    if leak:
        raise NotebookError(f"Ảnh val nằm cả trong train: {sorted(leak)[:3]}", "DATA")
    V1_VAL_SET = next((k for k, (s, _) in VAL_SETS.items() if _names_sha(s) == MAPILLARY_VAL["val_sha256"]), None)
    print("val chấm riêng:", {k: len(s) for k, (s, _) in VAL_SETS.items()}, "| val_frozen V1 =", V1_VAL_SET or "không có")
# ---- weighted sampling (V4 handoff: training_yaml + sampling_sha256): check the list BEFORE any training ----
WEIGHTED = None
if HANDOFF.get("training_yaml"):
    import yaml
    wy = yaml.safe_load((target / HANDOFF["training_yaml"]).read_text(encoding="utf-8"))
    wtxt = target / wy["train"]
    if _sha(wtxt) != HANDOFF["sampling_sha256"]:
        raise NotebookError(f"{wtxt.name} sha256 {_sha(wtxt)[:16]} khác sampling_sha256 {HANDOFF['sampling_sha256'][:16]} của handoff", "DATA")
    draws = [ln.strip() for ln in wtxt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    names = [Path(d).name for d in draws]
    if any(Path(d).parent.as_posix().lstrip("./") != "images/train" for d in draws):
        raise NotebookError(f"{wtxt.name}: mọi dòng phải trỏ vào images/train", "DATA")
    missing = set(names) - {q.name for q in (target / "images" / "train").iterdir()}
    per_src = {"V1": sum(n.startswith("mapillary_vistas__") for n in names)}
    per_src["V2"] = len(names) - per_src["V1"]
    exp_src = {k: v["epoch_draws"] for k, v in HANDOFF["sampling"]["source_draws"].items()}
    print(f"Sampling weighted: {len(draws)} lượt/epoch, {len(set(names))} ảnh unique, theo nguồn {per_src} | sha {HANDOFF['sampling_sha256'][:16]}")
    if missing or len(draws) != HANDOFF["train_epoch_draws"] or len(set(names)) != counts["train"] or per_src != exp_src:
        raise NotebookError(f"Sampling sai: {len(draws)} lượt (cần {HANDOFF['train_epoch_draws']}), {len(set(names))} unique "
                            f"(cần {counts['train']}), nguồn {per_src} (cần {exp_src}), thiếu ảnh {sorted(missing)[:3]}", "DATA")
    WEIGHTED = {"yaml": HANDOFF["training_yaml"], "names": names, "per_source": per_src, "sha256": HANDOFF["sampling_sha256"]}
marker.write_text(DATA_KEY)
print("Dataset OK: fingerprint", fp[:16], "| val khóa OK")

# ---- 2. Mapillary V1 val (separate comparison only, never trained on) ----
MAPILLARY_OK = False
try:
    if VAL_SETS and V1_VAL_SET:
        raise RuntimeError(f"Mapillary val V1 đã có trong dataset ('{V1_VAL_SET}', đúng val_frozen), không tải thêm")
    mv = Path(MAPILLARY_VAL["dir"])
    if not (mv / "images" / "val").is_dir() or names_sha(mv / "images" / "val") != MAPILLARY_VAL["val_sha256"]:
        print("Tải Mapillary val (V1, chỉ để so sánh)")
        fetch(MAPILLARY_VAL["source"], mv.parent / "_stage_mv")
        r = find_root(mv.parent / "_stage_mv", ("val",))
        shutil.rmtree(mv, ignore_errors=True)
        shutil.move(str(r), str(mv))
        shutil.rmtree(mv.parent / "_stage_mv", ignore_errors=True)
    n = check_split(mv, "val")
    assert n == MAPILLARY_VAL["n"] and names_sha(mv / "images" / "val") == MAPILLARY_VAL["val_sha256"], "Mapillary val khác val_frozen V1"
    MAPILLARY_OK = True
    print("Mapillary val OK:", n, "ảnh")
except Exception as e:  # comparison only: never blocks training
    print("CẢNH BÁO: bỏ qua đánh giá Mapillary val:", e)

INIT_MODE = "__INIT_MODE__"
if INIT_MODE in ("e25", "v3best"):
    # ---- 3. init weights = BASE_TAG best.pt (Drive copy is only read) ----
    INIT_SRC = Path(E25["best"])
    INIT_WEIGHTS = Path(f"/content/init/{BASE_TAG}_best.pt") if IN_COLAB else Path(WORK_DIR) / "init" / f"{BASE_TAG}_best.pt"
    INIT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(INIT_SRC, INIT_WEIGHTS)
    if _sha(INIT_WEIGHTS) != E25["best_sha256"]:
        raise NotebookError(f"{INIT_SRC} không phải best.pt {BASE_TAG} đã nộp (sha256 khác run_report)", "MODEL")
    print(f"Khởi tạo từ {BASE_TAG} best.pt OK:", E25["best_sha256"][:16])
else:
    # ---- 3. init weights = official COCO-pretrained yolo26n-seg.pt (Ultralytics release asset), sha256 pinned ----
    # Defined here, CALLED in the training cell: ultralytics is only importable after the official install cell
    # (this setup cell runs before it; importing ultralytics here was the V2 pretrained failure mode).
    PRETRAINED_SHA = "__PRETRAINED_SHA__"
    INIT_WEIGHTS = Path("/content/init/yolo26n-seg.pt") if IN_COLAB else Path(WORK_DIR) / "init" / "yolo26n-seg.pt"
    INIT_SRC = None

    def init_pretrained():
        INIT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        drive_copy = Path("__DRIVE__/init/yolo26n-seg.pt")
        if drive_copy.is_file():
            shutil.copyfile(drive_copy, INIT_WEIGHTS)
            src = drive_copy
        else:
            from ultralytics.utils.downloads import attempt_download_asset
            src = Path(attempt_download_asset(str(INIT_WEIGHTS)))
            if src.resolve() != INIT_WEIGHTS.resolve():
                shutil.copyfile(src, INIT_WEIGHTS)
        if _sha(INIT_WEIGHTS) != PRETRAINED_SHA:
            raise NotebookError(f"yolo26n-seg.pt sha256 {_sha(INIT_WEIGHTS)[:16]} khác bản chính thức đã ghim {PRETRAINED_SHA[:16]}. "
                                f"Upload yolo26n-seg.pt của repo lên {drive_copy} rồi chạy lại.", "MODEL")
        print("Khởi tạo từ yolo26n-seg.pt pretrained chính thức OK:", PRETRAINED_SHA[:16], "| nguồn", src)
        return src

import torch  # preinstalled by Colab; not part of the pinned specs
if REQUIRE_CUDA and not EXPORT_ONLY_CKPT:
    assert torch.cuda.is_available(), "Không có GPU: Runtime > Change runtime type > T4 GPU"
    print("CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0))
'''

TRAIN = r'''from ultralytics import YOLO

# [VietSight V4] weighted sampling: the official prepare step above wrote its own data.yaml (images/train, no repeats).
# Point training at the checked weighted list instead, mapped onto the prepared folder; the official data.yaml is untouched.
if WEIGHTED is not None:
    _out = Path(DATASET["out_dir"]).resolve()
    _have = {q.name for q in (_out / "images" / "train").iterdir()}
    if set(WEIGHTED["names"]) - _have:
        raise NotebookError(f"Bước chuẩn bị đổi tên ảnh: thiếu {sorted(set(WEIGHTED['names']) - _have)[:3]} trong {_out}/images/train", "DATA")
    (_out / "train_weighted.txt").write_text("".join(f"./images/train/{n}\n" for n in WEIGHTED["names"]), encoding="utf-8")
    _wyaml = _out / "data_weighted.yaml"
    _wyaml.write_text(f"path: {_out}\ntrain: train_weighted.txt\nval: images/val\nnames:\n"
                      + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)), encoding="utf-8")
    from ultralytics.data import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    _ds = YOLODataset(img_path=check_det_dataset(str(_wyaml))["train"], imgsz=TRAIN_IMGSZ, augment=False,
                      data=check_det_dataset(str(_wyaml)), task="segment")
    _lens = (len(_ds), len(set(_ds.im_files)))
    print(f"Loader Ultralytics (weighted): {_lens[0]} lượt/epoch, {_lens[1]} ảnh unique")
    if _lens != (HANDOFF["train_epoch_draws"], DATASET["counts"]["train"]["images"]):
        raise NotebookError(f"Loader thấy {_lens}, cần ({HANDOFF['train_epoch_draws']}, {DATASET['counts']['train']['images']})", "DATA")
    del _ds
    DATASET["data_yaml"] = str(_wyaml)

# [VietSight V2] Step 3 (free to edit per the official notebook): FINE-TUNE V1 e25 best.pt on the Vietnam-only V2 data.
# Fresh train() = new optimizer + new LR schedule (never resume=True across datasets). resume=True is used only to continue
# THIS run after a Colab disconnect, and only if its dataset, init weights and EPOCHS are the ones it started with.
SAVE_PERIOD = __SAVE_PERIOD__
RUN_NAME = Path(DRIVE_OUTPUT_DIR).name
# warmup_bias_lr = lr0: the Ultralytics default (0.1) would start the bias LR ~300x above lr0 and erode e25.
# freeze = first N layers kept at e25 (10 = backbone up to SPPF; None = train all) -> less forgetting on a small set.
FT_ARGS = __FT_ARGS__
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
    **FT_ARGS,
)
if TRAIN_ARGS["imgsz"] != IMGSZ:
    print(f"CẢNH BÁO: imgsz huấn luyện = {TRAIN_ARGS['imgsz']} khác {IMGSZ}. Hệ thống chấm chạy mô hình ở {IMGSZ}, điểm có thể thấp hơn.")
n_train, n_val = DATASET["counts"]["train"]["images"], DATASET["counts"]["val"]["images"]
print(f"Ước lượng thô (chưa đo): {n_train} ảnh train x {EPOCHS} epoch, cỡ {MODEL_SIZE} -> khoảng {fmt_minutes(estimate_train_minutes(n_train, n_val, EPOCHS, MODEL_SIZE))} trên T4.")

seed_everything(SEED)
RUN_DIR = Path(TRAIN_ARGS["project"]) / RUN_NAME
LAST = RUN_DIR / "weights" / "last.pt"
STAMP = RUN_DIR / "vietsight_run.json"
if INIT_MODE == "pretrained":
    INIT_SRC = init_pretrained()  # after the official install cell: ultralytics is importable now
INIT_SHA = sha256_file(INIT_WEIGHTS)
EXPORT_CK = RUN_DIR / "weights" / EXPORT_ONLY_CKPT if EXPORT_ONLY_CKPT else None
if EXPORT_CK is not None:
    if not EXPORT_CK.is_file():
        raise NotebookError(f"Không có checkpoint {EXPORT_CK} (có: {sorted(p.name for p in (RUN_DIR / 'weights').glob('*.pt'))})", "TRAIN")
    DRIVE_OUTPUT_DIR = str(Path(DRIVE_OUTPUT_DIR) / f"export_{EXPORT_CK.stem}")  # never overwrite the run's own outputs
    Path(DRIVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
if STAMP.is_file() and EXPORT_CK is None:
    stamp = json.loads(STAMP.read_text())
    if (stamp["dataset_sha256"], stamp["init_sha256"], stamp["epochs"], stamp.get("sampling_sha256")) != (DATASET["sha256"], INIT_SHA, EPOCHS, HANDOFF.get("sampling_sha256")):
        raise NotebookError(f"{RUN_DIR} đã bắt đầu với dataset/khởi tạo/EPOCHS khác. Đặt DRIVE_OUTPUT_DIR mới (run mới, "
                            "fine-tune mới); không resume sang dataset hay lịch học khác.", "TRAIN")
done_epochs = len((RUN_DIR / "results.csv").read_text().splitlines()) - 1 if (RUN_DIR / "results.csv").is_file() else 0
if EXPORT_CK is not None:
    print(f"CHỈ EXPORT {EXPORT_CK} ({done_epochs}/{EPOCHS} epoch đã xong trong results.csv) -> {DRIVE_OUTPUT_DIR}")
    model = YOLO(str(EXPORT_CK))
elif LAST.is_file() and done_epochs < EPOCHS:
    print(f"RESUME run {RUN_NAME} từ {LAST} (đã xong {done_epochs}/{EPOCHS} epoch, cùng dataset)")
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
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(json.dumps({"dataset_sha256": DATASET["sha256"], "sampling_sha256": HANDOFF.get("sampling_sha256"), "handoff": HANDOFF, "init_weights": str(INIT_SRC),
                                 "init_sha256": INIT_SHA, "epochs": EPOCHS, "ft_args": FT_ARGS}, indent=1))
    print(f"FINE-TUNE mới từ {INIT_SRC} -> {RUN_DIR} ({EPOCHS} epoch, {FT_ARGS})")
    model = YOLO(str(INIT_WEIGHTS))
    model.train(**TRAIN_ARGS)
BEST_WEIGHTS = EXPORT_CK if EXPORT_CK is not None else RUN_DIR / "weights" / "best.pt"
if EXPORT_CK is not None and not HANDOFF.get("submission_eligible"):
    (Path(DRIVE_OUTPUT_DIR) / "NOT_FOR_SUBMISSION.txt").write_text("Bản thử nghiệm: quyền nguồn chưa xác nhận.\n", encoding="utf-8")
trained_names = [model.names[i] for i in sorted(model.names)]
if trained_names != list(CLASSES):
    raise NotebookError(f"Tên lớp của mô hình {trained_names} khác danh sách lớp của cuộc thi {list(CLASSES)}. Hãy huấn luyện bằng data.yaml do Bước 1 tạo.", "TRAIN")
TRAIN_INFO = {"best_weights": str(BEST_WEIGHTS), "best_sha256": sha256_file(BEST_WEIGHTS), "best_tensors_sha256": weights_fingerprint(BEST_WEIGHTS),
              "train_args": {k: v for k, v in TRAIN_ARGS.items() if k != "project"}, "device": DEVICE, "gpu": GPU_NAME,
              "init_weights": {"path": str(INIT_SRC), "sha256": INIT_SHA}, "v2_handoff": HANDOFF}
print("Trọng số tốt nhất:", BEST_WEIGHTS, "| checkpoint:", sorted(p.name for p in (RUN_DIR / "weights").glob("*.pt")))
'''

XEVAL = r'''
# [VietSight V2] separate evaluation, same platform inference + scorer as above: e25 (baseline) vs this run,
# on the Vietnam val of V2 and on the Mapillary val of V1. -> DRIVE_OUTPUT_DIR/eval_compare.json
def _xscore(zip_path, imgs, lab_dir, tag):
    w = WORK / "xeval" / tag
    w.mkdir(parents=True, exist_ok=True)
    pred, _ = run_platform_inference(zip_path, imgs, w, SELFCHECK)
    r = score_predictions(pred, lab_dir, imgs, SELFCHECK).to_dict()
    return {"score_x100": round(r["score"], 4), "map50": r["map50"], "per_class_ap": r["per_class_ap"], "n_images": r["n_images"]}


e25_zip = WORK / "xeval" / "e25_submission.zip"
e25_zip.parent.mkdir(parents=True, exist_ok=True)
if Path(E25["zip"]).is_file() and _sha(E25["zip"]) == E25["zip_sha256"]:
    shutil.copyfile(E25["zip"], e25_zip)
elif INIT_MODE in ("e25", "v3best"):  # same official export of the same e25 best.pt, in a separate folder (V2 submission untouched)
    print(f"Không thấy submission.zip {BASE_TAG} trên Drive: xuất lại từ best.pt {BASE_TAG}")
    shutil.copyfile(make_submission(INIT_WEIGHTS, WORK / "xeval" / "e25_export", YOLO, SELFCHECK)["zip"], e25_zip)
else:  # pretrained init: no e25 weights here -> baseline only from the sha-checked e25 submission.zip
    print(f"Bỏ qua mốc e25: không có {E25['zip']} đúng sha256 (upload runs/v1_colab_e25/submission.zip của repo vào đó)")
    e25_zip = None
if VAL_SETS:  # mixed dataset: every validation set of the handoff, all its images, scored separately
    sets = dict(VAL_SETS)
else:
    sets = {"vn_val": (sample_stems(WORK / "dataset" / "images" / "val", SELFCHECK_MAX_IMAGES, SEED), WORK / "dataset" / "labels" / "val")}
    if MAPILLARY_OK:
        sets["mapillary_val"] = (sample_stems(Path(MAPILLARY_VAL["dir"]) / "images" / "val", SELFCHECK_MAX_IMAGES, SEED),
                                 Path(MAPILLARY_VAL["dir"]) / "labels" / "val")
EVAL_COMPARE = {"run": RUN_NAME, "dataset_sha256": DATASET["sha256"], "handoff": HANDOFF, "results": {}}
for model_tag, zp in [*([(BASE_TAG, e25_zip)] if e25_zip else []), (RUN_NAME, SUBMISSION["zip"])]:
    for set_tag, (imgs, lab_dir) in sets.items():
        EVAL_COMPARE["results"].setdefault(model_tag, {})[set_tag] = _xscore(zp, imgs, lab_dir, f"{model_tag}_{set_tag}")
print(f"{'model':<28}" + "".join(f"{s:>16}" for s in sets))
for model_tag, res in EVAL_COMPARE["results"].items():
    print(f"{model_tag:<28}" + "".join(f"{res[s]['score_x100']:>16.2f}" for s in sets))
print("AP theo lớp (x100):", "  ".join(f"{c[:6]}" for c in CLASSES))
for model_tag, res in EVAL_COMPARE["results"].items():
    for s in sets:
        print(f"  {model_tag[:20]:<20} {s:<14}", " ".join(f"{100 * res[s]['per_class_ap'].get(c, 0):6.1f}" for c in CLASSES))
xp = SUBMISSION["zip"].parent / "eval_compare.json"
xp.write_text(json.dumps(EVAL_COMPARE, indent=1, ensure_ascii=False), encoding="utf-8")
shutil.copyfile(xp, Path(DRIVE_OUTPUT_DIR) / xp.name)
print("Đã lưu", Path(DRIVE_OUTPUT_DIR) / xp.name)
if not HANDOFF.get("submission_eligible"):
    print("\n*** BẢN THỬ NGHIỆM: quyền nguồn V2 chưa xác nhận -> KHÔNG nộp", SUBMISSION["zip"].name, "của run này lên cuộc thi ***")
'''

CKPT = r'''
# [VietSight] intermediate checkpoints: official export (make_submission, separate folders; the run's submission.zip is
# untouched) + the same scorer on every val above -> DRIVE_OUTPUT_DIR/ckpt_eval.json, ckpt_submissions/*.zip.
# Ultralytics counts epochs from 0: epochN.pt = weights after N+1 epochs (results.csv row N+1).
EVAL_CKPT_FILES = __EVAL_CKPTS__   # e.g. ["epoch20.pt", "epoch30.pt"]; [] = skip (~3 phút / checkpoint)
DOWNLOAD_WEIGHTS = False         # True: also download the weights zip to this computer (Colab browser download)
wdir = RUN_DIR / "weights"
ckpt_json = Path(DRIVE_OUTPUT_DIR) / "ckpt_eval.json"
CKPT_EVAL = json.loads(ckpt_json.read_text(encoding="utf-8")) if ckpt_json.is_file() else {}
(Path(DRIVE_OUTPUT_DIR) / "ckpt_submissions").mkdir(exist_ok=True)
for name in EVAL_CKPT_FILES:
    ck, tag = wdir / name, Path(name).stem
    if not ck.is_file():
        print("Không có", ck)
        continue
    if tag in CKPT_EVAL and CKPT_EVAL[tag].get("sha256") == _sha(ck):
        print(tag, "đã chấm (ckpt_eval.json)")
        continue
    sub = make_submission(ck, WORK / "ckpt_eval" / tag, YOLO, SELFCHECK)
    dz = Path(DRIVE_OUTPUT_DIR) / "ckpt_submissions" / f"{tag}_submission.zip"
    shutil.copyfile(sub["zip"], dz)
    CKPT_EVAL[tag] = {"sha256": _sha(ck), "zip": str(dz), "zip_sha256": _sha(dz),
                      "results": {s: _xscore(sub["zip"], imgs, lab, f"{tag}_{s}") for s, (imgs, lab) in sets.items()}}
    print(tag, {s: round(r["score_x100"], 2) for s, r in CKPT_EVAL[tag]["results"].items()})
    ckpt_json.write_text(json.dumps(CKPT_EVAL, indent=1, ensure_ascii=False), encoding="utf-8")
if not HANDOFF.get("submission_eligible") and EVAL_CKPT_FILES:
    (Path(DRIVE_OUTPUT_DIR) / "ckpt_submissions" / "NOT_FOR_SUBMISSION.txt").write_text("Bản thử nghiệm: quyền nguồn chưa xác nhận.\n", encoding="utf-8")
arch = shutil.make_archive(str(Path(DRIVE_OUTPUT_DIR) / f"{RUN_NAME}_weights"), "zip", wdir)
print("Checkpoint:", sorted(p.name for p in wdir.glob("*.pt")), "| zip:", arch, f"{Path(arch).stat().st_size / 1e6:.0f} MB")
if IN_COLAB and DOWNLOAD_WEIGHTS:
    from google.colab import files
    files.download(arch)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--save-period", type=int, default=3)
    ap.add_argument("--lr0", type=float, default=0.0003)
    ap.add_argument("--close-mosaic", type=int, default=3)
    ap.add_argument("--freeze", type=int, default=None, help="freeze first N layers (small READY set: 10 = backbone w/o C2PSA)")
    ap.add_argument("--handoff", type=Path, help="JSON: status, version, source, images{train,val}, fingerprint_sha256, vn_val_sha256")
    ap.add_argument("--prep-kernel", default="mangashounen/vietsight-prep-vietsight-v1")
    ap.add_argument("--init", choices=["e25", "pretrained", "v3best"], default="e25",
                    help="pretrained = official COCO yolo26n-seg.pt (sha pinned), official default train args like V1 e25")
    ap.add_argument("--eval-ckpts", nargs="*", default=[], help="periodic checkpoints to export+score, e.g. epoch20.pt epoch30.pt")
    ap.add_argument("--out-name", help="notebook file name in notebooks/colab/ (default by --init)")
    a = ap.parse_args()
    handoff = {"status": "PENDING", "version": None, "source": None, "images": None, "fingerprint_sha256": None, "vn_val_sha256": None, "rights_confirmed": False}
    if a.handoff:
        handoff.update(json.loads(a.handoff.read_text(encoding="utf-8")))
    # rights_confirmed != true -> EXPERIMENT run: own Drive folder, NOT_FOR_SUBMISSION.txt, never a competition submission
    handoff["submission_eligible"] = handoff.get("rights_confirmed") is True
    kind = {"e25": "ft", "pretrained": "pretrained", "v3best": "ftv3best"}[a.init]
    run_name = f"{handoff['version'] or 'vietsight_v2'}_{kind}_e{a.epochs}" + ("" if handoff["submission_eligible"] else "_EXPERIMENT")
    if a.freeze:
        run_name += f"_frz{a.freeze}_lr{a.lr0:g}"
    base_dir = "v3_final002_e50" if a.init == "v3best" else "v1_colab_e25"
    e25 = json.loads((ROOT / "runs" / base_dir / "run_report.json").read_text(encoding="utf-8"))
    base_tag = "v3e50" if a.init == "v3best" else "e25"
    base_drive = (f"{DRIVE}/vietsight_mixed_final002_v3_pretrained_e50_EXPERIMENT" if a.init == "v3best" else f"{DRIVE}/vietsight_v1_e25")
    base_best = e25["train"]["best_weights"] if a.init == "v3best" else f"{base_drive}/runs/vietsight_v1_e25/weights/best.pt"
    frozen = [x.strip() for x in (ROOT / "configs" / "val_frozen.txt").read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    v1_val_sha = hashlib.sha256("\n".join(sorted(k.split("/", 1)[1] for k in frozen)).encode()).hexdigest()

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    out = copy.deepcopy(nb)
    cells = out["cells"]
    ip = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith("#@title Tham số của đội")][0]
    it = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "".join(c["source"]).startswith("from ultralytics import YOLO\n\nTRAIN_ARGS")][0]
    src = "".join(cells[ip]["source"])
    for old, new in [('EPOCHS = 60  #@param {type:"integer"}', f'EPOCHS = {a.epochs}  #@param {{type:"integer"}}'),
                     ('DATASET_PATH = ""  #@param {type:"string"}', f'DATASET_PATH = "{LOCAL_DATASET}"  #@param {{type:"string"}}'),
                     ('DRIVE_OUTPUT_DIR = ""', f'DRIVE_OUTPUT_DIR = "{DRIVE}/{run_name}"')]:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    cells[ip]["source"] = src.splitlines(True)
    if a.init == "e25":
        ft_args = ('dict(optimizer="AdamW", lr0=__LR0__, lrf=0.1, cos_lr=True, warmup_epochs=1, warmup_bias_lr=__LR0__,\n'
                   '               freeze=__FREEZE__, close_mosaic=__CLOSE_MOSAIC__)')
    elif a.init == "v3best":  # V3 final002 e50 best.pt: AdamW low LR, warmup 1, mosaic OFF, no freeze
        ft_args = ('dict(optimizer="AdamW", lr0=__LR0__, lrf=0.1, cos_lr=True, warmup_epochs=1, warmup_bias_lr=__LR0__,\n'
                   '               freeze=__FREEZE__, mosaic=0.0, close_mosaic=0)')
    else:  # from COCO pretrained: the official defaults, i.e. the same recipe as V1 e25 (optimizer=auto, warmup 3, ...)
        ft_args = "dict()  # pretrained COCO init: official default args (same recipe as V1 e25)"
    cells[it]["source"] = (TRAIN.replace("__FT_ARGS__", ft_args).replace("__SAVE_PERIOD__", str(a.save_period)).replace("__LR0__", repr(a.lr0))
                           .replace("__CLOSE_MOSAIC__", str(a.close_mosaic)).replace("__FREEZE__", repr(a.freeze))).splitlines(True)
    pretrained_sha = hashlib.sha256((ROOT / "yolo26n-seg.pt").read_bytes()).hexdigest()
    setup = (SETUP.replace("__HANDOFF__", repr(handoff)).replace("__INIT_MODE__", a.init).replace("__PRETRAINED_SHA__", pretrained_sha)
             .replace("__BASE_TAG__", base_tag).replace("__BASE_BEST__", base_best).replace("__BASE_ZIP__", f"{base_drive}/submission.zip")
             .replace("__DRIVE__", DRIVE)
             .replace("__E25_BEST_SHA__", e25["train"]["best_sha256"]).replace("__E25_ZIP_SHA__", e25["submission"]["zip_sha256"])
             .replace("__PREP_KERNEL__", a.prep_kernel).replace("__V1_NVAL__", str(len(frozen))).replace("__V1_VAL_SHA__", v1_val_sha))
    cells.insert(ip + 1, code(setup))
    init_vi = {"e25": "e25 best.pt", "v3best": "V3 final002 e50 best.pt", "pretrained": "yolo26n-seg.pt pretrained chính thức"}[a.init]
    cells.insert(ip + 1, md(f"## [VietSight V2] Bàn giao dataset (chỉ READY) → tải + kiểm tra → Mapillary val → khởi tạo {init_vi} → GPU"))
    cells.append(code(XEVAL))
    cells.append(code(CKPT.replace("__EVAL_CKPTS__", repr(list(a.eval_ckpts)))))
    dst = ROOT / "notebooks" / "colab" / (a.out_name or {"e25": "VietSight_TEAM55_colab_v2.ipynb", "pretrained": "VietSight_TEAM55_colab_v2_pretrained.ipynb", "v3best": "VietSight_TEAM55_colab_v4_ft_v3best.ipynb"}[a.init])
    data_vi = ("dữ liệu trộn, val chấm riêng: " + ", ".join(handoff["val_subsets"])) if handoff.get("val_subsets") else "chỉ dữ liệu Việt Nam"
    cells.insert(0, md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PHai237/VietSight/blob/main/notebooks/colab/{dst.name})\n\n"
                       f"VietSight TEAM 55 — `{run_name}` từ {init_vi}, {data_vi}, {a.epochs} epoch, checkpoint mỗi {a.save_period} epoch: "
                       "Runtime → T4 GPU → Run all. "
                       f"Kết quả: Drive `MyDrive/VietSight/{run_name}/`. Mất kết nối: Run all lại (tự resume)."
                       + ("" if handoff["submission_eligible"] else "\n\n**BẢN THỬ NGHIỆM — quyền nguồn chưa xác nhận: KHÔNG nộp submission.zip của run này.**")))
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    for i, oc in enumerate(nb["cells"]):  # guard: official cells unchanged except parameter + training cells
        if i in (ip, it):
            continue
        assert "".join(oc["source"]) == "".join(cells[i + 1 + (2 if i > ip else 0)]["source"]), f"official cell {i} changed"
    assert len(cells) == len(nb["cells"]) + 5
    out["metadata"]["accelerator"] = "GPU"
    out["metadata"].setdefault("colab", {})["gpuType"] = "T4"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(dst, "| run", run_name, "| handoff", handoff["status"])


if __name__ == "__main__":
    main()
