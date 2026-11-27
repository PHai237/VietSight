# VietSight — Vietnamese Streets. Intelligent Vision.

AIInAction RoboTaxi Data-Centric Competition · Team 50 · `TEAM_ID = 55` · deadline **10:30 28/11/2026 (GMT+7)**.
Instance segmentation 11 lớp, model cố định **YOLO26n-seg** (Ultralytics 8.4.214, imgsz 640), điểm = 100 × mask mAP@[.5:.95].
Cải thiện **dữ liệu**, không đổi kiến trúc. Notebook chính thức: [`notebooks/VietSight_TEAM55.ipynb`](notebooks/VietSight_TEAM55.ipynb) (không sửa).

## Luồng dữ liệu

```
public tar / COCO json / CVAT export
   │  fetch_sly_tar.py · download_sources.py           (data/raw/<source>/, không commit)
   ▼
convert_supervisely.py · convert_coco.py · convert_yolo.py · prelabel.py
   │  remap 11 lớp, ghép người lái + xe, mask nhìn thấy, polygon hợp lệ (validator của notebook)
   ▼
data/processed/<source>/  images/ labels/ review/ meta.jsonl summary.json
   │  review_queue.py (CVAT)  ← ảnh mơ hồ + mọi AI pre-label phải được người duyệt
   ▼
build_version.py configs/versions/<v>.yaml   → dedup sha256+dhash, val CỐ ĐỊNH (configs/val_frozen.txt), báo cáo notebook
   ▼
data/versions/<v>/ (images|labels)/(train|val), data.yaml   +  <v>_report/
   ▼
Kaggle: VietSight_TEAM55_kaggle.ipynb  → best.pt, submission.zip, run_report.json → collect_run.py → runs/RESULTS.md
```

## Chạy trên Kaggle (đường chính — mạng local quá chậm cho 21 GB)
1. `python scripts/make_kaggle_notebooks.py --version-config configs/versions/v1_first_gear.yaml --epochs 60` (đã sinh sẵn; thêm `--kaggle-user <tên>` để đẩy bằng `kaggle kernels push -p notebooks/kaggle/prep`).
2. Kaggle → New Notebook → Import `notebooks/kaggle/prep/VietSight_prep_kaggle.ipynb`. Accelerator **None**, Internet **On** → *Save Version → Save & Run All*.
   Ra: `versions/vietsight_v1/` + `versions/vietsight_v1_report/` + `versions/val_frozen.txt`.
3. Tải `val_frozen.txt` về `configs/val_frozen.txt` và commit (val cố định cho mọi phiên bản; lần build sau notebook dùng lại).
4. Import `notebooks/kaggle/train/VietSight_TEAM55_kaggle.ipynb`. Accelerator **GPU (T4/P100)**, Internet **On**, *Add Input → Notebook Output → VietSight prep*. Run All.
   DATASET_PATH được tự dò; ô cuối gom `best.pt, submission.zip, run_report.json`, biểu đồ, báo cáo val vào `/kaggle/working/vietsight_outputs/<version>/`.
5. Tải thư mục đó về → `python scripts/collect_run.py <thư mục> --name v1_e60` → cập nhật `runs/RESULTS.md`. Nộp `submission.zip` trên website, ghi điểm public: `--public-lb`.

Ước lượng (notebook, T4): ~22 ảnh/s → 6 000 ảnh × 60 epoch ≈ 4.5 giờ. Thiếu thời gian: giảm `--epochs` (40) khi sinh notebook.

## Chạy local
```
py -3.12 -m venv .venv && .venv\Scripts\pip install -r requirements.txt   (torch CPU: --index-url https://download.pytorch.org/whl/cpu)
.venv\Scripts\python scripts\convert_yolo.py --root data\raw\sample_data --source sample_test       # kiểm thử pipeline
.venv\Scripts\python scripts\run_notebook_local.py notebooks\kaggle\VietSight_TEAM55_kaggle.ipynb --set DATASET_PATH=<version dir> EPOCHS=1 BATCH=2 REQUIRE_GPU=False INSTALL_PACKAGES=False WORK_DIR=work WORKERS=0
```
`run_notebook_local.py` chạy toàn bộ notebook không cần Jupyter (smoke test CPU: train → export → self-check → chấm val).

## Scripts
| Script | Việc |
|---|---|
| `fetch_sly_tar.py` | stream tar Supervisely (Dataset Ninja), lọc ảnh, thu nhỏ |
| `download_sources.py` | COCO 2017 (annotation + chỉ ảnh được chọn) |
| `convert_supervisely.py` / `convert_coco.py` / `convert_yolo.py` | → YOLO-seg 11 lớp (`--workers N` cho Supervisely) |
| `prelabel.py new|fill` | AI pre-label ảnh mới / điền lớp thiếu → hàng review |
| `review_queue.py stats|export|accept|reject` | vòng duyệt CVAT |
| `preview.py` | ảnh overlay mask, lọc theo lớp |
| `build_version.py` | merge, dedup, val cố định, báo cáo chính thức |
| `package_kaggle.py` | đóng gói nguồn/phiên bản để upload Kaggle (private) |
| `make_kaggle_notebooks.py` | sinh notebook prep (nhúng code) + notebook train (chỉ sửa ô tham số) |
| `run_notebook_local.py` | chạy notebook không cần Jupyter |
| `collect_run.py` | gom kết quả run → `runs/RESULTS.md` |

Thư viện `scripts/vslib/` nạp **trực tiếp** các hàm kiểm tra của notebook chính thức (`notebook_lib.py`), không chép lại.

## Phân công
| # | Vai trò | Việc chính trong repo |
|---|---|---|
| 1 | Team Lead & ML | notebook Kaggle, `build_version.py`, train, nộp bài, chọn 2 checkpoint |
| 2 | Dataset Eng. — Mapillary | xác nhận giấy phép trang chính thức, rà mapping `configs/sources/mapillary_vistas.yaml`, preview riders |
| 3 | Dataset Eng. — IDD/Asia | đăng ký IDD, `configs/sources/idd.yaml` (+ converter Cityscapes-json nếu dùng), BDD fill |
| 4 | Data Collector | ảnh đường phố VN (đêm, mưa, đông đúc, nhiều xe máy) → `data/raw/vn_team/<lô>/` |
| 5 | Annotation Eng. | `prelabel.py new` → `review_queue.py export` → CVAT → `accept` |
| 6 | QA & Analyst | `preview.py`, kiểm tra chéo 10%, đọc Bước 5 (AP từng lớp, ảnh lỗi), slide |

Mỗi người làm trên `data/processed/<source>/` riêng → không đụng nhau; merge bằng version config. Tài liệu: [nhãn](docs/LABEL_GUIDELINE.md) · [nguồn dữ liệu](docs/DATA_SOURCES.md) · [phiên bản](docs/VERSIONS.md).
