# VietSight — kết quả các run

Dataset V1 `vietsight_v1` (fingerprint `5612ab42…`, train 1 469 / val 262 ảnh, val cố định `configs/val_frozen.txt`), seed 55,
YOLO26n-seg imgsz 640, notebook chính thức (validator/export/self-check không đổi). Điểm = 100 × mask mAP@[.5:.95].
Val = bộ chấm nhúng của notebook trên val của đội; Public = leaderboard.

| Run | Nền tảng | Epoch | Val score | Public score | Public mAP50 | Nộp | Ghi chú |
|---|---|---|---|---|---|---|---|
| v1_e3_cpu | Kaggle CPU | 3 | 17.55 | — | — | không | ~24 phút/epoch; artefact mất (WORK_DIR=/content/work), chỉ còn log |
| **v1_e25** | Colab T4 | 25 | **22.9707** | **12.90** | **23.68** | lượt 1/20 | checkpoint: Drive `MyDrive/VietSight/vietsight_v1_e25/` |
| v1_e50 | Colab T4 | 50 (tiếp từ e25 `epoch20.pt`) | | | | | `notebooks/colab/VietSight_TEAM55_colab_e50.ipynb` |

## AP từng lớp — public leaderboard, v1_e25

| ID | Lớp | Public AP |
|---|---|---|
| 0 | Car | 29.67 |
| 1 | Bus | 12.60 |
| 2 | Truck | 20.41 |
| 3 | Other vehicle | 2.57 |
| 4 | Motorcycle | 1.95 |
| 5 | Motorcyclist | 15.02 |
| 6 | Bicycle | 6.53 |
| 7 | Cyclist | 17.05 |
| 8 | Pedestrian | 12.54 |
| 9 | Sign | 6.05 |
| 10 | Traffic light | 17.56 |

Nhận xét: val 22.97 cao hơn public 12.90 khá nhiều (val là ảnh Mapillary cùng phân phối với train, test ẩn khác miền). Yếu nhất trên public:
Motorcycle 1.95, Other vehicle 2.57, Sign 6.05, Bicycle 6.53 → ưu tiên dữ liệu VN (xe máy đỗ, biển báo VN, xe ba gác) ở V2.
