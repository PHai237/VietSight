# VietSight — kết quả các run

Dataset V1 `vietsight_v1` (fingerprint `5612ab42…`, train 1 469 / val 262 ảnh, val cố định `configs/val_frozen.txt`), seed 55,
YOLO26n-seg imgsz 640, notebook chính thức (validator/export/self-check không đổi). Điểm = 100 × mask mAP@[.5:.95].
Val = bộ chấm nhúng của notebook trên val của đội; Public = leaderboard.

| Run | Nền tảng | Epoch | Val score | Public score | Public mAP50 | Nộp | Ghi chú |
|---|---|---|---|---|---|---|---|
| v1_e3_cpu | Kaggle CPU | 3 | 17.55 | — | — | không | ~24 phút/epoch; artefact mất (WORK_DIR=/content/work), chỉ còn log |
| **v1_e25** | Colab T4 | 25 | **22.9707** | **12.90** | **23.68** | lượt 1/20 | checkpoint: Drive `MyDrive/VietSight/vietsight_v1_e25/` |
| v1_e50 | Colab T4 | 50 (tiếp từ e25 `epoch20.pt`) | 23.3754 | 12.37 | 23.10 | lượt 2/20 | Drive `MyDrive/VietSight/vietsight_v1_e50/`; val ↑ nhưng public ↓ |
| v2_ft_e12 | Colab T4 | 12 (fine-tune từ e25 best.pt, chỉ ảnh VN) | | | | | `notebooks/colab/VietSight_TEAM55_colab_v2.ipynb`; chờ Codex bàn giao READY |

## AP từng lớp — e25 vs e50 (public = leaderboard; val = bộ chấm nhúng trên val Mapillary, ×100)

| ID | Lớp | Public e25 | Public e50 | Δ public | Val e25 | Val e50 | Δ val |
|---|---|---|---|---|---|---|---|
| 0 | Car | 29.67 | 29.65 | -0.02 | 48.98 | 49.62 | +0.64 |
| 1 | Bus | 12.60 | 10.46 | -2.14 | 32.74 | 35.32 | +2.58 |
| 2 | Truck | 20.41 | 19.88 | -0.53 | 23.46 | 20.66 | -2.80 |
| 3 | Other vehicle | 2.57 | 2.49 | -0.08 | 7.50 | 9.99 | +2.49 |
| 4 | Motorcycle | 1.95 | 1.83 | -0.12 | 8.48 | 9.67 | +1.19 |
| 5 | Motorcyclist | 15.02 | 13.44 | -1.58 | 30.31 | 27.89 | -2.42 |
| 6 | Bicycle | 6.53 | 8.22 | +1.69 | 6.08 | 5.50 | -0.58 |
| 7 | Cyclist | 17.05 | 14.75 | -2.30 | 20.74 | 22.50 | +1.76 |
| 8 | Pedestrian | 12.54 | 13.39 | +0.85 | 20.17 | 20.46 | +0.29 |
| 9 | Sign | 6.05 | 6.12 | +0.07 | 29.13 | 29.64 | +0.51 |
| 10 | Traffic light | 17.56 | 15.79 | -1.77 | 25.07 | 25.87 | +0.80 |

Nhận xét: val 22.97 cao hơn public 12.90 khá nhiều (val là ảnh Mapillary cùng phân phối với train, test ẩn khác miền). Yếu nhất trên public:
Motorcycle 1.95, Other vehicle 2.57, Sign 6.05, Bicycle 6.53 → ưu tiên dữ liệu VN (xe máy đỗ, biển báo VN, xe ba gác) ở V2.

## Vì sao e50 giảm điểm public (27/11)

**Bằng chứng (số đo):**
- Public giảm 12.90 → 12.37 (−0.53), mAP50 23.68 → 23.10; val Mapillary lại tăng 22.97 → 23.38 (+0.40). Chênh lệch val–public nới rộng 10.1 → 11.0.
- Ngược chiều theo lớp: Bus (val +2.6 / public −2.1), Cyclist (+1.8 / −2.3), Traffic light (+0.8 / −1.8). Val Mapillary **không dự báo** được public.
- Cùng dataset, cùng val, seed 55 và cùng tham số; best.pt của cả hai run được chọn theo val Mapillary.
- Theo code (đã kiểm khi làm notebook e50, chưa xem log Colab): optimizer `auto` → AdamW lr0 0.000667, giảm tuyến tính. e50 resume từ `epoch20.pt` với lịch học tính lại cho 50 epoch nên LR tăng lại. close_mosaic=10 nên mosaic bật lại ở epoch 22–40 (e25 đã tắt mosaic từ epoch 16).

**Giả thuyết (chưa kiểm chứng):**
1. Model khớp sâu hơn với miền Mapillary (ảnh, góc chụp, quy ước nhãn) nên kém đi trên miền test ẩn. Phù hợp với việc val tăng mà public giảm.
2. Nhiễu: −0.53 điểm, chỉ có một lần đo mỗi run, không rõ cỡ tập test. Chưa loại trừ được.
3. Ảnh hưởng của lịch học: LR tăng lại và mosaic bật lại khiến e50 khác một run 50 epoch train từ đầu. Không tách được nếu không chạy lại, và không chạy lại vì đã dừng tăng epoch trên Mapillary.
4. Sign public 6 so với val 29 (xem SIGN_FORENSICS của Codex) cho thấy lệch quy ước nhãn hoặc lệch miền. Điều này đúng cho cả hai run, nên không giải thích phần giảm.

**Quyết định:** dừng tăng epoch trên Mapillary. V2 khởi tạo từ **e25 best.pt** (public cao hơn) và fine-tune chỉ trên ảnh VN với optimizer và lịch LR mới (AdamW lr0 3e-4, cos, lrf 0.1, warmup 1, close_mosaic 3). Đánh giá trên cả val VN cố định lẫn val Mapillary (`eval_compare.json`).
