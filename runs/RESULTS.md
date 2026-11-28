# VietSight — kết quả các run

Dataset V1 `vietsight_v1` (fingerprint `5612ab42…`, train 1 469 / val 262 ảnh, val cố định `configs/val_frozen.txt`), seed 55,
YOLO26n-seg imgsz 640, notebook chính thức (validator/export/self-check không đổi). Điểm = 100 × mask mAP@[.5:.95].
Val = bộ chấm nhúng của notebook trên val của đội; Public = leaderboard.

## Leaderboard hiện tại (cập nhật 28/11, 5 lượt nộp)

Public score là con số tổng hợp do leaderboard hiển thị; công thức chưa rõ, không phải mask mAP50-95 ×100 như cột
"Public score" cũ bên dưới. Cột "Val Mapillary" và "Val VN" là bộ chấm nhúng của notebook (×100 mask mAP50-95).

| Lượt | Run | Public score | Public mask mAP50 | Public mask mAP50-95 | Val Mapillary | Val VN (283 ảnh) | zip sha256 | Xác định run |
|---|---|---|---|---|---|---|---|---|
| 1 | **v1_e25** | **50.18** | **25.18** | **13.49** | **22.97** | 6.94 | `f0b87463…` | người dùng xác nhận |
| 2 | v1_e50 | 49.54 | 24.54 | 12.96 | 23.38 | 7.19 | `84bd3dbf…` | suy ra: thứ tự nộp + chênh lệch với e25 giữ nguyên (xem dưới) |
| 3 | **chưa xác định** | 46.75 | 21.86 | 10.84 | | | | một trong hai lượt 3–4 là v2_pretrained_e25 |
| 4 | **chưa xác định** | 40.08 | 16.06 | 8.72 | | | | lượt còn lại không có run nào trong repo/máy này |
| 5 | **v3_final002_e50** | **52.20** | **27.25** | **14.89** | 23.09 | 13.20 | `f616e1ae…` | người dùng xác nhận; run EXPERIMENT (rights_confirmed=false) |
| 6 | v4_balanced_e15 (nộp 09:49 28/11) | 51.15 | 26.17 | 14.24 | 22.86 | 12.93 | `9d1a2114…` | người dùng xác nhận; run EXPERIMENT (rights_confirmed=false) |

| Run | Nền tảng | Epoch | Khởi tạo / dữ liệu | Ghi chú |
|---|---|---|---|---|
| v2_pretrained_e25 (`vietsight_vn_dashcam_final001_v2_pretrained_e25_EXPERIMENT`) | Colab T4 | 25 | `yolo26n-seg.pt` COCO (sha `361fbfab…`), chỉ dashcam VN 1070/283 | **EXPERIMENT, rights_confirmed=false: không đủ điều kiện nộp.** zip `ed371061…`, best.pt `0801ab13…`, val VN 12.48. Checkpoint trên Drive (save_period 3) |
| v3_final002_e50 (`vietsight_mixed_final002_v3_pretrained_e50_EXPERIMENT`) | Colab T4 | 50 | `yolo26n-seg.pt` COCO, mixed final002 2539/545 | EXPERIMENT, rights_confirmed=false. best.pt `8bd34af8…` (= epoch 50), backup `runs/v3_final002_e50/` (không commit weights) |
| v4_ft_v3best_e15 | Colab T4 | 15 | V3 best.pt, V4 weighted 50:50 Mapillary/VN (2539 ảnh unique, 4050 lượt/epoch) — ĐÃ CHẠY, public 51.15 | notebook `VietSight_TEAM55_colab_v4_ft_v3best.ipynb` (PENDING): AdamW lr0 1e-4 cos lrf 0.1, warmup 1, mosaic 0, không freeze, batch 16, save_period 5 |
| v2_ft_e12 | — | — | từ e25 best.pt | không chạy: best.pt e25/e50 không còn trên Drive lẫn máy (đã băm mọi `.pt`) |

**So với số cũ của cùng run:** e25 12.90 → 13.49 mAP50-95 (+0.59), mAP50 23.68 → 25.18 (+1.50); e50 12.37 → 12.96 (+0.59),
23.10 → 24.54 (+1.44). Hai run dịch cùng một lượng, chênh e25 − e50 giữ nguyên 0.53. Chưa có thông báo chính thức nào giải thích
chênh lệch so với số cũ; không kết luận là BTC chấm lại. Số cũ giữ trong bảng lịch sử bên dưới, không so trực tiếp số cũ với số mới.

### Val VN không dự báo được public (đo 27/11, `runs/vn_val_compare_local.json`)

Chấm lại 3 submission.zip trên cùng val VN (283 ảnh) bằng bộ chấm của notebook (CPU; trên Windows thay symlink bằng copy và
`resource` chỉ dùng để báo RAM → không đổi điểm; v2 cho 12.48 = đúng `val_estimate` của run_report).

| Lớp (AP ×100, val VN) | e25 | e50 | v2_pretrained_e25 |
|---|--:|--:|--:|
| Car | 14.3 | 14.3 | 27.0 |
| Bus | 6.4 | 12.1 | 18.5 |
| Truck | 28.5 | 22.9 | 35.0 |
| Other vehicle | 0.4 | 2.2 | 0.0 |
| Motorcycle | 0.7 | 0.7 | 3.0 |
| Motorcyclist | 9.7 | 9.9 | 23.1 |
| Bicycle | 0.0 | 0.0 | 0.0 |
| Cyclist | 2.0 | 3.5 | 5.9 |
| Pedestrian | 1.4 | 1.3 | 2.1 |
| Sign | 7.5 | 6.4 | 14.6 |
| Traffic light | 5.5 | 5.8 | 8.1 |
| **Tổng** | **6.94** | **7.19** | **12.48** |

v2 gần gấp đôi e25 trên val VN, nhưng public của v2 (10.84 hoặc 8.72) thấp hơn e25 (13.49). Tập dashcam này khác miền với test
ẩn (và val VN có 2 cặp gần trùng với train, chia theo thời gian cùng nguồn video), nên val VN cũng không dùng để chọn model nộp.

## V3 final002 e50 (28/11, `runs/v3_final002_e50/`)

Val ×100 mask mAP50-95, bộ chấm nhúng, chấm riêng hai val (`eval_compare.json`, `ckpt_eval.json`):

| Checkpoint | Val Mapillary (262) | Val VN (283) | Public mAP50-95 |
|---|--:|--:|--:|
| epoch25.pt (sau 26 epoch) | 21.53 | 11.68 | chưa nộp |
| epoch35.pt (sau 36 epoch) | 22.97 | **13.73** | chưa nộp |
| best.pt = epoch 50 | **23.09** | 13.20 | **14.89** |

- So với V1 e25: public +1.40 mAP50-95 (13.49 → 14.89), Mapillary val ngang (22.97 → 23.09), VN val gần gấp đôi (6.94 → 13.20).
- `results.csv` (val trộn 545 ảnh, box/mask mAP50-95): epoch 39 25.27/19.46, epoch 50 25.27/19.48 → **bão hòa** từ ~epoch 36; LR cuối 2e-5.
  Không resume tới 75 epoch. `epoch45.pt` còn optimizer; `best.pt`/`last.pt` đã strip.
- Lớp yếu trên VN val: Bicycle 0.0 (chỉ 2 instance), Other vehicle 0.8 (4 instance), Pedestrian 2.1, Motorcycle 2.2 → vấn đề dữ liệu, không phải số epoch.
- Bước tiếp: V4 cân bằng dữ liệu (Codex), fine-tune 15 epoch từ V3 best.pt trên V4.

## Lịch sử (số public cũ, trước khi leaderboard chấm lại)

| Run | Nền tảng | Epoch | Val score | Public score | Public mAP50 | Nộp | Ghi chú |
|---|---|---|---|---|---|---|---|
| v1_e3_cpu | Kaggle CPU | 3 | 17.55 | — | — | không | ~24 phút/epoch; artefact mất (WORK_DIR=/content/work), chỉ còn log |
| **v1_e25** | Colab T4 | 25 | **22.9707** | **12.90** | **23.68** | lượt 1/20 | best.pt không còn trên Drive (27/11 tối); còn submission.zip trong repo local |
| v1_e50 | Colab T4 | 50 (tiếp từ e25 `epoch20.pt`) | 23.3754 | 12.37 | 23.10 | lượt 2/20 | best.pt không còn trên Drive; val ↑ nhưng public ↓ |

## AP từng lớp — e25 vs e50 (public = leaderboard TRƯỚC khi chấm lại; val = bộ chấm nhúng trên val Mapillary, ×100)

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

## Final — 2 checkpoint chấm private (người dùng chọn 28/11)
| Checkpoint | Zip sha256 | Val Mapillary | Val VN | Public |
|---|---|---|---|---|
| v3_final002_e50 | `f616e1ae…` | 23.09 | 13.20 | 52.20 |
| v4_balanced_e15 (fine-tune từ V3 best.pt, V4 weighted) | `9d1a2114…` | 22.86 | 12.93 | 51.15 |

Cả hai là run EXPERIMENT: handoff ghi `rights_confirmed=false`, `submission_eligible=false`.

### V4 so với V3 (28/11)
- V4 = fine-tune 15 epoch từ V3 best.pt; sampling 50:50 Mapillary/VN (2539 ảnh unique, 4050 lượt/epoch); AdamW lr0 1e-4 cos, warmup 1, mosaic 0.
- Public: V4 51.15 / mAP50 26.17 / mAP50-95 14.24 so với V3 52.20 / 27.25 / 14.89. Val Mapillary 22.86 vs 23.09; val VN 12.93 vs 13.20.
- Theo người dùng: Bicycle và Truck cải thiện, nhưng tổng điểm giảm.
- Không quy toàn bộ mức giảm cho sampling: cấu hình training cũng đổi cùng lúc (khởi tạo, số epoch, LR, mosaic), chưa có run tách riêng từng yếu tố.
- **V3 vẫn là phiên bản tốt nhất trên public (52.20).**
- Tái lập: notebook V4 đã sửa để ánh xạ `train_weighted.txt` qua `source_map.json` khi bước chuẩn bị chính thức đổi tên ảnh. Backup run: `runs/v4_balanced_e15/` (submission.zip, weights.zip không commit).
