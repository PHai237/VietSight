# So sánh backend đề xuất mask — 9 ảnh Mapillary Việt Nam (27/11/2026)

Script: `scripts/qa/eval_proposals.py` (dùng lại `Backend` của `propose_labels.py`). JSON: `mapillary_vn9_e25_coco.json` (conf 0.25),
`mapillary_vn9_e25_coco_conf010.json` (conf 0.1). CPU, ~0.6 s/ảnh (e25 ONNX), ~0.35 s/ảnh (COCO).

**Tham chiếu ≠ ground truth.** Tham chiếu là draft chuyển từ mask pixel gốc Mapillary của Codex
(`VietSight-data-qa/.../mapillary_vn_native_pixels_intake001_REVIEW/converted_queue/review`, status
`BLOCKED_TARGETED_ASSOCIATION_GEOMETRY_AND_COVERAGE`): còn 19 ca ghép người lái mơ hồ, 9 người lái chưa có xe (đang nằm ở
Pedestrian), 5 ca Truck/Other vehicle cần chốt. Chỉ dùng để đo, **không train, không sửa**.

Cảnh báo thêm:
- 8/9 ảnh thuộc split *training* gốc của Mapillary; V1 train trên Mapillary nên **e25 có thể đã thấy một phần các ảnh này**
  (danh sách train V1 chỉ có trên Kaggle, chưa đối chiếu) → số của e25 có thể lạc quan.
- Mẫu rất nhỏ: Other vehicle 3, Traffic light 2 vật thể tham chiếu. Chỉ đọc như dấu hiệu, không như số đo ổn định.
- Vật < 64 px² ở thang 640 = ignore (không tính FN, dự đoán trúng không tính FP).
- Union = cách `propose_labels.py` gộp (điểm cao giữ, chồng IoU > 0.5 bỏ).

## conf 0.25 (mặc định của propose_labels)

Recall / precision tại IoU mask ≥ 0.5; R@0.3 = recall lỏng; mIoU = IoU trung bình của TP.

| Lớp (số tham chiếu) | e25 R / P | COCO R / P | union R / P | e25 R@0.3 | e25 mIoU |
|---|---|---|---|--:|--:|
| Sign (15) | **0.80 / 0.80** | 0.07 / 1.00 | 0.80 / 0.80 | 0.87 | 0.85 |
| Other vehicle (3) | 0 / – (0 dự đoán) | 0 / – | 0 / – | 0 | – |
| Motorcycle (21) | 0.05 / 0.14 | 0.05 / 0.25 | 0.05 / 0.17 | 0.05 | 0.74 |
| Motorcyclist (36) | 0.39 / 0.64 | 0.36 / 0.77 | 0.42 / 0.68 | 0.39 | 0.82 |
| Xe 2 bánh gộp (57) | 0.32 / 0.62 | 0.28 / 0.76 | 0.32 / 0.64 | 0.39 | 0.80 |
| Traffic light (2) | 0 / 0 | 0 / – | 0 / 0 | 0 | – |

Nhầm lớp (dự đoán trùng IoU ≥ 0.5 với tham chiếu lớp khác), e25: Motorcycle→Motorcyclist 3, Motorcyclist→Pedestrian 2
(có thể là người lái chưa ghép trong tham chiếu), Motorcyclist→Motorcycle 1.

## conf 0.1 (đề xuất cho người duyệt, chịu thêm FP)

| | e25 R / P | COCO R / P | union R / P |
|---|---|---|---|
| Sign | 0.80 / 0.57 | 0.07 / 1.00 | 0.80 / 0.57 |
| Xe 2 bánh gộp | **0.44 / 0.49** (R@0.3 0.56) | 0.39 / 0.61 | 0.39 / 0.50 |

## Đọc kết quả

- **Sign:** e25 là nguồn đề xuất chính; COCO gần như vô dụng (chỉ có `stop sign`). Hạ conf chỉ thêm FP.
- **Other vehicle:** cả hai backend **không đề xuất được gì** → cần vẽ tay hoặc SAM3 (prompt text "xe ba gác", "xích lô"...).
- **Motorcycle/Motorcyclist:** khoảng 2/3 xe 2 bánh bị bỏ sót ở conf 0.25; Motorcycle (xe không người) gần như không tìm
  được. Tách lớp người lái/không người lái là điểm yếu riêng (xem nhầm lớp). Khi gói đề xuất xe 2 bánh để duyệt, dùng
  **e25 đơn lẻ ở conf ~0.1** thay vì union: union bỏ mất một số mask e25 tốt hơn vì mask COCO điểm cao chồng lên.
- **SAM3:** không đo — nhóm quyết định không triển khai SAM3.
