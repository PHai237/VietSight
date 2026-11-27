# V2 — fine-tune từ e25 trên dữ liệu Việt Nam READY

Chỉ áp dụng khi Codex bàn giao `status: READY` (có `fingerprint_sha256`, `vn_val_sha256`, tập val VN khóa). Không dùng ảnh
BLOCKED, ảnh trong `review/`, hay pseudo-label chưa duyệt (`runs/proposals/*`, `pseudo_unreviewed`). Notebook vẫn dừng
trước khi train nếu handoff không READY.

Tập train V2 = **100% ảnh Việt Nam**. Không trộn ảnh Mapillary quốc tế vào train; Mapillary val V1 chỉ dùng để so sánh.

## Run chính

```
python scripts/make_colab_v2_notebook.py --epochs 12 --lr0 0.0003 --close-mosaic 3 --handoff configs/handoff/v2.json
```

| Tham số | Giá trị | Lý do |
|---|---|---|
| init | e25 `best.pt` (sha kiểm trong notebook) | giữ kiến thức V1 |
| optimizer / lr0 | AdamW, **3e-4**, cosine (`lrf` 0.1) | e25 train với `optimizer=auto` → AdamW lr ≈ 6.7e-4 (công thức 0.01/(4+nc) của Ultralytics 8.4.214) |
| warmup | 1 epoch, **`warmup_bias_lr = lr0`** | mặc định Ultralytics là 0.1: bias bắt đầu với LR gấp hàng trăm lần lr0 → làm trôi e25 |
| freeze | không | dataset VN mới đủ lớn để train toàn mạng |
| epochs / close_mosaic | 12 / 3 | |
| imgsz / model | 640 / YOLO26n-seg | không đổi (luật cuộc thi) |

## Thử nghiệm dự phòng (nếu run chính làm giảm Mapillary val hoặc không tăng VN val)

```
python scripts/make_colab_v2_notebook.py --epochs 12 --lr0 0.0001 --freeze 10 --close-mosaic 3 --handoff configs/handoff/v2.json
```

`--freeze 10` giữ nguyên lớp 0–9 (Conv/C3k2/SPPF, ~1.37M/3.13M tham số); chỉ C2PSA + neck + head học. Smoke test CPU
(1 epoch, sample_data): lớp 0–9 không đổi (cả BN), lớp 10 và head thay đổi.

## Chấp nhận kết quả

Ô cuối notebook so e25 và run mới trên **VN val** và **Mapillary val V1** bằng scorer chính thức (`eval_compare.json`).
Đề xuất (cần nhóm chốt): chỉ nộp khi VN val tăng **và** Mapillary val không giảm đáng kể. Chưa có số đo nào: chưa chạy.

## SAM3

Không triển khai (quyết định của nhóm 27/11/2026): không tải trọng số, không cài thêm thư viện.
