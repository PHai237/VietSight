# V2 — fine-tune từ e25 trên dữ liệu Việt Nam READY

Chỉ áp dụng khi Codex bàn giao `status: READY` (có `fingerprint_sha256`, `vn_val_sha256`, tập val VN khóa). Không dùng ảnh
BLOCKED, ảnh trong `review/`, hay pseudo-label chưa duyệt (`runs/proposals/*`, `pseudo_unreviewed`). Notebook vẫn dừng
trước khi train nếu handoff không READY.

Tập train V2 = **100% ảnh Việt Nam**. Không trộn ảnh Mapillary quốc tế vào train; Mapillary val V1 chỉ dùng để so sánh.

## Thử nghiệm hay đủ điều kiện nộp

Trường `rights_confirmed` trong `configs/handoff/v2.json` quyết định:

| `rights_confirmed` | Tên run / thư mục Drive | Được nộp cuộc thi? |
|---|---|---|
| `false` (mặc định, quyền nguồn chưa xác nhận) | `<version>_ft_e12_EXPERIMENT` + `NOT_FOR_SUBMISSION.txt`, cảnh báo đầu/cuối notebook | **Không** |
| `true` (chỉ khi có bằng chứng quyền nguồn) | `<version>_ft_e12` | Có, nếu đạt mục "Chấp nhận kết quả" |

Hai loại run ở hai thư mục khác nhau; không bao giờ ghi vào `vietsight_v1_e25` / `vietsight_v1_e50` (notebook chặn). Khi
quyền được xác nhận sau, tạo lại notebook với `rights_confirmed: true` và train lại vào thư mục mới, không đổi tên run cũ.

## Upload Drive và chạy

1. Drive phải có sẵn (từ V1): `MyDrive/VietSight/vietsight_v1_e25/runs/vietsight_v1_e25/weights/best.pt` và
   `MyDrive/VietSight/vietsight_v1_e25/submission.zip`. Notebook kiểm sha256 với `runs/v1_colab_e25/run_report.json`.
2. Upload ZIP của Codex lên `MyDrive/VietSight/datasets/<version>.zip`. ZIP phải chứa đúng một thư mục có
   `images/train`, `images/val`, `labels/train`, `labels/val` (ảnh nào cũng có nhãn).
3. Điền `configs/handoff/v2.json` (mẫu `v2.example.json`): `status` READY, `version`, `source` =
   `drive:/content/drive/MyDrive/VietSight/datasets/<version>.zip`, số ảnh train/val, `fingerprint_sha256` (hàm
   `dataset_fingerprint` chính thức), `vn_val_sha256`, `rights_confirmed`.
4. `python scripts/make_colab_v2_notebook.py --handoff configs/handoff/v2.json`, commit + push notebook.
5. Mở link Colab ở ô đầu notebook → Runtime → T4 GPU → Run all. Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY` chỉ cần cho
   Mapillary val (so sánh). Mất kết nối: Run all lại, tự resume cùng run.
6. Kết quả trên Drive `MyDrive/VietSight/<run>/`: `runs/<run>/weights/` (best/last + `epoch0/3/6/9.pt`; epochN = sau N+1 epoch), `submission.zip`,
   `run_report.json`, `eval_compare.json`.

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
| batch / save_period | 16 / 3 | T4; checkpoint mỗi 3 epoch trên Drive |
| imgsz / model | 640 / YOLO26n-seg | không đổi (luật cuộc thi) |

## V3 Mixed (Mapillary V1 + Việt Nam V2) — từ pretrained, 50 epoch

Chờ `configs/handoff/v3.json` của Codex (mẫu `configs/handoff/v3.example.json`). Khác V2: `val_subsets` chia `images/val`
thành các phần (regex trên tên file, số ảnh, sha256 danh sách tên). Notebook kiểm từng phần, kiểm các phần phủ đủ val, rời
nhau, không trùng tên với train; phần `mapillary_val` phải đúng `configs/val_frozen.txt` của V1. Mỗi phần được chấm riêng.

Handoff thật của Codex dùng schema `validation_sets` (mỗi tập val một thư mục: `images/val` = VN, dùng khi train;
`images/val_v1` = Mapillary V1 = đúng `configs/val_frozen.txt`, chỉ chấm riêng). Notebook nhận cả `validation_sets`
lẫn `val_subsets`. Mỗi bản annotation = một handoff = một notebook + một thư mục run riêng (tên chứa version):

```
# final001 (đã có): notebooks/colab/VietSight_TEAM55_colab_v3_final001.ipynb
python scripts/make_colab_v2_notebook.py --init pretrained --epochs 50 --save-period 5 --eval-ckpts epoch25.pt epoch35.pt \
    --out-name VietSight_TEAM55_colab_v3_final001.ipynb --handoff configs/handoff/v3_final001.json
# final002 (khi Codex bàn giao): copy handoff -> configs/handoff/v3_final002.json, cùng lệnh với _final002
```

Thứ tự ô: setup VietSight chạy TRƯỚC ô cài thư viện chính thức, nên không import ultralytics ở đó; tải `yolo26n-seg.pt`
được gọi trong ô train (sau khi cài). V2 pretrained từng import ultralytics trong setup.

**Export giữa chừng** (sắp hết giờ/GPU): trong ô setup đặt `EXPORT_ONLY_CKPT = "last.pt"` (hoặc `"epoch25.pt"`) rồi
Run all, chạy được trên runtime CPU. Không train; export + self-check chính thức trên checkpoint đó, kết quả ở
`<run>/export_<tên>/` (không ghi đè thư mục run). `last.pt` được Ultralytics ghi sau mỗi epoch.

(`--eval-ckpts` chỉ nhận checkpoint được lưu: `epoch0/5/10/…/45.pt`; `epoch25.pt` = sau 26 epoch.)

- Công thức chính thức (`dict()` = mặc định Ultralytics như V1 e25), batch 16, imgsz 640, T4, checkpoint mỗi 5 epoch trên Drive.
- Ultralytics đếm epoch từ 0: `epochN.pt` = trọng số sau N+1 epoch (dòng N+1 của `results.csv`). Với save_period 5 có
  `epoch0, 5, 10, …, 45.pt` + `best.pt`, `last.pt`.
- Ô cuối: export từng checkpoint trong `--eval-ckpts` bằng `make_submission` chính thức (thư mục riêng, không đụng
  `submission.zip` của run), chấm trên từng phần val → `ckpt_eval.json`, zip ở `ckpt_submissions/`. Đóng gói
  `weights/` → `<run>_weights.zip` trên Drive; `DOWNLOAD_WEIGHTS = True` để tải về máy.
- Run `<version>_pretrained_e50_EXPERIMENT` (thư mục mới; notebook từ chối thư mục e25/e50 và run cũ khác dataset).

## Khi không còn best.pt e25 (27/11/2026: không tìm thấy trên Drive lẫn trên máy)

```
python scripts/make_colab_v2_notebook.py --init pretrained --epochs 25 --handoff configs/handoff/v2.json
```

Notebook riêng `notebooks/colab/VietSight_TEAM55_colab_v2_pretrained.ipynb`: khởi tạo từ `yolo26n-seg.pt` COCO chính thức
(Ultralytics release v8.4.0, sha256 ghim `361fbfab…`, sai sha thì dừng; có thể đặt sẵn bản của repo ở
`MyDrive/VietSight/init/yolo26n-seg.pt`), 25 epoch, tham số mặc định chính thức (cùng công thức V1 e25), chỉ dữ liệu VN.
Mốc e25 trong `eval_compare.json` chỉ có khi `MyDrive/VietSight/vietsight_v1_e25/submission.zip` đúng sha (bản ở
`runs/v1_colab_e25/submission.zip` khớp).

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
