# VietSight — Phiên bản dữ liệu & kết quả

Val cố định: `configs/val_frozen.txt` (tạo ở lần build V1 đầu tiên, theo group, seed 55). Mọi so sánh dùng cùng val.
Ảnh val không bao giờ vào train (`build_version.py` loại theo key và theo group/video). Kết quả từng run: [`runs/RESULTS.md`](../runs/RESULTS.md).

| Phiên bản | Config | Nội dung | Trạng thái |
|---|---|---|---|
| V1 — First Gear | `configs/versions/v1_first_gear.yaml` | Mapillary Vistas v2 ≤ 6 500 ảnh (ưu tiên xe 2 bánh/người lái), val 10% | pipeline kiểm thử xong; chờ chạy notebook prep trên Kaggle |
| V2 — Local Vision | `configs/versions/v2_local_vision.yaml` | V1 + ảnh VN của nhóm (pre-label + CVAT) + BDD sau khi điền Sign/TL | chờ dữ liệu |
| V3 — Fine Focus | (tạo sau) | sửa nhãn theo ảnh lỗi Bước 5, tăng lớp AP thấp | |
| Final | — | chọn ≤ 2 checkpoint theo val + public LB | |

## Nhật ký
- 2026-11-27: dựng pipeline; smoke test CPU toàn notebook chính thức (5 ảnh, 1 epoch) → self-check máy chấm **ĐẠT**, zip 9.2 MiB. Chưa có số liệu train thật.
- Khi thay đổi val (chỉ khi bắt buộc, ví dụ thêm ảnh VN vào val): ghi lý do ở đây và chấm lại các checkpoint cũ trên val mới.
