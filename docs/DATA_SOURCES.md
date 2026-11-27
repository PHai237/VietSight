# VietSight — Nguồn dữ liệu (provenance)

Bản đọc được bằng script: [`data/sources_manifest.yaml`](../data/sources_manifest.yaml). Số liệu thực tế của mỗi lần build nằm ở
`data/versions/<version>_report/stats.json` (sinh tự động, copy về repo dưới tên `data/versions/<version>_stats.json`).
Nguyên tắc: **không ghi số liệu hay giấy phép chưa kiểm chứng** — ô nào chưa đo ghi "chưa đo".

Kiểm tra ngày 27/11/2026. Các trang chính thức Mapillary/COCO render bằng JavaScript và trang BDD bị lỗi chứng chỉ khi kiểm tra tự động →
giấy phép dưới đây lấy từ LICENSE.md của mirror Dataset Ninja (có dẫn nguồn gốc) — **Dataset Engineer cần mở trang chính thức xác nhận lại**.

## Tổng quan

| Nguồn | Trạng thái | Giấy phép | Tải | Nhãn | Ảnh gốc |
|---|---|---|---|---|---|
| Mapillary Vistas v2.0 | **Dùng (V1)** | CC BY-NC-SA 4.0 | mirror Dataset Ninja (.tar Supervisely, ~21 GB), không cần đăng ký | polygon chồng lớp, 124 lớp (70 instance) | 25 000 (18k train, 2k val, 5k test không nhãn) |
| BDD100K 10K | Ứng viên (V2) | BDD100K license (UC Regents): giáo dục/nghiên cứu/phi lợi nhuận | mirror Dataset Ninja (~1 GB) | instance mask 10 lớp, **không có Sign/Traffic light** | 10 000 (7k/1k/2k test) |
| COCO 2017 | Ứng viên | Annotation CC BY 4.0; ảnh theo giấy phép Flickr từng ảnh | images.cocodataset.org (tải từng ảnh theo URL) | polygon/RLE, chỉ có *stop sign* | 123 287 |
| IDD (India Driving) | Chưa dùng | **Chưa rõ** (mirror ghi "unknown") | cần đăng ký idd.insaan.iiit.ac.in | | 20k (theo trang dự án) |
| Ảnh VN của nhóm | Kế hoạch V2 | dữ liệu của nhóm | Data Collector | AI pre-label + CVAT | — |
| sample_data.zip (BTC) | **Chỉ kiểm thử** | — | — | 10 ảnh tổng hợp | 10 |
| Data-VF.zip (máy Team Lead) | Không dùng | không rõ nguồn/giấy phép | — | camera + LiDAR, không có mask 2D | 2 100 ảnh |

Đã khảo sát nhưng chưa dùng được: repo GitHub *VN-Traffic-Instance-Segmentation-YOLO11* (ảnh VN + polygon) — README không có link tải dữ liệu, không ghi giấy phép dữ liệu.

## Mapillary Vistas v2.0 (V1)
- URL: https://www.mapillary.com/dataset/vistas · mirror: https://datasetninja.com/mapillary-vistas-dataset
- Giấy phép: CC BY-NC-SA 4.0 → phi thương mại, ghi nguồn, chia sẻ cùng giấy phép. Dataset dẫn xuất trên Kaggle để **Private**.
- Chuyển đổi: `fetch_sly_tar.py` (stream tar; giữ mọi ảnh có xe 2 bánh/người lái/other vehicle/wheeled slow + 30% ảnh còn lại theo hash tất định; thu nhỏ ≤ 1024 px) → `convert_supervisely.py` (`z_order: painter`: mask nhìn thấy = polygon trừ mọi object vẽ sau, kể cả cột/cây; ghép người lái–xe; tô xám nhóm người/nhóm xe) → `build_version.py`.
- Class mapping: [`configs/sources/mapillary_vistas.yaml`](../configs/sources/mapillary_vistas.yaml).
- Nhãn bị loại và lý do: mặt sau biển báo, biển mơ hồ, biển quảng cáo/cửa hàng (không phải biển giao thông); tàu điện, thuyền, xe ego; mọi lớp nền (đường, trời, cột...) chỉ dùng làm vật che; vật < 64 px ở thang 640 (bộ chấm bỏ qua). Thống kê chi tiết: `data/processed/mapillary_vistas/summary.json` (`dropped`, `unmapped_labels`).
- Phương pháp duyệt: quy tắc tự động + ảnh mơ hồ vào review; QA xem preview theo lớp (riders, other vehicle) trước khi train.
- Ảnh thực tế sử dụng / instance theo lớp: **chưa đo** (điền từ `vietsight_v1_report/stats.json` sau khi chạy notebook prep).
- Phát hiện khi kiểm tra định dạng: polygon của mirror **chồng lớp** (đo trên 15 ảnh: polygon xe–xe chồng nhau trung vị 33%) → bắt buộc dùng painter order.

## BDD100K 10K (ứng viên)
- URL: https://www.bdd100k.com/ · mirror: https://datasetninja.com/bdd100k-10k · config: `configs/sources/bdd100k_10k.yaml`
- Lý do chưa dùng ở V1: không có nhãn Sign/Traffic light trong khi ảnh dashcam nào cũng có → dạy model coi biển/đèn là nền. Dùng sau khi `prelabel.py fill` (model V1) + người duyệt.
- Tách train/val theo video (tiền tố tên ảnh). Cần kiểm tra trên dữ liệu thật: mask có chồng nhau không.

## COCO 2017 (ứng viên)
- Chỉ lấy ảnh có motorcycle/bicycle (bổ sung lớp hiếm). Hạn chế: ảnh không phải góc nhìn xe, chỉ có biển STOP, `truck` gồm pickup. Vùng `iscrowd` → tô xám.

## Chia sẻ dữ liệu
- Dữ liệu không vào git. Dataset dẫn xuất chia sẻ qua **Kaggle Private dataset/Notebook output** trong nhóm, ghi nguồn đầy đủ.
- Không công khai lại dữ liệu Mapillary/BDD/IDD; ảnh nhóm tự chụp: làm mờ mặt/biển số nếu công bố ra ngoài.
