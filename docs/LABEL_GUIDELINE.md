# VietSight — Quy tắc gán nhãn (LABEL GUIDELINE)

Áp dụng cho: chuyển đổi public dataset, AI pre-label, chỉnh polygon trên CVAT, duyệt nhãn.
Ký hiệu: **[BTC]** = theo tài liệu/dữ liệu của ban tổ chức · **[NHÓM]** = nhóm tự thống nhất · **[HỎI BTC]** = chưa chắc, cần xác nhận trên website/với ban tổ chức.

## 1. Định dạng [BTC]
- YOLO segmentation: mỗi dòng `class x1 y1 x2 y2 ...` (≥ 3 điểm, tọa độ chuẩn hóa [0,1] theo ảnh **sau khi xoay EXIF**). Nhãn hộp 5 cột bị từ chối.
- **Một dòng = một instance = một polygon.** Vật bị che thành nhiều mảnh: các mảnh được nối thành một vòng bằng "cầu" độ rộng 0 (pipeline tự làm; trên CVAT vẽ một polygon cho phần lớn nhất hoặc gộp bằng công cụ mask).
- Bộ chấm rasterize polygon bằng pycocotools; mask < **64 px ở thang 640** (`pixels × (640/max(H,W))²`) bị **bỏ qua** → không cần gán vật quá nhỏ (pipeline tự loại).
- Mỗi ảnh tối đa 100 dự đoán được chấm; ảnh nền (không vật thể) hợp lệ.
- Thứ tự lớp cố định 0–10: Car, Bus, Truck, Other vehicle, Motorcycle, Motorcyclist, Bicycle, Cyclist, Pedestrian, Sign, Traffic light.

## 2. Xe 2 bánh và người lái (quan trọng nhất)
Căn cứ: `sample_data.zip` chính thức (ảnh tổng hợp) — Motorcyclist/Cyclist được vẽ là **một mask gộp người + xe**. [BTC]

| Tình huống | Lớp | Mask |
|---|---|---|
| Xe máy có ≥ 1 người ngồi/lái (kể cả người ngồi sau, trẻ em) | **Motorcyclist (5)** | hợp (union) của xe + tất cả người trên xe, **1 instance** |
| Xe máy không có người (đỗ, dựng chân chống) | **Motorcycle (4)** | chỉ xe |
| Xe đạp có người đạp/ngồi | **Cyclist (7)** | xe + người, 1 instance |
| Xe đạp không người | **Bicycle (6)** | chỉ xe |
| Người đi bộ, đứng, người dắt xe đi bộ bên cạnh | Pedestrian (8) + xe riêng (Motorcycle/Bicycle) | [NHÓM] — **[HỎI BTC]** trường hợp dắt xe |
| Người đứng sát/ngồi tạm trên xe đang đỗ (không lái) | **[HỎI BTC]** — tạm thời: đưa vào hàng review | |
| Xe máy chở hàng cồng kềnh | Motorcyclist, mask gồm xe + người; hàng hóa: **[HỎI BTC]** (tạm: không gồm) | |
| Xe đạp điện / xe máy điện | theo hình dáng: có bàn đạp kiểu xe đạp → Bicycle/Cyclist; kiểu xe máy/scooter → Motorcycle/Motorcyclist [NHÓM] | |

Quy tắc pipeline (không suy diễn) [NHÓM]:
- KHÔNG ánh xạ mọi `motorcycle` thành Motorcycle, KHÔNG ánh xạ mọi `rider` thành Motorcyclist.
- Nguồn có lớp `rider`/`motorcyclist`/`bicyclist` riêng (Mapillary, BDD): người lái được ghép với xe bằng hình học (hộp chồng nhau, tâm người nằm trong bề ngang xe, chân/thân dưới nằm trong chiều cao xe, mask chạm nhau). Ghép được → Motorcyclist/Cyclist (union). **Không tìm thấy xe** → ảnh vào **hàng review**, không vào dataset tự động.
- Nguồn chỉ có `person` (COCO): người được coi là người lái khi thỏa đủ điều kiện hình học "mạnh"; chồng lấn "yếu" với một xe đang không có người → ảnh vào hàng review.
- Xe 2 bánh không có ai ghép → Motorcycle/Bicycle.

## 3. Các lớp khác
| Lớp | Gồm | Không gồm / ghi chú |
|---|---|---|
| Car (0) | ô tô con, SUV, taxi, xe 7 chỗ | pickup: **[HỎI BTC]** (COCO gộp pickup vào truck) |
| Bus (1) | xe buýt, xe khách | |
| Truck (2) | xe tải, xe ben, container (đầu kéo + thùng) | |
| Other vehicle (3) | **[HỎI BTC]** định nghĩa chính xác. Tạm [NHÓM]: xe ba gác, xích lô, xe lôi, xe cải tiến, máy cày, caravan, rơ-moóc, xe chuyên dụng | tàu hỏa/tàu điện, thuyền: không gán |
| Pedestrian (8) | người đi bộ/đứng/ngồi không trên phương tiện | người trong ô tô/xe buýt: không gán. Đám đông không tách được: tô xám (xem §5) |
| Sign (9) | **mặt trước** biển báo giao thông (cấm, nguy hiểm, hiệu lệnh, chỉ dẫn, biển tạm) — chỉ mặt biển, không gồm cột | mặt sau biển, biển quảng cáo/biển hiệu cửa hàng: không gán. Mặt sau: **[HỎI BTC]** |
| Traffic light (10) | thân/hộp đèn (gồm đèn cho người đi bộ) — không gồm cột/cần treo [BTC: sample] | đèn nhìn từ phía sau: **[HỎI BTC]** |

## 4. Độ chính xác mask (điểm chấm mAP@[.5:.95] rất khắt khe)
- Polygon bám sát biên vật thể nhìn thấy, sai số ≤ 2 px ở ảnh gốc. Dùng SAM/AI tools của CVAT rồi chỉnh tay.
- Chỉ vẽ **phần nhìn thấy**; phần bị vật khác che không vẽ. (Mapillary dùng polygon chồng lớp → pipeline tự trừ phần bị che: `z_order: painter`.)
- Không gán vật bị cắt ở mép ảnh nếu nhìn thấy < ~20% và không nhận ra lớp. [NHÓM]
- Mỗi vật thể nhìn thấy rõ (≥ 64 px ở thang 640) đều phải có nhãn — thiếu nhãn làm model học sai "vật thể = nền". Không bỏ qua vật thể chỉ để làm nhanh.

## 5. Vùng không tách được instance
Đám đông người, hàng xe đỗ dày đặc ở xa (Mapillary `person group`/`vehicle group`, COCO `iscrowd`): pipeline **tô xám (114)** vùng đó, giữ lại ảnh [NHÓM]. Trên CVAT: nếu không tách được, gán nhãn tạm `ignore_region` (polygon) — script sẽ tô xám (cần bổ sung khi có dữ liệu thật).

## 6. Quy trình duyệt [NHÓM]
1. Converter/pre-label → ảnh rõ ràng vào `images/`; ảnh mơ hồ hoặc **toàn bộ ảnh AI pre-label** vào `review/` (`status: review`).
2. `python scripts/review_queue.py export ...` → CVAT (YOLO Segmentation 1.0) → người gán sửa → export → `review_queue.py accept --reviewer <tên>`.
3. QA kiểm tra chéo ≥ 10% ảnh mỗi lô bằng `scripts/preview.py` (lọc theo lớp: `--cls Motorcyclist Cyclist`), ghi lỗi vào issue/bảng theo dõi. Lô có > 5% ảnh lỗi → trả lại sửa.
4. Chỉ `status: ok` mới vào dataset version. Dữ liệu pre-label chưa duyệt **không bao giờ** dùng để train.
5. `sample_data.zip` chỉ dùng kiểm thử pipeline, không train.

## 7. Cần xác nhận với ban tổ chức (website "Quy ước gán nhãn")
1. Định nghĩa đầy đủ của **Other vehicle**; pickup thuộc Car hay Truck?
2. Mask Motorcyclist có gồm **hàng hóa** chở trên xe / người ngồi sau không? (mẫu: có gồm người)
3. Người **dắt xe**, người ngồi trên xe đang đỗ: Pedestrian + Motorcycle hay Motorcyclist?
4. **Mặt sau** biển báo, đèn giao thông nhìn từ sau, biển chỉ đường lớn trên giá long môn: có gán không?
5. Xe đạp điện/xe máy điện thuộc nhóm nào?
6. Vùng đám đông: bộ test có vùng ignore không?
