# V2 — Local Vision, chỉ ảnh giao thông Việt Nam

**L? ?ang giao ng??i ki?m:** 5 ZIP round001, 20 ?nh/g?i, t?ng 70 dashcam + 30 val d? ki?n t? ba video. 862 polygon draft, g?m 237 ?? xu?t b?ng c?ng c? Claude; 0 human-reviewed/READY. H??ng d?n: [V2_CVAT_REVIEW.md](../V2_CVAT_REVIEW.md). D? li?u c?n thi?u quy?n g?c gi? ri?ng RIGHTS_PENDING; source thay th? Sign VR-TSD ?? x?c minh metadata CC BY 4.0 v? m? t? t?c gi? t? thu th?p, nh?ng ch? bbox n?n c?n mask th?t.

Cập nhật 27/11/2026, Asia/Saigon. Bàn giao hiện tại ở **[docs/V2_DATA_HANDOFF.md](../V2_DATA_HANDOFF.md)**; đây là nguồn trạng thái/số liệu mới nhất. Làm trên branch `codex/data-qa`, không thao tác working directory của Claude hoặc luồng training V1.

## Kết quả DATA thực tế

- Đã đóng gói **377 ảnh ứng viên / 4.279 polygon draft**, train dự kiến **309**, val Việt Nam dự kiến **68**. Snapshot local và hai ZIP CVAT riêng đã có, hash/fingerprint ghi trong file bàn giao.
- **BLOCKED cho training, accepted train/val = 0/0**: còn sai/thiếu mapping rider, thiếu Other vehicle/Cyclist, missing Sign, mask/overlap cần sửa và quyền footage gốc chưa đối chiếu đủ. Human QA chưa được xác nhận. Không bypass gate để tạo dataset có vẻ READY.
- 100 ảnh/540 polygon GitHub đã tải và kiểm đúng số lượng; manifest hash dùng decoded BGR có shape, khớp 100/100. Source chỉ bốn class và có chính sách pickup khác; không nhận trực tiếp làm 11 class BTC.
- SideWalk v3: **279 ảnh/4.715 native masks**, đã kiểm bốn video tại Việt Nam và xem overview 279 ảnh. Nguồn phù hợp bổ sung xe máy đỗ/xe đạp; góc nhìn đi bộ. Toàn video Đà Nẵng được tách làm val dự kiến; không giữ split gốc đang rò rỉ frame cùng video.
- 73 ảnh Kaggle `pvd2201/data-vn` đã tải; **0 annotation gốc**. Tạo 733 COCO-pretrained mask nháp trên CPU, sau remap/geometry còn 709 polygon trong queue riêng. Không nhập lô 377 khi nguồn/quyền/nhãn chưa giải quyết.
- API key Roboflow đã có ở **User environment**, nạp vào từng phiên lệnh riêng, HTTP 200. Không in/commit key hoặc signed download links. Nguồn Jose được đọc ZIP bằng Range, chỉ lấy test annotations +37 ảnh, 6,86 MB mạng thay vì tải full 1,185 GB; mẫu thực có ảnh quốc tế nên giữ blocked cho intake mặc định.
- Nguồn Sign hoangthanh xuất 354 ảnh/368 segmentation, nhưng 348 polygon có bốn góc sau bỏ điểm đóng và spot-check `10040` chứa cột/hai biển. Không coi đây là mask sẵn đạt BTC và không đưa vào lô.
- Bổ sung **287 Sign +30 Traffic light draft** bằng teacher V1 e3 chạy CPU 2 threads; giữ 157 rider suggestions riêng, không tự gộp vào GT. Vẫn thấy missing Sign và gợi ý đèn trang trí sai, nên AI pre-label không thay kiểm coverage.

## Quy tắc đang áp dụng

1. Không nhập train V1 hoặc ảnh quốc tế làm nền. Mapillary chỉ nhận từng ảnh có bằng chứng chụp tại Việt Nam; ảnh Mapillary QA cũ chỉ là reference chuyển đổi.
2. Ưu tiên **Motorcycle, Other vehicle, Sign, Bicycle, Motorcyclist**. Builder riêng V2 đã thêm Motorcyclist vào danh sách chọn lô.
3. Giữ đầy đủ người ngồi sau trong rider instance; người dắt tách Pedestrian + xe; đầu kéo/rơ-moóc hai instance; 9–15 chỗ là Car. Không lấy source class làm BTC truth.
4. V2 giữ pixel-edge masks/round-trip gate, không largest-part fallback; raw/decoded overlap, ngưỡng cạnh 1,25% và area 64 px² báo riêng. Mask không biểu diễn được giữ review và source annotation để sửa.
5. Split theo canonical video/camera/chuyến chụp; VN val khóa ảnh+nhãn riêng khi phát hành, V1 val giữ độc lập để đối chiếu. Chưa có full V1 val local để so pixel, không nhận là đã kiểm xong bước đó.
6. Đóng gói mỗi phiên bản riêng; snapshot review 001 không ghi đè. Chỉ phát hành phiên bản mới sau giải quyết lỗi/gate và ghi đúng phương pháp/người kiểm tra, không tự gọi AI QA là human QA.

Thử builder với hai source thật trả `BLOCKED: no reviewed Vietnam images; no trainable dataset created`. Công cụ converter, audit, review queue và CVAT exporter hiện có đã được tái sử dụng; không phát triển UI mới. Validation checksum/ZIP/fingerprint và kết quả test lưu trong [evidence DATA](evidence/V2_DATA_INTAKE001.json).

V1 e25 public **12,90**, e50 public **12,37** do người dùng cập nhật; 2/20 lượt nộp. Không có V2 metric hoặc lần nộp mới. [Sign V1 forensic](SIGN_FORENSICS.md) vẫn là đo độ trung thực chuyển đổi trên Mapillary, không phải dữ liệu train V2 hoặc AP cải thiện dự đoán.

PR #2 giữ **chưa merge pipeline**. Cấu hình Claude cũ `configs/versions/v2_local_vision.yaml` vẫn có V1/Mapillary: không dùng cho yêu cầu VN-only mới, không sửa luồng đó trong phiên DATA này.
