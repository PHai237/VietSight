# Dashcam Việt Nam — kiểm định lô 1.407 ảnh

**BLOCKED, chưa phát hành READY.** Đã chuyển đổi đủ 1.407 ảnh sang YOLO segmentation; không gộp prediction vào GT, không giả human reviewer. Thành viên nhóm gán lại 11 lớp, theo xác nhận người dùng ngày 27/11/2026; chưa rõ có AI hỗ trợ. Không yêu cầu gán lại toàn bộ.

ZIP nguồn khớp SHA256 `8a3ff16fb78b22d8d0f00d73996c994efd17bdc2b96fa423eee617cb4df3a2a0`; `annotations/instances.json` khớp `81a17b3e0f4e5d2f0446ba8a399672c961f1e171d108c579fcda31a6f60c2379`.

[Nguồn Roboflow](https://universe.roboflow.com/d22cqcn01ns-workspace/dashcam_vietnam): D22CQCN01Ns Workspace, publisher công bố **CC BY 4.0**, 1.412 ảnh, bbox/5 lớp. API xác nhận, chưa có version export công khai. Ba preview tải nhỏ khớp trực quan và ảnh xám thu nhỏ với ba ảnh ZIP; không nhận là hash byte giống nhau hoặc đã đối chiếu cả 1.407 ảnh. Chưa xác định lý do thiếu 5 ảnh. Ghi attribution publisher, URL, license, đóng góp gán lại của thành viên và thay đổi chuyển đổi trong gói. Ảnh có watermark Camera Giao Thông/VOV; người dùng xác nhận **chưa có xác nhận quyền footage gốc riêng**. Không suy diễn quyền đó từ license của publisher, cũng không kết luận watermark tự chứng minh vi phạm.

Remap **theo tên**, không dùng category ID làm YOLO ID. Category ID1 thừa có 0 instance, bỏ khỏi ontology. Hai bánh nguồn đã phân thành Motorcycle/Motorcyclist/Bicycle/Cyclist nên giữ class/mask đã gán; không tự ghép lại tất cả theo bbox. Không đổi mọi Truck thành Other vehicle hoặc Car.

| Class | COCO nguồn | YOLO sau chính sách kích thước |
|---|---:|---:|
| Car | 1829 | 1778 |
| Bus | 203 | 202 |
| Truck | 443 | 441 |
| Other vehicle | 35 | 35 |
| Motorcycle | 518 | 517 |
| Motorcyclist | 2840 | 2687 |
| Bicycle | 19 | 19 |
| Cyclist | 91 | 90 |
| Pedestrian | 922 | 798 |
| Sign | 902 | 853 |
| Traffic light | 253 | 215 |
| **Tổng** | **8055** | **7635** |

- Kiểm đủ 1.407 ảnh/8.055 annotation. Shapely báo 72 polygon lỗi: **41** chuẩn hóa được mà giữ nguyên pixel theo COCO decoder; **30** dưới ngưỡng kích thước; **1** không pixel-exact (ID3575, image644), giữ riêng để xem. Không nhận sửa hình học là đã sửa đúng ngữ nghĩa.
- **420** instance dưới ngưỡng cạnh ngắn 1,25% theo bounding extent của mask raster được loại có log; COCO gốc giữ nguyên. Khác 600 bbox metadata dưới8 vì tọa độ vector và làm tròn pixel. Không dùng diện tích64 để quyết định loại. Nguồn có314 area metadata dưới64; nhãn sau chuyển đổi có131 theo phép đo polygon của notebook và125 theo pixel decode — ghi rõ phép đo, không đánh đồng.
- Validator notebook trên toàn bộ nhãn chuyển đổi: **0 polygon sai cấu trúc**. **16 ảnh có mask chồng nhau**, không tự chọn vật trước/sau khi chưa rõ. Nguồn có32 ảnh trống; sau chính sách kích thước là34 ảnh, không mặc nhiên coi là negative hợp lệ.
- Kiểm mẫu RGB/overlay, toàn bộ19 crop Bicycle và35 crop Other vehicle; không nhận các crop nhỏ là chứng nhận đầy đủ semantic QA. Ví dụ **image621** thiếu rider lớn giữa-phải, **image954** thiếu bus lớn, **image313** thiếu nhiều Car. Chuyển ngoại lệ cụ thể, không yêu cầu vẽ lại cả nguồn.
- COCO pretrained CPU/conf0,5 quét **đủ1.407 ảnh**:828 ảnh có2.103 đề xuất bổ sung. Đây là **ứng viên**, không phải828 ảnh lỗi được xác nhận; giữ riêng, không tự nhập GT hoặc xóa cả828 ảnh.562 ảnh qua bộ lọc bước đầu vẫn còn lỗi khi lấy mẫu; không gọi là562 ảnh READY. Kiểm e25+COCO bổ sung trên nhóm này được ghi riêng theo số ảnh thực chạy.

**Leakage thật đã tìm và sửa:** source val `frame_000344` (12/08/2020 16:25:15) và train `frame_000345` (16:25:20) cùng clip HIKVISION. Clip liên tục đến350, source351 đổi video; chuyển **345–350, sáu ảnh**, sang val trong bản YOLO riêng. Split mới **1120 train /287 val**. Không sửa ZIP/COCO nguồn. Cả clip quan sát ở ranh giới ở cùng phía sau sửa; chưa có manifest video gốc để chứng minh mọi clip khác. SHA256 không có trùng toàn bộ nguồn;18 cặp dHash≤4 chéo split là ứng viên, các cặp đã xem có false positive. Không dùng dHash làm bằng chứng lỗi hoặc bằng chứng không leakage. Không trùng tên với frozenV1; V1 không đổi.

Gói local: `C:\My Future\VietSight-data-qa\data\versions\team_dashcam_yolo_intake001_REVIEW.zip`, **97.049.751 byte**, CRC qua, SHA256 `fc0ac30c5ef477650c14a270ee65eeadd1f5c12b5e2f28edec26ceeb74f2c312`. Handoff: [configs/handoff/v2.json](../../configs/handoff/v2.json), đúng schema mẫu Claude; status BLOCKED. URI Drive trong JSON là **vị trí upload dự kiến**, chưa upload Drive/Kaggle và không cấp phép train. Fingerprint chính thức `3c13dcc7ca1510f725b3b3a0188c982fee717d03287a5ebfe8948d6b2e9273e6`; `vn_val_sha256=5a8507f364678c9559ab21c7e9b6b0ec09b5362062ea09b052d09abbf6830a2a` đo đúng công thức mẫu, chưa coi split là frozen đã duyệt. Có 52 dòng trong `TARGETED_EXCEPTIONS.csv`, tách khỏi model candidates; 1.407 JPEG giữ nguyên byte so với ZIP nguồn.

Kiểm bổ sung e25+COCO/conf0,25 đã xét đủ **562** ứng viên: **418** ảnh có **1.100** đề xuất, **144** không có đề xuất, vẫn không phải chứng nhận hoàn chỉnh. Rider thiếu ở image621 nay được cả hai backend đề xuất, lưu riêng. Outer PowerShell trả exit1 dù CLI in báo cáo cuối và đủ418 bộ ảnh/nhãn/proposal được kiểm chứng; không ghi là CLI exit0. Lượt CPU trước dừng do thread contention đánh dấu ABORTED, không dùng làm kết quả. QA bổ sung nằm ở sidecar của ZIP, không thay GT/fingerprint hoặc ghi đè ZIP đã chốt.

16 test V2 +8 test QA qua; converter chạy đủ lô thật. Builder trên source chưa duyệt từ chối đúng gate, không tạo bản trainable giả. Không chạy train/export notebook hoặc thay validator.

Còn chặn phát hành: ngoại lệ coverage/overlap cần xử lý và bằng chứng quyền footage. Đóng góp nhãn của thành viên đã xác nhận; không tiếp tục coi việc chưa biết tên công cụ AI là lý do bắt gán lại toàn bộ. Nhãn native đủ tin cậy có thể được chấp nhận qua source QA + tự động + lấy mẫu, không cần điền reviewer giả. Mọi cờ geometry/license thực tế vẫn chặn.

Evidence máy đọc: [TEAM_DASHCAM_EVIDENCE.json](TEAM_DASHCAM_EVIDENCE.json). Không training, GPU, SAM3, thêm nguồn hoặc đổi dependencies. Pipeline/notebook V1 và validator chính thức giữ nguyên.
