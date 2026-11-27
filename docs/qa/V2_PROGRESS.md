# V2 — chỉ ảnh giao thông đã xác minh chụp tại Việt Nam

Cập nhật 27/11/2026, GMT+7, theo yêu cầu mới. Làm trên `codex/data-qa`, worktree `VietSight-data-qa`. **Không nhập train V1 vào V2.** Mapillary chỉ được nhận từng ảnh có bằng chứng địa điểm Việt Nam; 80 ảnh Mapillary QA cũ chỉ là reference chuyển đổi, không phải train V2. V1 e50/notebook/Kaggle/validator/dependencies không thay đổi.

## Trạng thái dữ liệu thật

- **0 ảnh train V2 mới đã duyệt; 0 ảnh validation Việt Nam có mask hoàn chỉnh. Chưa có dataset V2 có thể train/chấm.** Không tạo nhãn giả hoặc duyệt hàng loạt để đổi trạng thái này.
- Đã tải **12 ảnh camera TP.HCM + nhãn bbox**, 1.197.068 byte, làm **ứng viên validation dự phòng**. Đã xem contact sheet đủ 12 ảnh; caption thấy các địa điểm Phạm Văn Đồng, Nam Kỳ Khởi Nghĩa–Lý Chính Thắng, Võ Văn Ngân–Đặng Văn Bi, Đinh Bộ Lĩnh… phù hợp mô tả TP.HCM của publisher. Chưa phải kiểm chéo độc lập hoặc đo đầy đủ nhãn.
- 12 ảnh có **170 bbox, 0 mask**. Converter V2 đã chạy thật: **0 accepted / 12 review**. Queue local: `data/processed/vn_hcm_validation_candidates_v2/review/`; ảnh/annotation không commit. Toàn nguồn này được dự kiến dành riêng validation, tránh cùng camera xuất hiện trong train; split chính thức chưa khóa vì chưa có mask/quyền đã xác nhận.
- Chưa lấy được annotation các nguồn Roboflow instance segmentation: API trả **401 yêu cầu API key**, môi trường hiện chưa có key Roboflow. Có thể nhận export COCO masks của nhóm thay cho API; không cần đổi dependencies.
- Reference V1 public **12,90**, e25 local val **22,97**, Sign local **29,13** / public **6,05**. Chi tiết/giới hạn tại [SIGN_FORENSICS.md](SIGN_FORENSICS.md); [public JSON](evidence/V1_PUBLIC_OFFICIAL.json). Chưa có kết quả V2.

## Nguồn ưu tiên và kết quả kiểm tra trực tiếp

Số catalog/API listing khác số đã tải và kiểm. Không nguồn nào dưới đây đã được cấp A.

| Nguồn chính thức | Quyền/địa điểm/khả năng tải | Mask, mapping và quyết định |
|---|---|---|
| [SideWalk — Khaipixel](https://universe.roboflow.com/khaipixel/sidewalk-srgyh) | Publisher: instance segmentation, CC BY 4.0, 279 ảnh catalog; một preview đã xem là phố/xe máy nhưng chưa chứng minh nơi chụp. Export cần key | **B kỹ thuật / chưa nhận vì chưa xác minh Việt Nam.** Ưu tiên lô nhỏ Motorcycle/Bicycle nếu có chứng cứ. Car→Car; bicycle/motorbike cần xác định có rider; person cần kiểm người dắt/ngồi sau. `object` không tự map Other vehicle. Thiếu Bus/Truck/Other vehicle/Sign/TL và coverage cần bổ sung. Chưa đo mask/instance |
| [Traffic sign — hoangthanh](https://universe.roboflow.com/hoangthanh/traffic-sign-hemee) | Instance segmentation, CC BY 4.0, 150 ảnh catalog; không mô tả nguồn/nơi chụp. Export chưa lấy được | **B kỹ thuật / chưa nhận vì chưa xác minh Việt Nam.** 7 class biển→Sign nếu mặt biển thật; kiểm loại quảng cáo/cột và đủ 10 lớp còn lại. Chưa đo mask/instance |
| [Vietnam-Traffic-5Classes — Jose](https://universe.roboflow.com/jose-89vn1/vietnam-traffic-5classes-xthnt) | Publisher: instance segmentation, CC BY 4.0, 13.792 ảnh catalog, không mô tả nguồn. **3/4 model preview xem được là thể thao**, ảnh còn lại là đường phố chưa rõ nước | **D đối với intake VN-only hiện tại**: không dùng tên dự án làm chứng cứ Việt Nam. Model preview chưa chứng minh tỷ lệ nhiễm trong archive; cần xem ảnh/annotation thật trước khi khẳng định. Config prototype đã đánh dấu blocked. Không bulk download |
| [Transportation in Ho Chi Minh City — khoakait](https://www.kaggle.com/datasets/khoakait/transportation-in-ho-chi-minh-city) | Publisher xác nhận camera TP.HCM; API listing thực tế 1.646 ảnh + 1.646 txt + 3 metadata. Kaggle license `other`, export `data.yaml` dẫn publisher Roboflow CC BY 4.0 nhưng trang gốc không truy cập được để đối chiếu | **C về nhãn / D về quyền chưa đối chiếu xong.** 12 ảnh đã tải có 170 bbox, tất cả hàng 5 tokens, không có mask. bus→Bus (kiểm 9–15 chỗ→Car), car→Car, truck cần kiểm pickup/đầu kéo/rơ-moóc, motorbike/person cần gộp/tách đúng rider. Thiếu Bicycle/Cyclist/Other vehicle/Sign/TL. Dự phòng validation, cần gán mask đủ 11 lớp |
| [Street Traffic Signs in Vietnam COCO — lvnduy](https://www.kaggle.com/datasets/lvnduy/street-traffic-signs-in-vietnam-coco) | Listing thực tế **10.933 ảnh** / 10.940 files; không tải ảnh. Kaggle license `unknown`; COCO metadata ghi Public Domain — chưa xác nhận quyền ảnh gốc | **C về nhãn / D về quyền.** Đã tải annotation test: **613 ảnh metadata, 967 bbox, 0 segmentation không rỗng**. Không được gọi là instance masks chỉ vì định dạng COCO. Tên clip/README cho thấy nhiều frame và 5 augmentation/source: phải group theo video gốc, không giữ split ngẫu nhiên. Chỉ Sign, thiếu 10 lớp còn lại |
| Lô mask nhóm tự thu tại VN | Chưa nhận file/manifest/quyền | Ưu tiên nếu có sẵn mask: **Motorcycle, Other vehicle, Sign, Bicycle**, kiểm đủ mọi lớp cùng ảnh. Có thể đi thẳng converter hiện tại; không cần thêm UI |

Các nguồn quốc tế BDD/COCO/IDD hoặc Mapillary chưa xác minh Việt Nam **không phải ứng viên train V2**, dù audit kỹ thuật trước đây có thể đạt B. [Evidence nguồn](evidence/VIETNAM_ONLY_INTAKE.json) giữ hash file probe và URL preview, không chứa key hoặc signed URL.

## Quy trình dùng lại công cụ hiện tại

1. Nhận **một lô nhỏ có mask sẵn**, ưu tiên 4 lớp yếu; kiểm provenance/quyền và nơi chụp từng ảnh. Không tăng số ảnh đại trà. Lô đầu có thể 60–100 ảnh, đây là mục tiêu, không phải số đã có.
2. Converter `scripts/v2/convert.py` giữ pixel-edge topology/round-trip, rider/occlusion/overlap gates đã phát triển. Nay yêu cầu `country=VN`, xác minh nơi chụp/quyền và canonical camera/video group; named reviewer + coverage đủ 11 class vẫn bắt buộc. Record cũ `v2_policy=1` phải review/reconvert, không tự nâng cấp.
3. Dùng `scripts/qa/label_audit.py`, overlay/review queue sẵn có để sửa mask/coverage. Không đổi bbox thành polygon chữ nhật. Không tạo reviewer giả để bypass. Ngưỡng cạnh ngắn 1,25% và area 64 px² giữ riêng.
4. Builder `scripts/v2/build.py` **không lấy train V1**. Chọn holdout VN khoảng 20% theo canonical camera/video/chuyến chụp (seed 55), dùng group chung giữa các export; train lấy có giới hạn và ưu tiên 4 lớp. Camera mới phát hiện thuộc holdout không được vào train. Dedup exact/near giữa split; lưu `frozen_vn.txt` riêng.
5. Khi dùng lại VN val: truyền frozen list **và dataset reference**, kiểm bytes ảnh+nhãn. V1 val giữ riêng để so sánh model; `--baseline` nay chỉ là reference val V1 để kiểm overlap, không được nhập vào V2. Thiếu reference này thì release bị blocked, không tuyên bố đã kiểm trùng với V1.
6. Khi đã qua gate dữ liệu, bàn giao dataset cho Lead chạy cùng model/evaluator. Không chạy training hoặc thay e50 trong phiên QA.

Ví dụ decisions (chỉ điền sau kiểm tra thật; các đường dẫn là ví dụ):

```json
{
  "frame001.jpg": {
    "country": "VN",
    "location_verified": true,
    "location_evidence": "manifest:vn_trip_01/frame001:GPS_and_capture_log",
    "license_verified": true,
    "license_evidence": "manifest:vn_trip_01/consent_and_license",
    "group": "vn_trip_01_camera_front",
    "reviewer": "TenNguoiDuyet",
    "coverage_reviewed": true,
    "instances": {"20": {"vehicle_id": "21"}}
  }
}
```

Source config phải có URL và giấy phép thực tế của lô; `btc_reviewed_coco.yaml` là template mapping, không tự cấp quyền hoặc giả định nguồn.

```powershell
$v2Python = 'C:\My Future\VietSight\.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $v2Python scripts\v2\convert.py --format coco --config 'D:\VN\source_verified.yaml' --ann 'D:\VN\instances.json' --images 'D:\VN\images' --source vn_batch01_v2 --out data\processed --decisions 'D:\VN\review_decisions.json'
# --baseline: chỉ đọc val V1, không cần và không nhập train V1.
& $v2Python scripts\v2\build.py --baseline 'D:\QA\vietsight_v1_reference' --sources data\processed\vn_batch01_v2 --out work\vn_v2_batch01 --max-additions 200
# Phiên tiếp theo giữ nguyên cả bytes ảnh/nhãn validation VN:
& $v2Python scripts\v2\build.py --baseline 'D:\QA\vietsight_v1_reference' --sources data\processed\vn_batch01_v2 --vn-val-list work\vn_v2_batch01\frozen_vn.txt --vn-val-reference work\vn_v2_batch01\versions\vietsight_vietnam_v2 --out work\vn_v2_batch02
```

Cấu hình cũ `configs/versions/v2_local_vision.yaml` vẫn có Mapillary/V1 và val V1: **không dùng cho yêu cầu VN-only mới**. Giữ file Claude đang quản lý, không sửa vào luồng đang chạy. PR pipeline V2 tiếp tục **chưa merge**. Việc còn cần để có dataset train sớm là export mask VN có provenance (hoặc quyền tải nguồn nhỏ phù hợp), bổ sung coverage, duyệt và khóa VN val; không cần phát triển thêm UI.
