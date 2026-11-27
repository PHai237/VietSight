# V2 — bản sửa tách biệt và ưu tiên theo public V1

Ngày 27/11/2026, GMT+7. Source baseline khi bắt đầu: `origin/main=ec0f821`. Làm tại worktree `VietSight-data-qa`, branch `codex/data-qa`. Không thay notebook/dataset V1, frozen validation, validator, dependencies; không gọi training/inference/GPU. Code V2 là các CLI mới dưới `scripts/v2/`, chưa tích hợp notebook V1.

## Public V1 đã xác nhận bởi người dùng

Các giá trị dưới giữ nguyên thang public **0..100**, không phải local val AP.

| Class | Public V1 | Instance train V1 | Ưu tiên dữ liệu V2 |
|---|---:|---:|---|
| Car | 29,67 | 7.427 | giữ coverage, rà pickup/9–15 chỗ |
| Bus | 12,60 | 342 | giữ coverage, không lẫn xe 9–15 chỗ |
| Truck | 20,41 | 463 | giữ coverage, tách đầu kéo/rơ-moóc |
| Other vehicle | **2,57** | **223** | **2 — ba gác/xích lô/xe công trình/rơ-moóc có mask đúng** |
| Motorcycle | **1,95** | **358** | **1 — xe máy không người, gồm xe đang được dắt nhưng tách Pedestrian** |
| Motorcyclist | 15,02 | 263 | giữ đủ passenger, không gán xe có người thành Motorcycle |
| Bicycle | **6,53** | **419** | **4 — xe đạp không người, kiểm người dắt và xe bị che** |
| Cyclist | 17,05 | 381 | giữ đủ rider(s) + xe một instance |
| Pedestrian | 12,54 | 2.795 | giữ coverage các ảnh thêm |
| Sign | **6,05** | **3.502** | **3 — sửa fidelity/scale/domain trước khi tăng số ảnh** |
| Traffic light | 17,56 | 2.463 | giữ coverage, mỗi hộp một instance |
| **Tổng public** | **12,90** | — | không tính lại từ AP làm tròn |

[Public JSON](evidence/V1_PUBLIC_OFFICIAL.json). Counts lấy từ stats V1 và đã đối soát với 1.469 nhãn train tải trực tiếp Kaggle. Lần fetch cuối nhận `run_report.json` và RESULTS mới của Claude, gắn public này với **v1_e25** (25 epoch). ONNX SHA đã có trong JSON; submission ID và SHA file `.pt` chưa có.

| Class ưu tiên | Local val e25 ×100 | Public V1 | Cách đọc |
|---|---:|---:|---|
| Motorcycle | 8,48 | 1,95 | Yếu trên cả hai, ưu tiên xe không người/association/domain |
| Other vehicle | 7,50 | 2,57 | Ít ví dụ, ontology rộng; ưu tiên mask đúng các loại xe BTC |
| Sign | 29,13 | 6,05 | Gap khoảng 23,08 điểm giữa hai tập; kiểm domain/scale/fidelity, không tăng ảnh đại trà |
| Bicycle | 6,08 | 6,53 | Yếu trên cả hai; không chỉ giải thích bằng domain shift |

Local lấy từ `runs/v1_colab_e25/run_report.json`, chấm 262 ảnh val, không đo lại. Dataset fingerprint ở report đã qua bước prepare notebook là `fb6eaedd…`; fingerprint built dataset là `5612ab42…`. Hai fingerprint có phạm vi/path khác nhau, không tự kết luận dữ liệu đổi chỉ vì hai chuỗi khác nhau.

Không chọn thêm ảnh chỉ vì có nhiều xe máy: Motorcycle là **không người**. Không lấy mọi bicycle nguồn làm Bicycle nếu xe đang có người. Không đổi mọi truck thành Car hoặc mọi Other vehicle nguồn thành ba gác. Xem [phân tích Sign](SIGN_FORENSICS.md) và [mapping review](CLASS_MAPPING_REVIEW.md).

## Bản sửa đã triển khai

- `scripts/v2/masks.py`: truy vết cạnh pixel, giữ lỗ/mảnh nhỏ, kiểm serialized polygon bằng helper BTC và pycocotools, gate IoU≥0,995; không largest-part fallback. Mask không đạt được biểu diễn giữ pending review, không nhận `ok`.
- `scripts/v2/policy.py`: known rider đơn ứng viên được union; person có thể là người dắt không được tự ghép bằng score mạnh. Tie/missing vehicle cần review; cho reviewer gán `vehicle_id` hoặc `walking_bike`. Metadata seats 9–15 ở Car/Bus đưa về Car. Bus/Truck/Other vehicle chưa được duyệt ngữ nghĩa cần review; tổ hợp đầu kéo/rơ-moóc phải tách bằng mask thật.
- `scripts/v2/convert.py`: đọc mask COCO bằng union của pycocotools, đọc Supervisely/Mapillary gốc; giữ named reviewer + xác nhận đủ 11 class, kể cả ảnh không có nhãn thêm. Kiểm raw/decoded overlap; ngưỡng cạnh và area báo riêng. Không ghi đè output; tên source mới kết thúc `_v2`.
- `scripts/v2/build.py`: chỉ lấy additions `status=ok`, có V2 policy/reviewer/coverage, chứa 4 class ưu tiên; round-robin và cap 200 ảnh/lô mặc định. Dùng builder có sẵn trên **bản sao riêng**, kiểm lại exact val image+label identity với V1 và frozen list. Không hardlink output tới input đang sửa. Polygon train lỗi làm release gate blocked; cảnh báo val cũ giữ nguyên và ghi riêng. Không tự train.
- `scripts/v2/audit_sample.py`, `compare_sign_reference.py`: đo và xuất evidence thay vì suy từ số lượng. `scripts/qa/fetch_v1_sample.py`: lấy mẫu output Kaggle có giới hạn, resume/retry, không lưu signed URLs.

## Dữ liệu Việt Nam có mask sẵn: shortlist có điều kiện

Ưu tiên mask có sẵn; không bulk download và không phê duyệt nguồn chỉ từ tên dataset. Số ảnh website công bố không phải số đã kiểm đếm local.

| Nguồn publisher | Điều đã kiểm chứng từ trang nguồn | Phù hợp ưu tiên | Điều kiện còn thiếu / phân loại |
|---|---|---|---|
| [Vietnam-Traffic-5Classes — Jose](https://universe.roboflow.com/jose-89vn1/vietnam-traffic-5classes-xthnt/dataset/1) | Publisher ghi Instance Segmentation, CC BY 4.0; class Car/Bus/Bicycle/Motorbike/Person; catalog 13.792 ảnh | Ứng viên trước cho Motorcycle/Bicycle, góc nhìn Việt Nam cần xem mẫu | **B**, chưa tải archive/mask để kiểm. Phải xác nhận provenance, loại mask Motorbike có rider hay không, bổ sung coverage Sign/TL/Truck/Other vehicle. Mapping prototype ở `configs/v2/vn_traffic_masks.yaml` không cấp quyền train |
| [Traffic sign — hoangthanh](https://universe.roboflow.com/hoangthanh/traffic-sign-hemee) | Publisher ghi Instance Segmentation, CC BY 4.0, 150 ảnh, 7 class biển | Có thể dùng để kiểm Sign sau khi xem ảnh | **B về mask / chưa xác nhận domain VN**: trang không có mô tả nguồn; không coi đây là nguồn VN đã duyệt. Chưa tải/gán số instance |
| Nhóm tự gán CVAT tại VN, xuất COCO masks | Chưa nhận lô cụ thể trong phiên này | Tốt nhất nếu có đủ xe máy/xe đạp không người, ba gác và Sign lớn | B khi có mask, C nếu chỉ ảnh/bbox; cần manifest/quyền/coverage và người kiểm chéo trước A |

Một candidate Roboflow được search gợi ý có ba gác/biển VN trả **404** khi mở URL dự án; giữ chưa xác minh, không cấu hình/tải hay đưa vào V2. Các bộ VN chỉ bbox không được coi là mask sẵn hoặc thay mask bằng hình chữ nhật. Không nhận toàn bộ 13.792 ảnh để bù số lượng; bắt đầu một lô nhỏ đã audit đủ class.

## Cách chạy tách biệt

Chạy trong worktree, dùng Python có sẵn và UTF-8. Output phải mới. Các đường dẫn `D:\...` là ví dụ input đã bàn giao, không phải dữ liệu đã tồn tại.

```powershell
$v2Python = 'C:\My Future\VietSight\.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'

# Dữ liệu có mask sẵn, remap theo config riêng, toàn bộ chưa duyệt sẽ vào review.
& $v2Python scripts\v2\convert.py --format coco --config configs\v2\vn_traffic_masks.yaml --ann 'D:\VN\instances.json' --images 'D:\VN\images' --source vn_batch01_v2 --out data\processed --decisions 'D:\VN\review_decisions.json'

# Export đã sửa tay theo 11 class BTC (mỗi rider đã gộp xe+người vào một mask).
& $v2Python scripts\v2\convert.py --format coco --config configs\v2\btc_reviewed_coco.yaml --ann 'D:\VN\reviewed.json' --images 'D:\VN\images' --source vn_corrected_v2 --out data\processed --decisions 'D:\VN\review_decisions.json'

# Build chỉ khi có full V1 immutable để giữ 262 ảnh val; sample 80 không thay full V1.
& $v2Python scripts\v2\build.py --baseline 'D:\QA\vietsight_v1_immutable' --frozen configs\val_frozen.txt --sources data\processed\vn_corrected_v2 --out work\v2_batch01 --max-additions 200

& $v2Python -m unittest discover -s scripts\v2 -p test_v2.py -v
```

Ví dụ reviewer record (key filename theo source; instance ID là COCO ID/Mapillary ID):

```json
{
  "frame001.jpg": {
    "coverage_reviewed": true,
    "reviewer": "TenNguoiDuyet",
    "group": "vn_trip_01",
    "instances": {
      "10": {"walking_bike": true},
      "20": {"vehicle_id": "21"},
      "30": {"target": "Car", "seats": 12, "semantic_reviewed": true}
    }
  }
}
```

Record là sự xác nhận thật của người duyệt, không được điền sẵn để bypass queue. Trường `target` không tự sửa segmentation hoặc tách đầu kéo/rơ-moóc; muốn tách/sửa mask phải cung cấp annotation đã sửa. Những object nhỏ/crowd/overlap chưa được xử lý vẫn làm ảnh pending dù có `coverage_reviewed=true`.

## Trạng thái thực tế và điều chưa xong

- Đã tải và kiểm 80 ảnh/1.469 nhãn train V1; 12 ảnh/42 Sign có đối chiếu source mask. Cả 42 V2 round-trip đạt IoU 1,0 trên reference tái dựng; chưa đo public V2.
- Đã chạy converter V2 trên 12 ảnh Mapillary reference thật; **0 accepted / 12 review**, vì chưa có named coverage review và còn object nhỏ/crowd/ontology cần xử lý. Đây là kết quả mong đợi của gate, không phải 12 ảnh đã sẵn sàng train.
- Test builder chạy fixture nhỏ, giữ val bytes và frozen text; chưa build dataset V2 thật. Local chỉ lấy 80 ảnh train theo phạm vi cho phép, không có full V1/val images; không tự mở rộng tải dataset.
- Chưa có lô mask Việt Nam mới đã được human QA để thêm train. Việc chọn nguồn, người duyệt và correction batch còn cần phối hợp nhóm; không bịa số ảnh mới/số nhãn sửa thật.
- PR có pipeline V2 nên **không tự merge**, theo quy tắc bàn giao trước. V1/Kaggle không bị cập nhật hoặc khởi động lại bởi phiên này.
