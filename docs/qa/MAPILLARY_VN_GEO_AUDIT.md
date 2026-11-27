# Mapillary Việt Nam theo tọa độ và audit nguồn mask bổ sung

Ngày kiểm: **27/11/2026, Asia/Saigon**. Không chạy model, training hoặc GPU trong đợt kiểm này. Không sửa V1, notebook/validator, ZIP CVAT đã phát hoặc kết quả Claude. Dữ liệu nằm trong worktree riêng.

## Kết quả chính

**10 ID nằm trong polygon Việt Nam; 10/10 khớp RGB + polygon + instance PNG + semantic PNG gốc trên Kaggle.** Có **8 training gốc + 2 validation gốc**. `JARwgsAxhLIpv5YnMJgZ_A` thuộc frozen validation V1, đã loại khỏi pool chuyển đổi V2. Còn **9 ảnh ứng viên**, chưa phải một tập train/val READY đủ 11 class. Tất cả 10 ảnh đều có bối cảnh đường phố Việt Nam khi xem overview; đây là AI visual inspection, không phải nhóm đã human QA.

Đã chuyển **9 ảnh từ mask pixel instance chính thức**, không pre-label lại. Có 201 polygon BTC draft sau ghép rider/serialize. Chưa phát hành trainable vì còn các ca association/crowd/topology thực tế cần xử lý và không có Bus sau khi loại ảnh V1 val; không phải vì bắt mọi ảnh nguồn phải được nhóm vẽ/duyệt lại.

## CSV và địa lý

- [Zenodo, DOI 10.5281/zenodo.11459554](https://zenodo.org/records/11459554), creator Arizona State University. Tải **861.604 byte**, MD5 `e6cf7f15f8e566a5b0afd2690af6a16e`, khớp checksum công bố. Schema `index, Mapillary_ID, Latitude, Longitude`; ID giữ nguyên chuỗi, gồm dấu `_`/`-`.
- **20.000 dòng / 20.000 ID duy nhất**; **11.300 tọa độ hợp lệ**, **8.700 dòng thiếu/không có tọa độ hợp lệ**. Không suy ra các ID thiếu tọa độ đều ở nước ngoài; 10 là số xác minh được từ CSV này, không phải tổng tuyệt đối ảnh Việt Nam trong Vistas.
- Dùng [geoBoundaries gbOpen VNM ADM0](https://www.geoboundaries.org/api/current/gbOpen/VNM/ADM0/), boundary `VNM-ADM0-46766057`, năm đại diện 2016, CC BY 4.0; lưu geometry snapshot và hash. Point-in-polygon `contains`, không dùng bbox làm quyết định cuối. Bbox sơ tuyển 10, polygon giữ cả 10. Điểm trên biên không được tự nhận là trong nước.
- Cả 10 ở vùng TP.HCM; điểm gần biên geometry nhất vẫn cách khoảng **42,9 km** theo ước lượng haversine tới điểm gần nhất trên biên lon/lat. Không có ca sát biên cần quyết định quốc gia. Tọa độ do bộ metadata công bố, chưa phải đo GPS độc lập.

## Mask nguồn và số lượng

Kaggle input được xác nhận từ metadata notebook prep: `kaggleprollc/mapillary-vistas-image-data-collection`. Chỉ tải file của 10 ID, README/config và PNG nhãn, không tải toàn Mapillary.

Kiểm **10/10** instance PNG `uint16`: `instance_value // 256` **khớp từng pixel** semantic label PNG; lower byte phân biệt instance. Mapping class lấy đúng thứ tự `config_v2.0.json`. Dùng ID nguyên từ PNG và RLE, không nối nhầm các object theo tên class. Giữ mask pixel gốc; ảnh/instance map xử lý ở cạnh dài 1024 với resize nearest-neighbor cho ID map. Số đo serialize dưới đây ở độ phân giải xử lý, không gọi là IoU tại ảnh gốc 4K.

| Class BTC | Polygon nguồn, cả 10 ảnh, trước union/occlusion | Draft từ pixel GT, 9 ảnh ngoài V1 val |
|---|---:|---:|
| Car | 13 | 12 |
| Bus | 1 | 0 |
| Truck | 1 | 1 |
| Other vehicle | 4 | 4 |
| Motorcycle | 95 | 33 |
| Motorcyclist | 107 rider polygons, xe tách riêng | 48 |
| Bicycle | 5 | 3 |
| Cyclist | 1 rider polygon, xe tách riêng | 1 |
| Pedestrian | 42 person polygons | 66 |
| Sign | 29 mặt trước | 27 |
| Traffic light | 6 | 6 |

Hai cột khác số ảnh và cách đếm, **không dùng để tính tỷ lệ mất nhãn**. Pedestrian draft có cả rider chưa ghép được theo policy hiện tại; Motorcycle draft cũng chưa đồng nghĩa mọi xe đã xác nhận không người. Giữ nguyên lớp nguồn và source instance value trong COCO để sửa liên kết, không nhận output draft là GT đã đúng BTC.

QA toàn bộ 9 ảnh/pixel masks phát hiện cờ: **19 association mơ hồ, 9 rider chưa tìm được xe, 6 crowd/group, 5 Truck/Other vehicle cần đối chiếu ngữ nghĩa, 4 round-trip dưới 0,995, 2 mask chưa biểu diễn được bằng một polygon trong gate 400 điểm, 7 overlap sau serialize**. Đây là cờ cần phân giải, không phải số lỗi source đã được human xác nhận. RLE gốc của mask không biểu diễn được vẫn lưu, không loại vật đó khỏi GT để qua kiểm tra.

80 cờ cạnh ngắn 1,25% và 80 cờ diện tích dưới 64 ở thang 640 được giữ **hai phép kiểm riêng**. Việc hai tổng bằng nhau trong lô này không làm hai quy tắc tương đương. Các cờ `coverage_review_required`/`license_evidence_required` của converter cũ là trạng thái hồ sơ, không tính thành lỗi mask hoặc bằng chứng bắt buộc human QA toàn bộ. Một ảnh không có cờ association/topology vẫn chưa đủ tạo tập train/val độc lập.

Gói local: `C:\My Future\VietSight-data-qa\data\versions\mapillary_vn_native_pixels_intake001_REVIEW.zip`. Bao gồm RGB/GT nguồn của **9 ảnh ngoài V1 val**, config/README, tọa độ, COCO RLE, queue converter và danh sách instance cần phân giải. Không có status READY hoặc data.yaml để train thẳng. Native 10 ảnh/reference V1 giữ riêng dưới `data/raw/mapillary_vn_geographic/`.

## Giấy phép — không gộp metadata với ảnh

CSV Zenodo **CC BY 4.0** chỉ cấp quyền metadata. Native Vistas README ghi **Mapillary Vistas Dataset Research Use License**, cấm sử dụng thương mại theo định nghĩa của điều khoản gốc; trang chính thức hiện yêu cầu đăng nhập. [Mirror Dataset Ninja](https://github.com/dataset-ninja/mapillary-vistas-dataset/blob/main/LICENSE.md) công bố **CC BY-NC-SA 4.0** cho dataset. Cần ghi attribution, giới hạn nghiên cứu/phi thương mại và điều khoản chia sẻ tương ứng; không tự gọi dataset ảnh là CC BY 4.0 hay MIT. License MIT của repo evaluation không cấp quyền ảnh/annotation. Endpoint PDF điều khoản trả HTML login, không nhận đó là PDF đã đọc được. Chưa xác nhận thêm điều khoản riêng của cuộc thi; không tự công khai lại bộ ảnh hoặc đưa link Kaggle public.

## Bốn nguồn Roboflow: metadata, annotation đầy đủ và ảnh lấy mẫu

Đọc API bằng key trong User environment, không log key/signed link. Chỉ đọc annotation và tải 8–12 ảnh/nguồn truy cập được; không tải toàn bộ archive. Class counts bên dưới là kiểm đếm annotation export thực, khác catalog và không đại diện bộ ảnh đã tải đủ.

| Nguồn | Kiểm thực tế | Kết luận hiện tại |
|---|---|---|
| [Zibot vehicle-segmentation v9](https://universe.roboflow.com/zibot/vehicle-segmentation-7picx/dataset/9) | MIT do publisher khai; **1.415 image records / 28.385 polygon annotations**, 8 RGB mẫu, 4,21 MB mạng. Bus 673, Car 7.795, Truck 3.547, Motorcycle 15.822, Tricycle 548. Geometry từng ring toàn bộ không thấy malformed/nonfinite/invalid/out-of-bounds theo phép kiểm này; 12 hình chữ nhật trục song song, không coi mọi tứ giác là sai | Mẫu có xe buýt biển/chữ châu Âu cùng ảnh camera Việt Nam → **mixed geography**, không nhập cả bộ. Chỉ whitelist ảnh/clip xác minh Việt Nam. Thiếu nhãn person/Sign/TL/Bicycle… của cảnh; không coi 5 class phương tiện là coverage đủ BTC. Tricycle cần phân biệt ba gác với loại khác trước map Other vehicle. Loại B/D theo ảnh/nguồn con, chưa đủ nhận nguyên trạng |
| [sidewalk-violation-detection-luhlt](https://universe.roboflow.com/has-workspace-0wbbm/sidewalk-violation-detection-luhlt) | API project HTTP 200, CC BY 4.0, catalog 5.002; **không có version export công khai trong response**; thử v1 export HTTP 404 | Chưa có annotation/RGB để đo; không bịa instance thực tế hay chất lượng. BLOCKED access/version, không tải bằng cách đoán hàng loạt phiên bản |
| [sidewalk-violation-detection-2 v7](https://universe.roboflow.com/has-workspace-0wbbm/sidewalk-violation-detection-2/dataset/7) | CC BY 4.0; **720 records / 3.906 segmentation**; 12 RGB mẫu, 1,45 MB mạng. **2.234 mask là hình chữ nhật trục song song**, gồm **1.153/1.820 Motorcycle** và **126/141 Car**. Mẫu annotation Motorcycle ID 13 phủ nền quanh xe. 13/19 filename clip groups xuất hiện trong nhiều split | Có bằng chứng mask bbox hình chữ nhật thật, không nhận nguyên trạng. `sign_board` là lớp bảng/biển hiệu cần kiểm, **không tự map Sign**; `street_cart` không tự map Other vehicle. Thiếu coverage các class khác. Loại B/C cho phần xe/bổ sung nhãn; không giữ split gốc vì trùng clip |
| [PBL5-segmentation v9](https://universe.roboflow.com/khanhs-workspace-tcnac/pbl5-segmentation/dataset/9) | CC BY 4.0; **1.517 records / 3.101 polygon**, tất cả `side-walk`; 12 RGB mẫu, 1,37 MB mạng | **0 annotation thuộc 11 class BTC**. Không dùng vỉa hè làm lớp BTC. Một số mẫu là screenshot street-view, quyền ảnh nền cũng cần đối chiếu. Loại D cho nhu cầu mask BTC sẵn; chỉ có thể là nguồn ảnh để gán thêm, không ưu tiên khi nhóm quá tải |

## Đã nhận kết quả Claude

Đã sao chép read-only `C:\My Future\VietSight\runs\proposals\e25_coco_intake001_not_round001\` sang worktree, lưu checksum. **148 ảnh có đề xuất**, 302 mask mới: Sign 170, Traffic light 18, Motorcycle 97, Motorcyclist 10, Other vehicle 7. README nói đã xét 277 ảnh ngoài round001; output chỉ có 148 ảnh có proposal, không tự đếm thành 277 ảnh đã bàn giao.

Chạy QA cấu trúc/geometry trên đủ 148 output: 132 ảnh có cờ cần xem; không phải 132 ảnh lỗi đã xác nhận. Giữ `proposals_flagged.jsonl`, `RISK_REVIEW.tsv` và crop evidence Claude. Không xóa đồng loạt theo WATERMARK?/RIDDEN?; chúng là heuristic và có cả ca đúng/sai. Không chạy inference trùng hoặc ghi đè 5 task đang sửa. Những mask này vẫn pseudo, không dùng bổ sung như GT để che lỗi nguồn.

## Quyết định tiếp theo

Ưu tiên giải quyết **instance/rider/topology cụ thể của 9 ảnh Mapillary**, tận dụng GT nguồn và sampling, thay vì vẽ lại 100 ảnh hoặc pre-label cả Mapillary. Chưa có căn cứ phát hành một V2 READY đủ điều kiện từ các nguồn mới kiểm: Mapillary xác minh được quá ít ảnh và còn chuyển đổi cần sửa; các Roboflow còn mixed geography, bbox masks, thiếu class hoặc không export được. 8.700 ID thiếu tọa độ không được tự gán Việt Nam để tăng số lượng.

Phân biệt trong hồ sơ: nhãn nguồn do publisher cung cấp; QA tự động toàn bộ phạm vi annotation tải được; AI visual sample; human QA của nhóm chỉ khi thực sự có kết quả. Không giả reviewer và không dùng đòi hỏi human toàn bộ làm lý do chặn mặc định một nguồn đã đạt QA.
