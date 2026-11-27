# VietSight — Dataset audit và intake V2

**Cập nhật lô thực tế:** [V2_DATA_HANDOFF.md](../V2_DATA_HANDOFF.md) ghi snapshot **377 ảnh ứng viên (309/68), 4.279 polygon draft, BLOCKED**, cùng nguồn đã tải/kiểm và gói CVAT. API Roboflow đã hoạt động; nguồn SideWalk có 279 ảnh/4.715 native masks và bốn video Việt Nam xác định được. Nguồn Jose được lấy mẫu ZIP thực tế và có ảnh quốc tế; nguồn Sign xuất segmentation nhưng spot-check phủ cả cột/nhiều biển. Không dùng kết luận catalog cũ thay kết quả kiểm dữ liệu mới.

**Yêu cầu V2 mới nhất: chỉ ảnh xác minh chụp tại Việt Nam.** Xem [V2_PROGRESS.md](V2_PROGRESS.md) và [VIETNAM_ONLY_INTAKE.json](evidence/VIETNAM_ONLY_INTAKE.json). Các nguồn quốc tế bên dưới là lịch sử audit/V1, không phải nguồn được phép train V2. Nguồn mang tên “Vietnam-Traffic-5Classes” vừa kiểm preview có 3/4 cảnh thể thao; chưa được xác nhận thuần Việt Nam và không được nhập mặc định.

Ngày xác minh: **27/11/2026, GMT+7**. Nguồn được tổng hợp từ manifest/config hiện có; chưa nhận thêm danh sách/link dataset khác từ thành viên. Bảng này là quy trình đánh giá và kết quả quan sát có phạm vi, không phê duyệt tự động cho train.

Cập nhật mapping tiếp theo: [CLASS_MAPPING_REVIEW.md](CLASS_MAPPING_REVIEW.md), gồm quy tắc người dùng xác nhận **xe 9–15 chỗ là Car**. Các số kiểm đếm bên dưới giữ nguyên snapshot đã đo; không phải số liệu dataset V1 đang chạy trên Kaggle.

Cập nhật sau public V1: đã lấy mẫu train V1 trực tiếp qua Kaggle API (1.469 nhãn, 80 ảnh; không lấy full Mapillary). [V2_PROGRESS.md](V2_PROGRESS.md) bổ sung shortlist nguồn Việt Nam có instance mask và ưu tiên Motorcycle/Other vehicle/Sign/Bicycle; nguồn mới chưa được cấp A hoặc tải hàng loạt. [SIGN_FORENSICS.md](SIGN_FORENSICS.md) ghi số đo mask Sign và giới hạn của kết luận.

## Quy ước phân loại

- **A**: có thể dùng sau kiểm tra và chuẩn hóa; cần qua label QA, coverage, provenance và split gate.
- **B**: cần remap/sửa/bổ sung mask rồi duyệt.
- **C**: chỉ ảnh/bbox, cần pre-label và người sửa đủ 11 class.
- **D**: không phù hợp hoặc chưa xác nhận được quyền sử dụng. D có thể được đánh giá lại khi có bằng chứng.

Quyền sử dụng và chất lượng là hai điều kiện riêng. Một nguồn có mask tốt vẫn giữ D nếu chưa xác nhận được điều khoản chính thức; sau khi xác nhận quyền có thể chuyển B/A. Các mức công sức thấp/vừa/cao là đánh giá kỹ thuật, **không phải giờ công đã đo**.

## Kiểm tra nguồn chính thức

| Nguồn | URL và giấy phép | Khả năng tải đã xác minh | Quyết định hiện tại |
|---|---|---|---|
| Mapillary Vistas v2.0 | [Trang chính thức](https://www.mapillary.com/dataset/vistas); local mirror LICENSE ghi CC BY-NC-SA 4.0, **chưa xác nhận điều khoản Vistas trên trang chính thức** | Trang trả `Not Logged In`; URL PDF giấy phép không truy cập được qua web tool. Local có 15 annotation peek, không có ảnh Mapillary | **D — quyền chưa xác nhận độc lập**; về kỹ thuật là B do topology/association cần QA. Không thay quyết định V1 của Claude; báo blocker này cho Lead |
| BDD100K 10K instance segmentation | [License trong repo chính thức](https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/license.rst): UC Regents, cho giáo dục/nghiên cứu/phi lợi nhuận kèm điều kiện thông báo; không nhầm với BSD của code | [Tài liệu tải chính thức](https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/download.rst) có endpoint; kiểm tra endpoint trả 502 qua web tool nên **chưa xác nhận full download từ endpoint**. Local mirror đang tải có ảnh/annotation đọc được | **B**, sau kiểm đủ mask và bổ sung Sign/TL; quyền dùng theo mục đích cuộc thi cần Lead đối chiếu điều khoản, không suy ra quyền công bố lại |
| COCO 2017 instances | [Terms chính thức](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/termsofuse.htm): annotation CC BY 4.0; ảnh theo quyền/Flickr Terms từng ảnh, không có một giấy phép ảnh chung | [Trang tải chính thức](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/master/dataset/download.htm) có ZIP và hỗ trợ lấy ảnh theo subset. Local chỉ có file `.part`; chưa có ảnh/annotation hoàn chỉnh để audit | **B có điều kiện quyền từng ảnh**; ảnh thiếu metadata/quyền chuyển D |
| India Driving Dataset (IDD) | [Trang chính thức](https://idd.insaan.iiit.ac.in/), [download/login](https://idd.insaan.iiit.ac.in/dataset/download/). Trang nói giấy phép hiển thị trước khi tải; chưa đăng nhập/chưa đọc được giấy phép | Download chuyển tới login; chưa tải và không có dữ liệu local | **D**, chờ tài khoản/terms chính thức và đúng variant instance segmentation |
| Ảnh Việt Nam của nhóm | Chưa có URL/lô/người chụp/quyền đồng ý được bàn giao; không mặc định mọi ảnh trong máy là ảnh tự chụp | Chưa có `vn_team` trong inventory local | **C** nếu có provenance/quyền của nhóm; **D** cho lô chưa xác nhận quyền |
| `sample_data.zip` BTC | README local nói 10 ảnh vẽ bằng mã, minh họa format; không có URL/giấy phép công bố được cung cấp | Có 10 ảnh + 10 label local, đọc được | **D cho dữ liệu train**: chỉ test pipeline; không gộp vào V1/V2 |
| `Data-VF.zip` | Manifest cũ ghi camera+LiDAR, không mask 2D, nguồn/quyền chưa rõ; không có archive trong project để kiểm chứng lại | Chưa có đường dẫn/link từ thành viên | **D**; không nhắc lại con số 2.100 như kết quả kiểm đếm phiên này |
| `VN-Traffic-Instance-Segmentation-YOLO11` được nhắc trong docs | Chỉ có tên, chưa có URL/release/license dataset cụ thể trong bàn giao | Chưa kiểm chứng được gói dữ liệu | **D**, yêu cầu URL + license dữ liệu, không dùng license code thay cho dataset |

Mapillary imagery nói chung và **Vistas training dataset** có thể có điều khoản khác nhau; không dùng một trang license ảnh Mapillary chung để đóng việc xác minh Vistas. Với COCO phải giữ cả bảng `licenses`, per-image license ID và URL gốc, không chỉ một số ID thiếu bảng tra cứu.

## Số liệu local đã đo

Inventory đọc **11:32:35–11:32:39 ngày 27/11/2026, GMT+7**. Download của Claude vẫn đang diễn ra; số file là quan sát theo thời điểm, không phải số hoàn tất. Script chỉ liệt kê tên file, đọc tối đa 60 annotation mỗi nguồn và hash file đã đọc; không tải gì. Evidence: [local_inventory.json](evidence/local_inventory.json).

| Folder local | Ảnh file thấy được | Annotation JSON ở ann/ | Phạm vi mẫu đã parse | Giới hạn |
|---|---:|---:|---|---|
| `bdd100k_10k` | 5.420 | 9.000 | 60 annotation có đường dẫn ảnh đi kèm, trải đều theo tên; 38 có objects, 22 rỗng | train 7.000 ann/3.420 paired paths; test 2.000 ann/2.000 paired paths. Chưa có val local ở snapshot. File tồn tại không đồng nghĩa decode thành công |
| `peek_bdd` | 0 | 40 | 40/40 ann rỗng, thuộc test | Không phải ảnh nền đã được xác nhận; không dùng làm nhãn âm |
| `peek_mapillary` | 0 | 15 | 15/15 validation ann có objects | Không thể đánh giá biên mask bằng ảnh thật |
| `coco2017` | 0 | 0 | Không có file hoàn chỉnh | Một `.part`, không tính là dataset |
| `sample_data` | 10 | 0 Supervisely ann | QA đọc 10 YOLO `.txt`, tổng 105 instance | Synthetic BTC, chỉ test pipeline |

Trong 60 annotation BDD đọc được: polygon records = **494**, rectangle records = **489**. Đây **không phải 983 instance**: mirror có bbox bên cạnh mask; polygon multipart hoặc record trùng cũng cần đối chiếu ID gốc. Breakdown polygon: Car 428, Bus 12, Truck 11, caravan 2, trailer 1, pedestrian 37, rider 1, bicycle 2; Motorcycle 0 trong mẫu này. Không ngoại suy tổng instance dataset.

Trong 12 ảnh BDD sao chép để chạy converter: output 121 instance = Car 100, Bus 4, Truck 5, Other vehicle 1, Pedestrian 11. Không có instance của Motorcycle/Motorcyclist/Bicycle/Cyclist/Sign/TL trong **output mẫu này**; không có nghĩa những lớp đó vắng trong toàn nguồn. Tất cả 12 được converter cho `ok`; 3 ảnh được xem trực tiếp đều thấy Traffic light chưa gán. Chưa sửa nhãn nào. Xem [review](INDEPENDENT_REVIEW.md).

Số ảnh catalog trong docs cũ không được ghi thành số đã đo: Mapillary 25k/BDD 10k/COCO 123.287/IDD 20k cần gắn đúng variant/release. Trang IDD chính thức hiện mô tả 10.000 ảnh, 34 class; con số này khác docs cũ, nên giữ **chưa kiểm đếm** thay vì ghép hai release. Full instance counts của tất cả nguồn: **chưa đo**.

## Mapping, coverage và công chuyển đổi

Thứ tự BTC: `0 Car, 1 Bus, 2 Truck, 3 Other vehicle, 4 Motorcycle, 5 Motorcyclist, 6 Bicycle, 7 Cyclist, 8 Pedestrian, 9 Sign, 10 Traffic light`. [Đoạn quy ước BTC người dùng cung cấp](BTC_GUIDELINE_EXCERPT.md) đã chốt: người dắt xe→Pedestrian + xe riêng; người ngồi sau gộp cùng rider; đầu kéo→Truck, rơ-moóc/container→Other vehicle, hai instance; ba gác/xích lô/xe công trình→Other vehicle; xe đẩy tay không gán. Pickup còn cần decision cụ thể theo loại xe/cây quyết định, không remap toàn bộ truck của nguồn.

| Nguồn | Annotation và mapping sang BTC | Lớp thiếu/không đầy đủ | Mask và độ phù hợp VN | Công sức đánh giá |
|---|---|---|---|---|
| Mapillary v2 mirror | Supervisely polygon/bitmap; car/bus/truck→0/1/2; other vehicle/caravan/trailer/wheeled slow→3 tạm; motorcycle+bộ rider→4/5; bicycle+bộ rider→6/7; person→8; sign front→9; traffic light subtypes→10 | Chưa thấy thiếu toàn class trong config; mặt sau/ambiguous sign bị ignore, còn phụ thuộc BTC; passenger association chưa xác nhận | Cảnh đường phố đa dạng, độ phù hợp VN cần mẫu VN/xe máy đông. Painter+bridge/hole có rủi ro R02/R03/R09; chưa xem ảnh thật local | Vừa/cao; ưu tiên QA source đã có, không tải lại trùng việc Claude |
| BDD10K mirror | Supervisely polygon + rectangle (bỏ rectangle trùng); car/bus/truck→0/1/2; caravan/trailer→3 tạm; rider ghép motorcycle/bicycle→5/7; xe không người→4/6; pedestrian→8 | Mirror instance không có Sign/TL; không kết luận BDD semantic segmentation cũng thiếu hai class này | Dashcam Mỹ; phù hợp góc nhìn nhưng khác mật độ xe máy VN. Đã thấy pickup Truck. `bottom` occlusion cần so mask gốc | Cao cho lô có Sign/TL; chỉ lấy lô nhỏ đã duyệt đủ. Không dùng test ann rỗng |
| COCO2017 | COCO polygon/RLE; car/bus/truck→0/1/2; motorcycle/bicycle + person→4/5/6/7, người không rider→8; stop sign→9; traffic light→10 | Other vehicle không được chú thích đầy đủ; Sign chỉ STOP; pickup truck và person cần review | Nhiều góc chụp không dashcam; dùng có chọn lọc cho lớp hiếm. Polygon union/crowd/topology phải kiểm tra | Cao; không phải ưu tiên trước hạn nếu chưa có subset và license ảnh |
| IDD | Phải nhận đúng variant, category table, instance IDs; đề xuất person/rider/motorcycle/bicycle/autorickshaw/vehicles remap sau khi có schema | Chưa biết coverage của gói được bàn giao; không áp mapping Cityscapes theo phỏng đoán | Cảnh giao thông Ấn Độ có thể gần VN về unstructured traffic; đây là suy luận về domain, chưa đo | Cao do quyền + tải + converter + QA chưa sẵn |
| Ảnh nhóm VN | JPEG/PNG và label nếu có; ảnh/bbox→prelabel nháp→CVAT sửa đủ 11 class; giữ group video/chuyến chụp | Mọi class chưa xác nhận cho đến khi người duyệt ảnh gốc | Ưu tiên thực tế cho V2: xe máy đông, người ngồi sau, đêm/mưa, pickup/rơ-moóc/biển nhỏ | Vừa/cao tùy lô; chưa ước lượng giờ khi chưa thấy dữ liệu |

Không nguồn nào hiện được cấp A độc lập cho train V2. Một lô đã chuẩn hóa + duyệt đầy đủ có thể được cấp A riêng, không cấp A cho cả tên dataset.

## Quy trình intake cho mỗi thành viên

1. Ghi source ID/release, URL chính thức và link mirror, người bàn giao, giờ nhận, license URL + bản điều khoản/ngày kiểm tra, quyền sử dụng ảnh/nhãn, checksum archive hoặc manifest. Không mở rộng tải khi chưa có quyền và phạm vi.
2. Nhận inventory từ thành viên: count file, ảnh decode được, annotation tồn tại, có/không mask, unique instance IDs. Bbox + polygon cùng instance chỉ tính một lần; test không nhãn không được ghi thành ảnh nền. File `.part`/hỏng tách riêng.
3. Chốt bảng mapping có category ID gốc và 11 class BTC; ghi missing/incomplete classes, decision log pickup/trailer/rider/passenger/walking-bike/pole. Dùng đoạn BTC đã nhận, bổ sung URL/revision khi có; sửa cách hiểu đầu kéo + container trong doc nhóm, không suy từ tên class nguồn.
4. Lấy mẫu tất định theo class và thêm mẫu rủi ro: ít nhất có rider nhiều người, xe đỗ/dắt xe, pickup/trailer, sign/cột, occlusion, đêm/mưa. Ghi rõ mẫu random hay chọn có chủ đích. Không có mẫu của một class thì ghi chưa kiểm class đó.
5. Kiểm ảnh gốc + overlay, đủ mọi vật thể của 11 class, visible mask, multipart/hole, topology round-trip. Cờ tự động chỉ là candidate; cần người kết luận. Người gán và người kiểm chéo ghi riêng.
6. Chạy QA CLI trong worktree, giữ JSON/CSV/hash/ảnh bằng chứng local. Đối chiếu dữ liệu xuất CVAT với input bằng tên + hash + class order; chưa coi CVAT roundtrip đã hoạt động khi chưa kiểm dịch vụ thật.
7. Gán A/B/C/D cho **lô**, kèm lý do, người duyệt, phần việc còn lại; lô B/C chưa được train trước khi hoàn tất sửa/coverage. Lô D không đưa vào V2.
8. Trước merge dữ liệu, kiểm duplicate/sequence xuyên nguồn và frozen val byte identity; chuyển cho Claude/Lead lô A + manifest, không tự sửa config V2/build pipeline chung.

## Phiếu một nguồn/lô mới (sao chép khi nhận)

| Trường | Giá trị |
|---|---|
| Source ID, release, batch, người bàn giao | chưa cung cấp |
| Official URL / mirror / download check / timestamp | chưa xác minh |
| License ảnh / nhãn / URL terms / bằng chứng | chưa xác minh |
| Quyền dùng cho mục đích cuộc thi / công bố | chưa xác nhận |
| Archive/manifest SHA256, snapshot bất biến | chưa đo |
| Ảnh có trên đĩa / decode được / thiếu/hỏng | chưa đo / chưa đo / chưa đo |
| Annotation / unique instances / duplicates bbox-mask | chưa đo |
| Format + category IDs + instance IDs | chưa kiểm |
| Mapping đủ 11 lớp, missing/incomplete classes | chưa kiểm |
| Mẫu xem / tổng ảnh lô / cách lấy mẫu / seed | chưa đo / chưa đo / chưa ghi / 55 |
| Geometry / topology / completeness / domain VN | chưa kiểm |
| Trùng ảnh / group video / val leakage | chưa kiểm |
| Phân loại A/B/C/D + lý do | D — chưa có đủ bằng chứng |
| Số ảnh lỗi xác nhận / ảnh kiểm; số label sửa | chưa đo / chưa đo; chưa đo |
| Người sửa / người kiểm chéo / quyết định lô | chưa ghi / chưa ghi / pending |
| Công sức chuyển đổi, đầu việc còn lại | chưa ước lượng |
# Cập nhật nguồn thay thế và lô CVAT — 27/11/2026

Đã phát 5 gói QA, mỗi gói 20 ảnh, không trùng nhau; xem [hướng dẫn](../V2_CVAT_REVIEW.md). Nguồn review100 và SideWalk được kiểm lại bằng API, đều công bố CC BY 4.0; chưa đủ bằng chứng quyền footage gốc. Nhóm xác nhận chưa có quyền trích Camera Giao Thông/VOV. Không chuyển những ảnh này sang READY chỉ vì đã sửa nhãn.

Nguồn thay thế ưu tiên **Sign — VR-TSD, loại C**: [repo tác giả](https://github.com/LinhNhat07/VR-TSD-Vietnamese-Traffic-Sign-Recognition) mô tả tự thu thập video/ảnh, camera Xiaomi 70mai A510/Teyes X5 và điện thoại, gán **bbox**, không phải mask. [Dataset gốc](https://universe.roboflow.com/vietnam-traffic-sign-recognition-benchmark/vr-tsd) API HTTP 200 xác nhận object-detection, public, **CC BY 4.0**, catalog 8.078 ảnh, bản v2 train/valid/test 5.668/1.259/1.151. Đây là số metadata, **chưa tải/kiểm đếm ảnh hoặc instance thực tế**. Nguồn gốc được tác giả mô tả rõ hơn các bộ tổng hợp, nhưng vẫn cần kiểm ảnh/quyền tương ứng trước admission. Chưa nhập vào 5 gói, chưa tải toàn bộ. Tiếp theo lấy mẫu theo chuỗi dashcam, tạo mask bằng SAM/AI + người kiểm, đồng thời gán mọi class BTC xuất hiện, không chỉ Sign. Không dựng mask hình chữ nhật từ bbox. Không coi license của bài báo/repo code là license ảnh.
