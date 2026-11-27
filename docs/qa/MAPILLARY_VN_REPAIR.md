# Mapillary Việt Nam — sửa lô 9 ảnh, repair004

Ngày 27/11/2026. **READY: 0 train / 0 validation.** Giữ nguyên 9 ảnh, mask native và snapshot intake001 chứa 201 polygon draft; không ghi đè dữ liệu cũ. 201 là kết quả serialize của bản chuyển đổi trước, không phải số instance BTC native đã được duyệt.

Gói mới tại `C:\My Future\VietSight-data-qa\data\versions\mapillary_vn_repair004_REVIEW.zip`. SHA256: `427bd984bd9ae06ab6b9d707afebf5cb74983f16d8e1f27ffddf7c1eae5d97ee`. Đây là gói local REVIEW, chưa có quyền công khai lại và chưa có split để train. Có RGB xử lý, mask native RLE của 9 ảnh, nhãn chuyển đổi, quyết định có bằng chứng, crop, checksum và `TARGETED_REMAINING.csv`. RGB/PNG native đầy đủ vẫn ở intake001.

## Sửa có kiểm chứng

- Chỉ ghép source-labelled rider khi có duy nhất một xe đạt riding score ≥ 0,10. Không coi bbox chạm xe bên cạnh là một liên kết mạnh; một ứng viên duy nhất nhưng score bằng 0 cũng không tự được ghép. Nhiều ứng viên mạnh vẫn giữ cờ. Người dắt xe và source person không được hưởng quy tắc tự ghép này. Mọi người được ghép vào cùng xe union thành một instance.
- Đơn giản hóa từng ring độc lập; nếu ring nhỏ bị xẹp thành đường thẳng thì giữ ring gốc. Thử epsilon chi tiết hơn, vẫn ≤400 điểm và IoU ≥0,995 theo decoder notebook, không dùng largest-component fallback. Hai mask lớn từng không serialize được nay đạt gate sau union; một mask person ID 110 vẫn bị chặn. Không sửa validator hoặc thư viện hình học V1.
- Áp dụng cạnh ngắn **<1,25% cạnh dài ảnh** lên instance cuối, sau union rider/xe. Crowd có cả bounding extent dưới ngưỡng cũng không thể chứa thành viên đủ ngưỡng. Ghi **81 quyết định loại theo kích thước**, gồm crowd và instance cuối, trong `exclusions`; mask native không bị xóa. Không loại theo diện tích 64 px². Còn 7 cờ diện tích dưới 64 ở thang 640 dù đã qua điều kiện cạnh ngắn: hai điều kiện không tương đương.
- AI visual đối chiếu RGB/mask: ID 87 → xe 103, ID 175 → xe 192; ba xe đẩy hàng ID 104/105/106 được bỏ khỏi BTC theo guideline. Mỗi quyết định lưu `assessment.method=ai_visual`, crop và lý do; `reviewer=null`, không ghi thành human QA. ID 275 ở rất xa, chưa kết luận Truck/pickup/đầu kéo/rơ-moóc; chuyển đúng instance này để kiểm, không remap toàn bộ Truck hoặc Other vehicle theo suy đoán.

| Cờ | intake001 | repair004 |
|---|---:|---:|
| Association mơ hồ | 19 | 10 |
| Rider không tìm được xe | 9 | 7 |
| Crowd/group chưa tách | 6 | 4 |
| Ontology cần phân giải | 5 | 1 |
| Mask serialize được nhưng IoU <0,995 | 4 | 0 |
| Mask chưa biểu diễn được trong gate | 2 | 1 |
| Overlap sau decode | 7 | 2 |

Đây là cờ QA, không đồng nghĩa lỗi native đã được human xác nhận. Giảm số cờ gồm cả sửa quy tắc và loại đối tượng dưới ngưỡng guideline; không gọi toàn bộ là số nhãn sửa bằng tay. Có thêm các trường hợp score yếu trước đây tự ghép nay bị chặn. Không xóa rider/xe mơ hồ để đổi status.

Bản mới có **109 polygon draft**: Car 10, Bus 0, Truck 1, Other vehicle 0, Motorcycle 16, Motorcyclist 41, Bicycle 2, Cyclist 1, Pedestrian 23, Sign 15, Traffic light 0. Một mask chưa serialize được vẫn giữ RLE; draft không được coi là annotation đầy đủ để train. Sign từ 27 còn 15 do ngưỡng kích thước; không dùng điểm public thấp làm lý do tự tăng/xóa nhãn. Không có mẫu pickup hoặc rơ-moóc đủ rõ để kết luận thêm lỗi mapping trong lô này.

## Phạm vi còn chặn

Bốn ảnh không còn cờ association/crowd/ontology/topology sau các phép kiểm hiện tại: `Qhd_ZpAkyRpB-pLxJs2Zhg`, `Ua_bgqpPpJRkHZqH-fU2tw`, `bpqabUdDEwaw9gVaiRoQUw`, `lFr49FF9fjS6doEXbt1Zkw`. Đây là **ứng viên chất lượng**, chưa phải bốn ảnh READY hoặc bằng chứng đầy đủ nhãn. Năm ảnh còn lại có hàng cụ thể trong `TARGETED_REMAINING.csv`; không yêu cầu vẽ lại cả 9 ảnh.

Native Vistas README yêu cầu Research Use License và giới hạn phi thương mại; mirror công bố CC BY-NC-SA 4.0. Chưa lấy được toàn văn điều khoản gốc, hai đường dẫn license PDF riêng trên Kaggle không tải được. Metadata Zenodo CC BY 4.0 không cấp quyền ảnh. Không kết luận cuộc thi bị cấm, cũng không tự xác nhận quyền công khai lại dataset. Đây là một điều kiện chưa đủ bằng chứng.

Cả 9 ảnh là Việt Nam theo polygon địa giới; một ảnh V1 frozen đã loại trước đó. Không có trùng SHA256 hoặc cặp dHash ≤4 giữa 9 RGB chuyển đổi; điều này không chứng minh độc lập sequence. Chưa có sequence/capture group đủ bằng chứng để khóa split Việt Nam; không dùng mỗi image ID như bằng chứng các chuyến chụp độc lập. Không trộn V1 validation vào train. Human QA toàn bộ: chưa có. Nhãn native đáng tin cậy có thể đi theo source QA + tự động + lấy mẫu; không bắt buộc nhóm duyệt lại từng ảnh chỉ để điền reviewer. Cờ coverage hiện tại của converter là gate hồ sơ, không phải bằng chứng nguồn thiếu annotation.

## Kiểm thử

14 test V2 và 8 test QA qua trên CPU, gồm round-trip fragment một pixel, association mạnh/yếu, giữ trường hợp mơ hồ, trace quyết định AI mà không cấp human approval, và tách quy tắc cạnh ngắn/diện tích. Chạy converter thật trên đủ 9 ảnh. Audit nhãn serialize: 109 polygon, 5/9 ảnh có cờ; không có polygon sai cấu trúc được phát hành như hợp lệ. Builder chạy trên lô thật và từ chối với `no reviewed Vietnam images; no trainable dataset created`; không tạo dataset giả READY. Build fixture trong test qua. Không chạy notebook train/export vì không sửa đường training/export, không GPU hay thay dependencies. SHA256 V1 frozen và ZIP intake001 khớp bản đã ghi trước đó.

Chi tiết máy đọc: [MAPILLARY_VN_REPAIR_EVIDENCE.json](MAPILLARY_VN_REPAIR_EVIDENCE.json).

## Hai nguồn được yêu cầu — đã kết thúc tìm kiếm

[Dovai traffic-opliy v11](https://universe.roboflow.com/dovai/traffic-opliy/dataset/11): publisher ghi CC BY 4.0, instance segmentation, 2 class car/motorcycle. Export thật **634 ảnh / 16.666 annotation**, train455/val99/test80; car2.358, motorcycle14.308. Đọc annotation tất cả split và 6 RGB, **1.209.482 byte mạng**, không tải archive17.970.890 byte. 16.660 RLE +6 polygon, giải mã được nhưng **139 mask rỗng**, 160 mask hình chữ nhật kín; không gọi mọi mask chữ nhật là bbox giả. Hai nhóm tên video `Video-Project` và `vd1` đều đi qua cả ba split. Sáu mẫu là cùng camera cố định, có dấu hiệu Việt Nam nhưng chưa xác định camera/chủ nguồn; nhiều mask chỉ phủ xe, thiếu người ngồi, cảnh còn người/biển không có class nguồn. **B, chưa nhập train**: cần nhãn rider/coverage và xác minh nguồn, không được đổi motorcycle thẳng thành Motorcyclist nếu mask thiếu người. Không phải nguồn mask BTC dùng ngay để giảm tải nhóm.

[Bài Sensors 25(22):7083](https://www.mdpi.com/1424-8220/25/22/7083) dẫn tới [repo tác giả](https://github.com/NgocHaiNguyen14/Auto_driving_car). Kiểm tree189 file và README/code gán nhãn, chưa thấy license dataset hoặc URL tải bộ instance BTC đã gán nhãn. Bài mô tả segmentation cho lane/marking/vehicle/person, Sign là **detection**. `gen_labels.py` chạy model rồi xuất prediction; không coi đó là GT human. CC BY của bài không tự cấp quyền dữ liệu riêng. **D cho nhu cầu mask BTC native có thể train ngay**, không tải model/video lớn hoặc tự lấy hình bài báo làm dataset.

## API tọa độ Mapillary — kiểm có giới hạn

[Demo chính thức](https://github.com/mapillary/api-demo) công bố metadata endpoint Graph và token demo dùng thử. Không có token riêng cấu hình trên máy; thử token công khai của demo trong bộ nhớ, không lưu/log token. ID Vistas có tọa độ đã biết và một ID thiếu tọa độ đều trả HTTP400, error100, không có geometry; unauthenticated control error190. Chưa xác nhận được crosswalk legacy Vistas ID sang Graph image ID, cũng chưa kết luận các ảnh không còn tồn tại. **0 tọa độ mới, 0 ảnh tải thêm, không quét 8.700 ID.** Cần crosswalk chính thức hoặc Graph ID có thể đối chiếu trước khi mở rộng. Không suy đoán quốc gia từ lỗi API.

Evidence nguồn/API: [BOUNDED_SOURCE_ROUND003_EVIDENCE.json](BOUNDED_SOURCE_ROUND003_EVIDENCE.json). Giới hạn khảo sát ở hai nguồn người dùng chỉ định và probe API trên; không mở thêm tìm kiếm dataset.
