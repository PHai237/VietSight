# V2 — năm lô kiểm nhãn song song, round001

Các gói là **REVIEW ONLY**, không phải dataset READY. Nguồn công bố CC BY 4.0 nhưng quyền video gốc chưa đủ bằng chứng; người dùng đã xác nhận nhóm chưa có quyền trích Camera Giao Thông/VOV. Giữ các gói trong `RIGHTS_PENDING/`, không đưa vào training hoặc tự công bố lại. Người dùng cho phép AI tạo đề xuất để người thật kiểm/sửa; chưa xác nhận quy định BTC đầy đủ cho model sinh nhãn.

Thư mục local: `C:\My Future\VietSight-data-qa\data\versions\v2_cvat_round001\`. Danh sách gói, số ảnh và SHA256 nằm trong `PACKAGES.json`. Không gửi `staging/` hay cả intake001 thay cho năm ZIP phân công. Không có ảnh được duyệt tay trước khi thành viên nhận việc.

## Import và nhận việc

1. Mỗi thành viên nhận **một ZIP `member_01` … `member_05`**, giải nén. Đọc `README_VI.md`, `TODO_PER_IMAGE.csv`, `ATTRIBUTION_AND_RIGHTS.md`.
2. Tạo project/task CVAT với 11 label trong `CVAT_LABELS.json`, đúng tên/thứ tự `CLASSES.txt`. Không thêm `ignore_region` vào bộ 11 class. Nếu cần đánh dấu vùng khó, ghi vào CSV và để ảnh `needs_fix`/`reject`.
3. Tạo task từ **chỉ ảnh trong `images/train/`**; có thể upload `images_only.zip` có sẵn trong gói. Không upload ZIP ngoài có lẫn tài liệu làm dữ liệu ảnh.
4. Actions → Upload annotations → **Ultralytics YOLO Segmentation 1.0** (một số bản ghi `YOLO Segmentation 1.0`) → chọn `cvat_upload.zip`. File này chỉ có `data.yaml`, `train.txt` và labels; không chứa ảnh. Chờ import xong và mở vài ảnh kiểm class/mask đúng trước khi làm toàn lô. Cấu trúc theo [CVAT Ultralytics YOLO](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/).
5. Nhãn import gồm mask nguồn và đề xuất AI **chưa duyệt**. `AI_PROPOSALS.jsonl` ghi dòng AI theo **index 0** của công cụ Claude; `TODO_PER_IMAGE.csv`/audit dùng **dòng 1-based**. `source_only_labels/` giữ nhãn trước AI để đối chiếu; đừng import cả hai bộ chồng lên nhau.
6. Làm theo `priority` trong CSV: ảnh dễ/ít cờ trước, rồi ảnh nhiều lớp/khó. Thứ tự CVAT có thể khác CSV; tìm theo filename, không dựa vào frame ID của CVAT. `intended_split` phân biệt train/val dự kiến; chữ `train` trong thư mục import chỉ là định dạng CVAT.

## Mỗi ảnh phải kiểm những gì

- Rà cả **11 class xuất hiện**, kể cả vật chưa có nhãn/AI không phát hiện. Đánh dấu từng cột class trong `RETURN_REVIEW.csv` là `checked_present` hoặc `checked_absent` sau khi thực sự xem. Không suy từ class count hiện tại rằng lớp đó không có trong ảnh.
- Xe máy có người ngồi, gồm người ngồi sau/trẻ em/dừng đèn đỏ: **một Motorcyclist = xe + mọi người trên xe**. Xe không người/đang được dắt: Motorcycle; người dắt là Pedestrian riêng. Xe đạp áp dụng Bicycle/Cyclist. Sửa mask nguồn đã gộp người nhưng còn mang tên Motorcycle; không chỉ thêm người/xe lần thứ hai.
- Sign: mỗi **mặt biển** một instance; thêm biển bị bỏ sót, bỏ cột, quảng cáo/cửa hàng. Đèn giao thông phủ hộp, không gồm cột, đèn xe hoặc đèn trang trí. AI Sign/TL có thể sai hoặc thiếu dù score cao.
- Other vehicle: tìm ba gác, xích lô, xe công trình và rơ-moóc. Đầu kéo là Truck, rơ-moóc/container kéo theo là Other vehicle **riêng**, cần tách mask nguồn gộp tổ hợp. Xe 9–15 chỗ thuộc Car. Kiểm pickup theo quy ước đội; nguồn đã đổi một số Car thành Truck, không nhận thẳng quyết định đó.
- Chỉ mask phần nhìn thấy. Xóa vùng chồng lấn theo vật phía trước; không phủ bóng, phản chiếu, vật qua kính cửa hàng hoặc người trong ô tô. Cùng một vật bị che thành nhiều phần vẫn **một instance**, không bỏ phần nhỏ chỉ để export dễ.
- Dùng mask/brush một instance có nhiều phần rời, hoặc polygon cùng label được group cho **cùng một vật**; không group các vật khác nhau. Giữ COCO export để lưu các phần: [CVAT COCO](https://docs.cvat.ai/docs/dataset_management/formats/format-coco/) hỗ trợ polygon/mask và nhóm shape. Không tự nối cầu dày xuyên vật che, không vẽ bbox thay mask.
- Cạnh ngắn dưới khoảng **1,25% cạnh dài ảnh** là ngưỡng gán nhãn trong guideline. **64 px² tại thang 640** là ngưỡng diện tích của bộ chấm, không phải cùng điều kiện. Ghi trường hợp sát ngưỡng, không sửa validator.
- Xem cờ `invalid_polygon`, overlap và AI alternative class trong TODO. Cờ tự động là yêu cầu xem lại, không phải kết luận lỗi đã xác nhận. SAM/AI chỉ hỗ trợ; người kiểm chịu trách nhiệm mask cuối.

## Trả từng phần, không chờ xong cả lô

1. Save CVAT sau mỗi nhóm khoảng 3–5 ảnh. Ghi tên thật/ngày kiểm, `decision=reviewed` cho **chỉ ảnh đã kiểm đủ**, `needs_fix` hoặc `reject` cho ảnh chưa đạt. Không ghi `reviewed` thay người khác. `issues_fixed` ghi việc thực tế đã sửa; `note` giữ ca phân vân.
2. Export task **COCO 1.0**, giữ `segmentation` polygon/RLE; không export chỉ Detection/bbox. Có thể tắt Save images để ZIP nhẹ. Giữ filename gốc và category names. Khuyến nghị kèm export YOLO Segmentation nếu tất cả mask biểu diễn được, nhưng COCO là bản nhãn chuẩn để phục hồi topology.
3. Trả ZIP COCO và `RETURN_REVIEW.csv`, kèm task URL/ID trong CSV. Có thể export toàn task nhưng chỉ ghi `reviewed` ở các ảnh đã làm; phần còn lại vẫn pending, không được tự nhận cả task đã duyệt. Kèm ít nhất một ảnh trước/sau sửa rider hoặc Sign nếu có sửa loại này.
4. Đặt file trả vào `returns/member_XX/` dưới thư mục round001, hoặc gửi đường dẫn local tương ứng. Không ghi đè gói đầu vào; mỗi lần trả dùng tên revision mới. Ảnh chưa giải quyết quyền vẫn ở `RIGHTS_PENDING` dù nhãn đã sửa đúng.

## Tiếp nhận và phát hành

Codex đối chiếu filename/hash và whitelist ảnh thực sự đã kiểm; xác minh đủ cột class, reviewer, quyền/nguồn/capture group, segmentation và geometry/topology/overlap. Dùng converter và builder **V2 hiện có**; không dùng lệnh `review_queue.py accept` chung như cách bỏ qua gate quyền/coverage V2. Không tự điền reviewer hoặc `license_verified=true` từ một lần export CVAT.

Validation dự kiến của round này lấy từ **ba video Đà Nẵng, Hội An, Hà Nội**; toàn bộ ba capture group dành riêng cho val, cả các frame chưa chọn cũng không được đưa vào train V2. Split này thay kế hoạch một video của intake001 cho **bản mới tương lai**, không sửa snapshot intake001. Đây là góc nhìn đi bộ; validation dashcam vẫn cần nguồn có quyền rõ ràng. Bộ review100 là compilation chưa biết từng clip nên giữ toàn nguồn ở phía train và tiếp tục kiểm độc lập cảnh trước READY. V1 val giữ riêng; cần đối chiếu pixel trước release.

Phát hành một version mới ngay khi có tập train/val đủ chất lượng **và** đủ quyền thực tế; không chờ đủ 100 hoặc 377 ảnh. Ghi số lớp thực tế/giới hạn coverage, fingerprint, hash, quyền và review evidence vào handoff mới. Không gọi một gói thiếu điều kiện là READY. Hiện chưa có nhãn người kiểm trả về hoặc nguồn trong round này đủ xác nhận quyền gốc.
