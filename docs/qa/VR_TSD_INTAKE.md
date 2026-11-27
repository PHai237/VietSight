# VR-TSD — lấy mẫu thật và đề xuất e25 + COCO

Nguồn: [dataset VR-TSD](https://universe.roboflow.com/vietnam-traffic-sign-recognition-benchmark/vr-tsd/dataset/2), [repo tác giả](https://github.com/LinhNhat07/VR-TSD-Vietnamese-Traffic-Sign-Recognition). Tác giả mô tả tự thu thập bằng dashcam/điện thoại và gán bbox; API/README export công bố **CC BY 4.0**. Đây là bằng chứng nguồn tốt hơn các ảnh tổng hợp báo/video chưa có quyền gốc. Giữ attribution và link license, không coi license code/bài báo là license ảnh. Chỉ nhận ảnh chụp đường phố có nguồn phù hợp, không nhận ảnh minh họa hoặc ảnh chưa rõ tác giả từ screen recording chỉ dựa vào tên dataset.

Ngày kiểm: **27/11/2026, Asia/Saigon**. Nguồn thuộc **C: ảnh/bbox, cần mask thực**, chưa phải nguồn instance segmentation sẵn. Không cần phát triển công cụ mới.

## Dữ liệu local và chọn mẫu

`C:\My Future\VietSight-data-qa\data\raw\vr_tsd_v2_sample001\` giữ 40 ảnh thật đã tải, COCO annotation nguồn, README license, manifest và checksum. ZIP nguồn 460.610.938 byte **không tải toàn bộ**; đọc qua Range reader có sẵn, tổng 3.749.552 byte mạng. Annotation test được đọc đủ để chọn mẫu; 40 ảnh mẫu có **45 bbox, 0 instance masks nguồn**. Không dùng bbox làm segmentation.

Đã xem hai contact sheets đủ 40 ảnh bằng AI visual inspection, **không phải human QA**. Ba ảnh sơ đồ hai biển xanh trên nền trắng bị loại khỏi batch mới. Một số ảnh screen recording của ảnh đường phố cũng không chọn vào batch này vì nguồn footage khó xác minh hơn ảnh camera `IMG_*`. Phần lớn mẫu chọn là close-up/điện thoại, **không phải dashcam**, nên chỉ là bổ sung Sign; không thay thế tập cảnh giao thông đầy đủ.

Input chọn riêng 20 ảnh chụp thật: `data/versions/vr_tsd_e25_coco_review001_input/`. Manifest giữ hash và capture group dự kiến theo video filename; tên prefix là bằng chứng gợi ý nhóm, chưa chứng minh mọi clip khác nhau độc lập cảnh. Bbox nguồn chuyển thành **hint 5 cột** class Sign, công cụ không copy các dòng này vào labels segmentation. Có kiểm không trùng hash với 100 ảnh round001 và không gần trùng dhash trong 20 ảnh ở ngưỡng 2.

## E25 + COCO đã chạy thật

Người dùng chuyển `data/raw/prelabel_models/vietsight_v1_e25.zip` từ Claude. ZIP SHA256 `f0b874634ed90bcacd23238c3eb517fa3169bcbefd149a34f6f103a62c56e0de`; ONNX SHA256 `87c81673e92feeea0e194d15bb7d69b81c783328b8ca7133b1a2ea5159b607eb`; cả hai **khớp run_report e25**. Không dùng checkpoint e3 thay e25.

Dùng đúng `scripts/propose_labels.py` của Claude, e25 ONNX + COCO YOLO26n-seg, CPU/2 threads, conf 0,20, đủ 11 class, `--all`; output mới `data/versions/vr_tsd_e25_coco_review001_proposals/`. Không chạy lại 100 ảnh round001, không thay ZIP/task CVAT nhóm đang sửa. Phân công batch Codex ghi ở [V2_CLAUDE_DATA_COORDINATION.md](../V2_CLAUDE_DATA_COORDINATION.md).

Kết quả: **20 ảnh / 8 ảnh có đề xuất / 8 mask AI: 7 Sign + 1 Car**; 12 ảnh chưa có mask. Đây không phải recall hoặc AP. QA kiểm cấu trúc 20 ảnh, đủ class order; cảnh không nhãn là cờ cần kiểm, không phải nền đã duyệt. Trong `IMG_4063-MOV_out0014...`, hai biển lớn rõ nhưng AI chỉ gán biển đỏ, bỏ biển xanh. Không được dùng empty labels hoặc đề xuất thiếu này để train thẳng.

## Sửa và tiếp nhận

Gói bổ sung: `data/versions/vr_tsd_e25_coco_review001_QA_ONLY.zip`. Đây là lô dự phòng để sửa Sign khi nhóm có khả năng nhận thêm, không phải thay thế năm lô đã phân công. Có `images_only.zip`, `cvat_upload.zip`, bbox hints/proposals để tham khảo, attribution và CSV reviewer trống.

Import/export theo [hướng dẫn CVAT](../V2_CVAT_REVIEW.md), nhưng **20 ảnh này đều chưa phân train/val**. Giữ nguyên nhóm video khi chia sau QA; không dùng split test của publisher làm bằng chứng độc lập với train mới. Mỗi tấm biển phụ cũng kiểm như một instance Sign nếu thuộc guideline; không nhầm hình đèn trên biển cảnh báo thành Traffic light. Kiểm cả các class khác nhìn thấy, không chỉ Sign.

Nhận từng 3–5 ảnh đã sửa; chỉ đưa vào READY sau khi coverage/mask, người kiểm thật, nguồn/quyền và split đều đạt. Bằng chứng nguồn công bố license và tác giả tự thu thập đã có; source/country còn phân vân ở từng ảnh thì tách khỏi release. Hiện **0 human-reviewed / 0 READY**, chưa có lô người kiểm trả về. V1 và frozen validation không đổi.
