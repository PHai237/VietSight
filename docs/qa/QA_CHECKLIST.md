# VietSight — Label QA và release checklist

Deadline: **10:30 ngày 28/11/2026, GMT+7**. Review baseline `565d05b`; static delta `bfd434f`. Trạng thái hiện tại: **chưa duyệt release V2**. Checklist không thay validator BTC hoặc việc người duyệt xem ảnh.

## Quy tắc chạy song song

- [x] Worktree riêng `C:\My Future\VietSight-data-qa`, branch `codex/data-qa`.
- [x] Source/pipeline/notebook/config train/validator/dependencies chung giữ nguyên trong phiên QA.
- [x] Không training/inference/GPU/download dataset; không push/reset/checkout/stash trong thư mục chung.
- [x] Báo bug bằng [independent review](INDEPENDENT_REVIEW.md); chỉ đề xuất patch cho Claude/Lead.
- [ ] Người nhận xác nhận commit hiện tại đã áp dụng/chặn các lỗi P1 liên quan lô V2. Diff mới sau mốc review cần đối chiếu lại.

## Cổng nhận nguồn

- [ ] Source ID/release/batch, URL chính thức, URL tải/mirror, người gửi và thời điểm nhận đã ghi.
- [ ] Có giấy phép **ảnh và nhãn**, đúng mục đích sử dụng, attribution và quyền chia sẻ; license code không thay license dataset.
- [ ] Có manifest/checksum và snapshot bất biến; `.part`, ảnh/ann hỏng tách riêng.
- [ ] Count ảnh tồn tại, decode được, thiếu annotation, empty label và unique instances có mẫu số; bbox+mask trùng không đếm đôi.
- [ ] Mapping có ID gốc/tên gốc, 11 ID BTC, quyết định lớp thiếu/không đầy đủ.
- [ ] Lô A/B/C/D có lý do/người duyệt; chỉ lô A đã QA mới bàn giao merge V2.

Mẫu intake và kiểm tra chính thức: [DATASET_AUDIT.md](DATASET_AUDIT.md). Không tăng tải Mapillary/Kaggle, trùng công việc Claude.

## Cổng ngữ nghĩa và completeness (người xem ảnh)

Căn cứ: [quy ước BTC do người dùng cung cấp](BTC_GUIDELINE_EXCERPT.md). Nội dung này đã chốt các quy tắc dưới đây; URL/revision web vẫn chưa được cung cấp.

- [ ] Mọi đối tượng đủ lớn thuộc 11 class có annotation; xem ảnh gốc trước để không bị overlay dẫn dắt. Ảnh không prediction vẫn phải kiểm đủ nhãn.
- [ ] Xe máy có người, kể cả dừng đèn đỏ, trẻ em và người ngồi sau: **một Motorcyclist = xe + mọi người ngồi trên xe**. Xe không người Motorcycle.
- [ ] Xe đạp áp dụng tương tự Cyclist/Bicycle; xe điện không bàn đạp dạng xe máy là Motorcycle/Motorcyclist.
- [ ] Người dắt: Pedestrian + xe riêng, không gộp rider. Không ghép người đứng sát/đi sau vào xe.
- [ ] Cảnh nhiều xe: kiểm xe thực sự của mỗi rider, người ngồi sau, người bị cột che; không tin association hình học khi nhiều ứng viên gần điểm.
- [ ] **Xe 9–15 chỗ thuộc Car** theo cập nhật người dùng; xe kiểu bus/truck theo guideline đã duyệt. Pickup ghi decision cụ thể, không tin `truck` nguồn luôn đúng BTC; không tự map mọi `bus` nguồn thành Bus nếu đó là xe 9–15 chỗ.
- [ ] Đầu kéo Truck; rơ-moóc/sơ-mi rơ-moóc kể cả container Other vehicle; **hai instance riêng**. Ba gác, xích lô, xe công trình Other vehicle; xe đẩy tay không gán.
- [ ] Mỗi tấm biển một Sign, chỉ mặt biển, không cột. Mỗi hộp đèn một Traffic light, phủ hộp; không gồm cột/đèn xe/đèn trang trí.
- [ ] Không gán quảng cáo/poster, vật sau kính cửa hàng, người trong ô tô thấy qua kính, động vật, reflection/bóng/ánh đèn pha.
- [ ] Vật nhỏ xem cạnh ngắn <~1,25% cạnh dài ảnh; ghi quyết định vùng sát ngưỡng. Giữ nguyên validator/ignore-area của notebook.
- [ ] Với BDD/COCO: rà Sign/TL/Other vehicle và lớp nguồn chưa gán, không chỉ sửa các mask AI đã tìm ra. Fill không prediction không được coi là completed.

## Cổng geometry và visible mask

- [ ] Đúng ID 0..10 và thứ tự tên class BTC trong file nguồn; không có bbox 5 cột, NaN/Inf, điểm ngoài miền, polygon suy biến/tự cắt.
- [ ] Chỉ phần nhìn thấy; pixel giao thuộc vật phía trước; candidate overlap từ raster phải xem lại ảnh gốc, có thể chỉ là sai khác biên.
- [ ] Một vật bị cột/dây che vẫn một instance; giữ các mảnh rời và lỗ. Không thay bằng hull hay chỉ mảnh lớn.
- [ ] Kiểm round-trip mask→polygon→decoder thực của bộ chấm, lưu phần mất/thêm pixel, không chỉ parser pass.
- [ ] Không mất mask khi resize/EXIF/clip/occlusion, không lấp lỗ/cột. `painter` và `bottom` cần mẫu có visible-mask truth hoặc xác nhận thủ công.
- [ ] Cờ tiny/sparse/extreme aspect/sign tall/mask lớn chỉ là heuristic; người duyệt xác nhận rồi mới ghi error count.

## Cổng sửa nhãn và kiểm chéo

- [ ] Lưu bản nháp bất biến/hash trước khi accept; không ghi đè bằng chứng trước sửa.
- [ ] Dùng `preview.py` hiện có nếu chỉ cần contact sheet theo class. Công cụ bổ sung chỉ dùng khi cần ID/CSV/before-after/geometry flags.
- [ ] Trước CVAT accept, chạy QA với label export và class order; accept hiện tại không chặn polygon self-cross (R05).
- [ ] Sau sửa, xuất panel gốc/before/after từ **cùng ảnh, cùng kích thước**, hash nhãn hai revision; không dùng nhãn giả lập làm bằng chứng sửa thật.
- [ ] Lưu reviewer/người sửa riêng; instance ID QA = `file:L<dòng>` theo hash label revision, không phải ID tracking xuyên revision.
- [ ] Kiểm chéo ít nhất 10% mỗi lô theo quy trình nhóm; ưu tiên thêm mọi lỗi P1/association/multipart/missing class. Chưa kiểm class nào ghi rõ chưa kiểm.
- [ ] Ghi `ảnh lỗi xác nhận / ảnh đã kiểm` và `instance lỗi / instance đã kiểm` riêng; lô >5% ảnh lỗi trả sửa theo guideline nhóm. Không lấy flag tự động làm confirmed error.
- [ ] Trước release, mọi cờ critical và ảnh AI pre-label đã được người duyệt; giữ pending/rejected nếu chưa xong. Không sửa trạng thái source chung bằng CLI QA này.

## Cổng split và thí nghiệm

- [ ] Frozen val sau dedup có đủ mọi key; thiếu một key phải dừng, không chỉ WARNING.
- [ ] Khóa cả bytes ảnh và label val; không hardlink nhãn frozen tới processed đang sửa.
- [ ] So sánh V2/V3 với reference val V1 bất biến bằng tên + SHA256 ảnh/label. Sửa GT val phải tạo revision và chấm lại các checkpoint cũ.
- [ ] Exact duplicates xuyên train/val = 0; near duplicates và cùng video/chuyến chụp đã review, kể cả giữa hai nguồn.
- [ ] Không dùng synthetic BTC, test không nhãn, lô quyền chưa rõ hoặc pre-label chưa duyệt trong train/val thực tế.
- [ ] Run report, config/data fingerprint, val revision, AP từng class và mask mAP50-95 có artifact gốc. Thiếu kết quả ghi “chưa đo”.

## Cách dùng CLI bổ sung

Mọi lệnh dưới chạy tại worktree **VietSight-data-qa**; dùng Python hiện có theo đường dẫn tuyệt đối, không cài dependencies. Output phải là đường dẫn mới và ngoài input. Nếu đã có output, chọn tên khác để bảo toàn evidence.

```powershell
$qaPython = 'C:\My Future\VietSight\.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'

# Bài test CPU cho công cụ QA, không chạy model.
& $qaPython -m unittest discover -s scripts\qa -p test_qa.py -v

# Xem lô đã chuẩn hóa; class names config phải là mapping đã được người xác nhận.
& $qaPython scripts\qa\label_audit.py data\processed\qa_bdd12 --names-from configs\classes.yaml --out runs\review\bdd-next --per-class 5

# Kiểm export sửa nhãn: supplied BEFORE và AFTER, không tự sửa file.
& $qaPython scripts\qa\label_audit.py data\processed\qa_bdd12 --labels 'D:\QA\before_labels' --after-labels 'D:\QA\after_labels' --names-from configs\classes.yaml --out runs\review\bdd-corrected

# Built version: kiểm val identity và exact leakage; near chỉ là candidate cần xem.
& $qaPython scripts\qa\compare_validation.py 'D:\QA\v2_snapshot' --reference 'D:\QA\v1_immutable' --near-hamming 2 --out runs\review\val-compare.json

# Tái hiện bug trên fixture nhỏ riêng trong work/ (không sửa source).
& $qaPython scripts\qa\reproduce_risks.py --out work\qa-repro-next

# Inventory nhẹ: không tải file, tối đa 60 annotation mỗi nguồn.
& $qaPython scripts\qa\source_inventory.py 'C:\My Future\VietSight\data\raw' --out runs\review\inventory-next.json --max-annotations 60
```

`label_audit.py` xuất `audit.json`, `review.csv`, thư mục `evidence/` có ảnh gốc + overlay; thêm panel after nếu có annotation sửa được cung cấp. JSON chứa class ID/name, instance line ID, hash ảnh/nhãn, per-class counts và sample lists theo seed 55. CSV giữ pending để người điền decision/reviewer/note; ảnh không có cờ hình học vẫn cần kiểm completeness. Không có UI/server và không ghi lại source labels.

Giới hạn: mặc định đọc toàn lô; có thể dùng `--limit N` (N ảnh đầu theo tên, **không đại diện thống kê**) và `--max-previews N`. Mẫu per-class lấy tất định từ class có nhãn; không thể tự phát hiện class bị bỏ sót nếu không xem ảnh. Raster overlay có sai khác biên so decoder bộ chấm, nên IoU chấm thật phải dùng decoder thật. Một instance ID line có thể thay sau sửa; đối chiếu bằng label hash + hình ảnh thay vì khẳng định tracking.

`compare_validation.py` trả exit 1 khi lệch val/thiếu label/exact leak hoặc có near candidate; exit 0 chỉ pass phạm vi kiểm tra, không chứng minh không sequence leakage. Near check có chi phí O(N²), chỉ bật khi cần. Không dùng thư mục val reference có hardlink tới source còn sửa.

## Bàn giao vòng QA đầu (trước yêu cầu commit/PR)

- [x] Báo cáo code review, dataset audit, checklist, experiment log đã chuẩn bị trong worktree.
- [x] Lỗi có kịch bản tái hiện và số đo fixture tách biệt dataset thật.
- [x] 12 ảnh BDD thực đã qua converter/QA hình thức; 3 ảnh đã self-review, chưa cross-review; chưa sửa nhãn thật.
- [ ] Bản export label thật sau sửa và ảnh before/after thật: chưa được bàn giao.
- [ ] Val V1 bất biến + run_report V1 thật + AP: chưa có trong snapshot local được review.
- [ ] Merge tài liệu/công cụ vào main: chưa thực hiện; chủ pipeline quyết định tích hợp khi phù hợp. Không commit/push tự động.

Trong yêu cầu tiếp theo, người dùng đã cho phép commit/push/PR và merge phần guideline/tài liệu QA/công cụ độc lập đã test nếu không conflict. Xem [đối chiếu cập nhật](CLASS_MAPPING_REVIEW.md); không thay pipeline chính hoặc notebook của V1 để thực hiện việc tích hợp này.
