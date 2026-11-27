# VietSight — Independent review

Ngày kiểm tra: 27/11/2026, GMT+7. Deadline: **10:30 ngày 28/11/2026, GMT+7**.

## Phạm vi và kết luận

Review độc lập tại `C:\My Future\VietSight-data-qa`, branch `codex/data-qa`, từ commit **565d05b55ab7a0c507934811b4808b5952e9a595**. Đây là snapshot đã commit, không phải xác nhận V1 production đã ổn định. Git ban đầu sạch; không sửa source của Claude, notebook, validator, cấu hình train hoặc dependencies; không chạy model/training/GPU, không tải dataset. Có sao chép 12 cặp ảnh/annotation BDD đang có local để kiểm tra riêng.

Đã đọc cấu trúc 41 file tracked: quy định, README, các module `scripts/vslib`, toàn bộ CLI, configs, provenance, docs; kiểm tra cấu trúc notebook, các helper dữ liệu, sự tương đồng của các cell chính thức và bundle prep. Không thực thi các cell training/export hoặc kiểm chứng lại nội dung scorer nhúng bằng một lượt chấm model. Danh sách và hash file: [review_scope.json](evidence/review_scope.json).

**Chưa nên đưa BDD/COCO vào V2 trước khi rà đủ nhãn 11 class.** Các lỗi dưới đây có bằng chứng logic/fixture; không suy ra tỷ lệ lỗi hoặc mức giảm mAP của dataset thật. Chưa có kết quả train thật để đo ảnh hưởng đến điểm.

Mức độ: P1 = có thể trực tiếp làm sai GT, cho dữ liệu chưa duyệt vào train, hoặc làm mất khả năng so sánh val; P2 = rủi ro có điều kiện/cần xác nhận dữ liệu hoặc guideline. Không phát hiện lỗi đảo thứ tự class ID. Sau review ban đầu, người dùng cung cấp [quy ước BTC](BTC_GUIDELINE_EXCERPT.md): đã chốt người dắt xe, người ngồi sau, đầu kéo/rơ-moóc, visible mask, biển/đèn và ngưỡng cạnh ngắn; không còn coi các điểm này là chưa có hướng dẫn. Chưa có URL/revision để xác minh online.

## Các phát hiện ưu tiên

### R01 — P1: fill thiếu class vẫn cho ảnh chưa kiểm đủ nhãn ở trạng thái ok

- Vị trí: `scripts/prelabel.py:68-94`, đặc biệt `if new:` tại dòng 87; `scripts/build_version.py:62-63`. Config khai báo thiếu class ở `configs/sources/bdd100k_10k.yaml` và `configs/sources/coco2017.yaml` nhưng builder không dùng các khai báo này làm điều kiện chặn.
- Nguyên nhân: chỉ chuyển sang review khi model tìm được mask mới. Khi model không phát hiện, dự đoán dưới confidence, hoặc bị chặn bởi overlap >30%, ảnh vẫn `ok`. Không có prediction không chứng minh vật thể vắng mặt.
- Tái hiện: `reproduce_risks.py`, khóa `fill_zero_predictions`: stub prediction rỗng cho một ảnh `ok`; sau fill vẫn `ok`, flags rỗng. Không chạy model.
- Bằng chứng thật: trong 3 overlay BDD đã xem, cả 3 có đèn giao thông nhìn thấy nhưng output không có Traffic light; mẫu 12 ảnh được converter đánh dấu `ok` toàn bộ. Xem bảng minh chứng bên dưới. Không kết luận các ảnh BDD còn lại đều có lỗi.
- Đề xuất patch cho Claude: mọi ảnh thuộc nguồn thiếu/không đầy đủ class phải vào review, kể cả khi không thêm prediction. Chỉ cho `ok` sau khi người duyệt xác nhận đủ **toàn bộ 11 class**, ghi reviewer/revision. Builder kiểm tra cờ hoàn tất coverage của nguồn/lô. Thêm test no-prediction, overlap-filter và có-prediction đều không tự được train.

### R02 — P1: mất topology và mảnh nhìn thấy khi mask chuyển thành polygon

- Vị trí: `scripts/vslib/geometry.py:65-81`, đặc biệt `RETR_EXTERNAL` dòng 69 và điều kiện giữ mảnh dòng 75; `scripts/vslib/writer.py:90-106`.
- Nguyên nhân: chỉ lấy contour ngoài nên lấp lỗ do occlusion; mảnh nhỏ hơn 2% mảnh lớn bị bỏ dù diện tích tuyệt đối còn đáng kể. Điều kiện thực tế là `a >= max(4, 0.02*big)`, khác mô tả "nhỏ hơn tỷ lệ **và** <4 px". Khi polygon lỗi, writer còn fallback phần lớn nhất hoặc bỏ instance, nhưng không thêm review flag.
- Tái hiện: fixture mask 200×200 có lỗ 80×80: `status=ok`, decode qua pycocotools thêm **6.400 pixel** vào lỗ, IoU round-trip **0,7375390625**. Fixture mảnh 10×10 tách khỏi mảnh 100×100: mất toàn bộ mảnh phụ, tổng mất 299 pixel gồm sai khác biên raster; status vẫn `ok`, không có bộ đếm mất mảnh. Đây là số đo fixture, không phải mAP.
- Đề xuất patch: đo round-trip bằng chính decoder của bộ chấm; nếu mất mảnh/lấp lỗ/thay đổi đáng kể thì review hoặc loại ảnh, không tự nhận `ok`. Nếu định dạng một ring biểu diễn được, giữ hole/component bằng bridge đã kiểm chứng với validator **và** decoder. Không chỉ thay bằng convex hull/largest part. Giữ mask gốc làm bằng chứng local; test pole occlusion, nhiều mảnh, lỗ, mảnh nhỏ, hình lõm.

### R03 — P1: ghép rider chọn một xe tùy ý khi có nhiều ứng viên; có thể bỏ người ngồi sau

- Vị trí: `scripts/vslib/instances.py:65-86`; chọn `max(strong, ...)` dòng 72; chỉ review chồng yếu với xe **chưa có owner** ở dòng 85.
- Nguyên nhân: không kiểm tra chênh lệch ứng viên tốt nhất/thứ hai; điểm hình học không chứng minh cùng instance. Người thứ hai có role `person` chồng yếu với xe đã có người được xuất thành Pedestrian mà không có flag. Nhiều rider chắc chắn cùng xe được union đúng ở dòng 90-94; vấn đề nằm ở quyết định association.
- Tái hiện: `ambiguous_pair` cho hai xe cùng điểm → một Motorcyclist + một Motorcycle, flags rỗng. `weak_passenger` → Pedestrian + Motorcyclist, flags rỗng. Hai fixture này chứng minh thiếu cảnh báo ambiguity; chưa chứng minh một ảnh thật cụ thể bị ghép nhầm.
- Đề xuất patch: đưa ảnh nhiều ứng viên gần điểm nhau vào review; lưu candidate scores/association; kiểm tra tất cả người quanh xe đã có owner; người ngồi sau/trẻ em phải được duyệt và gộp cùng một instance. Không tự ghép sang xe bên cạnh chỉ vì mask chạm.

### R04 — P1: frozen val chỉ cố định key, không cố định tập hiệu lực và nội dung GT

- Vị trí: `scripts/build_version.py:114-120` (thiếu ảnh chỉ WARNING); `150-171` (dedup sau khi ghi frozen list); `38-42,184-191` (hardlink); không lưu hash label của val trong frozen list.
- Tái hiện: fixture frozen list yêu cầu 2 key, trong đó 1 key vắng → build hoàn tất, train=5/val=1, official report `ok=True`. Sau build, sửa label ở processed fixture khiến label val đã build đổi theo ngay vì hardlink; key vẫn y nguyên. Xem `missing_frozen_key`, `hardlink_label_mutation`.
- Rủi ro khác: frozen list được viết trước dedup nên ảnh val trùng có thể bị loại khỏi tập thực tế; thay cấu hình dedup hoặc sửa nhãn val giữa V1/V2 có thể đổi metric dù tên file không đổi. Chưa đo drift trên V1 thật vì chưa có frozen val/results local lúc kiểm tra.
- Đề xuất patch: fail khi thiếu/đổi key; freeze **tập sau dedup** và lưu hash ảnh + label + chính sách group/dedup; không hardlink label tới nguồn đang sửa. Chụp immutable val reference trước train, so sánh bytes trước mỗi phiên bản và trước chấm. Nếu buộc sửa GT val, tạo revision mới, chấm lại checkpoint cũ, không so trực tiếp metric khác revision.
- Công cụ tách biệt: `scripts/qa/compare_validation.py` kiểm tra hash ảnh/nhãn giữa hai built versions và exact train/val leakage. Near dHash tùy chọn chỉ là candidate. Tool không thay builder.

### R05 — P1: review accept chỉ parse cú pháp, converter YOLO bỏ nhãn lỗi rồi vẫn chấp nhận ảnh

- Vị trí: `scripts/review_queue.py:85-93`; `scripts/convert_yolo.py:71-98`; trường hợp segmentation vắng: `scripts/convert_coco.py:102-104`; unsupported geometry: `scripts/convert_supervisely.py:145-148`.
- Nguyên nhân: parser không kiểm tra self-intersection; accept không gọi `polygon_problem`. Converter YOLO ghi dropped counter nhưng giữ `--status ok` mặc định, có thể biến một ảnh có vật thành ảnh nền. Các converter còn lại cũng không bắt buộc review khi bỏ một object không chuyển được.
- Tái hiện: `accept_invalid_polygon`: bow-tie được chuyển `ok`, parse không báo lỗi nhưng `polygon_problem` trả lỗi tự cắt. `malformed_yolo_approved`: một dòng bbox bị bỏ, ảnh vẫn `ok` và `instances={}`.
- Đề xuất patch: trước accept kiểm tra geometry/area, tên class, label đầy đủ theo từng ảnh; fail hoặc giữ review khi có object không đọc/chuyển được. Phân biệt background đã xác nhận với missing/corrupt labels. Không có `data.yaml` thì cần mapping được xác nhận; thiếu metadata lớp không được xem là đã chứng minh đúng thứ tự. Có thể chạy CLI QA độc lập trước accept.

### R06 — P1: nhiều polygon của cùng instance bị rasterize theo even-odd thay vì union

- Vị trí: `scripts/vslib/geometry.py:18-20`, được gọi từ `scripts/convert_coco.py:105-107`.
- Nguyên nhân: `cv2.fillPoly(m, pts, 1)` với tất cả contour cùng lúc làm phần giao thành rỗng. COCO yêu cầu union các polygon của một instance; [COCO API chính thức](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/coco.py) thực hiện `frPyObjects` rồi `merge` trong `annToRLE`.
- Tái hiện: `overlapping_coco_parts`, hai hình vuông giao nhau → mất **1.521 pixel** so với union raster từng phần.
- Đề xuất patch: raster từng exterior vào mask riêng rồi OR; trừ các hole thuộc từng component trước khi union. Kiểm tra thêm polygon lồng nhau, contour khác orientation. Không thay validator BTC.

### R07 — P2: người dắt xe có thể bị hình học tự nhận thành rider

- Vị trí: `scripts/vslib/instances.py:35-49,68-74`; COCO `person` được suy luận bằng overlap/centred/feet/touch.
- Tái hiện: `person_beside_bike` có điểm hình học mạnh → Cyclist, flags rỗng. Cùng hình chiếu có thể là người ngồi hoặc đi sát/dắt xe; fixture không chứa tư thế thật nên chỉ chứng minh thiếu bước xác nhận ngữ nghĩa.
- Đề xuất patch: COCO person+xe phải review cho các tình huống khó, ưu tiên các crop có người ngồi sau/dắt xe/xe đỗ; bổ sung quy tắc tư thế chỉ nếu có dữ liệu kiểm chứng. Theo quy ước người dùng vừa cung cấp, người dắt phải là **Pedestrian + Motorcycle/Bicycle riêng**, không được gộp rider.

### R08 — P2: source truck không phân biệt pickup; cần đối chiếu từng loại xe

- Vị trí: `configs/sources/coco2017.yaml:20`, `configs/sources/bdd100k_10k.yaml:21`: `truck: Truck`; `configs/sources/mapillary_vistas.yaml:22` và `configs/sources/bdd100k_10k.yaml:23`: `trailer: Other vehicle`; `docs/LABEL_GUIDELINE.md:36-42` và mục cần hỏi BTC.
- Bằng chứng: BDD `00e9be89-00001885.jpg` có pickup phía trước được output thành Truck. Đây là mapping thực tế; **chưa kết luận sai BTC**, vì tài liệu local vẫn để pickup `[HỎI BTC]`. Không có rule phân biệt pickup trong lớp truck của nguồn.
- Trailer đang map Other vehicle, đúng nhánh rơ-moóc trong quy ước BTC người dùng vừa cung cấp. Chưa có ảnh trailer được xem để đánh giá segmentation đầu kéo/rơ-moóc.
- Đề xuất: lưu quyết định pickup theo cây quyết định loại xe/nhỏ dưới khoảng 9 chỗ; đoạn trích chưa gọi đích danh pickup nên chưa coi mọi pickup→Truck là sai đã được BTC xác nhận. Đầu kéo→Truck và rơ-moóc/container→Other vehicle, hai instance riêng, đã có căn cứ. Lọc review truck/Other vehicle của V2; không đổi hàng loạt toàn bộ Truck thành Car.

### R09 — P2: thứ tự occlusion dựa heuristic có thể trừ nhầm mask

- Vị trí: `scripts/convert_supervisely.py:128-156`; `configs/sources/bdd100k_10k.yaml` đặt `z_order: bottom`, Mapillary đặt painter.
- Nguyên nhân: điểm thấp nhất không luôn xác định vật phía trước, nhất là xe bị cắt mép, xe cao, xe trên dốc/cầu; painter phụ thuộc thứ tự mirror và stuff occluder. Mỗi object vẽ sau đều làm mất pixel object trước.
- Tái hiện có điều kiện: đặt hai polygon giao nhau, đổi thứ tự/đáy làm vùng giao chuyển chủ sở hữu. Chưa có visible-mask ground truth gốc cho 12 ảnh BDD để kết luận tỷ lệ pixel sai; không coi heuristic là ground truth.
- Đề xuất: so sánh theo instance bitmask gốc của BDD trên các ảnh đại diện trước khi dùng rộng; review 100% ảnh overlap mạnh; ưu tiên bitmask gốc nếu được cung cấp. Kiểm tra before/after trên cột che xe và pedestrian che xe, không chỉ xem báo cáo polygon hợp lệ.

### R10 — P2: group/video và dHash chưa đủ chứng minh không leakage

- Vị trí: `scripts/build_version.py:68,145-168`; `configs/sources/bdd100k_10k.yaml` group regex tiền tố; `scripts/convert_yolo.py:96-97` parent/name mặc định.
- Hiện có: exact SHA256 + near dHash, ưu tiên val, tách group nội bộ nguồn. Các cơ chế này hữu ích và đã được đọc; chưa có bằng chứng leakage thật trong V1.
- Rủi ro: ảnh cùng video nhưng đổi tên/crop nhiều có thể vượt ngưỡng dHash; cùng scene giữa hai nguồn có group khác vì source prefix; hai scene khác nhau cũng có thể trùng dHash và bị loại nhầm. `--group-from-parent` chỉ có ý nghĩa khi parent thực sự là video/chuyến chụp, không phải thư mục `images` chung.
- Đề xuất: provenance canonical scene/video ID xuyên nguồn, giữ nhóm cùng phía; lưu danh sách duplicate candidates để người xác nhận; so hash ảnh decoded/near/crop và timestamp theo nguồn nếu có. Đừng coi hash pass là semantic/sequence leakage pass.

### R11 — P1: tài liệu nhóm hướng dẫn gộp đầu kéo + thùng vào Truck trái đoạn quy ước BTC mới được cung cấp

- Vị trí: `docs/LABEL_GUIDELINE.md:38`, hàng Truck mô tả “container (đầu kéo + thùng)”; `configs/sources/mapillary_vistas.yaml:23` gán `wheeled slow: Other vehicle`, ghi chú bao gồm carts.
- Căn cứ mới: [quy ước BTC người dùng cung cấp](BTC_GUIDELINE_EXCERPT.md) quy định đầu kéo Truck và rơ-moóc/sơ-mi rơ-moóc kể cả container Other vehicle thành **hai instance riêng**; xe đẩy tay không gán. Config `trailer→Other vehicle` đã phù hợp, nhưng một source `truck` mask gộp cả tổ hợp phải được tách thủ công, không đủ remap class ID.
- Cách tái hiện: đối chiếu hàng Truck trong guideline local với mục đầu kéo/rơ-moóc của đoạn trích; review ảnh xe đầu kéo để xác định source mask có gộp không. Chưa có ảnh tổ hợp được xem nên chưa có số instance sai thực tế. Với `wheeled slow`, review xem record là xe đẩy tay (không gán) hay xích lô/ba gác (Other vehicle), không coi mọi record của class nguồn đều sai.
- Đề xuất patch cho Claude: sửa guideline chung và decision log; đưa mask truck tổ hợp/wheeled slow chưa phân loại vào review; tách đầu kéo–rơ-moóc, cấm tự gộp theo doc cũ. Thêm kiểm tra visible-mask không chồng nhau, không lấy pixel che khuất; loại đèn xe, reflection, poster, người sau kính theo BTC.
- Ngưỡng vật nhỏ: code dùng diện tích 64 px ở 640; BTC còn mô tả cạnh ngắn <~1,25% cạnh dài ảnh. Hai điều kiện không tương đương cho vật rất mảnh. QA chỉ thêm candidate để người xem, **không sửa ngưỡng validator/bộ chấm**.

## Ma trận các rủi ro được yêu cầu

| Rủi ro | Kết luận kiểm tra |
|---|---|
| Sai thứ tự 11 class | Không thấy drift ID; semantic đầu kéo/rơ-moóc đã chốt BTC và tài liệu cũ mâu thuẫn R11; pickup còn cần decision cụ thể |
| Người ghép nhầm xe | R03, ambiguity đã tái hiện, chưa đo trên ảnh thật |
| Bỏ người ngồi sau | R03, weak person có thể thoát review; union nhiều known riders có triển khai |
| Người dắt xe thành rider | R07, phải tách Pedestrian + xe theo đoạn BTC mới cung cấp |
| Pickup thành Truck | Một ảnh BDD xác nhận đang output Truck; đúng/sai BTC chưa xác nhận |
| Rơ-moóc thành Truck | Config trailer đúng Other vehicle; doc gộp đầu kéo + thùng vào Truck sai quy ước, R11 |
| Biển chứa cột | Pole và traffic sign frame không map Sign trong Mapillary; hình học mask input vẫn phải xem. Không có ảnh Mapillary local để phê duyệt biên |
| Polygon nhiều mảnh nối sai | R02/R06; chưa kiểm chứng tất cả bridge, có mất mảnh/lỗ trong fixture |
| Occlusion mất/biến dạng mask | R02/R09 |
| Vật có mặt nhưng thiếu nhãn | R01/R05; 3 ảnh BDD đã xem có Traffic light không được gán |
| Trùng ảnh/leakage | R04/R10; chưa kiểm tra toàn bộ dataset V1 thật |
| Val thay đổi giữa phiên bản | R04; fixture chứng minh thiếu key vẫn build và hardlink làm đổi GT |

## Minh chứng dữ liệu thật (self-QA, chưa cross-review)

Đã chạy converter trên 12 cặp BDD thật sao chép local. **12 ok / 0 review / 0 skipped** theo converter; 121 instance output (Car=100, Bus=4, Truck=5, Other vehicle=1, Pedestrian=11; sáu lớp còn lại=0). Không phát hiện polygon sai cú pháp/tự cắt trong mẫu này. Sau khi thêm quy tắc BTC, CLI QA gắn candidate ở **9/12 ảnh** do overlap raster và một đối tượng cạnh ngắn nhỏ; đây không phải 9 lỗi đã xác nhận (biên raster có thể chồng vài pixel). Đây là output converter, không phải ground truth đã xác nhận và không được dùng để train.

Chỉ **3/12 ảnh** đã được xem trực tiếp trong phiên này:

| Ảnh gốc BDD | Nhận xét tự kiểm | Minh chứng local (không commit ảnh) |
|---|---|---|
| `00de5508-00000000.jpg` | Đèn xanh phía trên là đối tượng chưa có Traffic light annotation | `runs/review/bdd12-audit/evidence/8289d7dac9cc7466.png` |
| `00e9be89-00001885.jpg` | Có cụm đèn giao thông chưa gán; pickup output Truck, cần chốt BTC | `runs/review/bdd12-audit/evidence/84a007a7687271f4.png` |
| `1bdba2c3-292b3682.jpg` | Đèn xanh ở giao lộ chưa có Traffic light annotation | `runs/review/bdd12-audit/evidence/b06a3d6ba3d1a0d9.png` |

Không lấy 3 ảnh này để ước lượng error rate toàn BDD; không khẳng định đã rà đầy đủ tất cả object trong từng ảnh. Các nhãn chưa được sửa; **số nhãn sửa bởi phiên QA này = 0**.

## Delta của Claude trong khi review

Main đã tiến tới **bfd434f352c93684fc33da4888310bbd2f5d20b3** khi kiểm tra lại (working tree vẫn sạch). Không merge/rebase/checkout main. Đã đọc diff 8 file: fetch có cap/retry, V1 chuyển sang khoảng 2.000 ảnh mục tiêu và chọn val theo `orig_prefix: validation/`, notebook shell fail-fast. Đây là mục tiêu config, không phải số ảnh đã build.

R01/R02/R03/R05/R06/R07/R08/R09 không đổi source trong delta này. R04: nhánh missing-frozen-key vẫn chỉ WARNING, hardlink vẫn còn; các dòng sau split dịch +6 so với baseline. R10: group/dedup không được thay về logic. Tái hiện runtime trong báo cáo dùng **565d05b**; trạng thái trên **bfd434f** là kiểm tra diff tĩnh, không nhận là đã chạy lại toàn pipeline mới. V1 mới dùng official split giúp việc chọn val ban đầu rõ hơn nhưng không khóa nội dung label.

## Xác minh đã thực hiện

- **7/7 test độc lập của QA CLI pass**: self-cross/bbox, NaN class, bảo vệ input/output, deterministic sampling + Unicode + before/after + input hashes giữ nguyên, sai class order, đổi GT val + exact leakage, overlap + cạnh ngắn.
- `reproduce_risks.py`: hoàn tất 11 kịch bản chẩn đoán; builder thật trên fixture 6 ảnh hoàn tất, official report pass. Đây là bằng chứng validator hình thức không đảm bảo đúng ngữ nghĩa/frozen identity.
- CLI `compare_validation.py --near-hamming 2` chạy trên built fixture: exact/near train-val check pass (5 train/1 val). Lượt CLI này không truyền reference nên **không** xác nhận frozen identity; test riêng đã chứng minh tool bắt được thay đổi bytes label giữa hai reference snapshots.
- Converter Supervisely thật trên 12 ảnh BDD; QA thật trên 10 ảnh synthetic BTC (105 instance; lượt đầu 1 ảnh flag tiny, lượt cuối 10/10 ảnh có candidate overlap/cạnh ngắn/tiny). Đây là cờ tự động, không phải kết luận nhãn mẫu BTC sai; không sửa sample. Số liệu lượt cuối được ghi ở evidence.
- Notebook train baseline chỉ đổi source cell tham số index 6; mọi cell chính thức khác giữ nguyên. Bundle prep khớp source baseline sau chuẩn hóa CRLF/LF.
- Chưa chạy training/export/ONNX/Kaggle, chưa chấm mAP, chưa kiểm chứng CVAT import/export qua dịch vụ thật. Không dùng lại smoke-test cũ trong README như bằng chứng phiên này.
- Evidence số liệu: [reproductions.json](evidence/reproductions.json), [bdd12_conversion.json](evidence/bdd12_conversion.json), [bdd12_label_audit.json](evidence/bdd12_label_audit.json), [sample_label_audit.json](evidence/sample_label_audit.json).

## Thứ tự xử lý để kịp deadline

1. Claude/Lead xử lý hoặc cách ly R01/R02/R04/R05 trước khi nguồn/lô tương ứng vào V2; xem thêm R03 cho cảnh đông xe máy. Đây là đề xuất cho người sở hữu pipeline, không có patch source chính được áp dụng.
2. Khóa val V1 thực tế và bytes nhãn, lấy kết quả baseline thật; duyệt các lô V2 nhỏ có provenance rõ, đủ 11 class.
3. Sửa lỗi mask/association có minh chứng, đo lại cùng val; chỉ thu thêm đúng class AP thấp sau khi có AP thật.
4. Hoàn tất báo cáo số liệu thật và minh chứng; không dùng thời gian sát hạn cho UI mới hoặc dataset lớn chưa rõ quyền sử dụng.
