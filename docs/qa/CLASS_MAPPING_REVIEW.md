# Đối chiếu toàn bộ class mapping và converter với guideline BTC mới

Ngày: **27/11/2026, GMT+7**. Source đối chiếu: **origin/main `bfd434f352c93684fc33da4888310bbd2f5d20b3`**, fetch trong worktree `codex/data-qa`. Không thao tác trên working directory Claude, không sửa pipeline V1/notebook/validator, không gọi Kaggle hoặc model.

Guideline sử dụng: [đoạn BTC người dùng đã cung cấp](BTC_GUIDELINE_EXCERPT.md) và cập nhật trực tiếp **xe 9–15 chỗ thuộc Car**. Bản guideline ở commit remote được fetch vẫn còn các mục cũ; điều này không phủ nhận guideline mới người dùng/Claude đang có ngoài snapshot remote. Báo cáo không sửa file guideline chung đang do Claude quản lý.

**Kết luận:** thứ tự 11 class đúng, nhưng remap theo tên class nguồn chưa bảo đảm đúng ngữ nghĩa BTC. Cần cách ly để review các xe 9–15 chỗ, pickup, mask gộp đầu kéo/rơ-moóc, người dắt xe/passenger khó ghép, và mask mất topology. Chưa có dữ liệu để tính tỷ lệ sai toàn bộ V1 hoặc mức giảm mAP.

## Ma trận đủ 11 class

Tên source bên dưới là key YAML thật. Toàn bộ key, role và ID đích tiềm năng được xuất trong [mapping/probes JSON](evidence/guideline_probes_bfd434f_final.json), không chỉ các dòng đại diện của bảng này.

| ID/BTC | Mapillary Vistas | BDD10K | COCO2017 | Kết luận theo guideline mới |
|---|---|---|---|---|
| **0 Car** | `car→Car` | `car→Car` | `car→Car` | Đúng ID; thiếu nhánh đưa xe **9–15 chỗ** đang ở source `bus` về Car. Không có bước phân loại pickup bị source gộp vào truck |
| **1 Bus** | `bus→Bus` | `bus→Bus` | `bus→Bus` | Remap thẳng chưa đủ: source bus có thể gồm xe 9–15 chỗ. Cần thông tin loại xe/số chỗ hoặc người duyệt, không đoán số ghế từ bbox |
| **2 Truck** | `truck→Truck` | `truck→Truck` | `truck→Truck` | Đúng cho xe tải/đầu kéo đúng nghĩa. Không tách source mask đã gộp đầu kéo + rơ-moóc; pickup chưa có ngoại lệ review |
| **3 Other vehicle** | `other vehicle`, `caravan`, `trailer`, `wheeled slow` | `caravan`, `trailer` | Không có mapping đủ coverage | `trailer→Other vehicle` phù hợp. `wheeled slow` có thể chứa xe đẩy tay cần bỏ, không gán tất cả vào Other vehicle. Ba gác/xích lô/xe công trình cần kiểm coverage |
| **4 Motorcycle** | `motorcycle` role vehicle2 | Tương tự | Tương tự | Resolver xuất Motorcycle khi không ghép human. Xe đang được dắt phải riêng với Pedestrian; hiện person hình học mạnh có thể bị gộp sai |
| **5 Motorcyclist** | `motorcyclist` role rider, ghép motorcycle | `rider` ghép motorcycle | `person` suy luận + motorcycle | Có union driver + nhiều known riders đúng; thiếu bảo vệ ghép nhầm khi nhiều candidate, weak passenger có thể rơi thành Pedestrian |
| **6 Bicycle** | `bicycle` role vehicle2 | Tương tự | Tương tự | Cùng rủi ro xe bị dắt/association; nguồn không có nhãn subtype xe điện không bàn đạp nên không tự phân biệt được |
| **7 Cyclist** | `bicyclist` role rider, ghép bicycle | `rider` ghép bicycle | `person` suy luận + bicycle | Cùng quy tắc một xe + mọi người trên xe; không phải cứ person chạm bicycle là Cyclist |
| **8 Pedestrian** | `person→Pedestrian` trực tiếp | `pedestrian→Pedestrian` | `person` nếu không ghép | Không có kiểm ngữ nghĩa vật sau kính/người trong ô tô/hình poster nếu source đã gán person; phụ thuộc nguồn và human review |
| **9 Sign** | 4 key sign front/direction/temporary/parking | Không có trong mirror instance này | Chỉ `stop sign→Sign` | Đúng nhóm sign front nhưng cần mỗi tấm một instance, không cột. COCO thiếu biển khác; BDD thiếu Sign, không được train trực tiếp như đủ lớp |
| **10 Traffic light** | 6 subtype traffic light | Không có trong mirror instance này | `traffic light→Traffic light` | Cần mỗi hộp một instance, không đèn xe/trang trí/cột; mapper không tự tách source đã gộp nhiều hộp. 3 ảnh BDD self-QA trước có đèn chưa gán |

Không cấu hình IDD trong repo để kiểm converter/mapping thực thi; ảnh nhóm YOLO nhập theo tên class/alias chứ không có remap ontology riêng. Không tuyên bố hai nguồn này đã pass.

## G01 — P1: xe 9–15 chỗ chưa được đưa về Car khi source gọi là bus

- Code: `configs/sources/mapillary_vistas.yaml:18`, `bdd100k_10k.yaml:20`, `coco2017.yaml:19` đều `bus: Bus`; `scripts/vslib/mapping.py:49-54` chỉ tra label; `scripts/vslib/instances.py:97-99` giữ target trực tiếp.
- Tái hiện: với từng config, tạo instance source `bus` có semantic fixture `seat_count=9`, `12`, `15`; resolver đều trả **Bus**, flags rỗng. Metadata fixture cho biết tình huống dự định, không giả định dataset thật đã có trường `seat_count`; production converter hiện cũng không đọc các tag phân loại này.
- Đối chiếu: theo cập nhật người dùng, cả ba trường hợp phải là **Car**. Đây là lỗ hổng bảo vệ mapping khi nhãn nguồn khác BTC, không phải chứng minh mọi ảnh bus nguồn đều sai.
- Đề xuất cho Claude/V2: tạo review subset source bus/minibus/van, người duyệt hoặc metadata đáng tin quyết định Car cho 9–15 chỗ; lưu decision theo ảnh/instance. Không sửa toàn bộ `bus→Car`, vì bus lớn vẫn Bus. Chưa áp dụng patch pipeline.

## G02 — P1: đầu kéo + rơ-moóc gộp trong source truck không được tách

- Code: source `truck` tại Mapillary:19, BDD:21, COCO:20; `mapping.py:32-40`, `instances.py:97-99`.
- Tái hiện: một source truck mask có semantic fixture `tractor_and_semitrailer` vẫn ra **một Truck**, flags rỗng. Converter không biết ranh giới đầu kéo/rơ-moóc từ tên class.
- Quy tắc BTC: Truck cho đầu kéo và Other vehicle cho rơ-moóc/container, **hai instance riêng**, chỉ phần nhìn thấy. Config `trailer→Other vehicle` đang đúng nếu source đã tách sẵn; không đổi trailer sang Truck.
- Đề xuất: review/split mask tổ hợp trước V2; cần ảnh/segmentation để tách, remap ID không giải quyết được. Không đoán ảnh truck nào có trailer nếu chưa xem; chưa có count lỗi tổ hợp thật.

## G03 — P2: pickup bị kế thừa Truck từ ontology nguồn

- Code: cùng ba mapping `truck→Truck`; ghi chú COCO tự thừa nhận truck gồm pickup, `configs/sources/coco2017.yaml:11`.
- Bằng chứng: 3 fixture source truck có subtype pickup đều ra Truck, không flag; ảnh BDD `00e9be89-00001885.jpg` trong self-QA trước thực sự có pickup output Truck.
- Không có bước kiểm ngoại lệ pickup. Đoạn guideline được gửi trong chat chưa nêu đích danh pickup, nên báo cáo không tuyên bố mọi pickup đều phải Car chỉ dựa vào số ghế. Cần áp quyết định pickup trong guideline đầy đủ đã được duyệt theo đúng kiểu xe; không dùng source truck như kết luận BTC.
- Đề xuất: queue pickup trong truck/các loại xe liên quan, gán theo guideline hiện hành, giữ bằng chứng instance-level. Không remap mọi truck thành Car.

## G04 — P1: có union người ngồi sau, nhưng chưa xử lý đủ ambiguity/xe bị dắt

- Code: `instances.py:65-86,90-94`. Known rider được ghép rồi union nhiều người là logic đúng; không phụ thuộc xe có đang chuyển động nên trường hợp dừng đèn đỏ không bị loại bằng vận tốc.
- Positive control mới: một motorcycle + driver + passenger đều role rider → **một Motorcyclist**, mask bằng union đủ cả ba phần, không mất người.
- Các counterexample đã chạy lại: hai xe cùng điểm vẫn chọn một; person chồng yếu xe đã có owner được output Pedestrian không flag; person hình học mạnh bên bicycle có thể thành Cyclist không flag.
- Đề xuất: lưu candidate associations, review tie/weak person quanh xe đã có owner; người dắt xe bắt buộc Pedestrian + xe riêng. Fixture chỉ chứng minh quyết định/thiếu review flag; tư thế trong ảnh thật phải do người xác minh.

## G05 — P1: mask giao nhau chưa được chặn ở mọi đường chuyển đổi

- Supervisely: `convert_supervisely.py:128-156` có trừ occluder; painter phụ thuộc thứ tự mirror, BDD bottom phụ thuộc heuristic đáy. Không mặc định hai heuristic luôn đúng visible ownership.
- COCO: `convert_coco.py:105-110` nhận mask từng instance rồi resolve; `instances.py:97-105` không trừ/chặn overlap giữa direct objects. `convert_yolo.py:80-98` cũng không kiểm quan hệ overlap giữa các instance; `prelabel.py:80-95` fill chỉ lọc overlap >30%, không clip phần giao hoặc cập nhật occupied sau mỗi mask mới.
- Tái hiện resolver mới: Car và Truck giao nhau **3.600 pixel**, flags rỗng. Đây là input đã chồng, chứng minh thiếu gate; không đồng nghĩa converter phải tự chọn z-order khi input không đủ dữ kiện.
- Đề xuất: gửi review khi có overlap đáng kể, xác định vật phía trước theo ảnh/mask gốc; không tự erode cả hai mask hoặc xóa vật sau. QA độc lập có cảnh báo raster; cạnh chung vài pixel không tự tính thành lỗi GT xác nhận.

## G06 — P1: một instance nhiều mảnh/lỗ có thể mất pixel sau polygon hóa

- Code: `geometry.py:69` chỉ contour ngoài, `:75` loại mảnh nhỏ theo tỷ lệ; `writer.py:90-106` fallback largest part/bỏ polygon không chuyển trạng thái review.
- Positive control mới: hai component lớn, cùng instance, bridge hợp lệ theo notebook; cả hai phần còn pixel sau decoder, IoU fixture **0,96775**. Vì vậy không kết luận mọi multipart đều hỏng.
- Counterexamples chạy lại: lỗ 80×80 bị lấp **6.400 pixel**, round-trip IoU **0,7375390625**; mảnh phụ 10×10 mất dù là pixel visible của cùng instance. Code vẫn `ok`. Multiple COCO polygons còn có lỗi fill even-odd làm mất vùng giao giữa các phần của **cùng** instance (R06 review trước).
- Đề xuất: gate bằng decoder thật, giữ mọi visible component/hole, review thay vì tự giảm thành phần lớn nhất; không tách một vật bị cột che thành nhiều instance. Không sửa validator để hợp thức hóa polygon lỗi.

## G07 — P1: diện tích 64 px² không thay được cạnh ngắn 1,25%

- Code: `instances.py:100-104`, `writer.py:85-86,100-102`, `convert_yolo.py:85-88` dùng ngưỡng **diện tích** ở thang 640; chưa có gate/decision theo cạnh ngắn guideline. Config crowd/group còn dùng ngưỡng diện tích **256** tại `convert_supervisely.py:160`, không phải ngưỡng scorer 64: không khẳng định mọi vùng nhỏ hơn 256 đều được chấm bỏ qua.
- Cạnh ngắn guideline: `min(bbox_width,bbox_height)/max(image_width,image_height) <~0.0125`. Ở thang 640, tương ứng cạnh ngắn khoảng **8 px**, là đại lượng chiều dài.
- Scorer: raster mask area sau quy đổi thang 640 **<64 px²**; đây là diện tích foreground, không phải cạnh ngắn hoặc diện tích hộp. Notebook helper dùng shoelace polygon như phép kiểm hình thức; có thể khác raster ở biên.

| Fixture tại ảnh 640×640 | Cạnh ngắn / cạnh dài ảnh | Foreground decode pycocotools | Theo cạnh ngắn | Theo area scorer |
|---|---:|---:|---|---|
| Hình chữ nhật 100×4 | 0,625% | 400 px² | nhỏ hơn ngưỡng xấp xỉ 1,25% | không dưới 64 |
| Mask chữ L mảnh trong hộp 20×20 | 3,125% | 39 px² | không nhỏ hơn 1,25% | dưới 64 |

- Hai quy tắc **không tương đương theo cả hai chiều**. Không đổi validator 64 thành ngưỡng cạnh; không dùng bbox area làm mask area. SourceWriter có hai lần lọc theo raster area và shoelace area; cần ghi rõ vì sao bỏ object để không lẫn guideline với scorer.
- QA độc lập đã tách `short_side_ratio`, `short_side_at_640` và `area_at_640`; cạnh tính từ bbox tọa độ liên tục, không cộng 1 pixel do rounding. Các cờ chỉ là candidate ở vùng xấp xỉ ngưỡng, không tự xóa nhãn.

## Những class/đường nhập còn cần kiểm completeness

- BDD instance mirror thiếu Sign/TL; COCO thiếu coverage Other vehicle và các Sign không phải STOP. `prelabel.py:87` chỉ đưa review khi có prediction mới nên ảnh không prediction vẫn có thể `ok`. Đây là R01 còn hiện hữu trên source được fetch.
- Sign mapping không chủ động nhập pole/traffic sign frame; không thấy rule trực tiếp gán pole thành Sign. Tuy nhiên mapper không tự tách nhiều tấm biển/hộp đèn trong một source mask; phải xem nguồn đã instance đúng chưa.
- `unmapped: ignore` ở cả 3 nguồn: hợp lý cho nền nhưng nếu nguồn có subtype đối tượng BTC chưa nằm trong mapping thì object sẽ mất mà ảnh còn. Kiểm unmapped summary trước build; không coi "không có key" là "không có object".
- YOLO import remap theo tên, giữ quyết định ngữ nghĩa của người gán; tên đúng không chứng minh 9–15 chỗ/rider/occlusion đúng. Geometry hỏng bị bỏ nhưng ảnh vẫn `ok` là R05; review accept chỉ parse, chưa chặn tự cắt.
- Không có đường chuyển đổi nào tự chứng minh loại trừ đúng người sau kính, poster, reflection hoặc handcart nếu ontology nguồn đã gộp vào target class. Đưa tình huống này vào manual QA, không thêm classifier chưa đo.

## Xác minh và phạm vi tích hợp

- Xuất đầy đủ mapping YAML qua loader hiện có, so thứ tự class với notebook; runtime probes dùng resolver/geometry/decoder gốc, không patch source.
- **8/8 test QA pass**, gồm test hai ngưỡng độc lập; test trước đó bảo vệ source khỏi ghi đè, Unicode, class order, self-cross, overlap, frozen label identity và exact leakage.
- Đã chạy lại `reproduce_risks.py` trên code cập nhật từ `origin/main`, converter thật với 12 cặp BDD đã sao chép trong worktree và builder thật trên fixture nhỏ; xem [verification_bfd434f.json](evidence/verification_bfd434f.json). Không chạy training/export/inference/Kaggle.
- Các số probe là fixture, không là error rate/mAP hoặc label correction thật. Source data/weights/ảnh không được commit, chỉ báo cáo/hash/số đo QA.
- Diff PR chỉ được chứa `docs/qa/**` và `scripts/qa/**`; mọi đề xuất trên là báo cáo cho Claude/V2, **không được thực thi lên V1 đang chạy**. Chỉ merge khi diff và conflict check xác nhận đúng phạm vi người dùng đã cho phép.
