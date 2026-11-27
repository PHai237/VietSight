# VietSight V2 — Local Vision: bàn giao DATA lô 001

**Đã nhận e25 + khai thác VR-TSD:** SHA256 ZIP/ONNX khớp bản e25 public 12,90 trong run report. Đã tải mẫu VR-TSD 40 ảnh / 45 bbox / 0 mask nguồn, dùng 3,75 MB mạng; chọn riêng 20 ảnh camera không trùng round001 và chạy công cụ Claude e25 + COCO trên CPU. Kết quả 7 Sign + 1 Car draft, chưa đủ coverage. Gói dự phòng `C:\My Future\VietSight-data-qa\data\versions\vr_tsd_e25_coco_review001_QA_ONLY.zip`, chưa phân train/val và chưa READY. Chi tiết [VR_TSD_INTAKE.md](qa/VR_TSD_INTAKE.md), phân công [V2_CLAUDE_DATA_COORDINATION.md](V2_CLAUDE_DATA_COORDINATION.md). Thư mục `round001/returns/member_XX/` hiện chưa có lô người kiểm trả về; nhận từng 3–5 ảnh khi có, không đợi đủ cả task.

**Cập nhật round001 CVAT:** đã tách 5 gói, mỗi gói 20 ảnh không trùng nhau (14 train dự kiến + 6 val dự kiến). Tổng 70 ảnh dashcam từ review100 sau chọn/near-dedup, 30 ảnh val từ ba video Đà Nẵng/Hội An/Hà Nội. Đường dẫn gốc: `C:\My Future\VietSight-data-qa\data\versions\v2_cvat_round001\RIGHTS_PENDING\VietSight_V2_CVAT_member_01_REVIEW_ONLY.zip` … `member_05_REVIEW_ONLY.zip`. Hướng dẫn và tiếp nhận từng phần tại [V2_CVAT_REVIEW.md](V2_CVAT_REVIEW.md), hash/số liệu tại [evidence](qa/evidence/V2_CVAT_ROUND001.json). Có 862 polygon draft, gồm 237 đề xuất AI mới bằng công cụ Claude chạy CPU. Không sửa snapshot intake001 bên dưới; chưa có nhãn người kiểm trả về hoặc ảnh đủ quyền gốc để phát hành READY. Người dùng xác nhận chưa có bằng chứng quyền trích Camera Giao Thông/VOV; tất cả gói giữ **REVIEW ONLY / RIGHTS PENDING**.

**Trạng thái: BLOCKED cho training.** Đã có gói dữ liệu thật để kiểm/sửa nhãn, chưa có tập train V2 đạt yêu cầu 11 class BTC. Không đổi trạng thái thành READY chỉ vì polygon hợp lệ hoặc có pre-label. Human QA: **chưa thực hiện/chưa được xác nhận**. Không chạy training, không dùng GPU, không thay V1/validation V1.

Mốc người dùng cung cấp: V1 e25 public **12,90**; e50 **12,37**; đã dùng **2/20** lượt nộp. Không có lần nộp mới trong công việc DATA này. Deadline **10:30 ngày 28/11/2026, Asia/Saigon**.

## Lô đã đóng gói

| Hạng mục | Thực tế |
|---|---:|
| Ảnh ứng viên sau chọn và loại near-redundancy | **377** |
| Train dự kiến, chưa duyệt | **309** |
| Validation Việt Nam dự kiến, chưa duyệt | **68** |
| Ảnh train/val được phát hành READY | **0 / 0** |
| Polygon draft đang bàn giao | **4.279** |
| Mask Sign / Traffic light AI bổ sung trong số trên | **287 / 30** |
| Gợi ý Motorcyclist/Cyclist riêng, chưa nhập GT | **157** |

Nguồn ứng viên là 100 ảnh phương tiện Việt Nam trên GitHub và 279 ảnh SideWalk tại Việt Nam; bỏ 2 ảnh gần trùng/redundant trong quá trình chọn. Không nhập Mapillary, COCO quốc tế hoặc Cityscapes vào lô này. 73 ảnh `pvd2201/data-vn` có queue riêng và **không nằm trong 377 ảnh**.

Thư mục snapshot, từ worktree `C:\My Future\VietSight-data-qa`:

```text
data/versions/vietsight_v2_intake001_review/
  STATUS.json                       BLOCKED + số liệu, teacher, giới hạn
  images/{train_candidates,val_candidates}/
  labels/{train_candidates,val_candidates}/        draft sau AI bổ sung
  native_labels/{train_candidates,val_candidates}/ draft trước AI bổ sung
  source_annotations/               annotation nguồn + attribution/video metadata
  ai_auxiliary_proposals.json        gợi ý rider, không phải GT
  candidate_val_keys.txt             split dự kiến, chưa phải frozen val đã duyệt
  PRIORITY_REVIEW.csv                thứ tự kiểm, reviewer/decision còn trống
  provenance.jsonl, CHECKSUMS.json
  qa/audit.json, qa/review.csv, qa/evidence/
```

Snapshot và các ZIP không commit Git; không sửa đè snapshot này khi một phiên khác đang dùng. Sau sửa nhãn phải tạo phiên bản mới và fingerprint mới.

**Dataset fingerprint**, dùng đúng helper `dataset_fingerprint` của notebook BTC trên toàn bộ `images/` và `labels/` theo relative path + SHA256 bytes:

```text
85a16c8c6a33622ad63d91dff9e6df5483d680ad4cc906622c4969a50278c77b
```

## Đường dẫn và cách chuyển dữ liệu

Các file đã tạo, đường dẫn tính từ worktree:

| File | Byte | SHA256 |
|---|---:|---|
| `data/versions/vietsight_v2_intake001_review.zip` | 58.516.425 | `cf1f9198c5a6cb82ad470f303e71b591fe579fe1cbf93f60755af385731543b9` |
| `data/versions/v2_intake001_train_candidates_CVAT_QA_ONLY.zip` | 40.577.619 | `fdb47faa3632098685b328fdd60dfba5ca93912e5250ecfa3c579d5678fec74a` |
| `data/versions/v2_intake001_val_candidates_CVAT_QA_ONLY.zip` | 10.493.576 | `ea9bb27b320c431ec5e65b6e5ef55eeb7a130bcdd6e53920299935ccf8bd8c76` |

Để người kiểm tra nhận ngay: mở hai ZIP `CVAT_QA_ONLY`, tạo task bằng ảnh trong `images/train/`, rồi upload annotation định dạng **YOLO Segmentation 1.0**. Trong ZIP CVAT, `train` là tên thư mục của định dạng import; split train/val dự kiến của dự án được phân biệt bằng **hai ZIP riêng** và manifest. Không đưa ZIP validation sang task train hoặc dùng làm augmentation train.

Để chuyển sang Colab sau khi quyền sử dụng và nhãn được giải quyết: upload snapshot vào Drive/Colab, kiểm SHA256 trước khi giải nén. Chưa upload/re-publish snapshot lên Kaggle vì hiện còn BLOCKED và nguồn video gốc chưa được đối chiếu đủ quyền. Chưa có URL tải một bản V2 READY; không dùng một link tự tạo chưa tồn tại.

Nếu Claude cần tự tải lại nguồn tại Colab/Kaggle ngay để kiểm dữ liệu, dùng URL GitHub gắn commit ở bảng nguồn và API Roboflow với secret `ROBOFLOW_API_KEY`; **không đưa key vào notebook/public URL/log**. Export đã kiểm thành công: `GET https://api.roboflow.com/khaipixel/sidewalk-srgyh/3/coco-segmentation` với key qua tham số request trong bộ nhớ, lấy `export.link` rồi tải ZIP 25.102.680 byte. Không print response hoặc signed link. Source ZIP SHA256 lưu ở [evidence](qa/evidence/V2_DATA_INTAKE001.json). Tải nguồn không thay thế snapshot/fingerprint vì snapshot còn có chọn ảnh, remap và AI drafts.

## Số instance theo class — draft, chưa phải nhãn đã duyệt

| ID | Class BTC | Train dự kiến | Val dự kiến |
|---:|---|---:|---:|
| 0 | Car | 509 | 198 |
| 1 | Bus | 26 | 0 |
| 2 | Truck | 106 | 0 |
| 3 | Other vehicle | **0** | **0** |
| 4 | Motorcycle | **1.170** | **461** |
| 5 | Motorcyclist | **0** | **0** |
| 6 | Bicycle | 17 | 4 |
| 7 | Cyclist | **0** | **0** |
| 8 | Pedestrian | 1.317 | 154 |
| 9 | Sign | 226 | 61 |
| 10 | Traffic light | 23 | 7 |
| | **Tổng draft** | **3.394** | **885** |

**Không được hiểu 1.631 Motorcycle là 1.631 xe không người đã xác nhận.** Source `motorcycle/motorbike` còn trộn có người/không người, một số mask đã chứa người, một số còn tách person. Cần phân loại lại và gộp đầy đủ rider/passenger; người dắt phải tách Pedestrian + xe. Bicycle/Cyclist tương tự. Các số 0 của Other vehicle/Motorcyclist/Cyclist hiện phản ánh thiếu mapping/annotation đã hoàn chỉnh, không khẳng định ảnh không có đối tượng đó. Đây là blocker thực tế cho coverage 11 class.

287 Sign và 30 Traffic light là **AI drafts**, không phải số nhãn đã sửa đúng. Teacher là checkpoint **V1 e3** tải read-only từ output Kaggle `mangashounen/vietsight-train-vietsight-v1-e3`, không phải model e25/e50; chạy CPU, 2 threads, conf 0,20. SHA256 checkpoint: `d5e5e6ea9ed994a2ab5a5e9fa27946aa628fca458c3aceb5c7bbbfc2fb3c69b8`.

## Nguồn, giấy phép và những phần không nhập lô

| Nguồn | Kiểm đếm/tải thực tế | Nguồn gốc/quyền và quyết định |
|---|---|---|
| [levienbang/yolo11 — review 100](https://github.com/levienbang/yolo11/tree/de7f424f4ea205f5de6bb5243e06dd2d21a585ff/datasets/vietnam-seg-review-100) | 100 ảnh + 100 txt; **540 polygon**: car 179, truck 110, bus 26, motorcycle 225; tải 4.992.939 byte của 206 file | Commit `de7f424f4ea205f5de6bb5243e06dd2d21a585ff`. [Nguồn gốc phan-ky-dat v1](https://universe.roboflow.com/phan-ky-dat/vietnam-traffic-vehicle-detectio/dataset/1) công bố CC BY 4.0; README xác định Việt Nam. Đã xem overview 100/100; nhiều ảnh dashcam. Polygon thực, không phải bbox. Manifest dùng hash pixel BGR có shape, **100/100 khớp khi dùng đúng thuật toán**, không nhầm với SHA file. Nhãn gốc chỉ 4 class, đã bỏ Bicycle và đổi 9 pickup Car→Truck theo quy ước riêng của nguồn; chưa phù hợp để nhận thẳng 11 class BTC |
| [SideWalk — Khaipixel v3](https://universe.roboflow.com/khaipixel/sidewalk-srgyh/dataset/3) | **279 ảnh / 4.715 annotation có segmentation**, ZIP 25.102.680 byte. Gồm cả road/sidewalk/object… nên không đồng nhất với số instance BTC | Publisher công bố CC BY 4.0. Video ID trong tên file đối chiếu được metadata gốc: Đà Nẵng, Hội An, Hà Nội, TP.HCM; đã xem overview 279/279. Đây là góc nhìn người đi bộ, nguồn bổ sung cho xe đỗ/xe đạp; không coi là dashcam. Chưa xác nhận quyền trích video từ tác giả gốc. Mask `object` không tự map Other vehicle; lớp `car` vẫn cần kiểm bus/xe 9–15 chỗ. Cùng video xuất hiện trong cả 3 split gốc, nên bỏ split đó |
| [pvd2201/data-vn](https://www.kaggle.com/datasets/pvd2201/data-vn) | **73 ảnh**, 91.405.460 byte, 38 PNG + 35 JPG, **0 annotation gốc**. Đã tải đủ và xem overview 73/73 | Uploader khai CC0-1.0; nhiều ảnh có watermark báo/video nên cần xác minh quyền gốc và chuỗi ảnh. Không nhập lô 377. Queue riêng `data/processed/vn_team73_cpu_masks_v2/review/`: 73 ảnh, 709 polygon draft sau remap/geometry từ 733 mask COCO-pretrained CPU. Không phải human GT; không dùng làm val/train đã hoàn chỉnh |
| [Jose Vietnam-Traffic-5Classes v1](https://universe.roboflow.com/jose-89vn1/vietnam-traffic-5classes-xthnt/dataset/1) | Catalog/API 13.792 ảnh; ZIP 1.185.538.150 byte **không tải toàn bộ**. Đọc ZIP bằng Range, chỉ lấy annotation test 1.133 ảnh / 12.806 instance và **37 ảnh mẫu**, tổng dữ liệu mạng 6.859.609 byte | Key đã hoạt động, HTTP 200. Listing có 3.219 tên kiểu `leftImg8bit` và 4.395 tên số COCO 12 chữ số; mẫu thật có nhiều cảnh quốc tế cùng một số ảnh chưa rõ nguồn. Không kết luận mọi ảnh trong bộ đều quốc tế, nhưng **không nhập toàn bộ hoặc coi tên “Vietnam” là bằng chứng**. Cần whitelist theo ảnh/video Việt Nam và quyền của nguồn con, nên chưa chọn vào lô đầu |
| [Traffic sign — hoangthanh export v1](https://universe.roboflow.com/hoangthanh/traffic-sign-hemee) | Export thực **354 ảnh / 368 segmentation**, ZIP 16.293.798 byte, khác catalog gốc 150 do bản export/augmentation. **348/368 polygon còn 4 góc sau bỏ điểm đóng**, trong đó 173 axis-aligned | Publisher khai CC BY 4.0. Đã xem hai contact sheets và 2 crop có chữ Việt Nam. Tại `10040`, annotation `No parking` là hình chữ nhật chứa cột và hai tấm biển: vi phạm mask mặt biển/mỗi biển một instance. Không coi mọi tứ giác là bbox sai, nhưng **không nhận mask nguồn này khi chưa sửa**; không có ảnh của nguồn trong lô 377 |

SideWalk: [Đà Nẵng/Milky Walks](https://www.youtube.com/watch?v=7ocqS1NUED4), [TP.HCM/TRAVELLER](https://www.youtube.com/watch?v=Ue_Qm_UTOYw), [Hội An/Walking With Mario](https://www.youtube.com/watch?v=8kMuzBnyP98), [Hà Nội/Urbanza](https://www.youtube.com/watch?v=tb4DAW3BJeU). Các link và metadata xác minh bối cảnh Việt Nam; không tự cấp quyền tái công bố video hoặc xác nhận human QA nhãn.

## Split, QA và giới hạn

- Holdout dự kiến: **toàn bộ 68 frame video Đà Nẵng `7ocqS1NUED4`**, không đưa frame cùng video vào train dù Roboflow gốc chia train/valid/test. Các video khác có canonical group riêng; bộ review100 giữ toàn publication/compilation trong một phía vì chưa có ID từng clip gốc.
- Đã kiểm exact/near-dhash trên nguồn được chọn, loại 2 ảnh gần trùng/redundant, ưu tiên giữ val. Không khẳng định dhash chứng minh độc lập tuyệt đối giữa mọi crop hoặc mọi video chưa biết.
- Giữ V1 val riêng, không lấy làm train V2. SHA frozen list V1 không đổi: `04b9cbb2d7f3ca0058d2eb9e9d9ceac2ddd19c12be5607d2a7faead91b723514`. **Chưa so pixel với 262 ảnh val V1** vì chưa có reference đầy đủ local; cần làm trước release cuối.
- Audit cả 377 ảnh: **295 ảnh có ít nhất một cờ geometry/overlap/size**, đây là cờ review, không phải tỷ lệ nhãn sai đã được human xác nhận. 176 cờ cạnh ngắn khoảng 1,25%, 114 cờ diện tích dưới 64 tại thang 640; hai ngưỡng giữ riêng. Các polygon draft đã xuất qua được kiểm cấu trúc, không có polygon tự cắt trong lô serialized; điều đó không chứng minh đúng ngữ nghĩa/coverage.
- Native review100 trước chuyển đổi có **3 polygon tự cắt**. Trước dedup, hai converter có **9 mask không biểu diễn được** và **47 polygon draft dưới gate round-trip IoU 0,995** (review100: 5/8; SideWalk: 4/39). Các draft dưới gate vẫn là đề xuất trong queue, không được nhận train. Annotation gốc giữ trong `source_annotations/` để phục hồi/sửa; không coi mask mất là đối tượng đã loại hợp lệ.
- Đã xem sâu 3 evidence overlay sau AI bổ sung: vẫn có mặt biển lớn chưa được gán và gợi ý đèn trang trí/biển cửa hàng có thể nhầm class. **AI pre-label chưa bảo đảm coverage**, không được train thẳng.
- Thử builder V2 trên nguồn thật trả **BLOCKED: no reviewed Vietnam images; no trainable dataset created**. Không bypass gate hoặc ghi reviewer giả. Cần sửa các lỗi thực tế và ghi đúng phương pháp/người kiểm tra trước khi release. Human QA hiện chưa có; không dùng chữ “đã duyệt tay”.

## Việc cần làm để phát hành lô train đầu tiên

1. Xử lý trước `PRIORITY_REVIEW.csv`: xe không người, xe đạp, rider/passenger và Sign lớn rõ; kiểm **mọi lớp xuất hiện** trong ảnh. Có thể duyệt từng lô nhỏ, không cần chờ đủ 377 ảnh. Mục tiêu 200–500 ảnh là định hướng, không phải điều kiện để sửa ít thành nhiều bằng augmentation.
2. Sửa Motorcycle/Motorcyclist, Bicycle/Cyclist và người dắt; bổ sung Other vehicle (đặc biệt tách rơ-moóc khỏi đầu kéo), Sign/TL bị thiếu; xe 9–15 chỗ là Car; rà 9 pickup và mask bị che. Không vẽ bbox thành mask, không gán bóng/đèn trang trí/quảng cáo.
3. Xác nhận quyền nguồn video và ảnh nhóm; lưu bằng chứng và attribution. Lưu reviewer thật nếu đã kiểm tay; nếu QA bằng AI/công cụ phải ghi đúng phương pháp và giới hạn, không tự gọi là human QA.
4. Export nhãn đã sửa, dùng converter/QA V2 hiện tại và builder vào **phiên bản mới**; khóa ảnh+nhãn VN val riêng và so với V1 val. Chỉ khi các lỗi/gate đã giải quyết mới ghi **READY** cùng fingerprint và số liệu cập nhật.
5. Với lô READY, chuyển qua Drive hoặc Kaggle **private** phù hợp quyền đã xác nhận, bàn giao URL/hash cho Claude train. Tiếp tục xử lý lô kế tiếp; không ghi đè dữ liệu Claude đang dùng và không dùng thêm lượt nộp khi chỉ chuẩn bị DATA.

Code tiếp tục ở branch `codex/data-qa`, [PR #2](https://github.com/PHai237/VietSight/pull/2); chưa merge pipeline. [Evidence máy đọc](qa/evidence/V2_DATA_INTAKE001.json) chứa số liệu snapshot, hash gói và nguồn. Dataset/ảnh/key/weights không nằm trong Git.
