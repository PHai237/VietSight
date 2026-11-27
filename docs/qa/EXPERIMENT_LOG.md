# VietSight — Experiment log và mẫu báo cáo cuộc thi

Team seed/ID: **55**. Deadline: **10:30 ngày 28/11/2026, GMT+7**. Model/validator/export do Claude/Lead phụ trách, không thay trong nhánh QA.

Nguyên tắc: chỉ điền metric từ artifact thật; `chưa đo` không phải 0. Phân biệt kết quả GPU training, chấm ONNX theo notebook, public leaderboard và kiểm thử pipeline synthetic. Mọi so sánh cần cùng val revision, không chỉ cùng seed/tên file.

## Bản kê provenance mỗi dataset version

| Trường | V1 | V2 | V3 |
|---|---|---|---|
| Tên dataset/revision + commit pipeline | chưa nhận artifact thật | chưa build | chưa build |
| Source/release/batch IDs và URL chính thức | chờ manifest build | chờ lô A | chờ lô A |
| License ảnh/nhãn + bằng chứng quyền | xem DATASET_AUDIT, còn Mapillary pending | chưa chốt | chưa chốt |
| SHA256 archive/manifest/dataset | chưa đo | chưa đo | chưa đo |
| Converter/mapping revision + quyết định BTC | chưa gắn artifact build | chưa gắn | chưa gắn |
| Ảnh gốc/decode được/được dùng/bị loại | chưa đo | chưa đo | chưa đo |
| Ảnh train/val sau dedup | chưa đo | chưa đo | chưa đo |
| Group/video split và duplicate report | chưa nhận | chưa nhận | chưa nhận |
| Frozen val keys + ảnh SHA + label SHA | chưa nhận reference bất biến | phải khớp V1 | phải khớp V1 |
| Reviewers/correction batch/check chéo | chưa nhận | chưa đo | chưa đo |

Config V1 main quan sát ở `bfd434f` nhắm tới khoảng 2.000 ảnh; đây là **mục tiêu**, không được điền thành số thực tế. QA snapshot của 12 ảnh BDD và 10 ảnh synthetic không phải V1/V2/V3.

## Số ảnh và instance từng class (điền từ dataset sau QA)

Trong mỗi ô ghi `ảnh train / instance train; ảnh val / instance val`. Một ảnh có nhiều class được tính ở từng class, vì vậy tổng ảnh theo class không bằng tổng ảnh unique.

| ID | Class | V1 | V2 | V3 |
|---:|---|---|---|---|
| 0 | Car | chưa đo | chưa đo | chưa đo |
| 1 | Bus | chưa đo | chưa đo | chưa đo |
| 2 | Truck | chưa đo | chưa đo | chưa đo |
| 3 | Other vehicle | chưa đo | chưa đo | chưa đo |
| 4 | Motorcycle | chưa đo | chưa đo | chưa đo |
| 5 | Motorcyclist | chưa đo | chưa đo | chưa đo |
| 6 | Bicycle | chưa đo | chưa đo | chưa đo |
| 7 | Cyclist | chưa đo | chưa đo | chưa đo |
| 8 | Pedestrian | chưa đo | chưa đo | chưa đo |
| 9 | Sign | chưa đo | chưa đo | chưa đo |
| 10 | Traffic light | chưa đo | chưa đo | chưa đo |
| — | Tổng ảnh unique / tổng instances | chưa đo | chưa đo | chưa đo |

Nguồn điền: `<version>_report/stats.json`, kiểm lại actual label files sau accept và built manifest. `summary.json` của converter có thể cũ sau sửa/accept; không lấy làm số cuối nếu chưa đối soát.

## Nhật ký lỗi và sửa nhãn

| Issue/batch | Version/source | Ảnh + label hash + instance ID | Lỗi xác nhận và căn cứ | Mức độ | Người sửa/người QC chéo | Số label thêm/xóa/đổi class/đổi mask | Evidence before/after | Trạng thái |
|---|---|---|---|---|---|---|---|---|
| mẫu — chưa có correction batch thật | chưa ghi | chưa ghi | chưa ghi | chưa ghi | chưa ghi | chưa đo | chưa có | pending |

Quy tắc count: một instance có thể vừa đổi class vừa sửa mask; ghi cả breakdown nhưng không cộng đôi thành tổng instance unique đã sửa. Đếm ảnh lỗi theo ảnh đã kiểm, không theo số flag; tách self-QA và cross-QA. Nhãn thêm do thiếu đối tượng, nhãn xóa do poster/phản chiếu/xe đẩy tay và instance split đầu kéo/rơ-moóc phải có ảnh minh chứng.

| Vòng QA | Tổng ảnh lô | Ảnh đã kiểm | Instance đã kiểm | Ảnh lỗi xác nhận | Instance lỗi xác nhận | Instance unique sửa | Người self-QA/cross-QA | Quyết định |
|---|---:|---:|---:|---:|---:|---:|---|---|
| QA độc lập 27/11 — BDD snapshot (không phải version train) | 12 | 3 xem trực tiếp, 12 kiểm hình thức | chưa kiểm đầy đủ | 3 ảnh có TL chưa gán | chưa kiểm đếm đầy đủ | **0** | Codex self-QA / chưa cross-QA | Hold cho V2, cần bổ sung nhãn |
| Lô V2 thật | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa ghi | pending |

Ba ảnh được chọn từ contact sheets để kiểm, không phải mẫu random đủ ước lượng error rate. 12 ảnh có 121 instance **output converter**, không ghi 121 là số instance đã được người xác nhận đúng. Các lỗi pipeline tổng hợp ở [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md) không tính thành số label lỗi thật đã sửa.

## Kết quả từng vòng huấn luyện

| Run ID | Version/data SHA | Val revision/SHA | Code/model SHA | Seed | Epochs/batch/imgsz | GPU/thời gian thực | Mask mAP50-95 | Mask AP50 | Public LB | Artifacts | Trạng thái |
|---|---|---|---|---:|---|---|---|---|---|---|---|
| V1 | chưa nhận | chưa nhận | chưa nhận | 55 | chờ params thực | chưa đo | chưa đo | chưa đo | chưa đo | chưa nhận run_report.json | chờ kết quả Claude |
| V2 | chưa build | phải cùng V1 | chưa nhận | 55 | chưa chạy | chưa đo | chưa đo | chưa đo | chưa đo | chưa có | kế hoạch |
| V3 | chưa build | phải cùng V1 | chưa nhận | 55 | chưa chạy | chưa đo | chưa đo | chưa đo | chưa đo | chưa có | kế hoạch |

Ghi rõ evaluator: metric `Mask mAP50-95` trên đây dành cho kết quả ONNX chạy theo scorer notebook BTC; nếu ghi metric Ultralytics `.pt` thì thêm dòng/evaluator riêng, không trộn hai loại. AP50 ≠ AP50-95; bbox mAP ≠ mask mAP. Public LB là test server, không phải frozen val.

## AP từng class và so sánh V1/V2/V3

Các ô AP dùng cùng evaluator và thang **0..1**; điểm `100 × mAP` ghi cột riêng nếu cần. Lớp không có GT val hoặc evaluator không trả AP ghi `N/A (không GT/không trả metric)`, không thay bằng 0.

| Class/metric | V1 | V2 | V3 | Δ V2−V1 | Δ V3−V2 |
|---|---|---|---|---|---|
| Car AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Bus AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Truck AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Other vehicle AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Motorcycle AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Motorcyclist AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Bicycle AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Cyclist AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Pedestrian AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Sign AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Traffic light AP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| **Mask mAP50-95** | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Mask AP50 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |
| Điểm 100×mask mAP50-95 | chưa đo | chưa đo | chưa đo | chưa tính | chưa tính |

Trước khi điền Δ: kiểm cùng val key + image SHA + label SHA, cùng class order/scorer/version và cùng tập con chấm. Notebook có thể sample giới hạn ảnh self-check/val; ghi cả `n evaluated / n val` và hash danh sách ảnh thực chấm, không mặc định full val. Thiếu một điều kiện thì ghi **không so sánh trực tiếp**. Đổi epochs/batch/augmentation cùng lúc đổi dataset thì ghi yếu tố nhiễu, không quy toàn bộ tăng điểm cho sửa dữ liệu.

## Mẫu một run (copy khi có artifact)

```text
Run ID / thời điểm GMT+7:
Người chạy / notebook commit / scorer version:
Dataset version / SHA / provenance manifest:
Train images/instances; val images/instances:
Val revision / SHA ảnh / SHA nhãn / hash evaluated image list / n evaluated:
Thay đổi dữ liệu so baseline (giả thuyết kiểm chứng):
Số nhãn thêm / xóa / đổi class / sửa mask / unique instance sửa:
Checkpoint SHA / seed=55 / actual train args / environment:
Thời gian bắt đầu-kết thúc / epochs hoàn tất / GPU thực:
Mask mAP50-95 / AP50 / AP từng class + evaluator:
Public leaderboard + lần nộp + thời điểm (nếu có):
Run report / val_result / results.csv / dataset stats / zip SHA:
Ảnh lỗi trước/sau + đường dẫn + người duyệt:
Kết luận có dữ liệu hỗ trợ / giới hạn / quyết định vòng kế:
```

## Danh mục hình minh chứng

| Figure | Mục đích | Ảnh/source + quyền | Nhãn before/after SHA, instance IDs | Run/val revision nếu là dự đoán | File/ghi chú |
|---|---|---|---|---|---|
| QA-BDD-TL-01 | Chứng minh thiếu Traffic light ở output nguồn | BDD `00de5508-00000000.jpg`, xem license audit | xem `evidence/bdd12_label_audit.json`; **chưa có after thật** | không inference | `runs/review/bdd12-audit/evidence/8289d7dac9cc7466.png`, local |
| QA-BDD-PICKUP-02 | Pickup đang output Truck; cần quyết định cụ thể | BDD `00e9be89-00001885.jpg` | cùng evidence JSON; **chưa sửa** | không inference | `runs/review/bdd12-audit/evidence/84a007a7687271f4.png`, local |
| CORRECTION-01 | Nhãn thật trước/sau sửa | chưa có | chưa có | chưa có | pending |
| VAL-ERROR-01 | GT và dự đoán trên cùng ảnh val | chưa có | chưa có | chưa có | pending |

Ảnh nguồn/weights không commit vào Git. Các panel before/after trong test tự động là **fixture**, không được đưa vào báo cáo như nhãn thật đã sửa. Khi dùng hình trên slide, ghi source/license và phạm vi QA; không công bố dataset hoặc ảnh chưa có quyền.

## Bằng chứng phiên QA này

Không có training run mới trong nhánh QA. Đã thực thi converter thật trên 12 ảnh BDD local, build thật trên 6 ảnh fixture (5 train/1 val), kiểm QA trên 10 ảnh synthetic BTC và các test CLI. Các kết quả chỉ xác nhận phạm vi kiểm tra nêu trong [review](INDEPENDENT_REVIEW.md); không điền chúng vào ô điểm V1/V2/V3.
