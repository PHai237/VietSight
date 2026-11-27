# Sign V1 — kiểm tra trực tiếp và sửa chuyển đổi V2

Ngày: 27/11/2026, GMT+7. Public AP Sign **6,05**, tổng V1 **12,90**, do người dùng cung cấp là kết quả public chính thức. Trong lần fetch cuối, `runs/RESULTS.md` và `runs/v1_colab_e25/run_report.json` mới của Claude đã gắn public này với **v1_e25, Colab T4, 25 epoch**, ONNX SHA `87c81673e92feeea0e194d15bb7d69b81c783328b8ca7133b1a2ea5159b607eb`. Chưa có submission ID hoặc SHA file `.pt`; không gắn với e3/e20.

Artifact e25 ghi local val đầy đủ **262 ảnh**, mask mAP50-95 **0,2297068910** (score 22,970689), Sign AP **0,2912839787** (29,13 trên thang 100). Sign giảm khoảng **23,08 điểm** từ val sang public; đây là khác biệt hai tập đánh giá, **không phải delta của một thí nghiệm có kiểm soát**. Nó phù hợp với giả thuyết domain/scale/coverage cần ưu tiên hơn tăng ảnh ngẫu nhiên; chưa xác định nguyên nhân causal. Không chạy lại model trong phiên QA.

## Dữ liệu thực sự đã lấy

Kaggle output: `mangashounen/vietsight-prep-vietsight-v1`, prefix `versions/vietsight_v1/`. Đã đọc API listing, tải **1.469 nhãn train + 80 ảnh train**, tổng **27.313.117 byte**. Chọn round-robin seed 55: 20 ảnh cho Motorcycle, 20 Other vehicle, 20 Sign, 20 Bicycle; ảnh có thể chứa nhiều lớp. Đây là mẫu có chủ đích, không dùng 80 làm mẫu số ước lượng lỗi toàn dataset.

Đã xem overview gốc/overlay của **80/80 ảnh** qua 7 contact sheets; xem sâu các crop Sign đại diện, gồm 4 crop hao hụt biên và một ảnh có 2 Sign rất mảnh. Chưa kiểm chéo bởi người thứ hai và chưa xác nhận từng object trong cả 80 ảnh là đúng/sai.

Input notebook được metadata xác nhận là `kaggleprollc/mapillary-vistas-image-data-collection`. Chỉ tải config V2 và **13 JSON annotation gốc** cho Sign (12 ảnh đối chiếu + 1 ảnh anomaly), khoảng **998.601 byte** gồm config; thêm 4 JSON spot-check Other vehicle ở thư mục riêng. **Không tải toàn bộ Mapillary, không tải ảnh val, không ghi lại V1/frozen val, không dùng GPU.**

Dữ liệu/ảnh/nhãn gốc ở `data/raw/v1_kaggle_qa80/` và `data/raw/mapillary_sign_reference/`, đều gitignored. Không đưa dataset hoặc signed download URLs vào Git. Metadata/hash/số liệu tổng hợp: [V1_SAMPLE80_SUMMARY.json](evidence/V1_SAMPLE80_SUMMARY.json).

## 3.502 instance không tương đương 3.502 ví dụ độc lập, đủ lớn và đúng domain

Nhãn tải về khớp toàn bộ số đếm class trong `data/versions/vietsight_v1_stats.json`: Sign **3.502 trên 1.113 ảnh train**, trung bình khoảng **3,15 Sign/ảnh có Sign**. V1 gồm 1.469 ảnh train và 262 val. Không có căn cứ từ count để kết luận model đã thấy đủ mặt biển, kích thước, bối cảnh hoặc kiểu biển Việt Nam.

Trong **80 ảnh mẫu** có **233 Sign**:

| Cạnh ngắn Sign tại thang 640 | Instance trong mẫu |
|---|---:|
| <8 px (dưới khoảng 1,25% cạnh dài ảnh) | 43 |
| 8–<16 px | 104 |
| 16–<32 px | 62 |
| ≥32 px | 24 |

**147/233 Sign trong mẫu có cạnh ngắn dưới 16 px**; 43/233 dưới ngưỡng cạnh ngắn guideline. Đây là đo geometry trên mask V1 đã lưu, không phải số nhãn xác nhận sai. Không ngoại suy tỷ lệ này lên 3.502 instance.

V1 báo `tiny_total=0` vì lọc theo **diện tích 64 px²**, vẫn có thể giữ biển rất mảnh có chiều dài lớn. 43 instance trên không mâu thuẫn với báo cáo đó. Ngưỡng cạnh và diện tích không thay cho nhau; V2 báo riêng, không sửa scorer.

## Mapping Sign: không thấy lỗi đảo ID hoặc gán thẳng cột vào Sign

- ID 9 khớp notebook và config.
- 4 subtype được map Sign: `traffic sign (front)`, `traffic sign - direction (front)`, `traffic sign - temporary (front)`, `traffic sign - parking`.
- `pole`, `traffic sign frame`, `signage - advertisement`, `signage - store` không map Sign. Không có bằng chứng từ code rằng mọi Sign bị chứa cột.
- 42 source instance ghép được trong nhóm 12 ảnh gồm **29 front, 12 direction-front, 1 parking**. Không có temporary-front trong nhóm này; không nhận là đã kiểm đủ subtype đó.
- Thấy nhiều biển ngoài Việt Nam và kiểu direction/parking/biển rất nhỏ trong overview. Khác domain là giả thuyết hợp lý, chưa có phép đo causal hoặc thống kê quốc gia.
- Hai cờ `sign_tall_check_pole` cùng ở `Akns91f-PPxvqXYitVXIlQ`, dòng 4/5: bbox khoảng 14×93 và 5×39 pixel ở ảnh 1024×765. Đã lấy annotation gốc và xem ảnh; vẫn để review ngữ nghĩa/biên, **không tự kết luận cột bị gán Sign**. Poster quảng cáo lớn trong ảnh không được tô như một Sign mặt trước.

## Hao hụt biên khi chuyển sang YOLO đã đo được

Đã tái dựng visible mask từ polygon Mapillary gốc ở kích thước ảnh V1, dùng cùng rasterizer/painter order của pipeline để kiểm tra **độ trung thực chuyển đổi**. Reference này không chứng minh annotation Mapillary đúng tuyệt đối theo BTC.

Ghép theo IoU giữa source instance và mask V1 lưu: **42/42 Sign trên 12 ảnh ghép được**, không có source Sign đủ area được xét mà không ghép được trong nhóm này.

| Kiểm tra cùng 42 Sign | V1 hiện tại | Bộ mã hóa V2 |
|---|---:|---:|
| IoU trung vị với visible mask reference | **0,9229299425** | **1,0** |
| Số mask IoU <0,95 | **30/42** | **0/42** |
| Số mask IoU <0,75 | 0/42 | 0/42 |
| Qua gate round-trip ≥0,995 | chưa dùng gate này | **42/42** |

Ví dụ:

| Source ảnh / instance | IoU V1 | Pixel mất V1 | IoU V2 |
|---|---:|---:|---:|
| `1kRTBIZtpt_iexwaAVkbdQ` / 36 | 0,8339 | 46 | 1,0 |
| `0payhcBxl43WlJSdnrbOww` / 78 | 0,8450 | 31 | 1,0 |
| `2EG-9qqAB81t0bjWF0iS9Q` / 66 | 0,8491 | 43 | 1,0 |
| `10rQTVeg4rO8Cewm-HY-JA` / 24 | 0,8586 | 28 | 1,0 |

V1 lấy contour theo tâm pixel rồi normalize +0,5; decoder theo polygon-edge làm co biên đáng kể tương đối với biển nhỏ. V2 truy vết biên **cạnh ô pixel**, giữ orientation của lỗ/component, nối cùng instance bằng bridge rồi kiểm bằng validator và decoder thật **sau khi làm tròn tọa độ 6 chữ số**. Không fallback largest part. Nếu không thể biểu diễn bằng polygon hợp lệ ≤400 điểm đạt gate thì ảnh vào review, không âm thầm bỏ mask.

Số liệu: [SIGN_REFERENCE12_COMPARISON.json](evidence/SIGN_REFERENCE12_COMPARISON.json). Crop 4 panel (gốc/source visible/V1/V2) ở `runs/review/sign_reference12_zoom/crops/`, local và gitignored.

**Không được kết luận sửa biên sẽ đưa AP từ 6,05 lên một giá trị dự đoán.** Chưa chạy V2 hoặc public evaluation. Chưa đo lỗi chính xác trên test BTC, ảnh hưởng thay số epoch/checkpoint, lỗi dự đoán bỏ sót/nhầm lớp, hay tỷ trọng từng nguyên nhân. Đã kiểm toàn bộ nhãn train: **24 polygon tự cắt**, khớp phần train của báo cáo 29 lỗi cả train+val; **0 Sign tự cắt**. Vì vậy self-intersection không giải thích riêng AP Sign trong snapshot này.

## Completeness và topology

- Overview 80 ảnh và crop reference chưa đủ khẳng định mọi Sign đủ lớn đều được gán. Không ghi missing-label rate chưa đo.
- 42/42 reference Sign đều một connected component; nhóm này **không kiểm chứng lỗi multipart Sign thật**. Lỗi mất lỗ/mảnh ở V1 đã có fixture; V2 test donut, mảnh nhỏ và component chạm chéo giữ nguyên pixel.
- Mapping “mỗi source object→một instance” chưa chứng minh mỗi tấm biển/hộp đèn được tách đúng. Ảnh nhiều biển, biển chồng và góc nghiêng phải review từng tấm. Không gộp nhiều biển thành hull.
- Với Sign Việt Nam, chọn các biển trước mặt/cạnh ngắn đủ lớn, mask bám mặt biển, không cột, có nhiều kích thước và điều kiện chụp; vẫn kiểm các lớp còn lại trong cùng ảnh.

## Việc V2 đã thay đổi và bước đo tiếp

1. Bộ mã hóa pixel-edge + round-trip gate riêng V2 đã sửa hao hụt trên nhóm 42 Sign; V1 và validator không đổi.
2. Converter V2 hỗ trợ COCO mask, Supervisely và polygon Mapillary gốc. Ảnh thiếu named reviewer + xác nhận coverage 11 class ở trạng thái review, kể cả không có object/prediction thêm.
3. Gate ngữ nghĩa/rider/overlap/short-side/area tách biệt, không tự lấy source class làm BTC truth.
4. Batch V2 chỉ chọn ảnh **đã duyệt** chứa Motorcycle, Other vehicle, Sign hoặc Bicycle; round-robin, giới hạn lô, dedup và giữ bytes val V1. Chưa có batch nhãn VN đã duyệt trong phiên này.
5. Đo hiệu quả bằng cùng frozen val và checkpoint/evaluator rõ ràng sau khi Lead chạy V2. Giữ public V1 làm tham chiếu riêng, không trộn với local val.
