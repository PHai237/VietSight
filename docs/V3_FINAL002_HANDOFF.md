# V3 final002 — 28/11/2026

**READY cho train thử nghiệm, có cảnh báo V1 kế thừa; quyền nguồn chưa xác nhận đầy đủ.** Giữ nguyên V1, V2 và V3 final001 làm dự phòng. Không tạo pseudo-label, không thêm hay nhân đôi ảnh.

- ZIP: `C:\My Future\VietSight-data-qa\data\versions\vietsight_mixed_final002_v3.zip` — **447.546.521 bytes**, hiện local, **chưa upload Drive**.
- [Handoff Claude](../configs/handoff/v3_final002.json), khớp `v3.example.json`: dùng **val_subsets**, không dùng `validation_sets`. URI Drive là vị trí upload dự kiến.
- Train: **2.539 ảnh / 24.522 instance**, đủ 11 class BTC. Thay nhãn **44 ảnh train Việt Nam**, tăng **68 instance**; **1.026 ảnh** còn lại giữ nhãn V2. Ảnh train Mapillary giữ nguyên.
- Val: **545 ảnh**, hai subset đánh giá riêng: **262 Mapillary / 283 Việt Nam**. Cả ảnh và nhãn val giữ nguyên byte; hash tên và hash nội dung từng subset khớp final001. Cấu hình riêng: `data_mapillary_val.yaml`, `data_vn_val.yaml`. Chỉ số gộp 545 ảnh không so trực tiếp với chỉ số val 283 ảnh của final001; dùng kết quả từng subset.

## Đối chiếu nguồn và chọn nhãn

API key truy cập được `hngs-workspace-htjsl/hackathon-c1o0t`; không cần quyền bổ sung. Nguồn live có **1.407 ảnh / 13.145 instance / đúng 11 class**. Export version 1 có **20.062 instance**, thêm alias `car`, là bản cũ: **không dùng cho nhãn thay thế**. Snapshot live ổn định trước/sau khi đọc, SHA256 `805da9ffd306760a4e6b29f2d29aada995e670a0667664f2521feed4fdc10e6d`.

1.407 tên frame khớp bộ gốc, trong đó 1.353 khớp V2 đã phát hành. 8 ảnh export kiểm tra pixel khớp sau resize; **44/44 ảnh thực sự thay nhãn khớp RGB chính xác với ảnh gốc live**. Không nhập lại 54 ảnh đã loại khỏi V2.

Đã decode toàn bộ mask live; **1.037 ảnh có pixel mask chồng lấn**, nên không nhập nguyên bộ. Chỉ nhận ảnh train không chồng mask nguồn hoặc mask sau serialize, có số instance tăng, mỗi instance cũ có đối sánh cùng class riêng biệt (IoU ≥0,2 và phủ ≥75% mask cũ). Đây là gate duy trì coverage, không phải số đo độ chính xác GT. Dùng encoder hiện có: ≤400 đỉnh, kiểm tra geometry chính thức, roundtrip IoU ≥0,995. Các trường hợp không đạt giữ nhãn cũ.

Đã xem bằng AI 8 cặp nguồn cũ/live và 8 cặp ứng viên trước/sau serialize: live cải thiện coverage, rider hợp nhất, phân loại container ở mẫu; một ứng viên `frame_000653` giữ nhãn cũ vì chưa rõ người/xe. **Không khai human QA hoặc nhãn toàn bộ nguồn đã sạch.** V1 vẫn có 24 cảnh báo polygon train / 5 val, cùng 12 nhãn train / 3 val rỗng. [Số liệu và hạn chế](qa/evidence/V3_FINAL002.json); ZIP có quyết định từng ảnh và minh chứng trực quan.

| Class | V2 Việt Nam đã phát hành, 1.353 ảnh | Live mới, 1.407 ảnh | Phần Việt Nam trong final002, 1.353 ảnh |
|---|---:|---:|---:|
| Car | 1728 | 3690 | 1752 |
| Bus | 196 | 336 | 200 |
| Truck | 433 | 723 | 439 |
| Other vehicle | 35 | 108 | 36 |
| Motorcycle | 506 | 1048 | 507 |
| Motorcyclist | 2635 | 4796 | 2651 |
| Bicycle | 19 | 48 | 19 |
| Cyclist | 90 | 173 | 91 |
| Pedestrian | 780 | 977 | 785 |
| Sign | 844 | 991 | 852 |
| Traffic light | 210 | 255 | 212 |
| Tổng | 7476 | 13145 | 7544 |

## Quyền và kiểm chứng

Hngs Workspace công bố **CC BY 4.0**; chưa xác minh độc lập nhãn được tạo thủ công hay có AI hỗ trợ. Quyền footage Camera Giao Thông/VOV vẫn chưa được xác nhận riêng. Nguồn ảnh V2 và đóng góp gán nhãn trước đó giữ attribution; Mapillary giữ các hạn chế CC BY-NC-SA và điều khoản gốc chưa tái xác minh như final001. Handoff ghi `rights_confirmed: false`, `submission_eligible: false`; READY không xác nhận được phép nộp bài.

Đã qua gate dữ liệu thực tế của generator Claude, gồm `val_subsets` rời nhau và phủ đủ val; toàn bộ ảnh giữ nguyên hash; val và nhãn train Mapillary giữ nguyên hash; CRC/checksum ZIP và fingerprint tính lại từ payload đều đạt. Không sửa notebook, validator, cấu hình train hoặc chạy GPU.

- Dataset fingerprint Linux: `4df8a030527e3764a240abe2c36ab7787f4822974aa8ac0d4f45e245ff94851b`.
- Val Việt Nam filename SHA256: `22e40f3164c7f2c52a5038214518acba11bfd6f32233f46a28035e598c8505ac`.
- Val Mapillary filename SHA256: `60f5198e96f90522115984a616fc02e0dd666f8973d667b5e00d41481cf41421`.
- ZIP SHA256: `7739bcc5579a4a187c9737efda13c279c715f00cd0eb94108f0cf9b15ef67089`.

Fingerprint dùng thứ tự Path phân biệt hoa/thường của Linux Colab/Kaggle như final001; không dùng hash Path native Windows cho ID Mapillary có chữ hoa/thường. Claude dùng handoff **v3_final002.json**, tạo run mới; handoff **v3_final001.json** giữ làm dự phòng.
