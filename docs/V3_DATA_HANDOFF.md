# V3 Mixed — bàn giao dataset

**READY cho thử nghiệm về lắp ghép/định dạng**, có cảnh báo V1 kế thừa; không xác nhận nhãn sạch hoàn toàn hay quyền dùng để nộp bài. Không thay đổi ảnh/nhãn V1, V2; không tạo pseudo-label hoặc yêu cầu gán nhãn thêm.

- Version: `vietsight_mixed_final001_v3`.
- ZIP local: `C:\My Future\VietSight-data-qa\data\versions\vietsight_mixed_final001_v3.zip` — **445.889.627 bytes**.
- [Handoff cho Claude](../configs/handoff/v3.json), [số liệu và QA](qa/evidence/V3_MIXED_FINAL.json). ZIP có manifest, checksum, attribution và bằng chứng kiểm tra.
- Train: **2.539 ảnh = 1.469 Mapillary V1 + 1.070 Việt Nam V2**, **24.454 instance**; đủ 11 class BTC, giữ nguyên thứ tự class.
- Val V2: **283 ảnh / 1.658 instance**, `images/val`, cấu hình `data.yaml`.
- Val V1: **262 ảnh / 3.169 instance**, `images/val_v1`, cấu hình `data_val_v1.yaml`. Hai validation giữ nguyên cả ảnh và nhãn, không đưa vào train.
- ZIP hiện ở local; **chưa upload Drive**. URI Drive trong handoff là vị trí dự kiến. Claude dùng `configs/handoff/v3.json`, tạo run V3 riêng và đặt `path` YAML theo thư mục giải nén; không thay handoff/run V2.

## Kiểm tra và hạn chế

Không có ảnh trùng SHA256 hoặc trùng pixel giải mã, nên không loại ảnh qua exact dedup. Không có exact/pixel leakage với cả hai val. Cả 3 ứng viên dHash train–val được xem trực quan bằng AI và xác định khác cảnh/clip; không ghi thành human QA. Không có đầy đủ manifest video/sequence gốc để bảo đảm phát hiện mọi dạng leakage theo chuỗi. Giữ ranh giới clip V2 đã sửa và split chính thức Mapillary của V1.

Định dạng toàn bộ nhãn đã kiểm tra. Giữ nguyên **24 cảnh báo polygon tự cắt ở train V1, 5 ở val V1**, cùng **12 nhãn train và 3 nhãn val rỗng** của V1; không xem các ảnh rỗng là negative đã được kiểm chứng mới. Danh sách cảnh báo gốc nằm trong ZIP `qa/V1_ORIGINAL_STATS.json` và `qa/ASSEMBLY_AUDIT.json`. Đây là bản ghép thử nghiệm có cảnh báo kế thừa, không phải một vòng sửa nhãn V1. Không xác nhận human QA toàn bộ.

| Class BTC | Train V1 | Train V2 | Train V3 |
|---|---:|---:|---:|
| Car | 7427 | 1342 | 8769 |
| Bus | 342 | 176 | 518 |
| Truck | 463 | 390 | 853 |
| Other vehicle | 223 | 31 | 254 |
| Motorcycle | 358 | 375 | 733 |
| Motorcyclist | 263 | 1995 | 2258 |
| Bicycle | 419 | 17 | 436 |
| Cyclist | 381 | 78 | 459 |
| Pedestrian | 2795 | 602 | 3397 |
| Sign | 3502 | 642 | 4144 |
| Traffic light | 2463 | 170 | 2633 |

## Quyền sử dụng từng nguồn

- **Mapillary V1:** hồ sơ nguồn/mirror ghi **CC BY-NC-SA 4.0**, có yêu cầu attribution, phi thương mại và share-alike. [Trang gốc Vistas](https://www.mapillary.com/dataset/vistas) yêu cầu đăng nhập, chưa tái xác minh độc lập điều khoản bản phát hành gốc. [LICENSE của mirror](https://github.com/dataset-ninja/mapillary-vistas-dataset/blob/main/LICENSE.md) là bằng chứng ghi nhận; không thay bằng giấy phép MIT của toolkit hoặc giấy phép ảnh Mapillary nói chung.
- **Việt Nam V2:** [D22CQCN01Ns Workspace](https://universe.roboflow.com/d22cqcn01ns-workspace/dashcam_vietnam) công bố **CC BY 4.0**; thành viên đội gán lại 11 class. **Quyền footage bên thứ ba Camera Giao Thông/VOV chưa được xác nhận riêng**; chưa rõ mức AI hỗ trợ khi gán nhãn. Giữ attribution và các hạn chế từ bản V2.

Handoff ghi `rights_confirmed: false`, `submission_eligible: false`. READY chỉ áp dụng cho bản ghép thử nghiệm, không phải xác nhận được phép nộp hoặc phát hành công khai.

## Hash khóa phiên bản

- Dataset fingerprint: `ac67607864849d42579e4cfd0a695937148e3990adb9036aafb4933488f7a701`.
- Val V2 filename SHA256: `22e40f3164c7f2c52a5038214518acba11bfd6f32233f46a28035e598c8505ac`.
- Val V1 filename SHA256: `60f5198e96f90522115984a616fc02e0dd666f8973d667b5e00d41481cf41421`.
- ZIP SHA256: `3f6533c032618c10c22e787ed0dbe60af96d3ef44cdd125ca3c4257cc3407e16`.

Fingerprint dùng thứ tự Path phân biệt hoa/thường của **Linux Colab/Kaggle**. Đã kiểm chứng bằng helper notebook giữ nguyên bytecode với thứ tự Path Linux, đối chiếu cách tính độc lập và fingerprint V1 Kaggle gốc. Windows native Path sắp xếp ID Mapillary khác nên cho hash khác; không sửa validator. Hash ảnh/nhãn từng validation cũng được lưu trong handoff.
