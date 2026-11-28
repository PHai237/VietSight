# V4 balanced final001

- ZIP local: `C:\My Future\VietSight-data-qa\data\versions\vietsight_mixed_balanced_final001_v4.zip`. Chưa upload Drive.
- Handoff: [v4.json](../configs/handoff/v4.json). Kế thừa toàn bộ 2.539 ảnh train và nhãn V3 final002, đủ 11 class BTC; không sửa V3.
- Mỗi epoch: **4.050 lượt**, **2.025 Mapillary / 2.025 Việt Nam**. Giữ mọi ảnh ít nhất một lượt, tối đa hai lượt. Seed 55.
- Tăng ảnh chứa Motorcycle, Other vehicle, Bicycle theo tần suất ảnh dương tính từng nguồn, dùng Sign làm mốc; hệ số class chặn ở 2, hệ số ảnh lấy max, không cộng các class. Sign vốn phổ biến nên không tăng riêng bằng nghịch tần suất. Chi tiết trọng số và instance trước/sau lấy mẫu trong handoff.
- Giữ nguyên byte hai validation: 262 Mapillary, 283 Việt Nam. Không tạo pseudo-label, thêm nguồn hay xác nhận human QA mới.

**Claude cần chọn `data_weighted.yaml` trong thư mục giải nén**, đặt `path` thành đường dẫn tuyệt đối thư mục đó; giữ `train: train_weighted.txt`. Gán YAML này vào `DATASET["data_yaml"]` sau bước chuẩn bị folder nếu dùng notebook hiện tại. Loader chuẩn bị folder tự viết lại YAML nên chỉ đổi ZIP sẽ **không** bật sampling. Số ảnh unique vẫn 2.539; số mẫu loader/epoch là 4.050. Không cần đổi kiến trúc, loss hoặc trainer. Không chạy lại dedup trên các dòng lặp có chủ ý.

Đã kiểm tra loader YOLO segmentation thật trên CPU: giữ đủ lượt và ảnh unique, nhãn khớp. Không training/GPU. Một epoch dài khoảng **1,60 lần V3**; so sánh theo số optimizer step. Sampling không sửa lỗi coverage/geometry kế thừa và không tạo thêm độ đa dạng cảnh. Đây là định dạng sampling Ultralytics chuẩn; không khẳng định BTC đã phê duyệt riêng phương pháp này.

Fingerprint ảnh/nhãn giữ nguyên V3 vì payload không đổi; **phải kiểm tra thêm `sampling_sha256`** trong handoff để phân biệt V4. Quyền sử dụng và hạn chế từng nguồn giữ nguyên: Mapillary có điều khoản hạn chế chưa tái xác nhận; nguồn Việt Nam công bố CC BY 4.0 nhưng quyền footage bên thứ ba chưa xác nhận. `rights_confirmed=false`, `submission_eligible=false`; READY chỉ về kỹ thuật cho thử nghiệm.
