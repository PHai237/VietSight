# V2 — phối hợp DATA và đề xuất nhãn e25 + COCO

Mục tiêu người dùng: có bản READY đầu tiên trong tối **27/11/2026, Asia/Saigon**. Đây là mục tiêu, chưa phải cam kết đã có dữ liệu đủ điều kiện.

## Claude cần cung cấp

**Đã nhận e25:** `data/raw/prelabel_models/vietsight_v1_e25.zip`. ZIP SHA256 `f0b874634ed90bcacd23238c3eb517fa3169bcbefd149a34f6f103a62c56e0de`; `model.onnx` SHA256 `87c81673e92feeea0e194d15bb7d69b81c783328b8ca7133b1a2ea5159b607eb`, cả hai khớp `runs/v1_colab_e25/run_report.json`.

**Phân công inference:** Codex DATA nhận riêng batch `vr_tsd_e25_coco_review001`, 20 ảnh chụp thật mới, input tại `data/versions/vr_tsd_e25_coco_review001_input/manifest.json`; output riêng `data/versions/vr_tsd_e25_coco_review001_proposals/`. Không chạy lại batch này bên Claude. Không chạy lại 100 ảnh round001 đang sửa; đề xuất e3 trước đây vẫn được ghi đúng là e3. Input mới chỉ có bbox Sign làm hint, không có GT segmentation; ba ảnh minh họa biển trên nền trắng đã thấy trong mẫu bị loại khỏi batch này.

- Checkpoint **e25 `best.pt` hoặc `submission.zip`**, kèm run/epoch/hash để phân biệt với e3/e50. Nơi nhận riêng: `C:\My Future\VietSight-data-qa\data\raw\prelabel_models\`. Yêu cầu nhận file đã được đáp ứng qua người dùng, hash xác minh ở trên.
- Nếu Claude chạy ở Colab: nhận tập ảnh và nhãn nguồn ở worktree này, xuất proposals vào thư mục **mới**, giữ filename, `proposals.jsonl`, class names và checksum. Gửi lại đường dẫn thật, không sửa nhãn CVAT của nhóm hoặc ZIP đã phát.

Ảnh round001: `data/versions/v2_cvat_round001/staging/images/train/`; nhãn trước AI: `staging/labels/train/`. Đã có năm gói giao nhóm, nên **không import chồng đề xuất mới vào các task đang sửa**. So sánh e25 + COCO như tài liệu bổ sung cho ảnh chưa sửa hoặc lô tiếp theo. Dòng AI trong `proposals.jsonl` là index 0; không dùng để đánh dấu reviewer.

Dùng `scripts/propose_labels.py` hiện có, backend `vietsight:<e25 file>` + `coco:data/raw/prelabel_models/yolo26n-seg.pt`, `--classes` đủ 11 class, `--all`. Máy local chạy CPU, giới hạn threads, không train hoặc dùng GPU. Giữ tất cả `pseudo_unreviewed`; giải quyết topology và coverage bằng converter/QA V2 hiện có sau khi người kiểm sửa. Không cần phát triển thêm công cụ.

## Quyền nguồn và validation

- Review100/SideWalk: có license CC BY 4.0 của publisher; quyền footage gốc chưa đủ bằng chứng, giữ RIGHTS_PENDING dù nhãn đã sửa.
- VR-TSD: ưu tiên nguồn do tác giả mô tả tự thu thập và công bố CC BY 4.0. Chỉ có bbox; cần mask thực và kiểm nguồn từng ảnh, không chuyển bbox thành polygon chữ nhật. Giữ split chưa phân công cho đến khi xác định nhóm cảnh/video; không dựa vào train/test Roboflow để chứng minh độc lập.
- Round001 dành riêng cả ba group Đà Nẵng/Hội An/Hà Nội cho val; không lấy frame khác của chúng để train. Không đổi V1/frozen val. Bản release mới phải kiểm trùng với V1 val khi đủ ảnh reference.

## Nhận lô 3–5 ảnh

Nơi nhận: `C:\My Future\VietSight-data-qa\data\versions\v2_cvat_round001\returns\member_01\` … `member_05\`.

Mỗi lần trả: ZIP COCO segmentation revision mới + `RETURN_REVIEW.csv` có reviewer thật, chỉ đánh dấu các filename đã rà đủ 11 class. Giữ phần còn lại pending kể cả khi ZIP export toàn task. Codex kiểm ngay từng lô thực tế khi nhận; không mặc định cả task đã duyệt, không tự điền tên người kiểm.

Chỉ phát READY khi quyền/nguồn, mask, coverage, split và QA đạt yêu cầu; đóng version mới có fingerprint và URL/path thật. Không ghi đè snapshot/gói thành viên đang dùng. Các lô nhãn đạt nhưng quyền chưa giải quyết được giữ riêng để không làm mất công sửa của nhóm. Cập nhật [V2_DATA_HANDOFF.md](V2_DATA_HANDOFF.md) khi thực sự có bản phát hành.
