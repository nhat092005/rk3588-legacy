# Giai đoạn 2d: QAT và kiến trúc thân thiện lượng tử hóa (thiết kế nháp trước advisor)

Ngày 2026-10-31. Người dùng duyệt làm cả QAT và kiến trúc thân thiện lượng tử hóa (lựa chọn 2). Nguồn: `results/notes/phase2d_source_facts.md`.

## Câu hỏi và tiêu chí
Câu hỏi: có model INT8 thuần (không tensor FP16 trong đồ thị NPU) đạt độ chính xác của FP16 không, không cần `int8h5` lai?
Chuẩn so sánh (viết trước dữ liệu): trên `css_test`, AP50-95 nằm trong khoảng trải seed của FP16 SiLU (1.25 điểm, lấy làm 1.35 như quy tắc cũ để nhất quán) của FP16 seed 0 (49.76), tức từ 48.41 trở lên. Mốc tham chiếu: `int8h5` 49.66 (lai), `int8` mặc định 36.75.
Kỳ vọng thấp cho QAT: `int8h5` đã gần FP16 nên QAT chỉ có thể thay việc giữ FP16 15 tensor, không tăng quá FP16. Khoảng cách ReLU kém SiLU (khoảng 3 điểm) có ở FP16 nên không thể sửa bằng lượng tử hóa.

## Ứng viên
- A1 tách đầu (split head), không huấn luyện lại: xuất SiLU seed 0 với hai đầu ra thô `boxes` (1x64x8400, logit DFL) và `scores` (1x5x8400, logit class), INT8 PTQ cùng danh sách hiệu chuẩn, giải mã (DFL softmax kỳ vọng, dist2bbox, nhân stride, sigmoid) bằng numpy trên CPU board. Bám tiền lệ T1 (demo YOLOv5 chuyển hậu xử lý sang CPU, tr.168 guide). Đo: mAP trên board, latency phần NPU (P2), thời gian giải mã CPU.
- A2 bỏ DFL (reg_max = 1, hồi quy ltrb trực tiếp như YOLO26), huấn luyện lại seed 0 từ yolov8n.pt: INT8 xuất mặc định (giải mã trong đồ thị) và tách đầu. Đo mAP FP16 và INT8, so cổng C với SiLU baseline.
- B QAT: tinh chỉnh SiLU seed 0 với `prepare_qat_fx` (qconfig của guide tr.127), xuất TorchScript, nạp bằng `load_pytorch` với `do_quantization=False`. Khả thi chưa biết: guide khuyên torch 1.10 đến 1.13.1, repo dùng 2.4.0, YOLOv8n cần được FX trace (đầu ở chế độ train trả về dict).

## Thứ tự và ngân sách
1. A1 (rẻ nhất, không huấn luyện) rồi 2. B thử khả thi (trace và nạp một lượt, không huấn luyện) rồi 3. A2 (huấn luyện 100 epoch) rồi 4. B chạy QAT nếu khả thi (một lần). Mỗi ứng viên: tối đa hai thí nghiệm có rule viết trước. Hết ngân sách thì ghi "vẫn chưa biết".
Mọi kết quả ghi "1 seed". Cổng C và nhãn G4 đã duyệt không đổi, các ứng viên này là phần bổ sung dưới nhãn riêng.

## Công cụ cần viết
- `ai/core/rknn/split_head.py`: bọc mô hình thành hai đầu ra thô, hàm giải mã numpy (kiểm tương đương với `Detect._inference` bằng test).
- `ai/board/infer_dump.py`: nhận model tách đầu (metadata có `split_head`), giải mã bằng numpy, ghi cùng định dạng dump.
- `export_npu.py` thêm tùy chọn xuất tách đầu, `build` bằng cùng cấu hình.
