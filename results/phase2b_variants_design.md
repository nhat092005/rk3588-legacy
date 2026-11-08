# Giai đoạn 2b: các biến thể activation thay SiLU (thiết kế nháp trước advisor)

Ngày 2026-10-31. Người dùng duyệt: "làm đầy đủ các biến thể luôn" sau khi duyệt hướng 1 (ReLU seed 1, 2). Ngoài goal giai đoạn 2 gốc (mục 11 loại trừ), nay mở rộng theo duyệt này. Nhãn cổng G4 chuẩn không đổi: seed 0 so với SiLU seed 0, INT8 kết luận, 1 seed.

## Ứng viên (chọn theo danh sách op đa nhân của T1: Relu, Clip, Relu6, ThresholdedRelu, PRelu, LeakyRelu, UG232 tr.75, UG152 tr.27)
- V1 LeakyReLU (`nn.LeakyReLU(0.1)`), 100 epoch, seed 0. Op OPL: ConvolutionPRelu/LeakyRelu (3.2.9 tr.61, đa nhân 支持).
- V2 ReLU6 (`nn.ReLU6()`), 100 epoch, seed 0. Op OPL: ConvolutionClip (3.2.8 tr.61, đa nhân 支持) hoặc Relu6.
- V3 ReLU huấn luyện 200 epoch, seed 0 (cùng kiến trúc ReLU, chỉ khác số epoch).
Không thử: Hardswish, GELU, Mish (không nằm trong danh sách op đa nhân của T1, chưa có bằng chứng; có thể thêm bằng việc tìm nguồn riêng nếu người dùng muốn).
Điều kiện khởi tạo: như ReLU, trọng số `yolov8n.pt` (SiLU) nạp vào kiến trúc mới.

## Mỗi ứng viên qua G1 đến G4
- G1: tìm Operator List (doc-reader) mục LeakyRelu/PRelu, Clip/Relu6 và ràng buộc thuộc tính (ví dụ slope), tên op sau fuse trong bảng build (chưa biết trước build).
- G3: export INT8 (danh sách hiệu chuẩn seed 0) và FP16, đọc bảng layer: op CPU ngoài Input, Output, Transpose.
- G4 A, B: một lần chạy xen kẽ dưới P2: {SiLU s0, ReLU s0 (100ep), V1, V2, V3} x {CORE_0, CORE_0_1_2}, 2 phiên, cùng thứ tự hai vòng, 20 phiên. Quy tắc A, B giống rule onboard_g4b (5%, chênh hai phiên, 50% cycles, hàng chia khi ít nhất 2 trong 3 phần trăm WorkLoad > 0).
- G4 C: RKNNLite, INT8 kết luận, FP16 tham chiếu, FP32 mô tả, so với SiLU seed 0: AP50-95, NoHelmet, Vest mỗi cột giảm tối đa 1.0 điểm trên css_test.
- Giới hạn của V3: số epoch khác baseline (200 so 100), cổng không tách ảnh hưởng của activation khỏi ảnh hưởng của số epoch. Đối chứng SiLU 200 epoch được thêm sau advisor (xem cuối file). Nếu V3 đạt C, câu "đổi activation đạt" không đúng, câu đúng là "ReLU 200 epoch đạt".
- Nhiễu seed: giữ SiLU seed 0 đến 2 và ReLU seed 0 đến 2 (đang chạy). V1, V2, V3 chỉ 1 seed.

## Thứ tự và tài nguyên
GPU tuần tự (4 GB): ReLU s1 và s2 (đang chạy), V1, V2, V3 (200 epoch, khoảng 70 phút). Export và chấm mAP trên board ngay sau mỗi lần huấn luyện (board chỉ chạy một việc một lúc, không chạy khi đo latency). Sau cùng chạy xen kẽ latency 20 phiên dưới P2 (cần sudo).

## Nhãn
Mỗi ứng viên: Giữ khi qua G1 đến G4, Loại khi trượt một bước, Chưa biết khi chưa chạy hết. G1 cần mục T1 cùng tên op sau fuse, nếu không có thì tối đa Chưa biết dù A, B, C đạt (giống ReLU với ConvReluAdd).

## Chốt sau advisor (2026-10-31)
- Thêm đối chứng SiLU 200 epoch seed 0 (mô tả). Cổng C của V3 vẫn so với SiLU seed 0 100 epoch. Trend val mAP50-95 trong `metrics.csv` ghi trong `rule_variants.txt`. Patience 100, chạy đủ 200 epoch nếu val mAP còn cải thiện (đọc số epoch thật từ `metrics.csv`).
- Thứ tự cổng: G1 (doc-reader xong) và G3 sơ bộ (build FP16 trên yaml chưa huấn luyện, xong) trước khi huấn luyện. Pre-check: LeakyReLU cho 51 ConvLeakyRelu và 6 ConvLeakyReluAdd, ReLU6 cho 57 ConvClip và 8 Add riêng, CPU chỉ Input, Transpose, Output.
- G1 T1 (doc-reader): ConvolutionPRelu/LeakyRelu 3.2.9 tr.61 `多核联合: 支持`, không ràng buộc slope. ConvolutionClip 3.2.8 tr.61 `支持`. Op đứng riêng Relu, Clip/ReLU6 (tr.84), PRelu, LeakyRelu đều `暂不支持` (mâu thuẫn với UG232 tr.75, không giải quyết). Không có mục fuse với Add cho PRelu/LeakyRelu/Clip. `ConvClip` có trong ví dụ log UG232 tr.56 đến 57. UG232 tr.142 cột "Supported fusion rules": Conv+Relu, Conv+PRelu(LeakyRelu), Conv+Clip, Conv+Add, Conv+Activation+Add (đính chính 2026-10-31 sau fact-check: bản trước ghi nhầm Conv+Add là future plans), phù hợp với ConvReluAdd quan sát được, không phải mục OPL.
- Quy tắc ánh xạ G1 cố định: tên fuse không có Add ánh xạ theo tên, tên có Add (ConvLeakyReluAdd, ConvReluAdd) không có mục nên G1 qua một phần, nhãn tối đa Chưa biết dù A, B, C đạt. ReLU6 (ConvClip, không có fuse Add) có thể đạt Giữ.
- Độ dốc LeakyReLU 0.1: một điểm, quy ước của cfg Darknet YOLO, không phải nguồn T1 hay T2, không quét.
- Latency: lần chạy xen kẽ riêng `bench/onboard_g4v` (20 phiên: SiLU, ReLU, LeakyReLU, ReLU6, ReLU 200ep, mỗi cái `CORE_0` và `CORE_0_1_2`, 2 vòng), `analyze_g4v.py`, `results/g4v_latency_workload.json`. G4B gốc không đổi.
- Trang kết quả: trang sinh mới "Kết quả biến thể activation" (12) trong 02-tang-npu, câu "Mọi biến thể khác của ReLU chưa thử" ở trang 10 sửa trong generator.
- Thứ tự GPU: ReLU s1, s2 (đang chạy), LeakyReLU, ReLU6, ReLU 200ep, SiLU 200ep (hàng đợi `scratchpad/variants.sh`, một waiter nội bộ). Latency P2 chạy sau cùng (cần sudo, board rảnh).
