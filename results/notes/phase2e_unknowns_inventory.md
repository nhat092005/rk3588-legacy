# Kiểm kê điều chưa biết còn lại của giai đoạn 2 (2026-11-01)

Người dùng yêu cầu hoàn thành hết trước khi sang Phase 3. "Hoàn thành" ở đây có nghĩa là mỗi mục có một nhãn cuối kèm bằng chứng đã thử: ĐO, NGUỒN, hoặc "vẫn chưa biết" kèm điều đã tìm và đã đo. Không mục nào bị ép ra câu trả lời khi không có bằng chứng (CLAUDE.md). Mỗi phép đo mới có rule viết trước dữ liệu, tối đa hai thí nghiệm mỗi mục, một lượt advisor trước thiết kế.

Lớp: **U** cần người dùng, **S** ngân sách đã dùng và không có đường mới (nhãn cuối "vẫn chưa biết", ghi lại điều đã thử), **N** đường cụ thể chưa dùng (chạy dưới rule mới), **X** cần bộ công cụ thứ hai hoặc làm sai lệch danh tính hiệu chuẩn (không chạy ngầm, nhãn cuối kèm lý do).

| Mã | Trang nhà | Đã dùng | Đường còn lại | Chi phí | Lớp |
|---|---|---|---|---|---|
| G7 vì sao Resize `asymmetric` vẫn lên NPU | 01 mục 4.5 | Tìm toàn bộ T1 và T2 (2026-10-31), build giống hệt khi đổi giá trị | (a) onnxruntime: hai bộ thuộc tính có cho cùng Resize trên shape YOLO không, (b) micro model ở tỉ lệ ORT cho kết quả khác nhau, so đầu ra NPU với ORT | PC, nếu cần thì board | N (phần "vì sao lên NPU" vẫn không có văn bản) |
| G10 Reshape 8 nút ONNX, 7 dòng RKNN, Split | 01 mục 5.2 | Tìm nguồn | Đối chiếu nút ONNX mà toolkit đã nhận với tên dòng trong bảng layer (mô tả) | PC | N |
| G11 hai guide và Operator List khác nhau về Relu, Concat | 01 mục 6.1 | Concat đã đo chia 15/17 | Micro model Relu đứng riêng, `perf_debug` WorkLoad ở `CORE_0_1_2` | board | N (không phân xử tài liệu nào đúng) |
| M1 quy tắc chia job của `librknnrt` | 02 mục 2.2, 6.1 | Header driver, WorkLoad theo op | Mô tả mẫu chia quan sát được từ `_layers.json` của g4b | PC | N (chỉ mô tả, quy tắc nằm trong binary) |
| M2 `ALL` trong Table 4 của EDGE | 02 mục 2.3, 6.1 | Toàn bài báo và T1 (nghĩa trong bài: hết ngân sách) | Đo `NPU_CORE_ALL` trên board (trang 02 mục 6.1 nêu đường này, bản kiểm kê đầu xếp nhầm vào S): đã làm, chia như CORE_0_1_2 | board | N |
| M4 một context dùng chung nhiều luồng | 02 mục 3.1, 6.1 | SDK guide tr.153, S2 dùng mỗi luồng một context | Thăm dò: ba luồng dùng chung một context, so đầu ra với tham chiếu tuần tự | board, có thể làm kẹt runtime | N |
| M6 `async_mode` | 02 mục 4.2, 6.1 | Đọc `rknn_lite.py`, runtime là `.so` | Thăm dò `init_runtime(async_mode=True)`, so đầu ra với tham chiếu | board | N |
| M7 `rknn_batch_size` | 02 mục 1.3, 6.1 | không | Dựng batch N, so đầu ra với batch 1, thời gian chỉ mô tả | PC và board | N |
| M8 mục tiêu FPS | 02 mục 1.3, 5.4 | Tìm trong docs, không thấy | Người dùng chốt 60 FPS (2026-11-01), kiểm tra ở trang 11 mục 8 | | U (đã xong) |
| M16 số context tối đa | 02 mục 3.1 | không | Tạo context tới khi lỗi hoặc tới ngưỡng bộ nhớ lấy từ M14 | board, rủi ro | N |
| P2 và P3 duyệt | trang 16 | Đã trình bày | Người dùng chọn hướng 1 (2026-11-01) | | U (đã xong) |
| Nguyên nhân `h5` lấy lại điểm | 13 mục 7 | `h1` đến `h4`, `h6`, `c_all`, `kl`, `auto` (rule_diag, rule_recipes, ADDENDUM 3) | không có đường mới trong ngân sách | | S |
| Vì sao `h6` tệ hơn | 13 mục 7 | Một lần dựng, một lần chấm | Dựng lại `h6` một lần, chấm lại (kiểm tái lập) | board | N |
| `mmse` hết bộ nhớ | 13 mục 7 | Một lần dựng (mã 137) | Giảm số ảnh hiệu chuẩn làm sai lệch danh tính hiệu chuẩn (md5) | | X |
| Nguyên nhân 8 hàng FP16 của bản QAT | 15 mục 6 | Bảng layer | Truy vết từ đồ thị TorchScript và log dựng: node nào ứng với hàng nào (mô tả) | PC | N |
| Thời gian NPU và độ phủ đa nhân của bản QAT | 15 mục 3, 6 | không (ngoài ngân sách rule_qat) | `benchmark_npu --variant qat` dưới P2 (nhãn chưa duyệt), hai phiên mỗi mặt nạ | board | N |
| QAT với torch 1.10 đến 1.13.1 | 15 mục 6 | Chuỗi trace, nạp, dựng chạy trên torch 2.4.0 | Cần môi trường torch 1.13.1 cùng python tương thích, venv hiện dùng python 3.12 | | X |
| A2 tách đầu | 15 mục 4, 6 | Hết ngân sách thí nghiệm 1 (đã đóng) | Rule mới: tổng quát hóa `split_model`, dựng, chấm trên board | board | N |

Thứ tự thực hiện (theo advisor): PC trước (G7 a, G10, M1, truy vết QAT), rồi board từng việc một (A2 tách đầu, G11, `h6` lặp, thời gian QAT), cuối cùng M4, M6, M7, M16 (có thể làm kẹt runtime, chuẩn bị khởi động lại `rknn_server`). Kết quả vào trang sinh mới 17, trang 01 và 02 chỉ thêm dòng trạng thái có ngày.


Nhãn cuối của từng mục và số liệu: docs trang `02-tang-npu/17-ket-qua-muc-chua-biet.rst` (2026-11-01). Hai mục chờ người dùng: M8 (mục tiêu FPS) và duyệt P2 và P3 (trang 16). Hai mục không chạy có lý do: `mmse` (đổi danh tính hiệu chuẩn), QAT torch 1.10 đến 1.13.1 (cần bộ công cụ thứ hai).
