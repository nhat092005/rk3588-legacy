# Giai đoạn 2c: kết quả tìm nguồn lượng tử hóa (doc-reader, 2026-10-31)

Chưa qua fact-checker. Số trang là trang PDF. Bảng Q-YOLO bị lệch cột khi trích text, cần xác minh bằng mắt trên trang 8 trước khi dùng số.

## T1 (rknn-toolkit2 2.3.2), API = 3rdparty/rknn-toolkit2/doc/03_Rockchip_RKNPU_API_Reference_RKNN_Toolkit2_V2.3.2_EN.pdf, UG = 02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_EN.pdf
- `rknn.config` (API p6 đến 10): `quantized_dtype` w8a8 mặc định (p7), `quantized_algorithm` normal (mặc định), mmse ("usually has higher accuracy than normal", 20 đến 50 mẫu), kl_divergence, gdq (p8), `quantized_method` layer hoặc channel (mặc định channel, p8), `optimization_level` mặc định 3 (p8), `auto_hybrid_cos_thresh` mặc định 0.98 (p10).
- UG p120: "In RKNN-Toolkit2, Per-Channel quantization is only applied to the weights, while the activation values and intermediate values are still quantized using Per-Layer quantization."
- `rknn.accuracy_analysis(inputs, output_dir, target, device_id)` (API p22), đầu ra `error_analysis.txt`, cosine distance (entire_error, single_error), Euclidean distance (UG p8, p31). Chỉ gọi được sau build hoặc hybrid_quantization_step2.
- Hybrid: `hybrid_quantization_step1` và `step2` (API p20 đến 21), chọn lớp float16 trong mục `custom_quantize_layers` của `.quantization.cfg` (UG p124), tự động `build(..., auto_hybrid=True)` (UG p125, API p14).
- Tham số ép đầu ra float trong config hoặc build: không thấy (đã grep `output_tensor_type|output_dtype|output_type|float_out|dequant|per-tensor|per tensor`). Chỉ C API: `want_float` (UG p70, p83).
- QAT: API p13 (load_pytorch hỗ trợ model QAT), UG p125 đến 127 (``prepare_qat_fx``).
- Mâu thuẫn trong T1: `quantized_dtype` (API p7 nhiều giá trị, UG p122 chỉ asymmetric_quantized-8), `do_quantization` mặc định (API p14 True, UG p122 False).

## Ultralytics 8.4.154 (.venv/lib/python3.12/site-packages/ultralytics)
- `utils/export/rknn.py:74`: `mean_values [[0,0,0]]`, `std_values [[255,255,255]]`, `target_platform`. `rknn.py:77-80`: `do_quantization`, `dataset`, `rknn_batch_size`. Không đặt `quantized_algorithm`, `quantized_method`, `optimization_level`, `quantized_dtype`, `auto_hybrid` nên dùng mặc định toolkit (normal, channel, 3, w8a8).
- `engine/exporter.py:1101` (chú thích): "Normalize coordinates by input size so RKNN's per-tensor INT8 scale preserves class scores." Lớp `_NormalizeCoords` ở `utils/export/engine.py:17-45`, đảo ngược lúc suy luận ở `nn/backends/rknn.py:70-80`.

## T2
- QARepVGG (`08_npu_friendly_arch/qarepvgg_quant_2024chu.pdf`) p10: "Swish-like activations are known to have quantization collapse which either requires a delicate learnable quantization scheme to recover (Bhalgat et al. 2020), or to be replaced by RELU6 as in EfficientNet-Lite". p7: "YOLOv6s-RepVGG suffers a huge quantization degradation for about 7.4% mAP via PTQ" (COCO, 8-bit).
- Quant whitepaper (`02_quantization/quant_whitepaper_2021nagel.pdf`) p7: "Although newer activations like Swish functions provide accuracy improvement in floating-point, these may vanish after quantization or may be less efficient to deploy on fixed-point hardware."
- Q-YOLO (`02_quantization/qyolo_ptq_2023wang.pdf`) p8 Table 1 (COCO val2017, W8A8): YOLOv5s real 37.4, Percentile 36.9, MinMax 37.2, Q-YOLO 37.4 (YOLOv5m, YOLOv7 tương tự, mức rơi 0.0 đến 0.5). Cần xác minh bằng mắt.
- YOLOv6 (`yolov6_hwfriendly_2022li.pdf` p8), ActNAS (`actnas_yolo_2025sah.pdf` p2): chỉ đo float, không có INT8. Không bài nào đo SiLU, ReLU6, hard-swish lên mAP YOLO sau PTQ INT8 bằng số.
