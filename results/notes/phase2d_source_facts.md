# Giai đoạn 2d: nguồn cho QAT và kiến trúc thân thiện lượng tử hóa (doc-reader, 2026-10-31)

Chưa qua fact-checker. Trang PDF 1-based. Đường dẫn gốc từ gốc repo.

## QAT trên RKNN-Toolkit2 2.3.2 (T1)
- SDK guide `3rdparty/rknn-toolkit2/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_EN.pdf` mục 6.4, tr.125 đến 130. tr.127: "RKNN currently supports quantized models generated from the FX interface, specifically the prepare_qat_fx interface and its associated interfaces."
- API `03_Rockchip_RKNPU_API_Reference_RKNN_Toolkit2_V2.3.2_EN.pdf` tr.13 (load_pytorch): "Support the Quantization Aware Training (QAT) model, but need update torch version to '1.9.0' or higher." Mô hình phải ở dạng TorchScript.
- SDK guide tr.156: "For quantized models (QAT, PTQ), we recommend using PyTorch 1.10~1.13.1 to export the model and upgrade the PyTorch version that RKNN-Toolkit2 depends on to 1.10~1.13.1." Repo hiện dùng torch 2.4.0 (cho cả huấn luyện và rknn-toolkit2).
- Khi nạp model QAT, đặt `do_quantization=False` (`examples/pytorch/resnet18_qat/README.md:18`).
- qconfig của guide tr.127: activation FakeQuantize với MovingAverageMinMaxObserver, quant_min 0, quant_max 255, reduce_range False. Trọng số per-channel affine, int8 (-128 đến 127). Guide tr.128: không dùng per_channel_symmetric vì "RKNN hardware supports non-zero zero_point". Guide có mâu thuẫn (code 0 đến 255, lời văn -128 đến 127).
- Op không lượng tử hóa được giữ FP32, chèn `exDataConvert` làm tăng độ trễ (tr.128 đến 129). Không có danh sách op cụ thể.
- Ví dụ `examples/pytorch/resnet18_qat/test.py` nạp mô hình quantized torchvision đã trace, không gọi `prepare_qat_fx`.
- Guide khuyên QAT khi PTQ và hybrid không đạt yêu cầu (tr.125 đến 126).

## Tách giải mã khỏi đồ thị NPU (T1)
- rknn_model_zoo không có trong repo. Không thấy ví dụ YOLOv8 hay DFL.
- `3rdparty/rknn-toolkit2/rknpu2/examples/rknn_yolov5_demo/README.md:13`: "Due to hardware limitations, the demo model moves the post-processing part of the yolov5 model to the cpu implementation by default. The models attached to this demo all use relu as the activation function. Compared with the silu activation function, the accuracy is slightly lower, and the performance is greatly improved." Không có số đo tách decode.
- Guide tr.168: các op hậu xử lý như NonMaxSuppression RKNN chưa hỗ trợ, "The removed subgraphs are processed separately on the CPU side."

## T2
- QARepVGG (`docs/_assets/pdf/papers/08_npu_friendly_arch/qarepvgg_quant_2024chu.pdf` tr.7 Table 8, COCO, PTQ): YOLOv6s-RepVGG FP32 42.4, INT8 35.0 (rơi 7.4). YOLOv6s-QARepVGG 42.3, 41.0 (rơi 1.3). Tr.3: "we define a distribution of weights or activations as quantization friendly if it has a small variance and few outliers."
- YOLOv6 (`yolov6_hwfriendly_2022li.pdf` tr.6, 11 đến 12, T4 GPU và TensorRT): "We further improve the PTQ performance by partially converting quantization-sensitive operations into float computation." Table 15: YOLOv6-S 42.4 FP32, 35.0 PTQ, 40.9 với RepOptimizer.
- YOLO26 (`09_yolo_architecture/yolo26_official_2026jocher.pdf` tr.1, 6, 11): bỏ DFL, "removing DFL alone costs 0.6 AP on the YOLO11s baseline (47.0 to 46.4)", COCO, không nói INT8. `yolo26_arch_2025sapkota.pdf` tr.6: DFL "required specialized handling during inference and model export".
- RK3566 YOLO11 (`05_rk3588/rk3566_npu_yolo11_2026limaran.pdf` tr.12): "per-tensor quantization with activation outliers can saturate the scales, driving class logits below thresholds and yielding empty or degraded detections." Không có số mAP INT8.
- YOLOv4 tr.4, 6: "ReLU6 and hard-Swish ... specially designed for quantization networks" (khẳng định, không đo).
- Không thấy: thiết kế tách nhánh outlier, số đo INT8 khi bỏ DFL trên YOLO trong T2.
