# Giai đoạn 2: kết quả tìm nguồn (doc-reader, 2026-10-31)

Nguồn: hai subagent doc-reader, chỉ đọc. Số trang là trang PDF (pdftotext). Chưa qua fact-checker.

## G1 (Operator List, T1)
- Operator List V2.3.2 (`docs/_assets/pdf/rknn/03_RKNN_Compiler_Support_Operator_List_V2.3.2.pdf`): không có mục tên ConvReluAdd, ConvRelu, ConvolutionReluAdd, ConvExSwish. Có 3.2.7 ConvolutionRelu (tr.60), 3.2.19 ConvolutionAddRelu (tr.63), 3.2.69 exSwish (tr.98).
- 3.2.19 tr.63: "Convolution 与 Add 及 Relu 融合计算, Output = Relu(Add(Conv(Input0), Input1))", "说明及规格限制同Convolution", "多核联合: 支持".
- ConvRelu có trong ví dụ log của SDK guide V2.3.2 (`3rdparty/rknn-toolkit2/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_EN.pdf`) tr.75: "6 ConvRelu INT8 NPU 48.1%/51.9%/0.0% - Up:48.1%", và tr.109 đến 110 (bảng layer). Cũng ở `3rdparty/rknn-toolkit2/rknn-toolkit2/examples/functions/npu_device_test/README.md` dòng 45, 53.
- ConvReluAdd: không có trong PDF hay text T1. Chỉ có chuỗi trong tệp nhị phân `3rdparty/rknn-toolkit2/rknpu2/examples/rknn_yolov5_demo/model/RK3588/yolov5s-640-640.rknn` (không tính là tài liệu).
- ConvExSwish: không thấy. exSwish: `3rdparty/rknn-toolkit2/rknn-toolkit2/doc/changelog-2.3.2.txt` dòng 284 "增加exSwish的beta属性支持".
## G2 (T2)
- Không bài nào trong 46 PDF đo activation trên RK3588. Có số đo trên NPU khác: ActNAS (SiLU 22.35 ms, ReLU 17.46 ms, YOLO5n, NPU1 không nêu tên hãng, tr.6 Table 2). Không phải NPU RK3588.
## G11
- Mâu thuẫn Relu/Concat giữa Operator List và hai guide không được giải quyết ở T1 khác (changelog, header). SDK guide tr.75 và API guide V1.5.2 tr.27 liệt kê Relu, Concat đa nhân, Operator List tr.84 ghi Relu 暂不支持, Concat (tr.58) không có dòng.
## G7
- Operator List tr.86 đến 87 chỉ nêu giới hạn, không nói cách compiler xử lý `asymmetric`. Trong `rknn_toolkit2-2.3.2` (wheel, chuỗi trong .so, chưa đọc logic): tên pass `_p_convert_resize_to_deconv`, thông báo "Resize [{}] convert to Deconvolution for inference speedup, but may cause result driftting.", `_p_convert_resize_to_concat`, `_p_convert_1x1_resize_to_tile`. Không biết điều kiện kích hoạt: chưa biết.
## G10
- Không có giải thích văn bản. Tên pass trong binary (chỉ tên): `_p_remove_reshape_after_concat`, `_p_replace_parallel_slice_by_split`, `_p_fuse_two_slice`, `_p_fuse_reshape_transpose`.
## M2
- Kong 2026: không định nghĩa ALL. Table 4 tr.7 chỉ là tiêu đề cột. Data availability tr.11: dữ liệu theo yêu cầu từ tác giả liên hệ. Không có phụ lục.
- `rknn_api.h:253`: "RKNN_NPU_CORE_ALL = 0xffff, /* auto choice, run on NPU cores depending on platform */". RKNNRT API Reference V2.3.2 tr.21: "RKNN_NPU_CORE_ALL: Running on all of NPU cores depending on the platform;". Toolkit2 API Reference tr.17 (init_runtime): "RKNN.NPU_CORE_ALL: Indicates running on the number of NPU cores depending on the platform."
