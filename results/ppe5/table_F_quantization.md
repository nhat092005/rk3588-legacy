# Table F. Quantization Impact: `ppe5`

Δ = FP32 − NPU (positive = accuracy lost), mean ± std over seeds; AP = AP50-95.

## Eval set `css_test`

| Model | Precision | mAP50-95 | ΔmAP50-95 | ΔAP Helmet | ΔAP NoHelmet | ΔAP NoVest | ΔAP Vest | ΔAP Person |
|---|---|---|---|---|---|---|---|---|
| yolov8n_baseline | FP32 | 0.5064 ± 0.0058 | ref | ref | ref | ref | ref | ref |
| yolov8n_baseline | FP16 | 0.5059 ± 0.0072 | 0.0005 ± 0.0021 | -0.0003 ± 0.0059 | 0.0011 ± 0.0027 | 0.0011 ± 0.0003 | 0.0006 ± 0.0021 | 0.0001 ± 0.0023 |
| yolov8n_baseline | INT8 | 0.3753 ± 0.0070 | 0.1311 ± 0.0012 | 0.2600 ± 0.0274 | 0.0721 ± 0.0098 | 0.1015 ± 0.0099 | 0.1059 ± 0.0078 | 0.1160 ± 0.0167 |
| yolov8n_baseline | INT8H5 | 0.5038 ± 0.0063 | 0.0026 ± 0.0011 | 0.0031 ± 0.0112 | 0.0067 ± 0.0181 | 0.0081 ± 0.0079 | 0.0018 ± 0.0024 | -0.0069 ± 0.0040 |
| yolov8n_baseline_200ep | FP32 | 0.5378 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_baseline_200ep | FP16 | 0.5363 (1 seed) | 0.0015 (1 seed) | 0.0041 (1 seed) | 0.0004 (1 seed) | -0.0001 (1 seed) | 0.0028 (1 seed) | 0.0002 (1 seed) |
| yolov8n_baseline_200ep | INT8 | 0.3796 (1 seed) | 0.1582 (1 seed) | 0.2849 (1 seed) | 0.1093 (1 seed) | 0.1371 (1 seed) | 0.1165 (1 seed) | 0.1433 (1 seed) |
| yolov8n_leakyrelu | FP32 | 0.4840 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_leakyrelu | FP16 | 0.4792 (1 seed) | 0.0048 (1 seed) | 0.0010 (1 seed) | 0.0052 (1 seed) | 0.0052 (1 seed) | 0.0084 (1 seed) | 0.0045 (1 seed) |
| yolov8n_leakyrelu | INT8 | 0.3411 (1 seed) | 0.1429 (1 seed) | 0.2473 (1 seed) | 0.0856 (1 seed) | 0.0817 (1 seed) | 0.1401 (1 seed) | 0.1598 (1 seed) |
| yolov8n_nodfl | FP32 | 0.4918 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_nodfl | FP16 | 0.4933 (1 seed) | -0.0015 (1 seed) | 0.0016 (1 seed) | -0.0071 (1 seed) | 0.0000 (1 seed) | -0.0001 (1 seed) | -0.0014 (1 seed) |
| yolov8n_nodfl | INT8 | 0.3675 (1 seed) | 0.1243 (1 seed) | 0.2061 (1 seed) | 0.0903 (1 seed) | 0.0861 (1 seed) | 0.1034 (1 seed) | 0.1361 (1 seed) |
| yolov8n_relu | FP32 | 0.4611 ± 0.0052 | ref | ref | ref | ref | ref | ref |
| yolov8n_relu | FP16 | 0.4610 ± 0.0059 | 0.0002 ± 0.0015 | 0.0001 ± 0.0013 | -0.0024 ± 0.0032 | 0.0013 ± 0.0021 | 0.0011 ± 0.0027 | 0.0008 ± 0.0019 |
| yolov8n_relu | INT8 | 0.3366 ± 0.0087 | 0.1245 ± 0.0063 | 0.2451 ± 0.0144 | 0.0706 ± 0.0073 | 0.0703 ± 0.0083 | 0.1036 ± 0.0100 | 0.1330 ± 0.0227 |
| yolov8n_relu6 | FP32 | 0.4516 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_relu6 | FP16 | 0.4507 (1 seed) | 0.0009 (1 seed) | 0.0020 (1 seed) | 0.0024 (1 seed) | -0.0014 (1 seed) | 0.0015 (1 seed) | -0.0001 (1 seed) |
| yolov8n_relu6 | INT8 | 0.3312 (1 seed) | 0.1204 (1 seed) | 0.2296 (1 seed) | 0.0570 (1 seed) | 0.0862 (1 seed) | 0.1085 (1 seed) | 0.1204 (1 seed) |
| yolov8n_relu_200ep | FP32 | 0.4957 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_relu_200ep | FP16 | 0.4906 (1 seed) | 0.0051 (1 seed) | 0.0013 (1 seed) | 0.0065 (1 seed) | 0.0082 (1 seed) | 0.0045 (1 seed) | 0.0048 (1 seed) |
| yolov8n_relu_200ep | INT8 | 0.3618 (1 seed) | 0.1339 (1 seed) | 0.2558 (1 seed) | 0.0698 (1 seed) | 0.1053 (1 seed) | 0.1130 (1 seed) | 0.1256 (1 seed) |

## Eval set `ppe2286_ext`

| Model | Precision | mAP50-95 | ΔmAP50-95 | ΔAP Helmet | ΔAP NoHelmet | ΔAP NoVest | ΔAP Vest | ΔAP Person |
|---|---|---|---|---|---|---|---|---|
| yolov8n_baseline | FP32 | 0.2597 ± 0.0024 | ref | ref | ref | ref | ref | ref |
| yolov8n_baseline | FP16 | 0.2604 ± 0.0029 | -0.0007 ± 0.0006 | -0.0002 ± 0.0027 | -0.0002 ± 0.0002 | -0.0032 ± 0.0004 | -0.0008 ± 0.0017 | 0.0007 ± 0.0006 |
| yolov8n_baseline | INT8 | 0.2267 ± 0.0040 | 0.0330 ± 0.0016 | 0.0127 ± 0.0058 | 0.0023 ± 0.0009 | 0.0347 ± 0.0031 | 0.0360 ± 0.0079 | 0.0791 ± 0.0030 |
| yolov8n_baseline | INT8H5 | 0.2565 ± 0.0026 | 0.0032 ± 0.0008 | 0.0042 ± 0.0014 | -0.0005 ± 0.0009 | 0.0033 ± 0.0069 | 0.0080 ± 0.0036 | 0.0010 ± 0.0056 |
| yolov8n_baseline_200ep | FP32 | 0.2625 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_baseline_200ep | FP16 | 0.2639 (1 seed) | -0.0014 (1 seed) | -0.0001 (1 seed) | 0.0000 (1 seed) | -0.0053 (1 seed) | -0.0003 (1 seed) | -0.0014 (1 seed) |
| yolov8n_baseline_200ep | INT8 | 0.2255 (1 seed) | 0.0370 (1 seed) | 0.0108 (1 seed) | 0.0024 (1 seed) | 0.0410 (1 seed) | 0.0372 (1 seed) | 0.0937 (1 seed) |
| yolov8n_leakyrelu | FP32 | 0.2625 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_leakyrelu | FP16 | 0.2627 (1 seed) | -0.0002 (1 seed) | -0.0003 (1 seed) | -0.0001 (1 seed) | -0.0020 (1 seed) | -0.0007 (1 seed) | 0.0023 (1 seed) |
| yolov8n_leakyrelu | INT8 | 0.2278 (1 seed) | 0.0347 (1 seed) | 0.0013 (1 seed) | 0.0000 (1 seed) | 0.0303 (1 seed) | 0.0520 (1 seed) | 0.0897 (1 seed) |
| yolov8n_nodfl | FP32 | 0.2707 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_nodfl | FP16 | 0.2722 (1 seed) | -0.0015 (1 seed) | -0.0003 (1 seed) | -0.0005 (1 seed) | -0.0038 (1 seed) | -0.0018 (1 seed) | -0.0015 (1 seed) |
| yolov8n_nodfl | INT8 | 0.2377 (1 seed) | 0.0330 (1 seed) | -0.0035 (1 seed) | 0.0017 (1 seed) | 0.0406 (1 seed) | 0.0287 (1 seed) | 0.0973 (1 seed) |
| yolov8n_relu | FP32 | 0.2562 ± 0.0091 | ref | ref | ref | ref | ref | ref |
| yolov8n_relu | FP16 | 0.2565 ± 0.0088 | -0.0004 ± 0.0003 | -0.0006 ± 0.0017 | -0.0003 ± 0.0002 | -0.0002 ± 0.0018 | -0.0012 ± 0.0001 | 0.0005 ± 0.0004 |
| yolov8n_relu | INT8 | 0.2267 ± 0.0073 | 0.0294 ± 0.0020 | 0.0050 ± 0.0042 | 0.0037 ± 0.0025 | 0.0244 ± 0.0106 | 0.0323 ± 0.0190 | 0.0819 ± 0.0082 |
| yolov8n_relu6 | FP32 | 0.2572 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_relu6 | FP16 | 0.2577 (1 seed) | -0.0005 (1 seed) | 0.0005 (1 seed) | 0.0000 (1 seed) | -0.0014 (1 seed) | -0.0020 (1 seed) | 0.0004 (1 seed) |
| yolov8n_relu6 | INT8 | 0.2298 (1 seed) | 0.0274 (1 seed) | -0.0030 (1 seed) | 0.0019 (1 seed) | 0.0173 (1 seed) | 0.0290 (1 seed) | 0.0914 (1 seed) |
| yolov8n_relu_200ep | FP32 | 0.2416 (1 seed) | ref | ref | ref | ref | ref | ref |
| yolov8n_relu_200ep | FP16 | 0.2425 (1 seed) | -0.0009 (1 seed) | -0.0002 (1 seed) | -0.0002 (1 seed) | -0.0012 (1 seed) | -0.0035 (1 seed) | 0.0005 (1 seed) |
| yolov8n_relu_200ep | INT8 | 0.2240 (1 seed) | 0.0176 (1 seed) | -0.0029 (1 seed) | 0.0016 (1 seed) | 0.0196 (1 seed) | 0.0132 (1 seed) | 0.0565 (1 seed) |


_Generated by `make tables`. Do not edit by hand._
