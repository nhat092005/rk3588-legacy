# a4_board_3core: whole detectors on the board, CORE_0 against CORE_0_1_2 (on-board C bench)

Weights of the nets are random. Latency is the median NPU run duration of 300 runs per core mask (rep spread in results.json). Split = a layer with 2 or more cores above 0.0 percent in the CORE_0_1_2 WorkLoad column. Shares are over the layers without Input and Output.

| model | prec | ms CORE_0 | ms CORE_0_1 | ms CORE_0_1_2 | x2 | x3 | layers | split layers | cycles split share | time split share | balance-limited x | layers with largest share <= 34.0% | unsplit ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_convt_relu | fp16 | 20.269 | 13.303 | 9.781 | 1.52 | 2.07 | 40 | 38 | 0.896 | 0.837 | 2.47 | 14 of 40 | {'ConvTranspose': 2} |
| dense_convt_relu | int8 | 9.897 | 6.482 | 5.277 | 1.53 | 1.88 | 40 | 38 | 0.897 | 0.855 | 2.43 | 14 of 40 | {'ConvTranspose': 2} |
| dense_none_leakyrelu | fp16 | 14.462 | 8.809 | 5.990 | 1.64 | 2.41 | 34 | 34 | 1.000 | 1.000 | 2.86 | 12 of 34 | - |
| dense_none_leakyrelu | int8 | 7.191 | 4.273 | 3.238 | 1.68 | 2.22 | 34 | 34 | 1.000 | 1.000 | 2.87 | 12 of 34 | - |
| dense_none_relu | fp16 | 14.463 | 8.811 | 5.990 | 1.64 | 2.41 | 34 | 34 | 1.000 | 1.000 | 2.86 | 12 of 34 | - |
| dense_none_relu | int8 | 7.191 | 4.273 | 3.239 | 1.68 | 2.22 | 34 | 34 | 1.000 | 1.000 | 2.87 | 12 of 34 | - |
| dense_resize_relu | fp16 | 18.877 | 11.893 | 8.374 | 1.59 | 2.25 | 40 | 38 | 1.000 | 0.959 | 2.75 | 14 of 40 | {'Resize': 2} |
| dense_resize_relu | int8 | 9.203 | 5.790 | 4.582 | 1.59 | 2.01 | 40 | 38 | 1.000 | 0.957 | 2.67 | 14 of 40 | {'Resize': 2} |
| mbconv_none_relu6 | fp16 | 16.767 | 10.812 | 7.141 | 1.55 | 2.35 | 38 | 38 | 1.000 | 1.000 | 2.88 | 14 of 38 | - |
| mbconv_none_relu6 | int8 | 8.805 | 5.322 | 3.851 | 1.65 | 2.29 | 38 | 38 | 1.000 | 1.000 | 2.89 | 14 of 38 | - |
| mbconv_none_relu | fp16 | 16.767 | 10.809 | 7.141 | 1.55 | 2.35 | 38 | 38 | 1.000 | 1.000 | 2.88 | 14 of 38 | - |
| mbconv_none_relu | int8 | 8.804 | 5.323 | 3.852 | 1.65 | 2.29 | 38 | 38 | 1.000 | 1.000 | 2.89 | 14 of 38 | - |
| res_none_leakyrelu | fp16 | 14.480 | 8.826 | 6.001 | 1.64 | 2.41 | 34 | 34 | 1.000 | 1.000 | 2.86 | 12 of 34 | - |
| res_none_leakyrelu | int8 | 7.200 | 4.281 | 3.245 | 1.68 | 2.22 | 34 | 34 | 1.000 | 1.000 | 2.87 | 12 of 34 | - |
| res_none_relu | fp16 | 14.480 | 8.825 | 6.003 | 1.64 | 2.41 | 34 | 34 | 1.000 | 1.000 | 2.86 | 12 of 34 | - |
| res_none_relu | int8 | 7.200 | 4.281 | 3.243 | 1.68 | 2.22 | 34 | 34 | 1.000 | 1.000 | 2.87 | 12 of 34 | - |
| star_none_relu | fp16 | 20.350 | 12.804 | 9.043 | 1.59 | 2.25 | 50 | 50 | 1.000 | 1.000 | 2.90 | 24 of 50 | - |
| star_none_relu | int8 | 11.750 | 7.090 | 5.428 | 1.66 | 2.16 | 50 | 50 | 1.000 | 1.000 | 2.92 | 24 of 50 | - |
| uib_none_relu | fp16 | 17.462 | 11.170 | 7.384 | 1.56 | 2.36 | 42 | 42 | 1.000 | 1.000 | 2.88 | 16 of 42 | - |
| uib_none_relu | int8 | 9.340 | 5.595 | 4.044 | 1.67 | 2.31 | 42 | 42 | 1.000 | 1.000 | 2.90 | 16 of 42 | - |
| yolov8n_relu | fp16 | 19.648 | 16.644 | 13.723 | 1.18 | 1.43 | 111 | 80 | 0.993 | 0.653 | 2.11 | 33 of 111 | {'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Conv': 1, 'Transpose': 1, 'Sub': 2, 'Mul': 2, 'Concat': 2, 'Sigmoid': 1} |
| yolov8n_relu | int8 | 12.905 | 12.101 | 10.822 | 1.07 | 1.19 | 111 | 80 | 0.976 | 0.590 | 1.93 | 33 of 111 | {'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Conv': 1, 'Transpose': 1, 'Sub': 2, 'Mul': 2, 'Concat': 2, 'Sigmoid': 1} |
| yolov8n_silu | fp16 | 29.666 | 29.462 | 29.253 | 1.01 | 1.01 | 117 | 29 | 0.015 | 0.103 | 1.12 | 15 of 117 | {'ConvExSwish': 57, 'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Conv': 1, 'Transpose': 1, 'Sub': 2, 'Mul': 2, 'Concat': 2, 'Sigmoid': 1} |
| yolov8n_silu | int8 | 18.884 | 19.126 | 19.055 | 0.99 | 0.99 | 117 | 29 | 0.018 | 0.118 | 1.13 | 15 of 117 | {'ConvExSwish': 57, 'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Conv': 1, 'Transpose': 1, 'Sub': 2, 'Mul': 2, 'Concat': 2, 'Sigmoid': 1} |
