# a9c_fine: whole detectors on the board, CORE_0 against CORE_0_1_2 (on-board C bench)

Weights of the nets are random. Latency is the median NPU run duration of 300 runs per core mask (rep spread in results.json). Split = a layer with 2 or more cores above 0.0 percent in the CORE_0_1_2 WorkLoad column. Shares are over the layers without Input and Output.

| model | prec | ms CORE_0 | ms CORE_0_1 | ms CORE_0_1_2 | x2 | x3 | layers | split layers | cycles split share | time split share | balance-limited x | layers with largest share <= 34.0% | unsplit ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_w100_d2 | int8 | 6.397 | 3.872 | 2.934 | 1.65 | 2.18 | 43 | 43 | 1.000 | 1.000 | 2.88 | 15 of 43 | - |
| dense_w100_d6 | int8 | 10.490 | 6.021 | 4.475 | 1.74 | 2.34 | 59 | 59 | 1.000 | 1.000 | 2.87 | 23 of 59 | - |
| dense_w125_d3 | int8 | 14.727 | 8.111 | 6.647 | 1.82 | 2.22 | 47 | 46 | 0.953 | 0.898 | 2.59 | 17 of 47 | {'ConvRelu': 1} |
| dense_w125_d4 | int8 | 17.019 | 9.306 | 7.495 | 1.83 | 2.27 | 51 | 50 | 0.960 | 0.909 | 2.61 | 19 of 51 | {'ConvRelu': 1} |
| dense_w150_d1 | int8 | 11.568 | 6.722 | 5.175 | 1.72 | 2.24 | 39 | 39 | 1.000 | 1.000 | 2.98 | 25 of 39 | - |
| dense_w150_d2 | int8 | 14.257 | 8.236 | 6.268 | 1.73 | 2.27 | 43 | 43 | 1.000 | 1.000 | 2.98 | 29 of 43 | - |
| dense_w175_d1 | int8 | 14.309 | 8.554 | 5.934 | 1.67 | 2.41 | 39 | 39 | 1.000 | 1.000 | 2.80 | 13 of 39 | - |
| yolov8n_relu | int8 | 12.898 | 12.101 | 10.819 | 1.07 | 1.19 | 111 | 80 | 0.976 | 0.590 | 1.94 | 33 of 111 | {'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Conv': 1, 'Transpose': 1, 'Sub': 2, 'Mul': 2, 'Concat': 2, 'Sigmoid': 1} |
| yolov8n_relu_raw | int8 | 9.087 | 8.046 | 6.793 | 1.13 | 1.34 | 89 | 76 | 1.000 | 0.865 | 2.48 | 27 of 89 | {'Split': 8, 'MaxPool': 3, 'Resize': 2} |
