# a9_scale_grid: whole detectors on the board, CORE_0 against CORE_0_1_2 (on-board C bench)

Weights of the nets are random. Latency is the median NPU run duration of 300 runs per core mask (rep spread in results.json). Split = a layer with 2 or more cores above 0.0 percent in the CORE_0_1_2 WorkLoad column. Shares are over the layers without Input and Output.

| model | prec | ms CORE_0 | ms CORE_0_1 | ms CORE_0_1_2 | x2 | x3 | layers | split layers | cycles split share | time split share | balance-limited x | layers with largest share <= 34.0% | unsplit ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_w100_d2 | int8 | 6.395 | 3.871 | 2.934 | 1.65 | 2.18 | 43 | 43 | 1.000 | 1.000 | 2.88 | 15 of 43 | - |
| dense_w150_d2 | int8 | 14.251 | 8.231 | 6.264 | 1.73 | 2.28 | 43 | 43 | 1.000 | 1.000 | 2.98 | 29 of 43 | - |
| dense_w150_d3 | int8 | 16.945 | 9.736 | 7.350 | 1.74 | 2.31 | 47 | 47 | 1.000 | 1.000 | 2.98 | 33 of 47 | - |
| dense_w150_d4 | int8 | 19.644 | 11.252 | 8.448 | 1.75 | 2.33 | 51 | 51 | 1.000 | 1.000 | 2.98 | 37 of 51 | - |
| dense_w200_d2 | int8 | 20.592 | 11.757 | 8.338 | 1.75 | 2.47 | 43 | 43 | 1.000 | 1.000 | 2.82 | 15 of 43 | - |
| dense_w200_d3 | int8 | 24.633 | 13.894 | 9.868 | 1.77 | 2.50 | 47 | 47 | 1.000 | 1.000 | 2.82 | 17 of 47 | - |
| dense_w200_d4 | int8 | 28.675 | 16.027 | 11.391 | 1.79 | 2.52 | 51 | 51 | 1.000 | 1.000 | 2.81 | 19 of 51 | - |
| dense_w250_d2 | int8 | 43.127 | 22.440 | 16.590 | 1.92 | 2.60 | 43 | 43 | 1.000 | 1.000 | 2.83 | 15 of 43 | - |
| dense_w250_d3 | int8 | 50.573 | 26.213 | 19.315 | 1.93 | 2.62 | 47 | 47 | 1.000 | 1.000 | 2.83 | 17 of 47 | - |
| yolov8n_relu | int8 | 12.905 | 12.105 | 10.820 | 1.07 | 1.19 | 111 | 80 | 0.976 | 0.590 | 1.93 | 33 of 111 | {'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Conv': 1, 'Transpose': 1, 'Sub': 2, 'Mul': 2, 'Concat': 2, 'Sigmoid': 1} |
| yolov8n_relu_raw | int8 | 9.790 | 8.388 | 7.040 | 1.17 | 1.39 | 89 | 76 | 1.000 | 0.869 | 2.50 | 27 of 89 | {'Split': 8, 'MaxPool': 3, 'Resize': 2} |
| yolov8n_silu | int8 | 18.884 | 19.118 | 19.048 | 0.99 | 0.99 | 117 | 29 | 0.018 | 0.118 | 1.13 | 15 of 117 | {'ConvExSwish': 57, 'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Conv': 1, 'Transpose': 1, 'Sub': 2, 'Mul': 2, 'Concat': 2, 'Sigmoid': 1} |
| yolov8n_silu_raw | int8 | 15.548 | 15.456 | 15.377 | 1.01 | 1.01 | 95 | 25 | 0.027 | 0.131 | 1.12 | 9 of 95 | {'ConvExSwish': 57, 'Split': 8, 'MaxPool': 3, 'Resize': 2} |
