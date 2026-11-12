# a9b_refs: whole detectors on the board, CORE_0 against CORE_0_1_2 (on-board C bench)

Weights of the nets are random. Latency is the median NPU run duration of 300 runs per core mask (rep spread in results.json). Split = a layer with 2 or more cores above 0.0 percent in the CORE_0_1_2 WorkLoad column. Shares are over the layers without Input and Output.

| model | prec | ms CORE_0 | ms CORE_0_1 | ms CORE_0_1_2 | x2 | x3 | layers | split layers | cycles split share | time split share | balance-limited x | layers with largest share <= 34.0% | unsplit ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_w100_d2 | int8 | 6.398 | 3.872 | 2.933 | 1.65 | 2.18 | 43 | 43 | 1.000 | 1.000 | 2.88 | 15 of 43 | - |
| yolov8n_relu | int8 | 12.879 | 12.075 | 10.794 | 1.07 | 1.19 | 111 | 80 | 0.976 | 0.591 | 1.93 | 33 of 111 | {'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Conv': 1, 'Transpose': 1, 'Sub': 2, 'Mul': 2, 'Concat': 2, 'Sigmoid': 1} |
| yolov8n_relu_raw | int8 | 9.076 | 8.027 | 6.768 | 1.13 | 1.34 | 89 | 76 | 1.000 | 0.865 | 2.48 | 27 of 89 | {'Split': 8, 'MaxPool': 3, 'Resize': 2} |
| yolov8n_silu | int8 | 18.868 | 19.108 | 19.025 | 0.99 | 0.99 | 117 | 29 | 0.018 | 0.118 | 1.13 | 15 of 117 | {'ConvExSwish': 57, 'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Conv': 1, 'Transpose': 1, 'Sub': 2, 'Mul': 2, 'Concat': 2, 'Sigmoid': 1} |
| yolov8n_silu_raw | int8 | 14.856 | 14.798 | 14.729 | 1.00 | 1.01 | 95 | 25 | 0.018 | 0.133 | 1.12 | 9 of 95 | {'ConvExSwish': 57, 'Split': 8, 'MaxPool': 3, 'Resize': 2} |
