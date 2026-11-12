# a9d_deep: whole detectors on the board, CORE_0 against CORE_0_1_2 (on-board C bench)

Weights of the nets are random. Latency is the median NPU run duration of 300 runs per core mask (rep spread in results.json). Split = a layer with 2 or more cores above 0.0 percent in the CORE_0_1_2 WorkLoad column. Shares are over the layers without Input and Output.

| model | prec | ms CORE_0 | ms CORE_0_1 | ms CORE_0_1_2 | x2 | x3 | layers | split layers | cycles split share | time split share | balance-limited x | layers with largest share <= 34.0% | unsplit ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_w100_d10 | int8 | 14.583 | 8.169 | 6.008 | 1.79 | 2.43 | 75 | 75 | 1.000 | 1.000 | 2.87 | 31 of 75 | - |
| dense_w150_d2 | int8 | 14.254 | 8.233 | 6.265 | 1.73 | 2.28 | 43 | 43 | 1.000 | 1.000 | 2.98 | 29 of 43 | - |
| yolov8n_relu_raw | int8 | 9.088 | 8.041 | 6.789 | 1.13 | 1.34 | 89 | 76 | 1.000 | 0.872 | 2.48 | 27 of 89 | {'Split': 8, 'MaxPool': 3, 'Resize': 2} |
