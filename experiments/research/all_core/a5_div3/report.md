# a5_div3: whole detectors on the board, CORE_0 against CORE_0_1_2 (on-board C bench)

Weights of the nets are random. Latency is the median NPU run duration of 300 runs per core mask (rep spread in results.json). Split = a layer with 2 or more cores above 0.0 percent in the CORE_0_1_2 WorkLoad column. Shares are over the layers without Input and Output.

| model | prec | ms CORE_0 | ms CORE_0_1 | ms CORE_0_1_2 | x2 | x3 | layers | split layers | cycles split share | time split share | balance-limited x | layers with largest share <= 34.0% | unsplit ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_none_relu_v0_640 | fp16 | 14.464 | 8.809 | 5.983 | 1.64 | 2.42 | 34 | 34 | 1.000 | 1.000 | 2.86 | 12 of 34 | - |
| dense_none_relu_v0_640 | int8 | 7.190 | 4.271 | 3.235 | 1.68 | 2.22 | 34 | 34 | 1.000 | 1.000 | 2.87 | 12 of 34 | - |
| dense_none_relu_v1_640_w96 | fp16 | 46.952 | 27.402 | 21.313 | 1.71 | 2.20 | 34 | 34 | 1.000 | 1.000 | 3.00 | 28 of 34 | - |
| dense_none_relu_v1_640_w96 | int8 | 22.142 | 12.481 | 10.265 | 1.77 | 2.16 | 34 | 34 | 1.000 | 1.000 | 3.00 | 28 of 34 | - |
| dense_none_relu_v2_576 | fp16 | 11.809 | 7.268 | 4.926 | 1.62 | 2.40 | 34 | 34 | 1.000 | 1.000 | 2.91 | 27 of 34 | - |
| dense_none_relu_v2_576 | int8 | 5.915 | 3.571 | 2.799 | 1.66 | 2.11 | 34 | 34 | 1.000 | 1.000 | 2.86 | 27 of 34 | - |
| dense_none_relu_v2_768 | fp16 | 20.892 | 12.672 | 8.332 | 1.65 | 2.51 | 34 | 34 | 1.000 | 1.000 | 2.90 | 26 of 34 | - |
| dense_none_relu_v2_768 | int8 | 10.180 | 5.843 | 4.535 | 1.74 | 2.24 | 34 | 34 | 1.000 | 1.000 | 2.73 | 23 of 34 | - |
| dense_none_relu_v3_576_w96 | fp16 | 37.822 | 22.047 | 17.135 | 1.72 | 2.21 | 34 | 34 | 1.000 | 1.000 | 3.00 | 34 of 34 | - |
| dense_none_relu_v3_576_w96 | int8 | 19.897 | 10.611 | 8.586 | 1.88 | 2.32 | 34 | 34 | 1.000 | 1.000 | 3.00 | 34 of 34 | - |
| mbconv_none_relu6_v0_640 | fp16 | 16.773 | 10.817 | 7.149 | 1.55 | 2.35 | 38 | 38 | 1.000 | 1.000 | 2.88 | 14 of 38 | - |
| mbconv_none_relu6_v0_640 | int8 | 8.808 | 5.324 | 3.853 | 1.65 | 2.29 | 38 | 38 | 1.000 | 1.000 | 2.89 | 14 of 38 | - |
| mbconv_none_relu6_v1_640_w96 | fp16 | 46.982 | 30.288 | 24.482 | 1.55 | 1.92 | 38 | 38 | 1.000 | 1.000 | 2.99 | 30 of 38 | - |
| mbconv_none_relu6_v1_640_w96 | int8 | 23.494 | 13.688 | 11.697 | 1.72 | 2.01 | 38 | 38 | 1.000 | 1.000 | 2.99 | 30 of 38 | - |
| mbconv_none_relu6_v2_576 | fp16 | 13.691 | 8.893 | 5.867 | 1.54 | 2.33 | 38 | 38 | 1.000 | 1.000 | 2.93 | 31 of 38 | - |
| mbconv_none_relu6_v2_576 | int8 | 7.201 | 4.409 | 3.290 | 1.63 | 2.19 | 38 | 38 | 1.000 | 1.000 | 2.90 | 33 of 38 | - |
| mbconv_none_relu6_v2_768 | fp16 | 24.131 | 15.541 | 10.012 | 1.55 | 2.41 | 38 | 38 | 1.000 | 1.000 | 2.93 | 30 of 38 | - |
| mbconv_none_relu6_v2_768 | int8 | 12.519 | 7.374 | 5.355 | 1.70 | 2.34 | 38 | 38 | 1.000 | 1.000 | 2.86 | 30 of 38 | - |
| mbconv_none_relu6_v3_576_w96 | fp16 | 38.128 | 25.458 | 20.302 | 1.50 | 1.88 | 38 | 38 | 1.000 | 1.000 | 3.00 | 38 of 38 | - |
| mbconv_none_relu6_v3_576_w96 | int8 | 21.069 | 11.769 | 9.951 | 1.79 | 2.12 | 38 | 38 | 1.000 | 1.000 | 3.00 | 38 of 38 | - |
