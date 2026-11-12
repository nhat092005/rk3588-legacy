# a6_head: whole detectors on the board, CORE_0 against CORE_0_1_2 (on-board C bench)

Weights of the nets are random. Latency is the median NPU run duration of 300 runs per core mask (rep spread in results.json). Split = a layer with 2 or more cores above 0.0 percent in the CORE_0_1_2 WorkLoad column. Shares are over the layers without Input and Output.

| model | prec | ms CORE_0 | ms CORE_0_1 | ms CORE_0_1_2 | x2 | x3 | layers | split layers | cycles split share | time split share | balance-limited x | layers with largest share <= 34.0% | unsplit ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_none_relu_stock | fp16 | 12.371 | 7.725 | 5.224 | 1.60 | 2.37 | 43 | 43 | 1.000 | 1.000 | 2.87 | 15 of 43 | - |
| dense_none_relu_stock | int8 | 6.396 | 3.870 | 2.935 | 1.65 | 2.18 | 43 | 43 | 1.000 | 1.000 | 2.88 | 15 of 43 | - |
| dense_none_relu_stock_legacy | fp16 | 12.932 | 7.959 | 5.442 | 1.62 | 2.38 | 37 | 37 | 1.000 | 1.000 | 2.86 | 13 of 37 | - |
| dense_none_relu_stock_legacy | int8 | 6.588 | 3.941 | 3.007 | 1.67 | 2.19 | 37 | 37 | 1.000 | 1.000 | 2.88 | 13 of 37 | - |
| dense_none_relu_v6 | fp16 | 14.462 | 8.809 | 5.988 | 1.64 | 2.42 | 34 | 34 | 1.000 | 1.000 | 2.86 | 12 of 34 | - |
| dense_none_relu_v6 | int8 | 7.188 | 4.274 | 3.237 | 1.68 | 2.22 | 34 | 34 | 1.000 | 1.000 | 2.87 | 12 of 34 | - |
