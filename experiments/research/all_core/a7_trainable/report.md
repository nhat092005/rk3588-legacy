# a7_trainable: whole detectors on the board, CORE_0 against CORE_0_1_2 (on-board C bench)

Weights of the nets are random. Latency is the median NPU run duration of 300 runs per core mask (rep spread in results.json). Split = a layer with 2 or more cores above 0.0 percent in the CORE_0_1_2 WorkLoad column. Shares are over the layers without Input and Output.

| model | prec | ms CORE_0 | ms CORE_0_1 | ms CORE_0_1_2 | x2 | x3 | layers | split layers | cycles split share | time split share | balance-limited x | layers with largest share <= 34.0% | unsplit ops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| allcore_dense | fp16 | 12.374 | 7.722 | 5.220 | 1.60 | 2.37 | 43 | 43 | 1.000 | 1.000 | 2.87 | 15 of 43 | - |
| allcore_dense | int8 | 6.400 | 3.875 | 2.937 | 1.65 | 2.18 | 43 | 43 | 1.000 | 1.000 | 2.88 | 15 of 43 | - |
| allcore_mbconv | fp16 | 14.680 | 9.726 | 6.372 | 1.51 | 2.30 | 47 | 47 | 1.000 | 1.000 | 2.89 | 17 of 47 | - |
| allcore_mbconv | int8 | 8.011 | 4.912 | 3.546 | 1.63 | 2.26 | 47 | 47 | 1.000 | 1.000 | 2.91 | 17 of 47 | - |
