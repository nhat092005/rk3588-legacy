# C3 versus C2f, YOLOv8n, PPE5

Reading tables fixed in `experiments/npu/operator_support/c3_vs_c2f/rule.txt` before any C3 data. Arm A = C2f (`yolov8n_scratch`), arm B = C3 (`yolov8n_c3_scratch`).

## Size

| arm | parameters | GFLOPs (ultralytics get_flops, 640) |
|---|---|---|
| A C2f | 3,011,823 | 8.196 |
| B C3 | 2,487,023 | 6.885 |

## Accuracy (FP32, AP50-95 in pp)

### css_test

| arm | seeds | AP50-95 | NoHelmet | Vest |
|---|---|---|---|---|
| A | [0, 1, 2] | 41.04, 40.34, 42.30 | 22.98, 23.30, 28.81 | 49.32, 48.32, 51.58 |
| B | [0, 1, 2] | 35.82, 35.22, 35.37 | 19.02, 22.65, 19.51 | 44.30, 41.01, 44.21 |

Mean A 41.23, mean B 35.47, D = B minus A = -5.76 pp, S = 1.96 pp, D NoHelmet -4.64, D Vest -6.57. Reading: **C3 loses accuracy**.

### ppe2286_ext

| arm | seeds | AP50-95 | NoHelmet | Vest |
|---|---|---|---|---|
| A | [0, 1, 2] | 16.92, 18.05, 18.46 | 0.46, 0.23, 0.22 | 29.52, 33.23, 35.00 |
| B | [0, 1, 2] | 17.32, 14.78, 16.47 | 0.27, 0.28, 0.28 | 30.12, 25.13, 27.77 |

Mean A 17.81, mean B 16.19, D = B minus A = -1.62 pp, S = 2.54 pp, D NoHelmet -0.03, D Vest -4.91. Reading: **not distinguishable with three seeds**.

## Latency, CORE_0, taskset -c 4 (cell a4i4)

| round | A median (ms) | B median (ms) | B / A |
|---|---|---|---|
| 1 | 16.296 | 16.068 | 0.986 |
| 2 | 16.302 | 16.064 | 0.985 |
| 3 | 16.298 | 16.065 | 0.986 |
| 4 | 16.296 | 16.065 | 0.986 |
| 5 | 16.305 | 16.069 | 0.986 |
| 6 | 16.294 | 16.068 | 0.986 |
| 7 | 16.303 | 16.079 | 0.986 |
| 8 | 16.287 | 16.091 | 0.988 |

Median A 16.297 ms, median B 16.068 ms, median ratio 0.986 (range 0.985 to 0.988), noise floor 0.17 percent. Reading: **C3 faster**.

## Latency, CORE_0_1_2, taskset -c 4-7 (secondary, no reading)

| round | A median (ms) | B median (ms) | B / A |
|---|---|---|---|
| 1 | 15.865 | 15.742 | 0.992 |
| 2 | 15.848 | 15.751 | 0.994 |
| 3 | 15.873 | 15.765 | 0.993 |
| 4 | 15.867 | 15.767 | 0.994 |

Median A 15.866 ms, median B 15.758 ms, median ratio 0.993 (range 0.992 to 0.994), noise floor 0.16 percent. Reading: **C3 faster**.

## Per-layer detail, CORE_0

| arm | layer-time sum (us) | op | rows | time (us) | share |
|---|---|---|---|---|---|
| A | 16574 | ConvExSwish | 57 | 10380 | 62.63 percent |
| A | 16574 | exSoftmax13 | 1 | 2054 | 12.39 percent |
| A | 16574 | Concat | 17 | 1426 | 8.60 percent |
| A | 16574 | Transpose | 1 | 540 | 3.26 percent |
| A | 16574 | Split | 9 | 532 | 3.21 percent |
| A | 16574 | Conv | 7 | 488 | 2.95 percent |
| A | 16574 | Add | 8 | 328 | 1.98 percent |
| A | 16574 | Reshape | 7 | 308 | 1.86 percent |
| A | 16574 | Resize | 2 | 178 | 1.08 percent |
| A | 16574 | Mul | 2 | 118 | 0.71 percent |
| A | 16574 | Sigmoid | 1 | 76 | 0.46 percent |
| A | 16574 | Sub | 2 | 72 | 0.43 percent |
| A | 16574 | MaxPool | 3 | 69 | 0.42 percent |
| A | 16574 | OutputOperator | 1 | 3 | 0.02 percent |
| A | 16574 | InputOperator | 1 | 1 | 0.01 percent |
| B | 16450 | ConvExSwish | 65 | 10909 | 66.31 percent |
| B | 16450 | exSoftmax13 | 1 | 2054 | 12.49 percent |
| B | 16450 | Concat | 17 | 1192 | 7.24 percent |
| B | 16450 | Transpose | 1 | 540 | 3.28 percent |
| B | 16450 | Conv | 7 | 488 | 2.97 percent |
| B | 16450 | Add | 8 | 328 | 1.99 percent |
| B | 16450 | Reshape | 7 | 306 | 1.86 percent |
| B | 16450 | Resize | 2 | 178 | 1.09 percent |
| B | 16450 | Mul | 2 | 118 | 0.72 percent |
| B | 16450 | Split | 1 | 117 | 0.71 percent |
| B | 16450 | Sigmoid | 1 | 76 | 0.46 percent |
| B | 16450 | Sub | 2 | 70 | 0.42 percent |
| B | 16450 | MaxPool | 3 | 69 | 0.42 percent |
| B | 16450 | OutputOperator | 1 | 3 | 0.02 percent |
| B | 16450 | InputOperator | 1 | 1 | 0.01 percent |

WorkLoad of Split, Concat, Add rows, arm A (rows summed over all runs of all files): Add|single core x3200; Concat|single core x6800; Split|single core x3600
WorkLoad of Split, Concat, Add rows, arm B (rows summed over all runs of all files): Add|single core x3200; Concat|single core x6800; Split|single core x400

## Per-layer detail, CORE_0_1_2

| arm | layer-time sum (us) | op | rows | time (us) | share |
|---|---|---|---|---|---|
| A | 16046 | ConvExSwish | 57 | 10395 | 64.78 percent |
| A | 16046 | exSoftmax13 | 1 | 2055 | 12.81 percent |
| A | 16046 | Concat | 17 | 1014 | 6.32 percent |
| A | 16046 | Transpose | 1 | 540 | 3.37 percent |
| A | 16046 | Split | 9 | 534 | 3.33 percent |
| A | 16046 | Conv | 7 | 430 | 2.68 percent |
| A | 16046 | Reshape | 7 | 314 | 1.96 percent |
| A | 16046 | Add | 8 | 242 | 1.51 percent |
| A | 16046 | Resize | 2 | 179 | 1.12 percent |
| A | 16046 | Mul | 2 | 119 | 0.74 percent |
| A | 16046 | Sigmoid | 1 | 77 | 0.48 percent |
| A | 16046 | Sub | 2 | 72 | 0.45 percent |
| A | 16046 | MaxPool | 3 | 69 | 0.43 percent |
| A | 16046 | OutputOperator | 1 | 3 | 0.02 percent |
| A | 16046 | InputOperator | 1 | 1 | 0.01 percent |
| B | 16021 | ConvExSwish | 65 | 10924 | 68.19 percent |
| B | 16021 | exSoftmax13 | 1 | 2053 | 12.81 percent |
| B | 16021 | Concat | 17 | 886 | 5.53 percent |
| B | 16021 | Transpose | 1 | 540 | 3.37 percent |
| B | 16021 | Conv | 7 | 431 | 2.69 percent |
| B | 16021 | Reshape | 7 | 312 | 1.95 percent |
| B | 16021 | Add | 8 | 239 | 1.49 percent |
| B | 16021 | Resize | 2 | 178 | 1.11 percent |
| B | 16021 | Mul | 2 | 119 | 0.74 percent |
| B | 16021 | Split | 1 | 117 | 0.73 percent |
| B | 16021 | Sigmoid | 1 | 77 | 0.48 percent |
| B | 16021 | Sub | 2 | 69 | 0.43 percent |
| B | 16021 | MaxPool | 3 | 69 | 0.43 percent |
| B | 16021 | OutputOperator | 1 | 3 | 0.02 percent |
| B | 16021 | InputOperator | 1 | 1 | 0.01 percent |

WorkLoad of Split, Concat, Add rows, arm A (rows summed over all runs of all files): Add|split over cores x1600; Concat|single core x400; Concat|split over cores x3000; Split|single core x1800
WorkLoad of Split, Concat, Add rows, arm B (rows summed over all runs of all files): Add|split over cores x1600; Concat|single core x400; Concat|split over cores x3000; Split|single core x200

