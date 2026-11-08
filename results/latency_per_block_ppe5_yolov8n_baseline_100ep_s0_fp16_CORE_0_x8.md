# Latency per block: ppe5_yolov8n_baseline_100ep_s0 fp16 CORE_0

3 independent sessions. Block time = median over the eval_perf calls of a session, then median over sessions (perf_debug=True, so debug-mode time, not latency). model sha256 `bd4a8c8a33ab`, commit `d84836b0`.

## Totals

- latency (perf_debug=False), ms per session: [27.284, 27.282, 27.267], median 27.282, gap 0.06%
- debug total (Total Operator Elapsed Per Frame Time), ms per session: [27.389, 27.327, 27.432], median 27.389, gap 0.38%
- debug_ratio (debug total / latency): 1.004, sum_ratio (sum of layer times / total, per eval_perf call) min and max per session: [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
- per-row additive overhead of each session (intercept of time = a + b x est. cycles time over the Conv rows, us), numbered sessions then B2: [50.9, 51.2, 49.8]. The column 'share without per-row overhead' subtracts that intercept from every row with time above zero: a sensitivity bound, not a correction.

## Table 1: block L0-L22

| Block | rows | us (median) | B2 us | spread between sessions | spread between calls | share % | B2 share % | share without per-row overhead % (sensitivity bound) | share spread between sessions | cycle share % | GFLOPs | us per GFLOP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 1 | 1904 | - | 1.6% | 0.5% | 7.0 | - | 8.5 | 1.3% | 9.7 | 0.088 | 21521 |
| L1 | 1 | 1073 | - | 0.1% | 0.3% | 3.9 | - | 4.7 | 0.4% | 4.8 | 0.236 | 4548 |
| L2 | 7 | 3562 | - | 1.2% | 0.1% | 13.0 | - | 14.6 | 0.8% | 8.1 | 0.367 | 9706 |
| L3 | 1 | 553 | - | 0.7% | 0.2% | 2.0 | - | 2.3 | 0.3% | 2.4 | 0.236 | 2344 |
| L4 | 10 | 2503 | - | 0.5% | 0.2% | 9.1 | - | 9.1 | 0.3% | 6.7 | 0.629 | 3978 |
| L5 | 1 | 396 | - | 0.3% | 0.3% | 1.4 | - | 1.6 | 0.6% | 2.4 | 0.236 | 1678 |
| L6 | 10 | 1509 | - | 0.3% | 0.3% | 5.5 | - | 4.6 | 0.4% | 6.4 | 0.629 | 2398 |
| L7 | 1 | 335 | - | 0.3% | 0.6% | 1.2 | - | 1.3 | 0.1% | 2.4 | 0.236 | 1420 |
| L8 | 7 | 708 | - | 0.3% | 0.6% | 2.6 | - | 1.8 | 0.3% | 3.8 | 0.367 | 1929 |
| L9 | 6 | 462 | - | 0.2% | 0.9% | 1.7 | - | 0.9 | 0.5% | 1.3 | 0.131 | 3525 |
| L10 | 1 | 110 | - | 0.9% | 2.7% | 0.4 | - | 0.3 | 0.5% | 0.0 | - | - |
| L11 | 1 | 186 | - | 2.2% | 0.5% | 0.7 | - | 0.6 | 1.7% | 0.0 | - | - |
| L12 | 6 | 1129 | - | 0.3% | 0.3% | 4.1 | - | 3.8 | 0.2% | 4.8 | 0.472 | 2393 |
| L13 | 1 | 201 | - | 0.5% | 1.0% | 0.7 | - | 0.7 | 0.1% | 0.0 | - | - |
| L14 | 1 | 349 | - | 2.6% | 0.9% | 1.3 | - | 1.4 | 2.1% | 0.0 | - | - |
| L15 | 6 | 1913 | - | 0.2% | 0.2% | 7.0 | - | 7.3 | 0.5% | 4.9 | 0.472 | 4054 |
| L16 | 1 | 232 | - | 0.4% | 0.4% | 0.8 | - | 0.8 | 0.8% | 1.2 | 0.118 | 1967 |
| L17 | 1 | 98 | - | 1.0% | 1.0% | 0.4 | - | 0.2 | 1.0% | 0.0 | - | - |
| L18 | 6 | 1021 | - | 0.0% | 0.4% | 3.7 | - | 3.3 | 0.4% | 4.0 | 0.393 | 2597 |
| L19 | 1 | 192 | - | 0.0% | 0.5% | 0.7 | - | 0.6 | 0.4% | 1.2 | 0.118 | 1628 |
| L20 | 1 | 53 | - | 0.0% | 1.9% | 0.2 | - | 0.0 | 0.4% | 0.0 | - | - |
| L21 | 6 | 711 | - | 0.1% | 0.4% | 2.6 | - | 1.9 | 0.4% | 4.0 | 0.393 | 1808 |
| L22 | 39 | 8072 | - | 0.0% | 0.8% | 29.5 | - | 29.7 | 0.4% | 31.8 | 2.965 | 2722 |
| io | 2 | 12 | - | 25.0% | 21.4% | 0.0 | - | 0.0 | 25.2% | 0.0 | - | - |
| post_detect | 1 | 94 | - | 1.1% | 2.1% | 0.3 | - | 0.2 | 0.9% | 0.0 | - | - |

## Table 2: part inside block

| Block | Part | us (median) | spread between sessions | spread between calls | share % | share in block % |
|---|---|---|---|---|---|---|
| L0 | conv | 1904 | 1.6% | 0.5% | 7.0 | 100.0 |
| L1 | conv | 1073 | 0.1% | 0.3% | 3.9 | 100.0 |
| L2 | bottleneck | 1112 | 1.2% | 0.2% | 4.1 | 31.2 |
| L2 | concat | 356 | 1.4% | 0.3% | 1.3 | 10.0 |
| L2 | cv1 | 893 | 1.1% | 0.2% | 3.3 | 25.1 |
| L2 | cv2 | 957 | 1.6% | 0.3% | 3.5 | 27.0 |
| L2 | split | 240 | 1.7% | 0.8% | 0.9 | 6.7 |
| L3 | conv | 553 | 0.7% | 0.2% | 2.0 | 100.0 |
| L4 | bottleneck | 1145 | 1.0% | 0.4% | 4.2 | 45.7 |
| L4 | concat | 242 | 3.7% | 0.8% | 0.9 | 9.7 |
| L4 | cv1 | 457 | 0.4% | 0.2% | 1.7 | 18.3 |
| L4 | cv2 | 534 | 0.9% | 0.4% | 2.0 | 21.4 |
| L4 | split | 124 | 2.4% | 0.8% | 0.5 | 5.0 |
| L5 | conv | 396 | 0.3% | 0.3% | 1.4 | 100.0 |
| L6 | bottleneck | 813 | 0.2% | 0.5% | 3.0 | 53.8 |
| L6 | concat | 128 | 1.6% | 1.6% | 0.5 | 8.5 |
| L6 | cv1 | 232 | 0.0% | 0.4% | 0.8 | 15.4 |
| L6 | cv2 | 270 | 0.0% | 0.4% | 1.0 | 17.9 |
| L6 | split | 67 | 0.0% | 3.0% | 0.2 | 4.4 |
| L7 | conv | 335 | 0.3% | 0.6% | 1.2 | 100.0 |
| L8 | bottleneck | 337 | 0.3% | 0.9% | 1.2 | 47.7 |
| L8 | concat | 52 | 0.0% | 1.9% | 0.2 | 7.3 |
| L8 | cv1 | 125 | 0.0% | 0.8% | 0.5 | 17.7 |
| L8 | cv2 | 155 | 0.6% | 0.6% | 0.6 | 21.9 |
| L8 | split | 38 | 2.6% | 2.7% | 0.1 | 5.4 |
| L9 | concat | 68 | 1.5% | 1.5% | 0.2 | 14.7 |
| L9 | cv1 | 76 | 1.3% | 1.3% | 0.3 | 16.5 |
| L9 | cv2 | 203 | 0.0% | 1.0% | 0.7 | 43.9 |
| L9 | maxpool | 115 | 0.9% | 1.7% | 0.4 | 24.9 |
| L10 | resize | 110 | 0.9% | 2.7% | 0.4 | 100.0 |
| L11 | concat | 186 | 2.2% | 0.5% | 0.7 | 100.0 |
| L12 | bottleneck | 358 | 0.0% | 0.8% | 1.3 | 31.7 |
| L12 | concat | 95 | 1.1% | 1.1% | 0.3 | 8.4 |
| L12 | cv1 | 361 | 0.3% | 0.3% | 1.3 | 32.0 |
| L12 | cv2 | 249 | 0.4% | 0.4% | 0.9 | 22.0 |
| L12 | split | 66 | 3.0% | 1.5% | 0.2 | 5.9 |
| L13 | resize | 201 | 0.5% | 1.0% | 0.7 | 100.0 |
| L14 | concat | 349 | 2.6% | 0.9% | 1.3 | 100.0 |
| L15 | bottleneck | 483 | 0.2% | 0.2% | 1.8 | 25.2 |
| L15 | concat | 183 | 0.5% | 1.1% | 0.7 | 9.6 |
| L15 | cv1 | 626 | 0.3% | 0.3% | 2.3 | 32.7 |
| L15 | cv2 | 497 | 0.2% | 0.2% | 1.8 | 26.0 |
| L15 | split | 126 | 1.6% | 1.6% | 0.5 | 6.6 |
| L16 | conv | 232 | 0.4% | 0.4% | 0.8 | 100.0 |
| L17 | concat | 98 | 1.0% | 1.0% | 0.4 | 100.0 |
| L18 | bottleneck | 358 | 0.3% | 0.6% | 1.3 | 35.1 |
| L18 | concat | 96 | 2.1% | 1.1% | 0.4 | 9.4 |
| L18 | cv1 | 250 | 0.0% | 0.8% | 0.9 | 24.5 |
| L18 | cv2 | 248 | 0.4% | 0.4% | 0.9 | 24.3 |
| L18 | split | 68 | 1.5% | 1.5% | 0.2 | 6.7 |
| L19 | conv | 192 | 0.0% | 0.5% | 0.7 | 100.0 |
| L20 | concat | 53 | 0.0% | 1.9% | 0.2 | 100.0 |
| L21 | bottleneck | 310 | 0.0% | 0.6% | 1.1 | 43.6 |
| L21 | concat | 53 | 0.0% | 1.9% | 0.2 | 7.5 |
| L21 | cv1 | 155 | 0.6% | 0.6% | 0.6 | 21.8 |
| L21 | cv2 | 155 | 0.6% | 0.6% | 0.6 | 21.8 |
| L21 | split | 37 | 2.7% | 2.7% | 0.1 | 5.2 |
| L22 | box_branch | 2326 | 0.1% | 0.2% | 8.5 | 28.8 |
| L22 | cls_branch | 2241 | 0.0% | 0.2% | 8.2 | 27.8 |
| L22 | decode | 1039 | 0.6% | 1.2% | 3.8 | 12.9 |
| L22 | dfl | 2461 | 0.4% | 2.1% | 9.0 | 30.5 |
| io | InputOperator | 4 | 0.0% | 25.0% | 0.0 | 33.3 |
| io | OutputOperator | 7 | 42.9% | 28.6% | 0.0 | 63.6 |
| post_detect | Reshape | 94 | 1.1% | 2.1% | 0.3 | 100.0 |

## Table 3: by OpType

| OpType | Calls | us (median) | share % |
|---|---|---|---|
| ConvExSwish | 57 | 19768 | 72.26 |
| Concat | 17 | 2366 | 8.64 |
| exSoftmax13 | 1 | 1622 | 5.93 |
| Split | 9 | 874 | 3.19 |
| Conv | 7 | 605 | 2.21 |
| Transpose | 1 | 597 | 2.18 |
| Add | 8 | 544 | 1.98 |
| Reshape | 7 | 338 | 1.23 |
| Resize | 2 | 311 | 1.13 |
| MaxPool | 3 | 115 | 0.42 |
| Mul | 2 | 106 | 0.39 |
| Sub | 2 | 68 | 0.25 |
| Sigmoid | 1 | 43 | 0.16 |
| OutputOperator | 1 | 7 | 0.03 |
| InputOperator | 1 | 4 | 0.01 |

## Acceptance checks

| Check | Result | Detail |
|---|---|---|
| A1 | pass | build table 119 rows vs board table 119 rows, (OpType, FullName) sequence identical: True |
| A2 | pass | 6 frequency reports equal to {'CPU': [1800000, 2256000, 2304000], 'NPU': [1000000000], 'DDR': [2112000000]} |
| A3 | pass | NPU rows (not Reshape) with time > 0 and WorkLoad != 100.0%/0.0%/0.0%: [] |
| A4 | pass | rows outside model.N or with an unknown part: [('io', 'InputOperator'), ('io', 'OutputOperator'), ('post_detect', 'Reshape')] |
| A5 | pass | latency gap between 3 sessions 0.06% (max 5%) |
| A6 | pass | sum of block shares 0.9995, sum_ratio min/max per session [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], debug_ratio 1.004 |
| A8 | pass | sessions only: debug total gap 0.38% (max 5%), blocks with share >= 2% whose share spread exceeds 10%: none |
| A7 | pass | provenance keys ('model_sha256', 'git_commit', 'git_dirty', 'sdk_version') present in every session file |
