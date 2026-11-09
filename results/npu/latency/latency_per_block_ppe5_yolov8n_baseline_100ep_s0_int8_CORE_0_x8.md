# Latency per block: ppe5_yolov8n_baseline_100ep_s0 int8 CORE_0

3 independent sessions. Block time = median over the eval_perf calls of a session, then median over sessions (perf_debug=True, so debug-mode time, not latency). model sha256 `a501f94ee391`, commit `d84836b0`.

## Totals

- latency (perf_debug=False), ms per session: [16.536, 16.539, 16.502], median 16.536, gap 0.22%
- debug total (Total Operator Elapsed Per Frame Time), ms per session: [16.722, 16.755, 16.661], median 16.722, gap 0.56%
- debug_ratio (debug total / latency): 1.011, sum_ratio (sum of layer times / total, per eval_perf call) min and max per session: [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
- per-row additive overhead of each session (intercept of time = a + b x est. cycles time over the Conv rows, us), numbered sessions then B2: [5.1, 5.2, 6.3]. The column 'share without per-row overhead' subtracts that intercept from every row with time above zero: a sensitivity bound, not a correction.

## Table 1: block L0-L22

| Block | rows | us (median) | B2 us | spread between sessions | spread between calls | share % | B2 share % | share without per-row overhead % (sensitivity bound) | share spread between sessions | cycle share % | GFLOPs | us per GFLOP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 1 | 1851 | - | 1.0% | 1.7% | 11.0 | - | 11.4 | 0.6% | 16.3 | 0.088 | 20921 |
| L1 | 1 | 511 | - | 0.0% | 0.4% | 3.1 | - | 3.1 | 0.4% | 4.1 | 0.236 | 2166 |
| L2 | 7 | 2238 | - | 0.4% | 0.2% | 13.4 | - | 13.7 | 0.4% | 11.7 | 0.367 | 6098 |
| L3 | 1 | 261 | - | 0.4% | 0.4% | 1.6 | - | 1.6 | 0.3% | 2.0 | 0.236 | 1106 |
| L4 | 10 | 1277 | - | 0.5% | 0.3% | 7.6 | - | 7.6 | 0.4% | 5.9 | 0.629 | 2030 |
| L5 | 1 | 152 | - | 0.0% | 0.7% | 0.9 | - | 0.9 | 0.6% | 2.0 | 0.236 | 644 |
| L6 | 10 | 720 | - | 0.6% | 1.0% | 4.3 | - | 4.1 | 0.3% | 5.4 | 0.629 | 1144 |
| L7 | 1 | 137 | - | 0.0% | 0.7% | 0.8 | - | 0.8 | 0.6% | 2.0 | 0.236 | 581 |
| L8 | 7 | 361 | - | 0.0% | 1.4% | 2.2 | - | 2.0 | 0.4% | 3.2 | 0.367 | 984 |
| L9 | 6 | 228 | - | 0.4% | 0.9% | 1.4 | - | 1.2 | 0.6% | 1.1 | 0.131 | 1740 |
| L10 | 1 | 62 | - | 1.6% | 3.2% | 0.4 | - | 0.4 | 1.8% | 0.0 | - | - |
| L11 | 1 | 103 | - | 1.0% | 1.0% | 0.6 | - | 0.6 | 1.0% | 0.0 | - | - |
| L12 | 6 | 524 | - | 0.6% | 0.6% | 3.1 | - | 3.1 | 0.7% | 4.1 | 0.472 | 1111 |
| L13 | 1 | 116 | - | 0.0% | 0.9% | 0.7 | - | 0.7 | 0.6% | 0.0 | - | - |
| L14 | 1 | 198 | - | 0.5% | 1.5% | 1.2 | - | 1.2 | 0.3% | 0.0 | - | - |
| L15 | 6 | 950 | - | 0.3% | 0.2% | 5.7 | - | 5.7 | 0.3% | 4.3 | 0.472 | 2013 |
| L16 | 1 | 92 | - | 0.0% | 1.1% | 0.6 | - | 0.5 | 0.4% | 1.0 | 0.118 | 780 |
| L17 | 1 | 54 | - | 3.7% | 3.7% | 0.3 | - | 0.3 | 3.1% | 0.0 | - | - |
| L18 | 6 | 497 | - | 0.4% | 0.8% | 3.0 | - | 2.9 | 0.0% | 3.4 | 0.393 | 1264 |
| L19 | 1 | 79 | - | 0.0% | 1.3% | 0.5 | - | 0.5 | 0.6% | 1.0 | 0.118 | 670 |
| L20 | 1 | 31 | - | 3.2% | 6.5% | 0.2 | - | 0.2 | 3.0% | 0.0 | - | - |
| L21 | 6 | 347 | - | 0.6% | 0.9% | 2.1 | - | 2.0 | 0.4% | 3.4 | 0.393 | 882 |
| L22 | 37 | 5642 | - | 0.7% | 0.9% | 33.7 | - | 33.9 | 0.2% | 29.1 | 2.965 | 1903 |
| io | 2 | 10 | - | 10.0% | 20.0% | 0.1 | - | 0.0 | 9.7% | 0.0 | - | - |
| post_detect | 3 | 282 | - | 2.1% | 1.4% | 1.7 | - | 1.6 | 1.5% | 0.0 | - | - |

## Table 2: part inside block

| Block | Part | us (median) | spread between sessions | spread between calls | share % | share in block % |
|---|---|---|---|---|---|---|
| L0 | conv | 1851 | 1.0% | 1.7% | 11.0 | 100.0 |
| L1 | conv | 511 | 0.0% | 0.4% | 3.1 | 100.0 |
| L2 | bottleneck | 973 | 0.1% | 0.3% | 5.8 | 43.5 |
| L2 | concat | 190 | 1.6% | 0.5% | 1.1 | 8.5 |
| L2 | cv1 | 461 | 0.0% | 0.4% | 2.8 | 20.6 |
| L2 | cv2 | 487 | 0.2% | 0.2% | 2.9 | 21.8 |
| L2 | split | 127 | 3.1% | 3.1% | 0.8 | 5.7 |
| L3 | conv | 261 | 0.4% | 0.4% | 1.6 | 100.0 |
| L4 | bottleneck | 585 | 0.3% | 0.5% | 3.5 | 45.8 |
| L4 | concat | 128 | 2.3% | 1.5% | 0.8 | 10.0 |
| L4 | cv1 | 235 | 0.0% | 0.4% | 1.4 | 18.4 |
| L4 | cv2 | 260 | 0.0% | 0.4% | 1.6 | 20.4 |
| L4 | split | 68 | 4.4% | 3.0% | 0.4 | 5.3 |
| L5 | conv | 152 | 0.0% | 0.7% | 0.9 | 100.0 |
| L6 | bottleneck | 352 | 0.0% | 1.4% | 2.1 | 48.9 |
| L6 | concat | 70 | 4.3% | 1.4% | 0.4 | 9.7 |
| L6 | cv1 | 124 | 0.8% | 0.8% | 0.7 | 17.2 |
| L6 | cv2 | 137 | 0.0% | 0.7% | 0.8 | 19.0 |
| L6 | split | 38 | 5.3% | 13.5% | 0.2 | 5.3 |
| L7 | conv | 137 | 0.0% | 0.7% | 0.8 | 100.0 |
| L8 | bottleneck | 165 | 0.0% | 1.8% | 1.0 | 45.7 |
| L8 | concat | 31 | 0.0% | 3.2% | 0.2 | 8.6 |
| L8 | cv1 | 69 | 1.4% | 1.4% | 0.4 | 19.1 |
| L8 | cv2 | 73 | 1.4% | 5.5% | 0.4 | 20.2 |
| L8 | split | 23 | 0.0% | 4.3% | 0.1 | 6.4 |
| L9 | concat | 39 | 0.0% | 5.1% | 0.2 | 17.1 |
| L9 | cv1 | 43 | 0.0% | 2.3% | 0.3 | 18.9 |
| L9 | cv2 | 77 | 0.0% | 1.3% | 0.5 | 33.8 |
| L9 | maxpool | 69 | 0.0% | 1.4% | 0.4 | 30.3 |
| L10 | resize | 62 | 1.6% | 3.2% | 0.4 | 100.0 |
| L11 | concat | 103 | 1.0% | 1.0% | 0.6 | 100.0 |
| L12 | bottleneck | 148 | 0.7% | 1.4% | 0.9 | 28.3 |
| L12 | concat | 53 | 1.9% | 1.9% | 0.3 | 10.1 |
| L12 | cv1 | 157 | 0.6% | 0.6% | 0.9 | 30.0 |
| L12 | cv2 | 128 | 0.8% | 0.8% | 0.8 | 24.5 |
| L12 | split | 38 | 5.3% | 2.7% | 0.2 | 7.3 |
| L13 | resize | 116 | 0.0% | 0.9% | 0.7 | 100.0 |
| L14 | concat | 198 | 0.5% | 1.5% | 1.2 | 100.0 |
| L15 | bottleneck | 245 | 0.4% | 0.4% | 1.5 | 25.8 |
| L15 | concat | 99 | 0.0% | 1.0% | 0.6 | 10.4 |
| L15 | cv1 | 294 | 0.0% | 0.3% | 1.8 | 30.9 |
| L15 | cv2 | 243 | 0.4% | 0.4% | 1.5 | 25.7 |
| L15 | split | 68 | 4.4% | 3.0% | 0.4 | 7.2 |
| L16 | conv | 92 | 0.0% | 1.1% | 0.6 | 100.0 |
| L17 | concat | 54 | 3.7% | 3.7% | 0.3 | 100.0 |
| L18 | bottleneck | 148 | 0.7% | 1.4% | 0.9 | 29.8 |
| L18 | concat | 54 | 3.7% | 1.9% | 0.3 | 10.9 |
| L18 | cv1 | 128 | 0.0% | 0.8% | 0.8 | 25.8 |
| L18 | cv2 | 128 | 0.8% | 1.6% | 0.8 | 25.7 |
| L18 | split | 39 | 2.6% | 2.6% | 0.2 | 7.8 |
| L19 | conv | 79 | 0.0% | 1.3% | 0.5 | 100.0 |
| L20 | concat | 31 | 3.2% | 6.5% | 0.2 | 100.0 |
| L21 | bottleneck | 148 | 0.0% | 1.4% | 0.9 | 42.7 |
| L21 | concat | 32 | 3.1% | 3.1% | 0.2 | 9.2 |
| L21 | cv1 | 72 | 0.0% | 1.4% | 0.4 | 20.7 |
| L21 | cv2 | 73 | 1.4% | 1.4% | 0.4 | 21.0 |
| L21 | split | 24 | 4.2% | 4.3% | 0.1 | 6.9 |
| L22 | box_branch | 979 | 0.1% | 0.5% | 5.9 | 17.4 |
| L22 | cls_branch | 939 | 0.2% | 0.5% | 5.6 | 16.6 |
| L22 | decode | 851 | 2.0% | 1.1% | 5.1 | 15.1 |
| L22 | dfl | 2869 | 0.8% | 1.8% | 17.1 | 50.9 |
| io | InputOperator | 4 | 0.0% | 25.0% | 0.0 | 40.0 |
| io | OutputOperator | 6 | 16.7% | 33.3% | 0.0 | 60.0 |
| post_detect | Concat | 124 | 1.6% | 1.6% | 0.7 | 43.6 |
| post_detect | Mul | 91 | 1.1% | 1.1% | 0.5 | 32.3 |
| post_detect | Reshape | 67 | 4.5% | 3.0% | 0.4 | 23.9 |

## Table 3: by OpType

| OpType | Calls | us (median) | share % |
|---|---|---|---|
| ConvExSwish | 57 | 10445 | 62.45 |
| exSoftmax13 | 1 | 2051 | 12.27 |
| Concat | 17 | 1444 | 8.63 |
| Transpose | 1 | 580 | 3.46 |
| Split | 9 | 544 | 3.25 |
| Conv | 7 | 490 | 2.92 |
| Add | 8 | 330 | 1.98 |
| Reshape | 7 | 317 | 1.90 |
| Resize | 2 | 178 | 1.07 |
| Mul | 2 | 118 | 0.71 |
| Sigmoid | 1 | 76 | 0.46 |
| Sub | 2 | 72 | 0.43 |
| MaxPool | 3 | 69 | 0.41 |
| OutputOperator | 1 | 6 | 0.04 |
| InputOperator | 1 | 4 | 0.02 |

## Acceptance checks

| Check | Result | Detail |
|---|---|---|
| A1 | pass | build table 119 rows vs board table 119 rows, (OpType, FullName) sequence identical: True |
| A2 | pass | 6 frequency reports equal to {'CPU': [1800000, 2256000, 2304000], 'NPU': [1000000000], 'DDR': [2112000000]} |
| A3 | pass | NPU rows (not Reshape) with time > 0 and WorkLoad != 100.0%/0.0%/0.0%: [] |
| A4 | pass | rows outside model.N or with an unknown part: [('io', 'InputOperator'), ('io', 'OutputOperator'), ('post_detect', 'Concat'), ('post_detect', 'Mul'), ('post_detect', 'Reshape')] |
| A5 | pass | latency gap between 3 sessions 0.22% (max 5%) |
| A6 | pass | sum of block shares 0.9999, sum_ratio min/max per session [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], debug_ratio 1.011 |
| A8 | pass | sessions only: debug total gap 0.56% (max 5%), blocks with share >= 2% whose share spread exceeds 10%: none |
| A7 | pass | provenance keys ('model_sha256', 'git_commit', 'git_dirty', 'sdk_version') present in every session file |
