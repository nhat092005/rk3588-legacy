# Latency per block: ppe5_yolov8n_baseline_100ep_s0 int8 CORE_0

3 independent sessions. Block time = median over the eval_perf calls of a session, then median over sessions (perf_debug=True, so debug-mode time, not latency). model sha256 `a501f94ee391`, commit `d84836b0`.

## Totals

- latency (perf_debug=False), ms per session: [16.636, 16.632, 16.643], median 16.636, gap 0.07%
- debug total (Total Operator Elapsed Per Frame Time), ms per session: [16.801, 16.807, 16.815], median 16.807, gap 0.08%
- debug_ratio (debug total / latency): 1.01, sum_ratio (sum of layer times / total, per eval_perf call) min and max per session: [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
- per-row additive overhead of each session (intercept of time = a + b x est. cycles time over the Conv rows, us), numbered sessions then B2: [5.2, 5.0, 5.1]. The column 'share without per-row overhead' subtracts that intercept from every row with time above zero: a sensitivity bound, not a correction.

## Table 1: block L0-L22

| Block | rows | us (median) | B2 us | spread between sessions | spread between calls | share % | B2 share % | share without per-row overhead % (sensitivity bound) | share spread between sessions | cycle share % | GFLOPs | us per GFLOP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 1 | 1854 | - | 0.2% | 1.7% | 11.0 | - | 11.4 | 0.2% | 16.3 | 0.088 | 20955 |
| L1 | 1 | 511 | - | 0.0% | 0.4% | 3.0 | - | 3.1 | 0.1% | 4.1 | 0.236 | 2166 |
| L2 | 7 | 2251 | - | 0.1% | 0.2% | 13.4 | - | 13.7 | 0.2% | 11.7 | 0.367 | 6133 |
| L3 | 1 | 262 | - | 0.4% | 0.4% | 1.6 | - | 1.6 | 0.3% | 2.0 | 0.236 | 1111 |
| L4 | 10 | 1283 | - | 0.2% | 0.2% | 7.6 | - | 7.6 | 0.2% | 5.9 | 0.629 | 2039 |
| L5 | 1 | 152 | - | 0.0% | 0.7% | 0.9 | - | 0.9 | 0.2% | 2.0 | 0.236 | 644 |
| L6 | 10 | 725 | - | 0.1% | 0.4% | 4.3 | - | 4.2 | 0.2% | 5.4 | 0.629 | 1152 |
| L7 | 1 | 137 | - | 0.0% | 0.7% | 0.8 | - | 0.8 | 0.0% | 2.0 | 0.236 | 581 |
| L8 | 7 | 361 | - | 0.3% | 0.8% | 2.1 | - | 2.0 | 0.6% | 3.2 | 0.367 | 984 |
| L9 | 6 | 228 | - | 0.4% | 1.3% | 1.4 | - | 1.2 | 0.4% | 1.1 | 0.131 | 1740 |
| L10 | 1 | 63 | - | 0.0% | 1.6% | 0.4 | - | 0.4 | 0.3% | 0.0 | - | - |
| L11 | 1 | 105 | - | 1.0% | 1.0% | 0.6 | - | 0.6 | 0.9% | 0.0 | - | - |
| L12 | 6 | 527 | - | 0.0% | 0.6% | 3.1 | - | 3.1 | 0.1% | 4.1 | 0.472 | 1117 |
| L13 | 1 | 116 | - | 0.0% | 1.7% | 0.7 | - | 0.7 | 0.1% | 0.0 | - | - |
| L14 | 1 | 202 | - | 1.0% | 7.4% | 1.2 | - | 1.2 | 0.7% | 0.0 | - | - |
| L15 | 6 | 955 | - | 0.1% | 0.3% | 5.7 | - | 5.7 | 0.1% | 4.3 | 0.472 | 2024 |
| L16 | 1 | 92 | - | 0.0% | 1.1% | 0.5 | - | 0.5 | 0.0% | 1.0 | 0.118 | 780 |
| L17 | 1 | 56 | - | 1.8% | 3.6% | 0.3 | - | 0.3 | 1.7% | 0.0 | - | - |
| L18 | 6 | 499 | - | 0.2% | 0.6% | 3.0 | - | 2.9 | 0.2% | 3.4 | 0.393 | 1269 |
| L19 | 1 | 79 | - | 0.0% | 1.3% | 0.5 | - | 0.5 | 0.0% | 1.0 | 0.118 | 670 |
| L20 | 1 | 31 | - | 3.2% | 3.2% | 0.2 | - | 0.2 | 3.2% | 0.0 | - | - |
| L21 | 6 | 348 | - | 0.6% | 1.4% | 2.1 | - | 2.0 | 0.7% | 3.4 | 0.393 | 885 |
| L22 | 37 | 5680 | - | 0.1% | 1.0% | 33.8 | - | 34.0 | 0.0% | 29.1 | 2.965 | 1915 |
| io | 2 | 10 | - | 10.0% | 27.3% | 0.1 | - | 0.0 | 10.0% | 0.0 | - | - |
| post_detect | 3 | 283 | - | 1.4% | 1.1% | 1.7 | - | 1.7 | 1.4% | 0.0 | - | - |

## Table 2: part inside block

| Block | Part | us (median) | spread between sessions | spread between calls | share % | share in block % |
|---|---|---|---|---|---|---|
| L0 | conv | 1854 | 0.2% | 1.7% | 11.0 | 100.0 |
| L1 | conv | 511 | 0.0% | 0.4% | 3.0 | 100.0 |
| L2 | bottleneck | 974 | 0.2% | 0.2% | 5.8 | 43.3 |
| L2 | concat | 197 | 0.5% | 1.0% | 1.2 | 8.8 |
| L2 | cv1 | 461 | 0.0% | 0.2% | 2.7 | 20.5 |
| L2 | cv2 | 488 | 0.2% | 0.4% | 2.9 | 21.7 |
| L2 | split | 131 | 0.0% | 0.8% | 0.8 | 5.8 |
| L3 | conv | 262 | 0.4% | 0.4% | 1.6 | 100.0 |
| L4 | bottleneck | 585 | 0.2% | 0.3% | 3.5 | 45.6 |
| L4 | concat | 133 | 0.8% | 1.5% | 0.8 | 10.4 |
| L4 | cv1 | 235 | 0.0% | 0.4% | 1.4 | 18.3 |
| L4 | cv2 | 260 | 0.0% | 0.4% | 1.5 | 20.3 |
| L4 | split | 69 | 1.4% | 1.4% | 0.4 | 5.4 |
| L5 | conv | 152 | 0.0% | 0.7% | 0.9 | 100.0 |
| L6 | bottleneck | 353 | 0.3% | 0.9% | 2.1 | 48.7 |
| L6 | concat | 72 | 0.0% | 2.8% | 0.4 | 9.9 |
| L6 | cv1 | 124 | 0.8% | 0.8% | 0.7 | 17.1 |
| L6 | cv2 | 137 | 0.7% | 0.7% | 0.8 | 18.9 |
| L6 | split | 39 | 0.0% | 5.1% | 0.2 | 5.4 |
| L7 | conv | 137 | 0.0% | 0.7% | 0.8 | 100.0 |
| L8 | bottleneck | 165 | 1.2% | 1.2% | 1.0 | 45.7 |
| L8 | concat | 32 | 0.0% | 3.1% | 0.2 | 8.9 |
| L8 | cv1 | 69 | 1.4% | 1.4% | 0.4 | 19.1 |
| L8 | cv2 | 72 | 0.0% | 1.4% | 0.4 | 19.9 |
| L8 | split | 23 | 0.0% | 4.3% | 0.1 | 6.4 |
| L9 | concat | 39 | 2.6% | 5.1% | 0.2 | 17.1 |
| L9 | cv1 | 43 | 0.0% | 2.3% | 0.3 | 18.9 |
| L9 | cv2 | 77 | 0.0% | 1.3% | 0.5 | 33.8 |
| L9 | maxpool | 69 | 0.0% | 2.9% | 0.4 | 30.3 |
| L10 | resize | 63 | 0.0% | 1.6% | 0.4 | 100.0 |
| L11 | concat | 105 | 1.0% | 1.0% | 0.6 | 100.0 |
| L12 | bottleneck | 148 | 0.0% | 1.4% | 0.9 | 28.1 |
| L12 | concat | 55 | 0.0% | 1.8% | 0.3 | 10.4 |
| L12 | cv1 | 157 | 0.6% | 1.9% | 0.9 | 29.8 |
| L12 | cv2 | 128 | 0.0% | 0.8% | 0.8 | 24.3 |
| L12 | split | 39 | 0.0% | 2.6% | 0.2 | 7.4 |
| L13 | resize | 116 | 0.0% | 1.7% | 0.7 | 100.0 |
| L14 | concat | 202 | 1.0% | 7.4% | 1.2 | 100.0 |
| L15 | bottleneck | 245 | 0.4% | 0.4% | 1.5 | 25.7 |
| L15 | concat | 102 | 1.0% | 1.0% | 0.6 | 10.7 |
| L15 | cv1 | 294 | 0.3% | 0.3% | 1.7 | 30.8 |
| L15 | cv2 | 243 | 0.4% | 1.2% | 1.4 | 25.5 |
| L15 | split | 70 | 0.0% | 1.4% | 0.4 | 7.3 |
| L16 | conv | 92 | 0.0% | 1.1% | 0.5 | 100.0 |
| L17 | concat | 56 | 1.8% | 3.6% | 0.3 | 100.0 |
| L18 | bottleneck | 149 | 0.7% | 0.7% | 0.9 | 29.9 |
| L18 | concat | 55 | 0.0% | 1.8% | 0.3 | 11.0 |
| L18 | cv1 | 128 | 0.0% | 0.8% | 0.8 | 25.7 |
| L18 | cv2 | 128 | 0.0% | 0.8% | 0.8 | 25.7 |
| L18 | split | 39 | 0.0% | 2.6% | 0.2 | 7.8 |
| L19 | conv | 79 | 0.0% | 1.3% | 0.5 | 100.0 |
| L20 | concat | 31 | 3.2% | 3.2% | 0.2 | 100.0 |
| L21 | bottleneck | 148 | 0.7% | 1.4% | 0.9 | 42.4 |
| L21 | concat | 32 | 0.0% | 6.2% | 0.2 | 9.2 |
| L21 | cv1 | 72 | 0.0% | 1.4% | 0.4 | 20.7 |
| L21 | cv2 | 72 | 1.4% | 1.4% | 0.4 | 20.7 |
| L21 | split | 24 | 4.2% | 4.3% | 0.1 | 6.9 |
| L22 | box_branch | 980 | 0.1% | 0.4% | 5.8 | 17.2 |
| L22 | cls_branch | 940 | 0.2% | 1.1% | 5.6 | 16.5 |
| L22 | decode | 862 | 0.2% | 0.9% | 5.1 | 15.2 |
| L22 | dfl | 2901 | 0.3% | 1.7% | 17.2 | 51.0 |
| io | InputOperator | 4 | 0.0% | 25.0% | 0.0 | 40.0 |
| io | OutputOperator | 6 | 16.7% | 28.6% | 0.0 | 60.0 |
| post_detect | Concat | 124 | 0.0% | 0.8% | 0.7 | 43.8 |
| post_detect | Mul | 92 | 1.1% | 1.1% | 0.5 | 32.4 |
| post_detect | Reshape | 68 | 4.4% | 2.9% | 0.4 | 23.9 |

## Table 3: by OpType

| OpType | Calls | us (median) | share % |
|---|---|---|---|
| ConvExSwish | 57 | 10453 | 62.15 |
| exSoftmax13 | 1 | 2057 | 12.24 |
| Concat | 17 | 1480 | 8.79 |
| Transpose | 1 | 604 | 3.59 |
| Split | 9 | 554 | 3.30 |
| Conv | 7 | 490 | 2.91 |
| Add | 8 | 334 | 1.98 |
| Reshape | 7 | 317 | 1.89 |
| Resize | 2 | 178 | 1.06 |
| Mul | 2 | 120 | 0.71 |
| Sigmoid | 1 | 77 | 0.46 |
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
| A5 | pass | latency gap between 3 sessions 0.07% (max 5%) |
| A6 | pass | sum of block shares 0.9999, sum_ratio min/max per session [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], debug_ratio 1.01 |
| A8 | pass | sessions only: debug total gap 0.08% (max 5%), blocks with share >= 2% whose share spread exceeds 10%: none |
| A7 | pass | provenance keys ('model_sha256', 'git_commit', 'git_dirty', 'sdk_version') present in every session file |
