# Latency per block: ppe5_yolov8n_baseline_100ep_s0 int8 CORE_0

3 independent sessions. Block time = median over the eval_perf calls of a session, then median over sessions (perf_debug=True, so debug-mode time, not latency). model sha256 `a501f94ee391`, commit `d84836b0`.

## Totals

- latency (perf_debug=False), ms per session: [21.874, 17.254, 17.829], median 17.829, gap 25.91%
- debug total (Total Operator Elapsed Per Frame Time), ms per session: [23.081, 19.1, 21.285], median 21.285, gap 18.70%
- debug_ratio (debug total / latency): 1.194, sum_ratio (sum of layer times / total, per eval_perf call) min and max per session: [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
- per-row additive overhead of each session (intercept of time = a + b x est. cycles time over the Conv rows, us), numbered sessions then B2: [51.7, 13.8, 45.3]. The column 'share without per-row overhead' subtracts that intercept from every row with time above zero: a sensitivity bound, not a correction.

## Table 1: block L0-L22

| Block | rows | us (median) | B2 us | spread between sessions | spread between calls | share % | B2 share % | share without per-row overhead % (sensitivity bound) | share spread between sessions | cycle share % | GFLOPs | us per GFLOP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 1 | 1956 | - | 3.5% | 3.7% | 8.9 | - | 11.3 | 19.1% | 16.3 | 0.088 | 22108 |
| L1 | 1 | 552 | - | 14.1% | 13.4% | 2.6 | - | 3.1 | 8.0% | 4.1 | 0.236 | 2340 |
| L2 | 7 | 2540 | - | 6.2% | 8.8% | 11.9 | - | 13.8 | 8.8% | 11.7 | 0.367 | 6921 |
| L3 | 1 | 303 | - | 13.9% | 25.1% | 1.4 | - | 1.6 | 5.2% | 2.0 | 0.236 | 1284 |
| L4 | 10 | 1516 | - | 26.0% | 25.7% | 7.6 | - | 7.3 | 5.9% | 5.9 | 0.629 | 2410 |
| L5 | 1 | 193 | - | 19.7% | 20.2% | 0.9 | - | 0.9 | 4.8% | 2.0 | 0.236 | 818 |
| L6 | 10 | 1139 | - | 34.3% | 34.1% | 5.1 | - | 4.2 | 18.0% | 5.4 | 0.629 | 1810 |
| L7 | 1 | 179 | - | 21.2% | 21.8% | 0.8 | - | 0.8 | 6.3% | 2.0 | 0.236 | 759 |
| L8 | 7 | 652 | - | 41.6% | 41.9% | 3.0 | - | 2.1 | 24.3% | 3.2 | 0.367 | 1777 |
| L9 | 6 | 478 | - | 46.7% | 48.1% | 2.2 | - | 1.2 | 31.0% | 1.1 | 0.131 | 3647 |
| L10 | 1 | 104 | - | 36.5% | 39.4% | 0.5 | - | 0.4 | 20.6% | 0.0 | - | - |
| L11 | 1 | 147 | - | 26.5% | 26.5% | 0.6 | - | 0.6 | 11.4% | 0.0 | - | - |
| L12 | 6 | 776 | - | 30.3% | 29.9% | 3.5 | - | 3.1 | 13.5% | 4.1 | 0.472 | 1645 |
| L13 | 1 | 158 | - | 24.7% | 33.9% | 0.7 | - | 0.7 | 8.2% | 0.0 | - | - |
| L14 | 1 | 244 | - | 16.0% | 21.1% | 1.1 | - | 1.2 | 4.2% | 0.0 | - | - |
| L15 | 6 | 1202 | - | 19.4% | 22.3% | 5.4 | - | 5.6 | 4.1% | 4.3 | 0.472 | 2547 |
| L16 | 1 | 134 | - | 29.1% | 57.0% | 0.6 | - | 0.5 | 13.6% | 1.0 | 0.118 | 1136 |
| L17 | 1 | 97 | - | 40.2% | 77.4% | 0.4 | - | 0.3 | 22.7% | 0.0 | - | - |
| L18 | 6 | 750 | - | 18.7% | 55.2% | 3.5 | - | 3.0 | 3.2% | 3.4 | 0.393 | 1907 |
| L19 | 1 | 121 | - | 31.4% | 62.8% | 0.5 | - | 0.5 | 13.5% | 1.0 | 0.118 | 1026 |
| L20 | 1 | 73 | - | 52.1% | 101.2% | 0.3 | - | 0.2 | 34.1% | 0.0 | - | - |
| L21 | 6 | 598 | - | 39.1% | 48.8% | 2.8 | - | 2.0 | 21.7% | 3.4 | 0.393 | 1521 |
| L22 | 37 | 7061 | - | 13.7% | 24.6% | 33.8 | - | 34.4 | 4.1% | 29.1 | 2.965 | 2381 |
| io | 2 | 23 | - | 65.2% | 16.0% | 0.1 | - | 0.0 | 67.8% | 0.0 | - | - |
| post_detect | 3 | 409 | - | 28.6% | 46.1% | 1.8 | - | 1.6 | 13.5% | 0.0 | - | - |

## Table 2: part inside block

| Block | Part | us (median) | spread between sessions | spread between calls | share % | share in block % |
|---|---|---|---|---|---|---|
| L0 | conv | 1956 | 3.5% | 3.7% | 8.9 | 100.0 |
| L1 | conv | 552 | 14.1% | 13.4% | 2.6 | 100.0 |
| L2 | bottleneck | 1130 | 3.3% | 13.8% | 5.2 | 43.2 |
| L2 | concat | 238 | 16.0% | 20.4% | 1.1 | 9.4 |
| L2 | cv1 | 503 | 8.3% | 15.5% | 2.4 | 19.4 |
| L2 | cv2 | 529 | 7.0% | 14.9% | 2.5 | 20.2 |
| L2 | split | 173 | 22.5% | 43.6% | 0.8 | 6.8 |
| L3 | conv | 303 | 13.9% | 25.1% | 1.4 | 100.0 |
| L4 | bottleneck | 710 | 32.5% | 33.0% | 3.6 | 47.0 |
| L4 | concat | 144 | 27.8% | 28.0% | 0.8 | 10.2 |
| L4 | cv1 | 277 | 14.1% | 27.7% | 1.3 | 17.6 |
| L4 | cv2 | 272 | 16.5% | 16.1% | 1.4 | 17.5 |
| L4 | split | 111 | 35.1% | 41.4% | 0.5 | 6.6 |
| L5 | conv | 193 | 19.7% | 20.2% | 0.9 | 100.0 |
| L6 | bottleneck | 601 | 39.1% | 38.9% | 2.7 | 52.8 |
| L6 | concat | 114 | 34.2% | 34.2% | 0.5 | 9.9 |
| L6 | cv1 | 165 | 23.6% | 23.6% | 0.7 | 14.5 |
| L6 | cv2 | 178 | 21.9% | 21.9% | 0.8 | 15.6 |
| L6 | split | 80 | 48.8% | 48.8% | 0.4 | 7.0 |
| L7 | conv | 179 | 21.2% | 21.8% | 0.8 | 100.0 |
| L8 | bottleneck | 289 | 40.5% | 40.5% | 1.3 | 44.3 |
| L8 | concat | 73 | 52.1% | 54.8% | 0.3 | 11.2 |
| L8 | cv1 | 111 | 34.2% | 35.1% | 0.5 | 17.0 |
| L8 | cv2 | 114 | 34.2% | 34.2% | 0.5 | 17.5 |
| L8 | split | 65 | 60.0% | 60.0% | 0.3 | 10.0 |
| L9 | concat | 81 | 48.1% | 46.9% | 0.4 | 17.0 |
| L9 | cv1 | 85 | 44.7% | 45.9% | 0.4 | 18.0 |
| L9 | cv2 | 118 | 33.1% | 32.2% | 0.5 | 24.7 |
| L9 | maxpool | 193 | 54.4% | 60.1% | 0.9 | 40.0 |
| L10 | resize | 104 | 36.5% | 39.4% | 0.5 | 100.0 |
| L11 | concat | 147 | 26.5% | 26.5% | 0.6 | 100.0 |
| L12 | bottleneck | 231 | 34.2% | 33.3% | 1.0 | 29.8 |
| L12 | concat | 96 | 41.7% | 41.7% | 0.4 | 12.4 |
| L12 | cv1 | 198 | 20.2% | 19.7% | 0.9 | 25.5 |
| L12 | cv2 | 170 | 22.9% | 22.4% | 0.7 | 21.9 |
| L12 | split | 80 | 50.0% | 48.8% | 0.4 | 10.3 |
| L13 | resize | 158 | 24.7% | 33.9% | 0.7 | 100.0 |
| L14 | concat | 244 | 16.0% | 21.1% | 1.1 | 100.0 |
| L15 | bottleneck | 327 | 23.9% | 33.8% | 1.4 | 27.2 |
| L15 | concat | 143 | 26.6% | 41.6% | 0.6 | 11.9 |
| L15 | cv1 | 336 | 11.3% | 15.0% | 1.6 | 28.0 |
| L15 | cv2 | 285 | 13.7% | 27.2% | 1.3 | 23.7 |
| L15 | split | 111 | 33.3% | 54.9% | 0.5 | 9.2 |
| L16 | conv | 134 | 29.1% | 57.0% | 0.6 | 100.0 |
| L17 | concat | 97 | 40.2% | 77.4% | 0.4 | 100.0 |
| L18 | bottleneck | 232 | 14.7% | 47.8% | 1.1 | 31.1 |
| L18 | concat | 97 | 39.2% | 78.1% | 0.4 | 12.9 |
| L18 | cv1 | 170 | 22.9% | 44.1% | 0.8 | 22.4 |
| L18 | cv2 | 170 | 22.4% | 45.8% | 0.8 | 22.1 |
| L18 | split | 81 | 46.9% | 91.0% | 0.4 | 10.8 |
| L19 | conv | 121 | 31.4% | 62.8% | 0.5 | 100.0 |
| L20 | concat | 73 | 52.1% | 101.2% | 0.3 | 100.0 |
| L21 | bottleneck | 230 | 33.9% | 33.0% | 1.0 | 38.5 |
| L21 | concat | 74 | 52.7% | 52.7% | 0.3 | 12.4 |
| L21 | cv1 | 114 | 33.3% | 66.4% | 0.5 | 19.1 |
| L21 | cv2 | 114 | 35.1% | 34.2% | 0.5 | 19.1 |
| L21 | split | 65 | 60.0% | 142.9% | 0.3 | 10.9 |
| L22 | box_branch | 1354 | 19.5% | 39.9% | 6.2 | 18.9 |
| L22 | cls_branch | 1315 | 19.6% | 39.9% | 6.0 | 18.4 |
| L22 | decode | 1399 | 33.4% | 53.4% | 6.5 | 19.8 |
| L22 | dfl | 3170 | 9.2% | 5.0% | 14.1 | 43.3 |
| io | InputOperator | 9 | 55.6% | 33.3% | 0.0 | 39.1 |
| io | OutputOperator | 14 | 71.4% | 16.7% | 0.1 | 60.9 |
| post_detect | Concat | 165 | 23.6% | 46.0% | 0.7 | 40.3 |
| post_detect | Mul | 133 | 29.3% | 28.6% | 0.6 | 32.5 |
| post_detect | Reshape | 111 | 35.1% | 70.1% | 0.5 | 27.1 |

## Table 3: by OpType

| OpType | Calls | us (median) | share % |
|---|---|---|---|
| ConvExSwish | 57 | 12704 | 59.69 |
| Concat | 17 | 2148 | 10.09 |
| exSoftmax13 | 1 | 2099 | 9.86 |
| Split | 9 | 927 | 4.31 |
| Transpose | 1 | 847 | 3.80 |
| Conv | 7 | 781 | 3.64 |
| Add | 8 | 586 | 2.75 |
| Reshape | 7 | 482 | 2.24 |
| Resize | 2 | 262 | 1.17 |
| Mul | 2 | 203 | 0.92 |
| MaxPool | 3 | 193 | 0.87 |
| Sub | 2 | 156 | 0.72 |
| Sigmoid | 1 | 118 | 0.53 |
| OutputOperator | 1 | 14 | 0.07 |
| InputOperator | 1 | 9 | 0.04 |

## Acceptance checks

| Check | Result | Detail |
|---|---|---|
| A1 | pass | build table 119 rows vs board table 119 rows, (OpType, FullName) sequence identical: True |
| A2 | pass | 6 frequency reports equal to {'CPU': [1800000, 2256000, 2304000], 'NPU': [1000000000], 'DDR': [2112000000]} |
| A3 | pass | NPU rows (not Reshape) with time > 0 and WorkLoad != 100.0%/0.0%/0.0%: [] |
| A4 | pass | rows outside model.N or with an unknown part: [('io', 'InputOperator'), ('io', 'OutputOperator'), ('post_detect', 'Concat'), ('post_detect', 'Mul'), ('post_detect', 'Reshape')] |
| A5 | FAIL | latency gap between 3 sessions 25.91% (max 5%) |
| A6 | pass | sum of block shares 1.0011, sum_ratio min/max per session [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], debug_ratio 1.194 |
| A8 | FAIL | sessions only: debug total gap 18.70% (max 5%), blocks with share >= 2% whose share spread exceeds 10%: {'L0': 0.191, 'L6': 0.18, 'L8': 0.243, 'L9': 0.31, 'L12': 0.135, 'L21': 0.217} |
| A7 | pass | provenance keys ('model_sha256', 'git_commit', 'git_dirty', 'sdk_version') present in every session file |
