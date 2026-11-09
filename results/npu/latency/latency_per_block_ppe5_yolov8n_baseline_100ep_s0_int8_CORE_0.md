# Latency per block: ppe5_yolov8n_baseline_100ep_s0 int8 CORE_0

3 independent sessions. Block time = median over the eval_perf calls of a session, then median over sessions (perf_debug=True, so debug-mode time, not latency). model sha256 `a501f94ee391`, commit `33212040`.

## Totals

- latency (perf_debug=False), ms per session: [19.418, 22.352, 20.157], median 20.157, gap 14.56%
- debug total (Total Operator Elapsed Per Frame Time), ms per session: [22.945, 23.4, 24.004], median 23.4, gap 4.53%
- debug_ratio (debug total / latency): 1.161, sum_ratio (sum of layer times / total, per eval_perf call) min and max per session: [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
- per-row additive overhead of each session (intercept of time = a + b x est. cycles time over the Conv rows, us), numbered sessions then B2: [50.7, 50.3, 57.2, 15.8]. The column 'share without per-row overhead' subtracts that intercept from every row with time above zero: a sensitivity bound, not a correction.
- reference points, not in the statistics: {'dry_run_B2': {'latency_ms': 19.238, 'debug_total_ms': 18.842}, 'extra_probe_debug_0': {'debug_total_ms': 21.361}, 'extra_probe_debug_1': {'debug_total_ms': 21.336}}

## Table 1: block L0-L22

| Block | rows | us (median) | B2 us | spread between sessions | spread between calls | share % | B2 share % | share without per-row overhead % (sensitivity bound) | share spread between sessions | cycle share % | GFLOPs | us per GFLOP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 1 | 1993 | 1911 | 0.5% | 3.5% | 8.5 | 10.1 | 11.3 | 4.2% | 16.3 | 0.088 | 22526 |
| L1 | 1 | 604 | 565 | 0.0% | 7.3% | 2.6 | 3.0 | 3.2 | 4.7% | 4.1 | 0.236 | 2560 |
| L2 | 7 | 2732 | 2442 | 1.4% | 9.9% | 11.5 | 13.2 | 13.4 | 3.5% | 11.7 | 0.367 | 7444 |
| L3 | 1 | 315 | 314 | 1.0% | 24.7% | 1.4 | 1.7 | 1.5 | 5.3% | 2.0 | 0.236 | 1335 |
| L4 | 10 | 1792 | 1435 | 0.2% | 24.3% | 7.4 | 7.6 | 7.5 | 6.6% | 5.9 | 0.629 | 2848 |
| L5 | 1 | 203 | 164 | 0.5% | 20.7% | 0.8 | 0.9 | 0.9 | 4.2% | 2.0 | 0.236 | 860 |
| L6 | 10 | 1228 | 831 | 0.2% | 32.2% | 5.1 | 4.4 | 4.2 | 4.2% | 5.4 | 0.629 | 1952 |
| L7 | 1 | 188 | 149 | 0.5% | 21.4% | 0.8 | 0.8 | 0.8 | 3.8% | 2.0 | 0.236 | 797 |
| L8 | 7 | 714 | 440 | 0.1% | 38.6% | 3.0 | 2.3 | 2.1 | 4.2% | 3.2 | 0.367 | 1945 |
| L9 | 6 | 529 | 293 | 0.2% | 44.7% | 2.2 | 1.6 | 1.3 | 4.3% | 1.1 | 0.131 | 4036 |
| L10 | 1 | 113 | 74 | 0.0% | 35.4% | 0.5 | 0.4 | 0.4 | 4.4% | 0.0 | - | - |
| L11 | 1 | 155 | 114 | 0.6% | 29.0% | 0.6 | 0.6 | 0.6 | 3.7% | 0.0 | - | - |
| L12 | 6 | 828 | 593 | 0.0% | 28.7% | 3.4 | 3.2 | 3.1 | 4.2% | 4.1 | 0.472 | 1755 |
| L13 | 1 | 166 | 127 | 0.0% | 24.1% | 0.7 | 0.7 | 0.7 | 4.1% | 0.0 | - | - |
| L14 | 1 | 252 | 208 | 0.4% | 15.9% | 1.1 | 1.1 | 1.2 | 3.0% | 0.0 | - | - |
| L15 | 6 | 1255 | 1012 | 6.0% | 38.2% | 5.3 | 5.4 | 5.3 | 3.3% | 4.3 | 0.472 | 2660 |
| L16 | 1 | 142 | 103 | 27.5% | 78.8% | 0.6 | 0.5 | 0.5 | 12.3% | 1.0 | 0.118 | 1204 |
| L17 | 1 | 106 | 65 | 36.8% | 117.6% | 0.4 | 0.3 | 0.3 | 23.7% | 0.0 | - | - |
| L18 | 6 | 802 | 563 | 16.2% | 45.7% | 3.6 | 3.0 | 3.0 | 11.5% | 3.4 | 0.393 | 2040 |
| L19 | 1 | 133 | 91 | 30.1% | 61.2% | 0.7 | 0.5 | 0.5 | 22.0% | 1.0 | 0.118 | 1127 |
| L20 | 1 | 82 | 42 | 6.1% | 101.2% | 0.4 | 0.2 | 0.2 | 1.9% | 0.0 | - | - |
| L21 | 6 | 649 | 414 | 14.6% | 54.7% | 2.8 | 2.2 | 2.1 | 10.2% | 3.4 | 0.393 | 1650 |
| L22 | 37 | 7877 | 6517 | 5.9% | 23.6% | 34.0 | 34.6 | 34.4 | 2.1% | 29.1 | 2.965 | 2656 |
| io | 2 | 28 | 21 | 7.1% | 16.7% | 0.1 | 0.1 | 0.0 | 9.2% | 0.0 | - | - |
| post_detect | 3 | 435 | 314 | 0.7% | 45.6% | 1.8 | 1.7 | 1.6 | 8.7% | 0.0 | - | - |

## Table 2: part inside block

| Block | Part | us (median) | spread between sessions | spread between calls | share % | share in block % |
|---|---|---|---|---|---|---|
| L0 | conv | 1993 | 0.5% | 3.5% | 8.5 | 100.0 |
| L1 | conv | 604 | 0.0% | 7.3% | 2.6 | 100.0 |
| L2 | bottleneck | 1130 | 3.8% | 17.4% | 5.0 | 41.3 |
| L2 | concat | 252 | 1.2% | 16.9% | 1.1 | 9.2 |
| L2 | cv1 | 553 | 6.7% | 14.3% | 2.3 | 20.2 |
| L2 | cv2 | 578 | 6.4% | 7.4% | 2.4 | 21.1 |
| L2 | split | 185 | 1.1% | 42.7% | 0.8 | 6.8 |
| L3 | conv | 315 | 1.0% | 24.7% | 1.4 | 100.0 |
| L4 | bottleneck | 889 | 0.1% | 31.4% | 3.7 | 49.7 |
| L4 | concat | 183 | 0.5% | 23.6% | 0.8 | 10.2 |
| L4 | cv1 | 286 | 0.3% | 27.9% | 1.2 | 16.0 |
| L4 | cv2 | 311 | 0.0% | 13.2% | 1.3 | 17.4 |
| L4 | split | 119 | 0.8% | 36.1% | 0.5 | 6.7 |
| L5 | conv | 203 | 0.5% | 20.7% | 0.8 | 100.0 |
| L6 | bottleneck | 654 | 0.3% | 36.4% | 2.7 | 53.3 |
| L6 | concat | 122 | 0.0% | 32.8% | 0.5 | 9.9 |
| L6 | cv1 | 174 | 0.6% | 22.4% | 0.7 | 14.2 |
| L6 | cv2 | 187 | 0.0% | 21.4% | 0.8 | 15.2 |
| L6 | split | 90 | 0.0% | 45.6% | 0.4 | 7.3 |
| L7 | conv | 188 | 0.5% | 21.4% | 0.8 | 100.0 |
| L8 | bottleneck | 316 | 0.0% | 37.7% | 1.3 | 44.3 |
| L8 | concat | 81 | 0.0% | 49.4% | 0.3 | 11.3 |
| L8 | cv1 | 120 | 0.8% | 33.3% | 0.5 | 16.8 |
| L8 | cv2 | 122 | 0.8% | 32.8% | 0.5 | 17.1 |
| L8 | split | 74 | 0.0% | 54.1% | 0.3 | 10.4 |
| L9 | concat | 89 | 0.0% | 43.8% | 0.4 | 16.8 |
| L9 | cv1 | 93 | 0.0% | 43.0% | 0.4 | 17.6 |
| L9 | cv2 | 127 | 0.8% | 31.0% | 0.5 | 24.0 |
| L9 | maxpool | 220 | 0.5% | 54.3% | 0.9 | 41.6 |
| L10 | resize | 113 | 0.0% | 35.4% | 0.5 | 100.0 |
| L11 | concat | 155 | 0.6% | 29.0% | 0.6 | 100.0 |
| L12 | bottleneck | 249 | 0.4% | 31.7% | 1.0 | 30.1 |
| L12 | concat | 105 | 0.0% | 37.1% | 0.4 | 12.7 |
| L12 | cv1 | 207 | 0.0% | 19.8% | 0.9 | 25.0 |
| L12 | cv2 | 178 | 0.0% | 22.5% | 0.7 | 21.5 |
| L12 | split | 89 | 0.0% | 46.1% | 0.4 | 10.7 |
| L13 | resize | 166 | 0.0% | 24.1% | 0.7 | 100.0 |
| L14 | concat | 252 | 0.4% | 15.9% | 1.1 | 100.0 |
| L15 | bottleneck | 345 | 0.3% | 48.0% | 1.4 | 27.5 |
| L15 | concat | 152 | 25.0% | 72.8% | 0.6 | 12.1 |
| L15 | cv1 | 344 | 0.3% | 11.6% | 1.5 | 27.5 |
| L15 | cv2 | 293 | 12.6% | 32.0% | 1.3 | 23.3 |
| L15 | split | 120 | 0.0% | 68.3% | 0.5 | 9.6 |
| L16 | conv | 142 | 27.5% | 78.8% | 0.6 | 100.0 |
| L17 | concat | 106 | 36.8% | 117.6% | 0.4 | 100.0 |
| L18 | bottleneck | 259 | 17.0% | 47.9% | 1.2 | 31.5 |
| L18 | concat | 106 | 0.0% | 40.6% | 0.4 | 13.2 |
| L18 | cv1 | 180 | 23.3% | 46.4% | 0.8 | 22.4 |
| L18 | cv2 | 179 | 0.6% | 22.5% | 0.7 | 22.2 |
| L18 | split | 90 | 47.8% | 93.3% | 0.4 | 11.2 |
| L19 | conv | 133 | 30.1% | 61.2% | 0.7 | 100.0 |
| L20 | concat | 82 | 6.1% | 101.2% | 0.4 | 100.0 |
| L21 | bottleneck | 248 | 0.4% | 32.1% | 1.1 | 38.2 |
| L21 | concat | 82 | 2.4% | 49.4% | 0.3 | 12.5 |
| L21 | cv1 | 127 | 30.7% | 65.9% | 0.6 | 19.6 |
| L21 | cv2 | 123 | 0.0% | 65.0% | 0.5 | 19.0 |
| L21 | split | 83 | 48.2% | 106.8% | 0.4 | 12.8 |
| L22 | box_branch | 1435 | 11.6% | 43.3% | 6.2 | 18.4 |
| L22 | cls_branch | 1399 | 6.2% | 43.4% | 6.2 | 18.5 |
| L22 | decode | 1580 | 13.2% | 50.8% | 6.8 | 20.1 |
| L22 | dfl | 3265 | 0.2% | 6.4% | 14.1 | 41.4 |
| io | InputOperator | 9 | 11.1% | 22.2% | 0.0 | 32.1 |
| io | OutputOperator | 19 | 5.3% | 20.0% | 0.1 | 67.9 |
| post_detect | Concat | 176 | 1.1% | 46.6% | 0.7 | 40.3 |
| post_detect | Mul | 142 | 0.7% | 29.4% | 0.6 | 32.7 |
| post_detect | Reshape | 118 | 1.7% | 66.9% | 0.5 | 27.2 |

## Table 3: by OpType

| OpType | Calls | us (median) | share % |
|---|---|---|---|
| ConvExSwish | 57 | 13513 | 58.88 |
| Concat | 17 | 2341 | 10.03 |
| exSoftmax13 | 1 | 2114 | 9.12 |
| Split | 9 | 1052 | 4.58 |
| Conv | 7 | 884 | 3.83 |
| Transpose | 1 | 837 | 3.72 |
| Add | 8 | 787 | 3.37 |
| Reshape | 7 | 520 | 2.19 |
| Resize | 2 | 279 | 1.16 |
| MaxPool | 3 | 220 | 0.91 |
| Mul | 2 | 220 | 0.96 |
| Sub | 2 | 214 | 0.97 |
| Sigmoid | 1 | 126 | 0.52 |
| OutputOperator | 1 | 19 | 0.08 |
| InputOperator | 1 | 9 | 0.04 |

## Acceptance checks

| Check | Result | Detail |
|---|---|---|
| A1 | pass | build table 119 rows vs board table 119 rows, (OpType, FullName) sequence identical: True |
| A2 | pass | 6 frequency reports equal to {'CPU': [1800000, 2256000, 2304000], 'NPU': [1000000000], 'DDR': [2112000000]} |
| A3 | pass | NPU rows (not Reshape) with time > 0 and WorkLoad != 100.0%/0.0%/0.0%: [] |
| A4 | pass | rows outside model.N or with an unknown part: [('io', 'InputOperator'), ('io', 'OutputOperator'), ('post_detect', 'Concat'), ('post_detect', 'Mul'), ('post_detect', 'Reshape')] |
| A5 | FAIL | latency gap between 3 sessions 14.56% (max 5%) |
| A6 | pass | sum of block shares 0.9931, sum_ratio min/max per session [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], debug_ratio 1.161 |
| A8 | FAIL | sessions only: debug total gap 4.53% (max 5%), blocks with share >= 2% whose share spread exceeds 10%: {'L18': 0.115, 'L21': 0.102} |
| A8+B2 | FAIL | with the dry run: debug total gap 22.28%, blocks over 10%: {'L0': 0.215, 'L1': 0.185, 'L2': 0.143, 'L6': 0.163, 'L8': 0.243, 'L9': 0.321, 'L12': 0.108, 'L18': 0.253, 'L21': 0.303} |
| A7 | pass | provenance keys ('model_sha256', 'git_commit', 'git_dirty', 'sdk_version') present in every session file |
