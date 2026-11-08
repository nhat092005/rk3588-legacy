# Throughput under P3

All runs realized: True

| code | config | FPS runs | median | spread | realized |
|---|---|---|---|---|---|
| T1 | 1 context CORE_0 | [54.7, 54.66, 54.62] | 54.66 | 0.08 | True |
| T2 | 1 context CORE_0_1 | [55.86, 55.76, 55.72] | 55.76 | 0.14 | True |
| T3 | 1 context CORE_0_1_2 | [56.09, 56.1, 56.15] | 56.10 | 0.06 | True |
| T4 | 2 context CORE_0 và CORE_1 | [102.09, 101.62, 102.0] | 102.00 | 0.47 | True |
| T5 | 3 context (S2) | [140.67, 141.47, 141.18] | 141.18 | 0.80 | True |
| T6 | S3 pipeline | [147.39, 146.88, 147.64] | 147.39 | 0.76 | True |
| T7 | 3 tiến trình (chẩn đoán) | [143.51, 143.44, 143.6] | 143.51 | 0.16 | True |

M12: {'best': 'T3', 'gain_fps': 85.08000000000001, 'gain_pct': 151.65775401069521, 'larger_spread': 0.8000000000000114, 'verdict': 'hơn'}
M13: {'gain_fps': 6.2099999999999795, 'gain_pct': 4.398640033999135, 'larger_spread': 0.8000000000000114, 'cpu_t5': 67.5, 'cpu_t6': 72.4, 'cpu_gain': -4.900000000000006, 'cpu_spread': 2.299999999999997, 'verdict': 'không đáng giữ', 'workers': {'pre': 1, 'post': 1}}
M3: {'diff_fps': 2.329999999999984, 'diff_pct': 1.6503754072814734, 'larger_spread': 0.8000000000000114, 'verdict': 'không phân biệt được'}
M14: {'t1_after_init': 172.265625, 't5_after_init': 197.91796875, 'per_added_context': 12.826171875, 't1_peak': 242.5, 't5_peak': 285.6}
