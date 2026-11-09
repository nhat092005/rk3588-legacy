# X4: tier-1 eval_perf over the LAN, ppe5_yolov8n_baseline_100ep_s0

Rule: experiments/npu/latency/onboard_x4/rule.txt (written before the data).

## p1lan

All sessions realized: True. Gates (sessions 1 to 3): {'A1': True, 'A2': True, 'A3': True, 'A4': True, 'A5': False, 'A6': True, 'A8': False, 'A7': True}
| session | realized | latency ms | level | first minus rest ms | debug total ms | row intercept us |
|---|---|---|---|---|---|---|
| 1 | True | 21.874 | slow | -3.062 | 23.081 | 51.7 |
| 2 | True | 17.254 | mid | +2.064 | 19.100 | 13.8 |
| 3 | True | 17.829 | mid | +0.786 | 21.285 | 45.3 |
| 4 | True | 19.285 | mid | +0.320 | 18.667 | 13.1 |
| 5 | True | 19.192 | mid | -1.782 | 19.443 | 19.2 |
| 6 | True | 17.574 | mid | -0.632 | 17.426 | 8.3 |
| 7 | True | 19.299 | mid | +0.014 | 21.473 | 46.9 |
| 8 | True | 19.308 | mid | -0.688 | 18.217 | 9.3 |
| 9 | True | 19.316 | mid | +0.622 | 19.084 | 15.8 |
| 10 | True | 19.299 | mid | +0.008 | 21.513 | 46.8 |

U6: không ổn định (median first minus rest +0.011 ms). U10 spread 43.4 us. U13 debug below latency: 4

## p2

All sessions realized: True. Gates (sessions 1 to 3): {'A1': True, 'A2': True, 'A3': True, 'A4': True, 'A5': True, 'A6': True, 'A8': True, 'A7': True}
| session | realized | latency ms | level | first minus rest ms | debug total ms | row intercept us |
|---|---|---|---|---|---|---|
| 1 | True | 16.636 | fast | +0.168 | 16.801 | 5.2 |
| 2 | True | 16.632 | fast | +0.019 | 16.807 | 5.0 |
| 3 | True | 16.643 | fast | +0.011 | 16.815 | 5.1 |
| 4 | True | 16.645 | fast | +0.026 | 16.817 | 5.0 |
| 5 | True | 16.650 | fast | -0.006 | 16.815 | 5.1 |
| 6 | True | 16.653 | fast | +0.180 | 16.812 | 4.9 |
| 7 | True | 16.639 | fast | +0.036 | 16.818 | 5.1 |
| 8 | True | 16.645 | fast | +0.005 | 16.802 | 5.0 |
| 9 | True | 16.641 | fast | -0.009 | 16.798 | 5.0 |
| 10 | True | 16.658 | fast | +0.166 | 16.808 | 5.0 |

U6: không ổn định (median first minus rest +0.022 ms). U10 spread 0.3 us. U13 debug below latency: 0

Gate reading: {'A5': 'trên LAN, ghim lõi làm cổng đạt', 'A8': 'trên LAN, ghim lõi làm cổng đạt'}
Labels: {'U25': 'ĐO', 'U7': 'ĐO'}
FP16 allowed: True
U10: ghim lõi thu hẹp chênh lệch hằng số mỗi dòng giữa các phiên

Masks: {"CORE_0": {"session_medians": [16.676, 16.652, 16.647, 16.657], "diff_vs_core0_ms": 0.0, "realized": true, "irq_lines_per_session": [{"36": 315, "37": 0, "38": 0}, {"36": 315, "37": 0, "38": 0}, {"36": 315, "37": 0, "38": 0}, {"36": 315, "37": 0, "38": 0}], "irq_cpu_per_session": [[0, 0, 0, 0, 315, 0, 0, 0], [0, 0, 0, 0, 315, 0, 0, 0], [0, 0, 0, 0, 315, 0, 0, 0], [0, 0, 0, 0, 315, 0, 0, 0]], "verdict": null}, "AUTO": {"session_medians": [16.643, 16.652, 16.646, 16.641], "diff_vs_core0_ms": -0.00999999999999801, "realized": true, "irq_lines_per_session": [{"36": 315, "37": 0, "38": 0}, {"36": 315, "37": 0, "38": 0}, {"36": 315, "37": 0, "38": 0}, {"36": 315, "37": 0, "38": 0}], "irq_cpu_per_session": [[0, 0, 0, 0, 315, 0, 0, 0], [0, 0, 0, 0, 315, 0, 0, 0], [0, 0, 0, 0, 315, 0, 0, 0], [0, 0, 0, 0, 315, 0, 0, 0]], "verdict": "kh\u00f4ng ph\u00e2n bi\u1ec7t \u0111\u01b0\u1ee3c"}, "CORE_0_1_2": {"session_medians": [16.096, 16.105, 16.119, 16.118], "diff_vs_core0_ms": -0.5429999999999993, "realized": true, "irq_lines_per_session": [{"36": 385, "37": 145, "38": 145}, {"36": 385, "37": 145, "38": 145}, {"36": 385, "37": 145, "38": 145}, {"36": 385, "37": 145, "38": 145}], "irq_cpu_per_session": [[0, 0, 0, 0, 675, 0, 0, 0], [0, 0, 0, 0, 675, 0, 0, 0], [0, 0, 0, 0, 675, 0, 0, 0], [0, 0, 0, 0, 675, 0, 0, 0]], "verdict": "kh\u00e1c"}}

Log: {"False": {"session_medians": [16.642, 16.648, 16.644], "diff_vs_quiet_ms": 0.0, "realized": false, "verdict": null}, "True": {"session_medians": [16.65, 16.643, 16.64], "diff_vs_quiet_ms": -0.0009999999999976694, "realized": true, "verdict": "ch\u01b0a k\u1ebft lu\u1eadn (P2 kh\u00f4ng th\u1ef1c hi\u1ec7n \u0111\u00fang)"}}
