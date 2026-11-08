# X8: FP16 against INT8 under P2, ppe5_yolov8n_baseline_100ep_s0

| precision | session | realized | latency ms | debug total ms |
|---|---|---|---|---|
| int8 | 1 | True | 16.536 | 16.722 |
| fp16 | 1 | True | 27.284 | 27.389 |
| int8 | 2 | True | 16.539 | 16.755 |
| fp16 | 2 | True | 27.282 | 27.327 |
| int8 | 3 | True | 16.502 | 16.661 |
| fp16 | 3 | True | 27.267 | 27.432 |

Gates: {'int8': {'A1': True, 'A2': True, 'A3': True, 'A4': True, 'A5': True, 'A6': True, 'A8': True, 'A7': True}, 'fp16': {'A1': True, 'A2': True, 'A3': True, 'A4': True, 'A5': True, 'A6': True, 'A8': True, 'A7': True}}
Latency medians: {'int8': 16.536, 'fp16': 27.282}, FP16 minus INT8 10.745999999999999 ms, ratio 1.649854862119013

| block | INT8 share | FP16 share |
|---|---|---|
| L0 | 0.110 | 0.070 |
| L1 | 0.031 | 0.039 |
| L2 | 0.134 | 0.130 |
| L4 | 0.076 | 0.091 |
| L6 | 0.043 | 0.055 |
| L8 | 0.022 | 0.026 |
| L12 | 0.031 | 0.041 |
| L15 | 0.057 | 0.070 |
| L18 | 0.030 | 0.037 |
| L21 | 0.021 | 0.026 |
| L22 | 0.337 | 0.295 |
