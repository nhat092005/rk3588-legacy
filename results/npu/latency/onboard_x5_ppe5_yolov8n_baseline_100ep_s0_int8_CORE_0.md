# X5: inputs and weights, ppe5_yolov8n_baseline_100ep_s0

| block | cond | half | median ms | std ms | realized |
|---|---|---|---|---|---|
| 1 | s0-zeros | 1 | 16.332 | 0.013 | True |
| 2 | s0-random | 1 | 16.323 | 0.125 | True |
| 3 | s0-img1 | 1 | 16.315 | 0.016 | True |
| 4 | s0-img2 | 1 | 16.322 | 0.376 | True |
| 5 | s1-zeros | 1 | 16.312 | 0.009 | True |
| 6 | s1-img1 | 1 | 16.329 | 0.013 | True |
| 7 | s0-zerosb | 1 | 16.313 | 0.016 | True |
| 8 | s0-zerosb | 2 | 16.316 | 0.027 | True |
| 9 | s1-img1 | 2 | 16.317 | 0.016 | True |
| 10 | s1-zeros | 2 | 16.309 | 0.013 | True |
| 11 | s0-img2 | 2 | 16.315 | 0.331 | True |
| 12 | s0-img1 | 2 | 16.309 | 0.013 | True |
| 13 | s0-random | 2 | 16.314 | 0.011 | True |
| 14 | s0-zeros | 2 | 16.308 | 0.013 | True |

Inputs (difference vs s0-zeros, ms, half 1 and 2): {'s0-random': [0.0005000000000023874, 0.0020000000000024443], 's0-img1': [-0.0074999999999967315, -0.002999999999996561], 's0-img2': [-0.0004999999999988347, 0.0030000000000036664]}
Verdict: thời gian không phụ thuộc giá trị đầu vào trong các đầu vào đã thử

Weights: {'s1-zeros vs s0-zeros': [-0.00999999999999801, -0.002999999999996561], 's1-img1 vs s0-img1': [0.013999999999999346, 0.007999999999999119]}
Verdict: thời gian không phụ thuộc giá trị trọng số trong hai mô hình đã thử
