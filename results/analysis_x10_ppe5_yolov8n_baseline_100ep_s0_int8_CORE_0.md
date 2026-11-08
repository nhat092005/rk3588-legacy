# X10: re-analysis of B3 data, ppe5_yolov8n_baseline_100ep_s0

Fit: time = 1.960 x est + 51.8 us

| row | op | est us | time us | ratio | predicted us |
|---|---|---|---|---|---|
| Conv:/model/model.6/cv1/conv/Conv | ConvExSwish | 25.6 | 174.0 | 6.8 | 102.0 |
| Conv:/model/model.9/cv1/conv/Conv | ConvExSwish | 12.8 | 93.0 | 7.3 | 76.9 |
| Conv:/model/model.22/cv2.1/cv2.1.2/Conv | Conv | 12.8 | 82.0 | 6.4 | 76.9 |
| Conv:/model/model.22/cv3.1/cv3.1.2/Conv | Conv | 6.4 | 73.0 | 11.4 | 64.4 |
| Conv:/model/model.22/cv2.2/cv2.2.2/Conv | Conv | 3.2 | 64.0 | 20.0 | 58.1 |
| Conv:/model/model.22/cv3.2/cv3.2.2/Conv | Conv | 1.6 | 62.0 | 38.8 | 55.0 |

U11: chưa kết luận

| stage | time share | without row constant | GFLOPs share | gap before | gap after |
|---|---|---|---|---|---|
| backbone L0-L9 | 0.444 | 0.462 | 0.390 | +0.054 | +0.071 |
| neck L10-L21 | 0.204 | 0.177 | 0.243 | -0.039 | -0.066 |
| head L22 | 0.332 | 0.344 | 0.367 | -0.034 | -0.022 |

U12: chưa kết luận
