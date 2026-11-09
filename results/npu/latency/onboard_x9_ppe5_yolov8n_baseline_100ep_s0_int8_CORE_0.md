# X9: irqbalance and the NPU interrupt affinity, ppe5_yolov8n_baseline_100ep_s0

| trial | samples | samples not 4/4 | first change | CPU4 share of interrupts |
|---|---|---|---|---|
| 1 | 55 | 54 | 36:0-7/0 | 0.019 |
| 2 | 49 | 0 | None | 1.000 |
| 3 | 49 | 0 | None | 1.000 |

Verdict: irqbalance đổi affinity ngắt NPU trong lúc chạy (đã quan sát ở 1/3 lần). U4: ĐO.
