# Activation variants latency, coverage, G3 (P2, interleaved)

All sessions realized: False

| model | CORE_0 ms | CORE_0_1_2 ms | drop % | gap ms | B | A coverage | CPU ops |
|---|---|---|---|---|---|---|---|
| silu | 16.508 | 16.024 | 2.93 | 0.014 | chưa kết luận (P2 không thực hiện đúng) | 0.018 | ['Transpose'] |
| relu | 12.837 | 8.549 | 33.40 | 0.018 | nhanh hơn | 0.976 | ['Transpose'] |
| leakyrelu | 12.850 | 8.541 | 33.53 | 0.038 | nhanh hơn | 0.976 | ['Transpose'] |
| relu6 | 13.050 | 8.744 | 33.00 | 0.037 | nhanh hơn | 0.976 | ['Transpose'] |
| relu200 | 12.843 | 8.540 | 33.50 | 0.027 | nhanh hơn | 0.976 | ['Transpose'] |
