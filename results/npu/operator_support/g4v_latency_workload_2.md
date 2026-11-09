# Activation variants latency, coverage, G3 (P2, interleaved)

All sessions realized: False

| model | CORE_0 ms | CORE_0_1_2 ms | drop % | gap ms | B | A coverage | CPU ops |
|---|---|---|---|---|---|---|---|
| silu | 16.542 | 16.038 | 3.05 | 0.012 | không nhanh hơn | 0.018 | ['Transpose'] |
| relu | 12.857 | 8.561 | 33.41 | 0.021 | chưa kết luận (P2 không thực hiện đúng) | 0.976 | ['Transpose'] |
| leakyrelu | 12.862 | 8.555 | 33.49 | 0.014 | chưa kết luận (P2 không thực hiện đúng) | 0.976 | ['Transpose'] |
| relu6 | 13.059 | 8.772 | 32.83 | 0.029 | nhanh hơn | 0.976 | ['Transpose'] |
| relu200 | 12.854 | 8.543 | 33.54 | 0.012 | nhanh hơn | 0.976 | ['Transpose'] |
