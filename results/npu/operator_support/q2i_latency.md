# int8h5 latency cost (P2, interleaved)

All realized: True

- silu|CORE_0: int8 16.684, int8h5 16.581, cost -0.103 ms (-0.62%), gap 0.006, đổi latency, material False
- silu|CORE_0_1_2: int8 16.148, int8h5 16.045, cost -0.103 ms (-0.64%), gap 0.002, đổi latency, material False
- relu|CORE_0: int8 12.995, int8h5 12.886, cost -0.110 ms (-0.85%), gap 0.013, đổi latency, material False
- relu|CORE_0_1_2: int8 8.663, int8h5 8.554, cost -0.109 ms (-1.26%), gap 0.010, đổi latency, material False
- B silu|int8: 3.21% không nhanh hơn
- B silu|int8h5: 3.23% không nhanh hơn
- B relu|int8: 33.34% nhanh hơn
- B relu|int8h5: 33.62% nhanh hơn
- A silu|int8: 0.018 không đạt
- A silu|int8h5: 0.018 không đạt
- A relu|int8: 0.976 đạt
- A relu|int8h5: 0.988 đạt
