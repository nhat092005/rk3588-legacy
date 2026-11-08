# h5 latency cost (P2, interleaved)

All realized: True

- silu|CORE_0: base 16.530, h5 16.410, cost -0.121 ms (-0.73%), gap 0.065, đổi latency, material False
- silu|CORE_0_1_2: base 16.017, h5 15.937, cost -0.081 ms (-0.51%), gap 0.029, đổi latency, material False
- relu|CORE_0: base 12.866, h5 12.736, cost -0.129 ms (-1.00%), gap 0.041, đổi latency, material False
- relu|CORE_0_1_2: base 8.548, h5 8.442, cost -0.106 ms (-1.24%), gap 0.022, đổi latency, material False
- B silu|base: 3.10% không nhanh hơn
- B silu|h5: 2.88% không nhanh hơn
- B relu|base: 33.56% nhanh hơn
- B relu|h5: 33.72% nhanh hơn
- A silu: 0.018 không đạt
- A relu: 0.988 đạt
