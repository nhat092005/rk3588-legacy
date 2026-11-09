# G4 latency, workload and S1 masks (P2, interleaved)

All sessions realized: True

| model | mask | session medians ms | median | gap ms |
|---|---|---|---|---|
| silu | CORE_0 | [16.507, 16.516] | 16.511 | 0.009 |
| silu | CORE_0_1 | [16.141, 16.161] | 16.151 | 0.020 |
| silu | CORE_0_1_2 | [16.025, 16.047] | 16.036 | 0.022 |
| silu | AUTO | [16.509, 16.575] | 16.542 | 0.066 |
| relu | CORE_0 | [12.839, 12.828] | 12.834 | 0.011 |
| relu | CORE_0_1 | [9.888, 9.889] | 9.889 | 0.001 |
| relu | CORE_0_1_2 | [8.524, 8.524] | 8.524 | 0.000 |
| relu | AUTO | [12.861, 12.856] | 12.858 | 0.005 |

## B

- silu|CORE_0_1: drop 0.360 ms (2.18%), gap 0.020 ms, không nhanh hơn
- silu|CORE_0_1_2: drop 0.475 ms (2.88%), gap 0.022 ms, không nhanh hơn
- relu|CORE_0_1: drop 2.945 ms (22.95%), gap 0.011 ms, nhanh hơn
- relu|CORE_0_1_2: drop 4.310 ms (33.58%), gap 0.011 ms, nhanh hơn

## A

- silu: coverage [0.017788077294136975, 0.017788077294136975], không đạt
- relu: coverage [0.9762825636078174, 0.9762825636078174], đạt
