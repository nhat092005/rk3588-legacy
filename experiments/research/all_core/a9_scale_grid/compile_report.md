# a9: width x depth grid of the dense all-core detector (PC compile, 640x640, nc=5)

| model | prec | layers | params | GFLOPs | cycles M | blockers |
|---|---|---|---|---|---|---|
| dense_w100_d2 | int8 | 43 | 3379771 | 7.218 | 4.8 | - |
| dense_w150_d2 | int8 | 43 | 7593867 | 16.085 | 11.4 | - |
| dense_w150_d3 | int8 | 47 | 9357867 | 20.332 | 13.9 | - |
| dense_w150_d4 | int8 | 51 | 11121867 | 24.579 | 16.4 | - |
| dense_w200_d2 | int8 | 43 | 13490779 | 28.459 | 16.0 | - |
| dense_w200_d3 | int8 | 47 | 16626139 | 36.009 | 19.7 | - |
| dense_w200_d4 | int8 | 51 | 19761499 | 43.558 | 23.4 | - |
| dense_w250_d2 | int8 | 43 | 21070507 | 44.338 | 27.4 | - |
| dense_w250_d3 | int8 | 47 | 25968907 | 56.135 | 33.8 | - |
