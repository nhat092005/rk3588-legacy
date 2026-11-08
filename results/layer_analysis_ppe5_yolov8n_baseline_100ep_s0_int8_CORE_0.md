# Layer analysis: ppe5_yolov8n_baseline_100ep_s0 int8 CORE_0

3 sessions, perf_debug=True time per layer (median over calls, then over sessions). Total 23106 us. est. cycles time = Total cycles / 1000 us (compiler estimate at the locked NPU clock, not a hardware bound).

## 1. Conv layers against the compiler's cycle estimate

64 Conv rows use 14425 us (62.4% of the total) against 5667 us of estimated cycles time. Per-row ratio time / est.: median 2.66, min 1.35, max 38.75. Least squares fit time = 1.960 x est. + 51.8 us (R2 0.873, descriptive fit over 64 rows, not a hardware model). Rows with DDR cycles above NPU cycles: [21].

| ratio band | rows | time us | est. us |
|---|---|---|---|
| <= 1.5 | 4 | 1258 | 922 |
| 1.5 - 3 | 32 | 7911 | 3603 |
| 3 - 6 | 22 | 4708 | 1080 |
| > 6 | 6 | 548 | 62 |

| ID | block/part | OpType | cycles DDR/NPU/Total | est. us | time us | ratio | MacUsage | RW KB |
|---|---|---|---|---|---|---|---|---|
| 2 | L0/conv | ConvExSwish | 121323/921600/921600 | 922 | 1993 | 2.16 | 2.24/0.00/0.00 | 1202 |
| 3 | L1/conv | ConvExSwish | 104104/230400/230400 | 230 | 604 | 2.62 | 20.35/0.00/0.00 | 1604 |
| 10 | L2/cv2 | ConvExSwish | 86658/102400/102400 | 102 | 578 | 5.64 | 7.10/0.00/0.00 | 1201 |
| 4 | L2/cv1 | ConvExSwish | 69320/102400/102400 | 102 | 553 | 5.40 | 5.39/0.00/0.00 | 801 |
| 6 | L2/bottleneck | ConvExSwish | 34741/230400/230400 | 230 | 494 | 2.14 | 12.72/0.00/0.00 | 402 |
| 7 | L2/bottleneck | ConvExSwish | 34741/230400/230400 | 230 | 494 | 2.14 | 11.61/0.00/0.00 | 402 |
| 57 | L15/cv1 | ConvExSwish | 69807/76800/76800 | 77 | 344 | 4.48 | 25.18/0.00/0.00 | 1212 |
| 71 | L22/cls_branch | ConvExSwish | 36213/230400/230400 | 230 | 318 | 1.38 | 72.45/0.00/0.00 | 436 |
| 11 | L3/conv | ConvExSwish | 52750/115200/115200 | 115 | 315 | 2.73 | 41.89/0.00/0.00 | 818 |
| 66 | L22/cls_branch | ConvExSwish | 36213/230400/230400 | 230 | 315 | 1.37 | 83.78/0.00/0.00 | 436 |
| 69 | L22/box_branch | ConvExSwish | 36213/230400/230400 | 230 | 313 | 1.36 | 84.40/0.00/0.00 | 436 |
| 65 | L22/box_branch | ConvExSwish | 36213/230400/230400 | 230 | 312 | 1.35 | 84.71/0.00/0.00 | 436 |
| 21 | L4/cv2 | ConvExSwish | 52317/51200/52317 | 52 | 311 | 5.94 | 16.36/0.00/0.00 | 808 |
| 103 | L22/dfl | Conv | 45463/134400/134400 | 134 | 294 | 2.19 | 0.18/0.00/0.00 | 525 |
| 62 | L15/cv2 | ConvExSwish | 43573/51200/51200 | 51 | 293 | 5.72 | 15.12/0.00/0.00 | 606 |
| 12 | L4/cv1 | ConvExSwish | 34828/51200/51200 | 51 | 286 | 5.59 | 10.36/0.00/0.00 | 404 |
| 49 | L12/cv1 | ConvExSwish | 36754/76800/76800 | 77 | 207 | 2.70 | 45.99/0.00/0.00 | 649 |
| 22 | L5/conv | ConvExSwish | 29135/115200/115200 | 115 | 203 | 1.76 | 69.82/0.00/0.00 | 473 |
| 33 | L7/conv | ConvExSwish | 25542/115200/115200 | 115 | 188 | 1.63 | 77.84/0.00/0.00 | 490 |
| 32 | L6/cv2 | ConvExSwish | 27404/51200/51200 | 51 | 187 | 3.65 | 34.59/0.00/0.00 | 433 |
| 79 | L22/box_branch | ConvExSwish | 16126/115200/115200 | 115 | 184 | 1.60 | 79.45/0.00/0.00 | 272 |
| 80 | L22/cls_branch | ConvExSwish | 16126/115200/115200 | 115 | 183 | 1.59 | 79.45/0.00/0.00 | 272 |
| 67 | L18/cv1 | ConvExSwish | 22728/38400/38400 | 38 | 180 | 4.69 | 27.63/0.00/0.00 | 325 |
| 76 | L18/cv2 | ConvExSwish | 22728/38400/38400 | 38 | 179 | 4.66 | 27.23/0.00/0.00 | 325 |
| 54 | L12/cv2 | ConvExSwish | 22728/38400/38400 | 38 | 178 | 4.64 | 27.63/0.00/0.00 | 325 |
| 23 | L6/cv1 | ConvExSwish | 18053/25600/25600 | 26 | 174 | 6.80 | 18.82/0.00/0.00 | 217 |
| 14 | L4/bottleneck | ConvExSwish | 17717/57600/57600 | 58 | 173 | 3.00 | 43.31/0.00/0.00 | 209 |
| 17 | L4/bottleneck | ConvExSwish | 17717/57600/57600 | 58 | 173 | 3.00 | 43.31/0.00/0.00 | 209 |
| 18 | L4/bottleneck | ConvExSwish | 17717/57600/57600 | 58 | 173 | 3.00 | 42.99/0.00/0.00 | 209 |
| 59 | L15/bottleneck | ConvExSwish | 17717/57600/57600 | 58 | 173 | 3.00 | 42.99/0.00/0.00 | 209 |
| 15 | L4/bottleneck | ConvExSwish | 17717/57600/57600 | 58 | 172 | 2.99 | 43.31/0.00/0.00 | 209 |
| 60 | L15/bottleneck | ConvExSwish | 17717/57600/57600 | 58 | 172 | 2.99 | 43.31/0.00/0.00 | 209 |
| 70 | L22/box_branch | Conv | 34828/51200/51200 | 51 | 155 | 3.03 | 16.52/0.00/0.00 | 404 |
| 63 | L16/conv | ConvExSwish | 23226/57600/57600 | 58 | 142 | 2.47 | 55.92/0.00/0.00 | 436 |
| 77 | L19/conv | ConvExSwish | 17100/57600/57600 | 58 | 133 | 2.31 | 43.31/0.00/0.00 | 345 |
| 73 | L18/bottleneck | ConvExSwish | 10239/57600/57600 | 58 | 130 | 2.26 | 44.31/0.00/0.00 | 136 |
| 74 | L18/bottleneck | ConvExSwish | 10239/57600/57600 | 58 | 128 | 2.22 | 44.65/0.00/0.00 | 136 |
| 46 | L9/cv2 | ConvExSwish | 18616/51200/51200 | 51 | 127 | 2.48 | 58.18/0.00/0.00 | 330 |
| 81 | L21/cv1 | ConvExSwish | 15066/38400/38400 | 38 | 127 | 3.31 | 30.24/0.00/0.00 | 248 |
| 83 | L22/box_branch | ConvExSwish | 10239/57600/57600 | 58 | 127 | 2.20 | 44.65/0.00/0.00 | 136 |
| 91 | L22/box_branch | ConvExSwish | 11667/57600/57600 | 58 | 127 | 2.20 | 65.45/0.00/0.00 | 244 |
| 92 | L22/cls_branch | ConvExSwish | 11667/57600/57600 | 58 | 127 | 2.20 | 44.31/0.00/0.00 | 244 |
| 25 | L6/bottleneck | ConvExSwish | 10239/57600/57600 | 58 | 125 | 2.17 | 67.76/0.00/0.00 | 136 |
| 51 | L12/bottleneck | ConvExSwish | 10239/57600/57600 | 58 | 125 | 2.17 | 67.76/0.00/0.00 | 136 |
| 72 | L22/cls_branch | Conv | 21673/25600/25600 | 26 | 125 | 4.88 | 1.60/0.00/0.00 | 400 |
| 85 | L22/cls_branch | ConvExSwish | 10239/57600/57600 | 58 | 125 | 2.17 | 65.45/0.00/0.00 | 136 |
| 26 | L6/bottleneck | ConvExSwish | 10239/57600/57600 | 58 | 124 | 2.15 | 67.76/0.00/0.00 | 136 |
| 28 | L6/bottleneck | ConvExSwish | 10239/57600/57600 | 58 | 124 | 2.15 | 68.57/0.00/0.00 | 136 |
| 29 | L6/bottleneck | ConvExSwish | 10239/57600/57600 | 58 | 124 | 2.15 | 66.98/0.00/0.00 | 136 |
| 36 | L8/bottleneck | ConvExSwish | 10607/57600/57600 | 58 | 124 | 2.15 | 67.76/0.00/0.00 | 195 |
| 37 | L8/bottleneck | ConvExSwish | 10607/57600/57600 | 58 | 124 | 2.15 | 67.76/0.00/0.00 | 195 |
| 52 | L12/bottleneck | ConvExSwish | 10239/57600/57600 | 58 | 124 | 2.15 | 66.98/0.00/0.00 | 136 |
| 87 | L21/bottleneck | ConvExSwish | 10607/57600/57600 | 58 | 124 | 2.15 | 67.76/0.00/0.00 | 195 |
| 88 | L21/bottleneck | ConvExSwish | 10607/57600/57600 | 58 | 124 | 2.15 | 66.98/0.00/0.00 | 195 |
| 90 | L21/cv2 | ConvExSwish | 15066/38400/38400 | 38 | 123 | 3.20 | 45.71/0.00/0.00 | 248 |
| 40 | L8/cv2 | ConvExSwish | 15066/38400/38400 | 38 | 122 | 3.18 | 46.27/0.00/0.00 | 248 |
| 34 | L8/cv1 | ConvExSwish | 11516/25600/25600 | 26 | 120 | 4.69 | 31.22/0.00/0.00 | 166 |
| 41 | L9/cv1 | ConvExSwish | 7923/12800/12800 | 13 | 93 | 7.27 | 23.70/0.00/0.00 | 133 |
| 84 | L22/box_branch | Conv | 8853/12800/12800 | 13 | 82 | 6.41 | 7.53/0.00/0.00 | 104 |
| 95 | L22/cls_branch | ConvExSwish | 3745/14400/14400 | 14 | 78 | 5.42 | 17.56/0.00/0.00 | 61 |
| 93 | L22/box_branch | ConvExSwish | 3745/14400/14400 | 14 | 77 | 5.35 | 36.00/0.00/0.00 | 61 |
| 86 | L22/cls_branch | Conv | 5439/6400/6400 | 6 | 73 | 11.41 | 1.43/0.00/0.00 | 100 |
| 94 | L22/box_branch | Conv | 2360/3200/3200 | 3 | 64 | 20.00 | 4.85/0.00/0.00 | 29 |
| 96 | L22/cls_branch | Conv | 1380/1600/1600 | 2 | 62 | 38.75 | 0.46/0.00/0.00 | 25 |

## 2. Rows without NPU cycles (not Conv)

8681 us (37.6% of the total). Least squares over 34 Concat, Split and Add rows: time = 144 us per MB read and written + 68.4 us (R2 0.708).

| OpType | calls | time us | share % | RW KB | us per MB |
|---|---|---|---|---|---|
| Concat | 17 | 2336 | 10.11 | 7530 | 318 |
| exSoftmax13 | 1 | 2114 | 9.15 | 525 | 4123 |
| Split | 9 | 1029 | 4.45 | 2531 | 416 |
| Transpose | 1 | 837 | 3.62 | 525 | 1633 |
| Add | 8 | 746 | 3.23 | 2624 | 291 |
| Reshape | 7 | 517 | 2.24 | 943 | 561 |
| Resize | 2 | 279 | 1.21 | 300 | 952 |
| Sub | 2 | 225 | 0.97 | 524 | 440 |
| Mul | 2 | 225 | 0.97 | 393 | 586 |
| MaxPool | 3 | 219 | 0.95 | 150 | 1495 |
| Sigmoid | 1 | 126 | 0.55 | 131 | 985 |
| OutputOperator | 1 | 19 | 0.08 | 73 | 267 |
| InputOperator | 1 | 9 | 0.04 | 0 | - |

## 3. Detect head and decode chain

L22 plus post_detect: 8114 us (35.1%).

| group | time us | share % |
|---|---|---|
| L22/dfl | 3247 | 14.1 |
| L22/decode | 1584 | 6.9 |
| L22/box_branch | 1441 | 6.2 |
| L22/cls_branch | 1406 | 6.1 |
| post_detect/Concat | 176 | 0.8 |
| post_detect/Mul | 142 | 0.6 |
| post_detect/Reshape | 118 | 0.5 |

DFL rows (in graph order):

| ID | OpType | Target | time us |
|---|---|---|---|
| 101 | Reshape | NPU | 2 |
| 102 | exSoftmax13 | NPU | 2114 |
| 103 | Conv | NPU | 294 |
| 104 | Transpose | CPU | 837 |

## 4. Time share against GFLOPs share

| stage | time us | time share % | GFLOPs (Conv2d) | GFLOPs share % |
|---|---|---|---|---|
| backbone L0-L9 | 10257 | 44.4 | 3.156 | 39.0 |
| neck L10-L21 | 4707 | 20.4 | 1.966 | 24.3 |
| head L22 | 7678 | 33.2 | 2.965 | 36.7 |
| io and post_detect | 464 | 2.0 | 0.000 | 0.0 |

## 5. First eval_perf call against the others (per session)

| session | latency first ms | latency rest median ms | debug total first ms | debug total rest median ms |
|---|---|---|---|---|
| 1 | 19.293 | 19.422 | 19.705 | 23.572 |
| 2 | 20.844 | 22.477 | 23.143 | 23.431 |
| 3 | 18.423 | 20.512 | 24.004 | 23.887 |

## 6. Difference of each session against the dry run (B2), per layer

Regression of (time in session - time in B2) over the 119 layers on 1, est. cycles time (us) and RW (KB). A constant intercept with slopes near zero means a per-layer overhead that does not depend on size.

| session | total delta us | mean delta per layer us | intercept us | per est. us | us per MB | R2 |
|---|---|---|---|---|---|---|
| 1 | 4344 | 36.5 | 34.6 | 0.016 | 3.6 | 0.027 |
| 2 | 4485 | 37.7 | 36.1 | 0.031 | 0.4 | 0.056 |
| 3 | 5257 | 44.2 | 41.6 | 0.045 | 1.2 | 0.056 |

Mean delta per layer by group (us):

| session | Conv | Split | Add | Concat | MaxPool | Reshape |
|---|---|---|---|---|---|---|
| 1 | 36.9 | 38.1 | 41.1 | 37.1 | 39.0 | 22.6 |
| 2 | 37.8 | 37.6 | 41.1 | 41.5 | 39.0 | 21.7 |
| 3 | 45.7 | 50.0 | 50.0 | 42.5 | 39.0 | 21.9 |
