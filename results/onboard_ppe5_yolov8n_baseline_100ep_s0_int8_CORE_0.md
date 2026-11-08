# On-board C API measurement: ppe5_yolov8n_baseline_100ep_s0 int8 CORE_0

rknn_perf_bench.c on the board (RKNN API 2.3.2, driver 0.9.8, frequencies locked), no PC link. wall_us is the CPU clock around rknn_run, run_duration_us is RKNN_QUERY_PERF_RUN.

## 1. Back-to-back runs (3 independent processes, 1000 runs after 50 warm-up runs)

| batch | n | median us | mean us | min | max | p95 | p99 | CV | drift us per run |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1000 | 19194 | 18964 | 16537 | 22134 | 21144 | 21906 | 5.91% | -0.2157 |
| 2 | 1000 | 18994 | 18978 | 16520 | 22129 | 21799 | 21961 | 7.68% | 1.3078 |
| 3 | 1000 | 20911 | 20416 | 16595 | 22179 | 21935 | 22003 | 6.59% | -0.8025 |

Gap between batch medians: 9.99%. wall_us equals run_duration_us within 5 us on every run: True.

## 2. NPU activity during batch 1 (rknpu debugfs, 0.1 s sampling)

189 samples: power on in 188 (99.5%), Core0 load median 88%, min 0%, max 94%.

## 3. Cold start test (gap between runs inside one process)

| gap | n | median us | min | max | first run us | CV |
|---|---|---|---|---|---|---|
| gap2s | 40 | 19131 | 17316 | 19256 | 19103 | 1.60% |
| gap10s | 20 | 19264 | 17413 | 19277 | 17413 | 2.17% |

## 4. cpuidle counters during each batch (sum over CPUs)

| batch | state | entries | time us |
|---|---|---|---|
| hot_1 | state0 | 88781 | 90585789 |
| hot_1 | state1 | 49377 | 64803998 |
| hot_2 | state0 | 77871 | 90118709 |
| hot_2 | state1 | 51900 | 67529275 |
| hot_3 | state0 | 47427 | 91981013 |
| hot_3 | state1 | 79173 | 76624020 |
| gap2s | state0 | 51389 | 367053296 |
| gap2s | state1 | 15005 | 357174832 |
| gap10s | state0 | 103207 | 809320096 |
| gap10s | state1 | 31348 | 784681907 |
| detail_1 | state0 | 29696 | 20534462 |
| detail_1 | state1 | 22866 | 19249746 |

## 5. Per-layer batches (200 runs each after 20 warm-up runs, detail mode)

| batch | total us median | CV | Conv fit slope | Conv fit intercept us |
|---|---|---|---|---|
| 1 | 21344 | 4.38% | 1.818 | 50.6 |
| 2 | 18588 | 8.94% | 1.818 | 21.6 |
| 3 | 22504 | 7.77% | 1.819 | 59.0 |

| block | us (median over batches) | share % | share spread between batches |
|---|---|---|---|
| L0 | 1833 | 8.6 | 18.4% |
| L1 | 551 | 2.6 | 13.1% |
| L2 | 2529 | 11.9 | 9.3% |
| L3 | 303 | 1.4 | 6.8% |
| L4 | 1696 | 7.9 | 5.0% |
| L5 | 194 | 0.9 | 2.3% |
| L6 | 1139 | 5.3 | 17.3% |
| L7 | 179 | 0.8 | 3.8% |
| L8 | 654 | 3.1 | 25.5% |
| L9 | 477 | 2.2 | 33.4% |
| L10 | 104 | 0.5 | 20.0% |
| L11 | 145 | 0.7 | 8.4% |
| L12 | 774 | 3.6 | 12.6% |
| L13 | 157 | 0.7 | 5.8% |
| L14 | 238 | 1.1 | 3.5% |
| L15 | 1200 | 5.5 | 2.0% |
| L16 | 134 | 0.6 | 11.8% |
| L17 | 97 | 0.5 | 24.1% |
| L18 | 746 | 3.5 | 13.7% |
| L19 | 121 | 0.6 | 14.3% |
| L20 | 73 | 0.3 | 35.9% |
| L21 | 597 | 2.8 | 22.0% |
| L22 | 6984 | 33.2 | 6.2% |
| io | 6 | 0.0 | 119.5% |
| post_detect | 407 | 1.9 | 15.5% |

Blocks above 2% whose share spread between batches exceeds 10%: {'L0': 0.18356803274631467, 'L1': 0.1305579318056489, 'L6': 0.17266780560576403, 'L8': 0.2546909772651926, 'L9': 0.3339367396268422, 'L12': 0.12580840383553302, 'L18': 0.13723016475038585, 'L21': 0.21984211531700756}. DFL (L22/dfl) 2928 us.

| OpType | us | share % |
|---|---|---|
| ConvExSwish | 12760 | 59.77 |
| Concat | 2153 | 10.08 |
| exSoftmax13 | 2096 | 9.82 |
| Split | 917 | 4.30 |
| Conv | 779 | 3.65 |
| Add | 669 | 3.13 |
| Transpose | 550 | 2.58 |
| Reshape | 487 | 2.28 |
| Resize | 261 | 1.22 |
| Mul | 203 | 0.95 |
| MaxPool | 194 | 0.91 |
| Sub | 156 | 0.73 |
| Sigmoid | 118 | 0.55 |
| OutputOperator | 4 | 0.02 |
| InputOperator | 2 | 0.01 |

## 6. Against eval_perf over the PC link (same .rknn, same locked frequencies)

| source | latency ms | note |
|---|---|---|
| on-board hot batch 1 | 19.195 | median of 1000 runs |
| on-board hot batch 2 | 18.994 | median of 1000 runs |
| on-board hot batch 3 | 20.911 | median of 1000 runs |
| eval_perf B2 | 19.238 | debug total 18.842 ms |
| eval_perf B3-1 | 19.418 | debug total 22.945 ms |
| eval_perf B3-2 | 22.352 | debug total 23.400 ms |
| eval_perf B3-3 | 20.157 | debug total 24.004 ms |
| eval_perf E1-1 | 19.881 | median of 20 calls, min 18.349, max 20.365 |
| eval_perf E1-2 | 20.008 | median of 20 calls, min 18.337, max 20.467 |

## 7. E1: two eval_perf sessions of 20 calls (perf_debug=False)

| session | n | median ms | min | max | range | CV | drift us per call | first call | median of the rest | init_runtime s | wall s per call (median) | distinct printed frequency reports |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 19.881 | 18.349 | 20.365 | 10.1% | 4.1% | 41.4 | 18.349 | 19.924 | 286 | 37.0 | 1 |
| 2 | 20 | 20.008 | 18.337 | 20.467 | 10.6% | 3.3% | 1.3 | 18.337 | 20.009 | 252 | 35.4 | 1 |

Gap between the two session medians: 0.64%. 22% of the 40 calls lie below 19.0 ms (median 18.396 ms), the others have median 20.028 ms.

Board sampler (1 Hz, 2108 rows over 2134 s) read only these values, which stayed constant: {'npu_hz': ['1000000000'], 'npu_gov': ['userspace'], 'dmc_hz': ['2112000000'], 'dmc_gov': ['userspace'], 'cpu0_khz': ['1800000'], 'cpu4_khz': ['2256000'], 'cpu6_khz': ['2304000']}; SoC temperature 37.9 to 39.8 C, NPU temperature 37.0 to 38.8 C. A flat trace of these values does not prove that the real clocks were stable.

## 8. Latency levels of the back-to-back runs

Level thresholds from the valleys of the pooled histogram: fast below 17.0 ms, slow from 20.0 ms, mid in between.

| batch | fast share | mid share | slow share | longest run in one level | lag-1 autocorrelation |
|---|---|---|---|---|---|
| 1 | 9.3% | 82.4% | 8.3% | 175 | 0.72 |
| 2 | 17.7% | 66.2% | 16.1% | 90 | 0.91 |
| 3 | 0.7% | 42.6% | 56.7% | 184 | 0.88 |

## 9. Block shares of the 600 detail runs by level of the run total (low below 20.0 ms, high from it)

| level | runs | total us median |
|---|---|---|
| low | 178 | 18188 |
| high | 422 | 21355 |

| block | on-board all % | on-board low % | on-board high % | eval_perf B3 median % | eval_perf B2 % |
|---|---|---|---|---|---|
| L0 | 8.6 | 9.9 | 8.6 | 8.5 | 10.1 |
| L1 | 2.6 | 2.9 | 2.6 | 2.6 | 3.0 |
| L2 | 11.9 | 12.8 | 11.8 | 11.5 | 13.2 |
| L3 | 1.4 | 1.5 | 1.4 | 1.4 | 1.7 |
| L4 | 7.9 | 7.6 | 7.9 | 7.4 | 7.6 |
| L5 | 0.9 | 0.9 | 0.9 | 0.8 | 0.9 |
| L6 | 5.3 | 4.5 | 5.3 | 5.1 | 4.4 |
| L7 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| L8 | 3.1 | 2.4 | 3.1 | 3.0 | 2.3 |
| L9 | 2.2 | 1.6 | 2.2 | 2.2 | 1.6 |
| L10 | 0.5 | 0.4 | 0.5 | 0.5 | 0.4 |
| L11 | 0.7 | 0.6 | 0.7 | 0.6 | 0.6 |
| L12 | 3.6 | 3.2 | 3.6 | 3.4 | 3.2 |
| L13 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |
| L14 | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 |
| L15 | 5.5 | 5.5 | 5.6 | 5.3 | 5.4 |
| L16 | 0.6 | 0.6 | 0.6 | 0.6 | 0.5 |
| L17 | 0.5 | 0.4 | 0.5 | 0.4 | 0.3 |
| L18 | 3.5 | 3.1 | 3.5 | 3.6 | 3.0 |
| L19 | 0.6 | 0.5 | 0.6 | 0.7 | 0.5 |
| L20 | 0.3 | 0.2 | 0.3 | 0.4 | 0.2 |
| L21 | 2.8 | 2.3 | 2.8 | 2.8 | 2.2 |
| L22 | 33.2 | 34.9 | 32.7 | 34.0 | 34.6 |
| io | 0.0 | 0.1 | 0.0 | 0.1 | 0.1 |
| post_detect | 1.9 | 1.9 | 1.9 | 1.8 | 1.7 |

Largest spread (max minus min, in percentage points) of a block share across the five sources (on-board all, low, high, eval_perf B3 median, B2): 2.2 pp over the blocks with at least 2% of the time; per block: L0 1.6, L1 0.4, L2 1.7, L3 0.3, L4 0.5, L5 0.1, L6 0.9, L7 0.1, L8 0.7, L9 0.7, L10 0.1, L11 0.1, L12 0.5, L13 0.1, L14 0.1, L15 0.4, L16 0.1, L17 0.1, L18 0.6, L19 0.2, L20 0.1, L21 0.6, L22 2.2, io 0.1, post_detect 0.2

## 10. E2: CPU placement, NPU interrupt placement, cpuidle (1000 runs per block, order ABCD then DCBA)

Intended conditions: A baseline, B app and NPU IRQs on CPU4, C app and NPU IRQs on CPU0, D cpuidle state1 (cpu-sleep) off. The realized factors below come from the per-run CPU and the NPU interrupt counts, and differ from the intended ones in block 2.

| block | intended | order | n | median ms | p95 ms | CV | fast | mid | slow | app CPU | NPU interrupts: CPU with most (share) | cpu-sleep off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | ABCD | 1000 | 19.323 | 21.983 | 6.8% | 0.4% | 56.7% | 42.9% | migrating | CPU0 (100%) | False |
| 2 | B | ABCD | 1000 | 18.869 | 18.880 | 4.5% | 13.1% | 86.7% | 0.2% | CPU4 | CPU0 (89%) | False |
| 3 | C | ABCD | 1000 | 19.135 | 19.382 | 4.4% | 0.0% | 99.6% | 0.4% | CPU0 | CPU0 (100%) | False |
| 4 | D | ABCD | 1000 | 17.378 | 17.454 | 2.4% | 29.6% | 70.4% | 0.0% | migrating | CPU0 (100%) | True |
| 5 | D | DCBA | 1000 | 17.408 | 17.489 | 0.5% | 0.5% | 99.5% | 0.0% | migrating | CPU0 (100%) | True |
| 6 | C | DCBA | 1000 | 19.165 | 19.371 | 1.6% | 0.0% | 99.8% | 0.2% | CPU0 | CPU0 (100%) | False |
| 7 | B | DCBA | 1000 | 16.321 | 16.373 | 1.9% | 99.4% | 0.3% | 0.3% | CPU4 | CPU4 (100%) | False |
| 8 | A | DCBA | 1000 | 19.388 | 21.908 | 6.6% | 0.0% | 54.2% | 45.8% | migrating | CPU0 (100%) | False |

Median of each window of 100 runs (ms):

| block | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 500-599 | 600-699 | 700-799 | 800-899 | 900-999 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 A | 19.18 | 21.83 | 18.96 | 18.95 | 18.95 | 19.36 | 19.28 | 21.67 | 21.75 | 21.80 |
| 2 B | 16.31 | 18.87 | 18.87 | 18.87 | 18.87 | 18.87 | 18.87 | 18.87 | 18.87 | 18.87 |
| 3 C | 19.10 | 19.12 | 19.17 | 19.16 | 19.15 | 19.12 | 19.11 | 19.15 | 19.11 | 19.18 |
| 4 D | 17.41 | 16.52 | 16.52 | 17.41 | 17.39 | 16.52 | 17.43 | 17.43 | 17.43 | 17.37 |
| 5 D | 17.45 | 17.45 | 17.40 | 17.39 | 17.39 | 17.40 | 17.45 | 17.43 | 17.40 | 17.40 |
| 6 C | 19.13 | 19.15 | 19.15 | 19.15 | 19.19 | 19.16 | 19.21 | 19.19 | 19.18 | 19.17 |
| 7 B | 16.32 | 16.32 | 16.32 | 16.32 | 16.32 | 16.32 | 16.32 | 16.32 | 16.32 | 16.32 |
| 8 A | 21.34 | 21.68 | 19.20 | 19.99 | 19.24 | 19.29 | 21.03 | 19.38 | 19.37 | 19.39 |

Blocks grouped by realized factors (app placement, where most NPU interrupts landed, cpu-sleep):

| app | NPU interrupts | cpu-sleep off | blocks | medians ms | slow share |
|---|---|---|---|---|---|
| migrating | CPU0 | False | [1, 8] | 19.32, 19.39 | 42.9%, 45.8% |
| CPU4 | CPU0 (89%) | False | [2] | 18.87 | 0.2% |
| CPU0 | CPU0 | False | [3, 6] | 19.14, 19.16 | 0.4%, 0.2% |
| migrating | CPU0 | True | [4, 5] | 17.38, 17.41 | 0.0%, 0.0% |
| CPU4 | CPU4 | False | [7] | 16.32 | 0.3% |

Labels. The rule was fixed before the E2 data existed and is not from a source: a condition is labelled 'ủng hộ' when it moves the slow-level share by more than 0.20 in the same direction in both orders, 'không ủng hộ' under 0.10 in both, otherwise 'chưa kết luận'. Blocks that did not realize the intended condition are excluded, so condition B (block 2 not realized) has one valid block and cannot be labelled. 'ủng hộ' here only means that the intervention removes the slow level. It does not identify a mechanism.

| intended condition | slow-share shift vs baseline (ABCD, DCBA) | label |
|---|---|---|
| B app and NPU IRQs on CPU4 (big) | DCBA: -0.46 | chưa kết luận (chỉ một khối thực hiện đúng điều kiện) |
| C app and NPU IRQs on CPU0 (little) | ABCD: -0.42, DCBA: -0.46 | ủng hộ |
| D cpuidle state1 off | ABCD: -0.43, DCBA: -0.46 | ủng hộ |

Consistency check only, not a mechanism:

- NPU interrupts per run: 63 (66150 interrupts in block 7 over 1050 runs).
- Medians of the pooled levels: fast 16.625 ms, mid 19.127 ms, slow 21.661 ms. Gaps 2502 us and 2535 us, which is 40 us and 40 us per interrupt.
- Read on the board after E2 (bench/onboard_e2/board_facts.txt): cpu-sleep exit latency 220 us and minimum residency 1000 us, WFI latency 1 us; irqbalance is active.

Board state log of the run (IRQ affinity saved and restored):

```
npu irqs: 36
37
38
saved irq 36 affinity 0-7
saved irq 37 affinity 0-7
saved irq 38 affinity 0-7
block 1 cond A 04:26:37 pin='' state1_disable=0 irq_aff=0-7 0-7 0-7 
block 2 cond B 04:27:18 pin='taskset -c 4' state1_disable=0 irq_aff=4 4 4 
block 3 cond C 04:27:57 pin='taskset -c 0' state1_disable=0 irq_aff=0 0 0 
block 4 cond D 04:28:38 pin='' state1_disable=1 irq_aff=0-7 0-7 0-7 
block 5 cond D 04:29:16 pin='' state1_disable=1 irq_aff=0-7 0-7 0-7 
block 6 cond C 04:29:54 pin='taskset -c 0' state1_disable=0 irq_aff=0 0 0 
block 7 cond B 04:30:34 pin='taskset -c 4' state1_disable=0 irq_aff=4 4 4 
block 8 cond A 04:31:12 pin='' state1_disable=0 irq_aff=0-7 0-7 0-7 
restored: 0-7 0-7 0-7  idle_state1_disable=0
```

## 11. E2b: condition B with irqbalance stopped (1000 runs per block, order ABBA ABBA)

Another boot than E2, irqbalance stopped for the whole run, so B is compared only with the A blocks of this run. A is also measured with irqbalance stopped. B puts the app and all NPU interrupts on CPU4, which changes three things at once (app placement, interrupt placement, both on the same core).

| block | cond | half | n | median ms | p95 ms | std ms | fast | mid | slow | app CPU (share) | NPU interrupts: CPU with most (share) | interrupts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 1 | 1000 | 20.755 | 21.900 | 1.500 | 1.3% | 45.3% | 53.4% | migrating (31%) | CPU0 (100%) | 66150 |
| 2 | B | 1 | 1000 | 16.303 | 16.313 | 0.022 | 100.0% | 0.0% | 0.0% | CPU4 (100%) | CPU4 (100%) | 66150 |
| 3 | B | 1 | 1000 | 16.289 | 16.298 | 0.025 | 100.0% | 0.0% | 0.0% | CPU4 (100%) | CPU4 (100%) | 66150 |
| 4 | A | 1 | 1000 | 20.694 | 22.117 | 1.517 | 0.3% | 46.8% | 52.9% | migrating (27%) | CPU0 (100%) | 66150 |
| 5 | A | 2 | 1000 | 18.928 | 19.304 | 0.861 | 7.1% | 89.3% | 3.6% | migrating (51%) | CPU0 (100%) | 66150 |
| 6 | B | 2 | 1000 | 16.290 | 16.306 | 0.009 | 100.0% | 0.0% | 0.0% | CPU4 (100%) | CPU4 (100%) | 66150 |
| 7 | B | 2 | 1000 | 16.297 | 16.314 | 0.040 | 99.9% | 0.1% | 0.0% | CPU4 (100%) | CPU4 (100%) | 66150 |
| 8 | A | 2 | 1000 | 21.392 | 21.921 | 1.375 | 0.1% | 33.9% | 66.0% | migrating (60%) | CPU0 (100%) | 66150 |

Slow-share shift of B versus A under each pairing (halves are blocks 1 to 4 and 5 to 8). The pairing was not spelled out in rule.txt, which was written before the data. The half mean is used for the label and was chosen after the data existed.

| pairing | half 1 | half 2 |
|---|---|---|
| half mean of the two A blocks | -0.532 | -0.348 |
| last A block of each half | -0.529 | -0.660 |
| adjacent A block (b2 and b3 vs b1, b6 and b7 vs b5) | -0.534 | -0.036 |

B realized in every B block (app on CPU4 in at least 99% of runs, at least 99% of interrupts on CPU4): True. Label for B: ủng hộ. 'ủng hộ' here only means that putting the app and the NPU interrupts on CPU4 removes the slow level. It does not say which of the three changes does it.

Slow runs by the CPU the app ended on (post-hoc, exploratory, not used for the label), slow runs / runs on that CPU:

| block | cond | CPU0 | CPU1 | CPU2 | CPU3 | CPU4 | CPU5 | CPU6 | CPU7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 5/224 | 215/228 | 270/307 | 44/46 | 0/86 | 0/77 | - | 0/32 |
| 2 | B | - | - | - | - | 0/1000 | - | - | - |
| 3 | B | - | - | - | - | 0/1000 | - | - | - |
| 4 | A | 0/190 | 247/272 | 19/25 | 263/270 | 0/169 | 0/74 | - | - |
| 5 | A | 0/129 | - | - | 36/44 | 0/316 | 0/511 | - | - |
| 6 | B | - | - | - | - | 0/1000 | - | - | - |
| 7 | B | - | - | - | - | 0/1000 | - | - | - |
| 8 | A | 1/221 | 570/596 | 89/96 | - | - | 0/87 | - | - |

Rule file written before the data (bench/onboard_e2b/rule.txt):

```
E2b rule, written 2026-10-30T23:05:22+07:00 before any E2b data exists. Not from a source.
Condition B is realized in a block only when the per-run CPU of the app is CPU4, the NPU interrupts landed on CPU4, and the top-CPU share of interrupts is at least 0.99 (same realized() as E2 in ai/automation/analyze_onboard.py).
Label for B: 'ung ho' when the slow-level share moves by more than 0.20 in the same direction in both order of realized blocks, 'khong ung ho' under 0.10 in both, otherwise 'chua ket luan' (same as E2).
B is compared only with A blocks of E2b (same boot, irqbalance stopped), not with E2 blocks.
```

Note added after the data (bench/onboard_e2b/rule_note.txt):

```
Note written 2026-10-30T23:30:16+07:00, after the E2b data existed.
rule.txt says 'in both orders' but E2b runs ABBA twice, so each half holds two A blocks and the same order. The pairing was therefore chosen after the data:
the label uses the mean of the two A blocks of each half, and the other pairings are reported next to it (ai/automation/analyze_onboard.py, section 11).
Per adjacent block, b6 and b7 against b5 alone would shift the slow share by only 0.036 because block 5 (A) had 3.6 percent slow runs.
Fast share and standard deviation are given as description only and are not part of the label.
```

Board state log of the run (IRQ affinity and irqbalance saved and restored):

```
npu irqs: 36
37
38 ; irqbalance was: active
saved irq 36 affinity 0-7
saved irq 37 affinity 0-7
saved irq 38 affinity 0-7
irqbalance now: inactive
block 1 cond A 00:19:34 pin='' irqbalance=inactive readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 1 post readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 2 cond B 00:20:15 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 2 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 3 cond B 00:20:53 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 3 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 4 cond A 00:21:30 pin='' irqbalance=inactive readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 4 post readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 5 cond A 00:22:11 pin='' irqbalance=inactive readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 5 post readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 6 cond B 00:22:51 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 6 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 7 cond B 00:23:28 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 7 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 8 cond A 00:24:06 pin='' irqbalance=inactive readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 8 post readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
restored (read before irqbalance restarts): irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0  idle_state1_disable=0
irqbalance now: active ; readback after restart: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0
```

## 12. E2c: app placement versus NPU interrupt placement (irqbalance stopped, 1000 runs per block, order ABCEFG GFECBA)

Conditions (app CPU / NPU interrupt CPU): A none / default, B 4 / 4, C 0 / 0, E 4 / 0, F 0 / 4, G 1 / 0. Outcomes follow bench/onboard_e2c/rule.txt, written before the data.

| block | cond | half | n | median ms | p95 ms | std ms | fast | mid | slow | app CPU (share) | NPU interrupts: CPU (share) | interrupts | realized |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 1 | 1000 | 21.422 | 21.914 | 1.290 | 0.3% | 27.3% | 72.4% | CPU1 (76%) | CPU0 (100%) | 66150 | True |
| 2 | B | 1 | 1000 | 16.294 | 16.303 | 0.011 | 100.0% | 0.0% | 0.0% | CPU4 (100%) | CPU4 (100%) | 66150 | True |
| 3 | C | 1 | 1000 | 18.832 | 19.278 | 0.538 | 0.0% | 99.7% | 0.3% | CPU0 (100%) | CPU0 (100%) | 66150 | True |
| 4 | E | 1 | 1000 | 18.816 | 18.856 | 0.998 | 27.9% | 72.1% | 0.0% | CPU4 (100%) | CPU0 (100%) | 66150 | True |
| 5 | F | 1 | 1000 | 18.758 | 19.221 | 0.461 | 0.0% | 100.0% | 0.0% | CPU0 (100%) | CPU4 (100%) | 66150 | True |
| 6 | G | 1 | 1000 | 21.507 | 21.864 | 0.994 | 0.0% | 18.3% | 81.7% | CPU1 (100%) | CPU0 (100%) | 66150 | True |
| 7 | G | 2 | 1000 | 21.573 | 21.882 | 0.853 | 0.0% | 13.2% | 86.8% | CPU1 (100%) | CPU0 (100%) | 66150 | True |
| 8 | F | 2 | 1000 | 18.777 | 19.220 | 0.467 | 0.0% | 100.0% | 0.0% | CPU0 (100%) | CPU4 (100%) | 66150 | True |
| 9 | E | 2 | 1000 | 18.855 | 18.865 | 0.580 | 4.7% | 94.9% | 0.4% | CPU4 (100%) | CPU0 (100%) | 66150 | True |
| 10 | C | 2 | 1000 | 18.818 | 19.286 | 0.654 | 0.0% | 98.8% | 1.2% | CPU0 (100%) | CPU0 (100%) | 66150 | True |
| 11 | B | 2 | 1000 | 16.299 | 16.319 | 0.013 | 100.0% | 0.0% | 0.0% | CPU4 (100%) | CPU4 (100%) | 66150 | True |
| 12 | A | 2 | 1000 | 18.929 | 21.863 | 1.800 | 22.5% | 50.4% | 27.1% | CPU6 (25%) | CPU0 (100%) | 66150 | True |

| condition | primary outcome | secondary: shift vs A of the same half (half 1, half 2) and label |
|---|---|---|
| A | còn mức chậm | - |
| B | mất mức chậm | -0.724, -0.271 ủng hộ |
| C | lẫn lộn | -0.721, -0.259 ủng hộ |
| E | mất mức chậm | -0.724, -0.267 ủng hộ |
| F | mất mức chậm | -0.724, -0.271 ủng hộ |
| G | còn mức chậm | +0.093, +0.597 chưa kết luận |

Validity: A has a slow share of at least 0.20 in one block: True. B lost the slow level: True. C lost it: False. Valid: False.

Reading E and F: không đọc được yếu tố.
Reading E and G: không đọc được yếu tố.
G against the E2b exploratory hint: không đọc được yếu tố.

Rule file written before the data (bench/onboard_e2c/rule.txt):

```
E2c rule, written 2026-10-30T23:42:11+07:00 before any E2c data exists. Final: do not edit after the run starts. Not from a source.
Question: which of the two placements (app, NPU interrupts) removes the slow level that B (both on CPU4) and C (both on CPU0) removed in E2 and E2b. Mechanism is out of scope.
Setup: irqbalance stopped, 1000 runs per block after 50 warm-up runs, app pinned with taskset, NPU interrupts 36, 37, 38 all set to the target CPU. Order: A B C E F G then G F E C B A (12 blocks, each condition twice, one per half, so each half has exactly one A).
Conditions (app CPU / NPU interrupt CPU): A none / default (0-7, lands on CPU0), B 4 / 4, C 0 / 0, E 4 / 0, F 0 / 4, G 1 / 0.
Realized block: the app ran on its pinned CPU in at least 99 percent of runs (A exempt) and at least 99 percent of NPU interrupts landed on the target CPU (A: CPU0, E: CPU0, F: CPU4, G: CPU0, B: CPU4, C: CPU0). Unrealized blocks are excluded and reported.
Slow level: a run of at least 20.0 ms (same threshold as E2 and E2b, read from the pooled histogram of E1).
Primary outcome per condition, from its realized blocks only; both blocks must be realized, otherwise 'chua ket luan (khoi khong thuc hien dung)':
  'mat muc cham' when slow share is at most 0.01 in both blocks (the 0.01 cutoff is new and was chosen for this rule);
  'con muc cham' when slow share is at least 0.20 in both blocks;
  'lan lon' otherwise, including the two blocks disagreeing.
The primary rule is absolute, not a shift against A, because A ranged from 3.6 to 66 percent slow in E2b.
Secondary outcome, for comparison with E2 and E2b: shift of the slow share against the A block of the same half; 'ung ho' when the shift is below -0.20 in both halves (same rule as E2), 'khong ung ho' when its size is under 0.10 in both, otherwise 'chua ket luan'. Reported next to the primary outcome, never replaces it.
Validity checks (all needed before any reading below is allowed):
  1. A shows a slow share of at least 0.20 in at least one of its two blocks;
  2. B and C both have the primary outcome 'mat muc cham' in this run (replication of E2 and E2b).
  If either fails, every reading is 'khong doc duoc yeu to' whatever E, F and G show.
Reading table (allowed statements only, no mechanism claim). Rows use the primary outcomes of E, F, G:
  E 'mat muc cham' and F 'con muc cham': the app on CPU4 removes the slow level whatever the interrupt CPU is.
  E 'con muc cham' and F 'mat muc cham': the interrupts on CPU4 remove it whatever the app CPU is.
  E and F both 'mat muc cham': either placement alone is enough.
  E and F both 'con muc cham': app and interrupts on the same CPU are needed (B and C worked, E and F are the split cases).
  Any other combination of E and F: no statement about which factor.
  E 'mat muc cham' and G 'con muc cham': with the interrupts on CPU0, the app on CPU4 removes the slow level and the app on CPU1 does not. This is about where the app sits, not about same CPU versus different CPU.
  E 'con muc cham' and G 'con muc cham': same-CPU placement is needed.
  E 'con muc cham' and G 'mat muc cham': no statement.
  Other combinations of E and G: no statement.
G also tests the E2b exploratory hint (slow runs only when the app ended on CPU1 to CPU3): G 'con muc cham' supports it for an app pinned to CPU1, 'mat muc cham' does not.
Descriptive only, not part of any outcome and not to be worded as an outcome: median, standard deviation, fast share, per block.
Pairing: none, each condition is judged on its own two blocks, so the half-mean choice of E2b does not arise.
State differences with E2b, disclosed: E2b ran before rknn_server and adbd on 5555 were restarted after the reboot, E2c runs with both up. E2c relies on its own B and C blocks as controls.
```

Board state log of the run:

```
npu irqs: 36
37
38 ; irqbalance was: active
state: kernel 6.1.99-rockchip-rk3588 ; up 1 hour, 3 minutes ; rknn_server pid '10380' ; adbd pid '11257' ; adb port 5555 listeners 1
policy0: performance 1800000 kHz
policy4: performance 2256000 kHz
policy6: performance 2304000 kHz
npu: performance 1000000000 Hz
ddr: performance 2112000000 Hz
9285f251d6870ab67d60d79f16bade69  best-rk3588.rknn
d22a17c2723475ea95e2b620c089d63c  rknn_perf_bench
saved irq 36 affinity 0-7
saved irq 37 affinity 0-7
saved irq 38 affinity 0-7
irqbalance now: inactive
block 1 cond A 00:42:47 pin='' irqbalance=inactive readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 1 post readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 2 cond B 00:43:29 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 2 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 3 cond C 00:44:06 pin='taskset -c 0' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 3 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 4 cond E 00:44:46 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 4 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 5 cond F 00:45:25 pin='taskset -c 0' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 5 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 6 cond G 00:46:05 pin='taskset -c 1' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 6 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 7 cond G 00:46:47 pin='taskset -c 1' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 7 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 8 cond F 00:47:29 pin='taskset -c 0' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 8 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 9 cond E 00:48:09 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 9 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 10 cond C 00:48:49 pin='taskset -c 0' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 10 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 11 cond B 00:49:29 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 11 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 12 cond A 00:50:06 pin='' irqbalance=inactive readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 12 post readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
restored (read before irqbalance restarts): irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0  idle_state1_disable=0
irqbalance now: active ; readback after restart: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0
```

## 13. E2d: app CPU x NPU interrupt CPU map (irqbalance stopped, 1000 runs per block, 40 blocks)

Cell name aXiY: app pinned to CPU X, NPU interrupts on CPU Y. Outcomes follow bench/onboard_e2d/rule.txt, written before the data.

| block | cond | half | median ms | std ms | fast | mid | slow | app CPU (share) | NPU interrupts: CPU (share) | interrupts | realized |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 1 | 21.495 | 1.341 | 0.6% | 24.4% | 75.0% | CPU1 (53%) | CPU0 (100%) | 66150 | True |
| 2 | a0i0 | 1 | 18.821 | 0.455 | 0.0% | 100.0% | 0.0% | CPU0 (100%) | CPU0 (100%) | 66150 | True |
| 3 | a1i0 | 1 | 21.509 | 0.712 | 0.0% | 8.5% | 91.5% | CPU1 (100%) | CPU0 (100%) | 66150 | True |
| 4 | a2i0 | 1 | 21.599 | 0.509 | 0.0% | 2.8% | 97.2% | CPU2 (100%) | CPU0 (100%) | 66150 | True |
| 5 | a3i0 | 1 | 21.758 | 0.792 | 0.0% | 9.1% | 90.9% | CPU3 (100%) | CPU0 (100%) | 66150 | True |
| 6 | a4i0 | 1 | 18.858 | 0.631 | 6.2% | 93.8% | 0.0% | CPU4 (100%) | CPU0 (100%) | 66150 | True |
| 7 | a5i0 | 1 | 16.612 | 0.993 | 66.3% | 33.7% | 0.0% | CPU5 (100%) | CPU0 (100%) | 66150 | True |
| 8 | a6i0 | 1 | 18.871 | 0.560 | 4.4% | 95.6% | 0.0% | CPU6 (100%) | CPU0 (100%) | 66150 | True |
| 9 | a7i0 | 1 | 18.866 | 0.478 | 2.6% | 97.4% | 0.0% | CPU7 (100%) | CPU0 (100%) | 66150 | True |
| 10 | a0i4 | 1 | 18.708 | 0.537 | 0.0% | 99.7% | 0.3% | CPU0 (100%) | CPU4 (100%) | 66150 | True |
| 11 | a1i4 | 1 | 19.035 | 0.283 | 0.0% | 100.0% | 0.0% | CPU1 (100%) | CPU4 (100%) | 66150 | True |
| 12 | a2i4 | 1 | 19.065 | 0.261 | 0.0% | 99.9% | 0.1% | CPU2 (100%) | CPU4 (100%) | 66150 | True |
| 13 | a3i4 | 1 | 19.194 | 0.309 | 0.0% | 99.9% | 0.1% | CPU3 (100%) | CPU4 (100%) | 66150 | True |
| 14 | a4i4 | 1 | 16.292 | 0.263 | 98.5% | 1.5% | 0.0% | CPU4 (100%) | CPU4 (100%) | 66150 | True |
| 15 | a5i4 | 1 | 16.372 | 0.025 | 99.9% | 0.1% | 0.0% | CPU5 (100%) | CPU4 (100%) | 66150 | True |
| 16 | a6i4 | 1 | 16.375 | 0.062 | 99.8% | 0.2% | 0.0% | CPU6 (100%) | CPU4 (100%) | 66150 | True |
| 17 | a7i4 | 1 | 16.358 | 0.176 | 99.3% | 0.7% | 0.0% | CPU7 (100%) | CPU4 (100%) | 66150 | True |
| 18 | a5i5 | 1 | 16.306 | 0.017 | 100.0% | 0.0% | 0.0% | CPU5 (100%) | CPU5 (100%) | 66150 | True |
| 19 | a6i6 | 1 | 16.311 | 0.011 | 100.0% | 0.0% | 0.0% | CPU6 (100%) | CPU6 (100%) | 66150 | True |
| 20 | a7i7 | 1 | 16.318 | 0.015 | 100.0% | 0.0% | 0.0% | CPU7 (100%) | CPU7 (100%) | 66150 | True |
| 21 | a7i7 | 2 | 16.307 | 0.046 | 99.8% | 0.2% | 0.0% | CPU7 (100%) | CPU7 (100%) | 66150 | True |
| 22 | a6i6 | 2 | 16.315 | 0.185 | 99.5% | 0.5% | 0.0% | CPU6 (100%) | CPU6 (100%) | 66150 | True |
| 23 | a5i5 | 2 | 16.315 | 0.030 | 99.9% | 0.1% | 0.0% | CPU5 (100%) | CPU5 (100%) | 66150 | True |
| 24 | a7i4 | 2 | 16.379 | 0.011 | 100.0% | 0.0% | 0.0% | CPU7 (100%) | CPU4 (100%) | 66150 | True |
| 25 | a6i4 | 2 | 16.377 | 0.041 | 99.9% | 0.1% | 0.0% | CPU6 (100%) | CPU4 (100%) | 66150 | True |
| 26 | a5i4 | 2 | 16.385 | 0.016 | 100.0% | 0.0% | 0.0% | CPU5 (100%) | CPU4 (100%) | 66150 | True |
| 27 | a4i4 | 2 | 16.325 | 0.025 | 100.0% | 0.0% | 0.0% | CPU4 (100%) | CPU4 (100%) | 66150 | True |
| 28 | a3i4 | 2 | 19.383 | 0.304 | 0.0% | 99.8% | 0.2% | CPU3 (100%) | CPU4 (100%) | 66150 | True |
| 29 | a2i4 | 2 | 19.080 | 0.391 | 0.0% | 99.8% | 0.2% | CPU2 (100%) | CPU4 (100%) | 66150 | True |
| 30 | a1i4 | 2 | 19.050 | 0.285 | 0.0% | 100.0% | 0.0% | CPU1 (100%) | CPU4 (100%) | 66150 | True |
| 31 | a0i4 | 2 | 18.779 | 0.474 | 0.0% | 99.9% | 0.1% | CPU0 (100%) | CPU4 (100%) | 66150 | True |
| 32 | a7i0 | 2 | 18.878 | 0.575 | 4.8% | 95.2% | 0.0% | CPU7 (100%) | CPU0 (100%) | 66150 | True |
| 33 | a6i0 | 2 | 18.862 | 0.700 | 8.8% | 91.2% | 0.0% | CPU6 (100%) | CPU0 (100%) | 66150 | True |
| 34 | a5i0 | 2 | 18.870 | 0.679 | 7.9% | 92.1% | 0.0% | CPU5 (100%) | CPU0 (100%) | 66150 | True |
| 35 | a4i0 | 2 | 18.867 | 0.646 | 7.4% | 92.6% | 0.0% | CPU4 (100%) | CPU0 (100%) | 66150 | True |
| 36 | a3i0 | 2 | 21.773 | 0.768 | 0.0% | 5.2% | 94.8% | CPU3 (100%) | CPU0 (100%) | 66150 | True |
| 37 | a2i0 | 2 | 21.600 | 0.559 | 0.0% | 3.9% | 96.1% | CPU2 (100%) | CPU0 (100%) | 66150 | True |
| 38 | a1i0 | 2 | 21.579 | 0.716 | 0.0% | 7.2% | 92.8% | CPU1 (100%) | CPU0 (100%) | 66150 | True |
| 39 | a0i0 | 2 | 18.814 | 0.475 | 0.0% | 100.0% | 0.0% | CPU0 (100%) | CPU0 (100%) | 66150 | True |
| 40 | A | 2 | 19.507 | 1.558 | 3.2% | 48.1% | 48.7% | CPU1 (50%) | CPU0 (100%) | 66150 | True |

| cell | primary outcome | fast outcome |
|---|---|---|
| A | còn mức chậm | không đạt mức nhanh |
| a0i0 | mất mức chậm | không đạt mức nhanh |
| a1i0 | còn mức chậm | không đạt mức nhanh |
| a2i0 | còn mức chậm | không đạt mức nhanh |
| a3i0 | còn mức chậm | không đạt mức nhanh |
| a4i0 | mất mức chậm | không đạt mức nhanh |
| a5i0 | mất mức chậm | không đạt mức nhanh |
| a6i0 | mất mức chậm | không đạt mức nhanh |
| a7i0 | mất mức chậm | không đạt mức nhanh |
| a0i4 | mất mức chậm | không đạt mức nhanh |
| a1i4 | mất mức chậm | không đạt mức nhanh |
| a2i4 | mất mức chậm | không đạt mức nhanh |
| a3i4 | mất mức chậm | không đạt mức nhanh |
| a4i4 | mất mức chậm | đạt mức nhanh |
| a5i4 | mất mức chậm | đạt mức nhanh |
| a6i4 | mất mức chậm | đạt mức nhanh |
| a7i4 | mất mức chậm | đạt mức nhanh |
| a5i5 | mất mức chậm | đạt mức nhanh |
| a6i6 | mất mức chậm | đạt mức nhanh |
| a7i7 | mất mức chậm | đạt mức nhanh |

Validity: A has a slow share of at least 0.20 in one block: True. Cell a4i4 loses the slow level and reaches the fast level: True. Valid: True.

Not reproducing E2c (a0i0 should lose the slow level, a1i0 should keep it): none.

S0 (app CPUs with the slow level, interrupts on CPU0): [1, 2, 3], ambiguous or unresolved: [].
S4 (interrupts on CPU4): [], ambiguous or unresolved: [].
Comparison on CPUs definite in both rows ([0, 1, 2, 3, 4, 5, 6, 7]): không đổi trên các CPU [0, 4, 5, 6, 7], khác nhau trên các CPU [1, 2, 3] (mức chậm ở một hàng và không có ở hàng kia). CPU ngắt 0 và 4 cũng khác cụm và khác xung nhịp, nên không đọc thành vai trò của ngắt.
F (cells that reach the fast level): ['a4i4', 'a5i4', 'a6i4', 'a7i4', 'a5i5', 'a6i6', 'a7i7'].

Rule file written before the data (bench/onboard_e2d/rule.txt):

```
E2d rule, written 2026-10-31T00:00:07+07:00 before any E2d data exists. Final: do not edit after the run starts. Not from a source.
Approved by the user on 2026-10-30: the two validity checks, the thresholds 0.05 and 0.95 (taken from E2c data, so chosen after seeing E2c), and the three same-core big cells (5, 5), (6, 6), (7, 7).
Question 1: with which app CPUs does the slow level appear, with the NPU interrupts on CPU0 and on CPU4. Question 2: with which (app, interrupt) pairs does the fast level (16.3 ms, std about 0.01 ms in B) appear. Mechanism is out of scope.
Setup: irqbalance stopped, 1000 runs per block after 50 warm-up runs, 20 s rest between blocks, app pinned with taskset, NPU interrupts 36, 37, 38 all set to the target CPU. 20 conditions: A (no pin, default interrupts), the 16 cells (app CPU 0 to 7) x (interrupt CPU 0 or 4), and (5, 5), (6, 6), (7, 7).
Order: half 1 is the fixed list L = A, (0,0) to (7,0), (0,4) to (7,4), (5,5), (6,6), (7,7). Half 2 is L reversed. 40 blocks, each condition has exactly one block per half.
Mức chậm: một lần chạy từ 20.0 ms trở lên (cùng ngưỡng E2, E2b, E2c). Mức nhanh: dưới 17.0 ms.
Khối thực hiện đúng: ứng dụng chạy ở CPU đã ghim trong ít nhất 99% số lần (A được miễn) và ít nhất 99% ngắt NPU rơi vào CPU đích (A: CPU0). Khối không thực hiện đúng bị loại và được báo.
Kết quả chính theo ô, chỉ từ các khối thực hiện đúng và cần cả hai khối, nếu không là 'chưa kết luận (khối không thực hiện đúng)'.
  'mất mức chậm' khi tỉ lệ mức chậm không quá 0.05 ở cả hai khối.
  'còn mức chậm' khi tỉ lệ mức chậm từ 0.20 trở lên ở cả hai khối.
  'lẫn lộn' trong các trường hợp còn lại.
Kết quả nhanh theo ô, chỉ từ khối thực hiện đúng: 'đạt mức nhanh' khi tỉ lệ mức nhanh từ 0.95 trở lên ở cả hai khối, ngược lại 'không đạt mức nhanh'.
Nguồn gốc ngưỡng: 0.05 gấp bốn lần giá trị lớn nhất (0.012) trong các ô đã thấy mất mức chậm ở E2c. 0.20 là ngưỡng của E2. 0.95 nằm giữa 0.279 (giá trị mức nhanh lớn nhất ngoài B ở E2c, E khối 4) và 1.000 (B). Hai ngưỡng 0.05 và 0.95 do người viết quy tắc chọn sau khi đã thấy dữ liệu E2c.
Kiểm tra hợp lệ, chỉ có hai.
  1. A có tỉ lệ mức chậm từ 0.20 trở lên ở ít nhất một khối.
  2. Ô (4, 4) 'mất mức chậm' và 'đạt mức nhanh' (tái hiện B của E2b và E2c).
  Nếu một trong hai không đạt, mọi phát biểu bên dưới là 'không đọc được'.
Các ô (0, 0) và (1, 0) là ô thường. Nếu kết quả khác E2c (ví dụ (1, 0) không 'còn mức chậm' hoặc (0, 0) không 'mất mức chậm') thì báo là 'không tái hiện E2c', và vẫn báo ô đó như các ô khác.
Phát biểu cho phép, chỉ nói vị trí, không nói cơ chế hay nguyên nhân, và không được viết thành hiệu ứng của cụm CPU hay của tần số (CPU0 đến 3 chạy 1.8 GHz, CPU4 đến 5 chạy 2.256 GHz, CPU6 đến 7 chạy 2.304 GHz nên mọi mẫu theo cụm cũng là mẫu theo xung nhịp).
  S0: tập CPU ứng dụng có ô (CPU, ngắt CPU0) 'còn mức chậm'. S4: tập tương ứng khi ngắt ở CPU4. Báo S0 và S4, kèm danh sách CPU 'lẫn lộn' hoặc 'chưa kết luận' của từng hàng.
  So sánh S0 với S4 chỉ trên các CPU có kết quả rõ ràng ('mất' hoặc 'còn') ở cả hai hàng. 'Mức chậm không đổi giữa hai CPU ngắt' chỉ được viết khi cả 8 CPU đều rõ ràng ở cả hai hàng và S0 = S4. Nếu không, viết 'không đổi trên các CPU {danh sách}' và nêu riêng các CPU còn lại.
  F: tập các ô 'đạt mức nhanh' (gồm cả (5, 5), (6, 6), (7, 7)). Báo F như đo được. Nếu F chỉ gồm (4, 4) thì viết đúng như vậy, không suy ra lý do.
  Nếu S0 = {1, 2, 3} thì được viết 'với ngắt ở CPU0, mức chậm chỉ có khi ứng dụng ghim ở CPU1 đến CPU3 trong các CPU đã thử'. Nếu S0 khác thì báo đúng tập đó, không giải thích.
  Nếu S4 khác tập rỗng thì báo S4, không giải thích.
Mô tả, không phải kết quả và không viết như kết quả: trung vị, độ lệch chuẩn, tỉ lệ ba mức theo khối.
Không ghép khối, không so với A ngoài kiểm tra hợp lệ 1, không có kết quả phụ theo độ lệch.
Mỗi ô chỉ có hai khối và các khối kề nhau theo thời gian, nên mọi phát biểu chỉ áp cho lần chạy này trên một board và một lần khởi động.
Ngoài phạm vi: cơ chế, lý do một CPU chậm, ảnh hưởng của cụm CPU hay xung nhịp, tải CPU khác.
```

Board state log of the run:

```
npu irqs: 36
37
38 ; irqbalance was: active
state: kernel 6.1.99-rockchip-rk3588 ; up 1 hour, 21 minutes ; rknn_server pid '10380' ; adbd pid '11257' ; adb port 5555 listeners 1
policy0: performance 1800000 kHz
policy4: performance 2256000 kHz
policy6: performance 2304000 kHz
npu: performance 1000000000 Hz
ddr: performance 2112000000 Hz
9285f251d6870ab67d60d79f16bade69  best-rk3588.rknn
d22a17c2723475ea95e2b620c089d63c  rknn_perf_bench
saved irq 36 affinity 0-7
saved irq 37 affinity 0-7
saved irq 38 affinity 0-7
irqbalance now: inactive
blocks: 40
block 1 cond A 01:00:37 pin='' irqbalance=inactive readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 1 post readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 2 cond a0i0 01:01:19 pin='taskset -c 0' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 2 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 3 cond a1i0 01:01:58 pin='taskset -c 1' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 3 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 4 cond a2i0 01:02:41 pin='taskset -c 2' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 4 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 5 cond a3i0 01:03:24 pin='taskset -c 3' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 5 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 6 cond a4i0 01:04:06 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 6 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 7 cond a5i0 01:04:46 pin='taskset -c 5' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 7 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 8 cond a6i0 01:05:24 pin='taskset -c 6' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 8 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 9 cond a7i0 01:06:04 pin='taskset -c 7' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 9 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 10 cond a0i4 01:06:43 pin='taskset -c 0' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 10 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 11 cond a1i4 01:07:23 pin='taskset -c 1' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 11 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 12 cond a2i4 01:08:03 pin='taskset -c 2' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 12 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 13 cond a3i4 01:08:43 pin='taskset -c 3' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 13 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 14 cond a4i4 01:09:24 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 14 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 15 cond a5i4 01:10:01 pin='taskset -c 5' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 15 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 16 cond a6i4 01:10:38 pin='taskset -c 6' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 16 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 17 cond a7i4 01:11:16 pin='taskset -c 7' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 17 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 18 cond a5i5 01:11:53 pin='taskset -c 5' irqbalance=inactive readback: irq 36 smp=5 effective=5 irq 37 smp=5 effective=5 irq 38 smp=5 effective=5 
block 18 post readback: irq 36 smp=5 effective=5 irq 37 smp=5 effective=5 irq 38 smp=5 effective=5 
block 19 cond a6i6 01:12:30 pin='taskset -c 6' irqbalance=inactive readback: irq 36 smp=6 effective=6 irq 37 smp=6 effective=6 irq 38 smp=6 effective=6 
block 19 post readback: irq 36 smp=6 effective=6 irq 37 smp=6 effective=6 irq 38 smp=6 effective=6 
block 20 cond a7i7 01:13:08 pin='taskset -c 7' irqbalance=inactive readback: irq 36 smp=7 effective=7 irq 37 smp=7 effective=7 irq 38 smp=7 effective=7 
block 20 post readback: irq 36 smp=7 effective=7 irq 37 smp=7 effective=7 irq 38 smp=7 effective=7 
block 21 cond a7i7 01:13:45 pin='taskset -c 7' irqbalance=inactive readback: irq 36 smp=7 effective=7 irq 37 smp=7 effective=7 irq 38 smp=7 effective=7 
block 21 post readback: irq 36 smp=7 effective=7 irq 37 smp=7 effective=7 irq 38 smp=7 effective=7 
block 22 cond a6i6 01:14:22 pin='taskset -c 6' irqbalance=inactive readback: irq 36 smp=6 effective=6 irq 37 smp=6 effective=6 irq 38 smp=6 effective=6 
block 22 post readback: irq 36 smp=6 effective=6 irq 37 smp=6 effective=6 irq 38 smp=6 effective=6 
block 23 cond a5i5 01:15:00 pin='taskset -c 5' irqbalance=inactive readback: irq 36 smp=5 effective=5 irq 37 smp=5 effective=5 irq 38 smp=5 effective=5 
block 23 post readback: irq 36 smp=5 effective=5 irq 37 smp=5 effective=5 irq 38 smp=5 effective=5 
block 24 cond a7i4 01:15:37 pin='taskset -c 7' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 24 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 25 cond a6i4 01:16:14 pin='taskset -c 6' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 25 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 26 cond a5i4 01:16:52 pin='taskset -c 5' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 26 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 27 cond a4i4 01:17:29 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 27 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 28 cond a3i4 01:18:06 pin='taskset -c 3' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 28 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 29 cond a2i4 01:18:47 pin='taskset -c 2' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 29 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 30 cond a1i4 01:19:27 pin='taskset -c 1' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 30 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 31 cond a0i4 01:20:07 pin='taskset -c 0' irqbalance=inactive readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 31 post readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
block 32 cond a7i0 01:20:47 pin='taskset -c 7' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 32 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 33 cond a6i0 01:21:26 pin='taskset -c 6' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 33 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 34 cond a5i0 01:22:06 pin='taskset -c 5' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 34 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 35 cond a4i0 01:22:46 pin='taskset -c 4' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 35 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 36 cond a3i0 01:23:25 pin='taskset -c 3' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 36 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 37 cond a2i0 01:24:08 pin='taskset -c 2' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 37 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 38 cond a1i0 01:24:51 pin='taskset -c 1' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 38 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 39 cond a0i0 01:25:33 pin='taskset -c 0' irqbalance=inactive readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 39 post readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
block 40 cond A 01:26:13 pin='' irqbalance=inactive readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
block 40 post readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
restored (read before irqbalance restarts): irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0  idle_state1_disable=0
irqbalance now: active ; readback after restart: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0
```

## 14. E2e: eval_perf path (PC Toolkit2 to rknn_server), rknn_server and NPU interrupts pinned (500 timed calls per block, 14 blocks)

Cell name sXiY: rknn_server pinned to CPU X, NPU interrupts on CPU Y. Outcomes follow bench/onboard_e2e/rule.txt, written before the data. Sample = Total Time(us) of one eval_perf call (one inference).

| block | cond | half | n | median ms | std ms | fast | mid | slow | busy task (name) final CPU | all tasks pinned | NPU interrupts: CPU (share) | realized |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 1 | 500 | 19.694 | 1.540 | 1.8% | 57.8% | 40.4% | 10380 (rknn_server) CPU0 | True | CPU0 (100%) | True |
| 2 | s4i4 | 1 | 500 | 16.516 | 0.043 | 100.0% | 0.0% | 0.0% | 10380 (rknn_server) CPU4 | True | CPU4 (100%) | True |
| 3 | s0i0 | 1 | 500 | 17.765 | 0.096 | 0.0% | 100.0% | 0.0% | 10395 (listener) CPU0 | True | CPU0 (100%) | True |
| 4 | s4i0 | 1 | 500 | 19.081 | 0.369 | 0.4% | 99.6% | 0.0% | 10380 (rknn_server) CPU4 | True | CPU0 (100%) | True |
| 5 | s0i4 | 1 | 500 | 17.858 | 0.084 | 0.0% | 100.0% | 0.0% | 10395 (listener) CPU0 | True | CPU4 (100%) | True |
| 6 | s1i0 | 1 | 500 | 20.314 | 0.499 | 0.0% | 14.4% | 85.6% | 10395 (listener) CPU1 | True | CPU0 (100%) | True |
| 7 | s1i4 | 1 | 500 | 17.862 | 0.086 | 0.0% | 100.0% | 0.0% | 10395 (listener) CPU1 | True | CPU4 (100%) | True |
| 8 | s1i4 | 2 | 500 | 17.840 | 0.109 | 0.0% | 100.0% | 0.0% | 10395 (listener) CPU1 | True | CPU4 (100%) | True |
| 9 | s1i0 | 2 | 500 | 20.371 | 0.434 | 0.0% | 9.6% | 90.4% | 10395 (listener) CPU1 | True | CPU0 (100%) | True |
| 10 | s0i4 | 2 | 500 | 17.891 | 0.135 | 0.0% | 100.0% | 0.0% | 10395 (listener) CPU0 | True | CPU4 (100%) | True |
| 11 | s4i0 | 2 | 500 | 19.219 | 0.437 | 0.0% | 100.0% | 0.0% | 10380 (rknn_server) CPU4 | True | CPU0 (100%) | True |
| 12 | s0i0 | 2 | 500 | 17.939 | 0.129 | 0.0% | 100.0% | 0.0% | 10395 (listener) CPU0 | True | CPU0 (100%) | True |
| 13 | s4i4 | 2 | 500 | 16.659 | 0.049 | 99.8% | 0.2% | 0.0% | 10380 (rknn_server) CPU4 | True | CPU4 (100%) | True |
| 14 | A | 2 | 500 | 19.770 | 1.354 | 0.0% | 62.2% | 37.8% | 10380 (rknn_server) CPU0 | True | CPU0 (100%) | True |

| cell | E2e primary | E2e fast | E2d cell primary | E2d cell fast | median E2e ms | median E2d ms | difference ms | verdict |
|---|---|---|---|---|---|---|---|---|
| s4i4 | mất mức chậm | đạt mức nhanh | mất mức chậm | đạt mức nhanh | 16.587 | 16.309 | +0.279 | cùng kết quả |
| s0i0 | mất mức chậm | không đạt mức nhanh | mất mức chậm | không đạt mức nhanh | 17.852 | 18.818 | -0.966 | cùng kết quả |
| s4i0 | mất mức chậm | không đạt mức nhanh | mất mức chậm | không đạt mức nhanh | 19.150 | 18.863 | +0.287 | cùng kết quả |
| s0i4 | mất mức chậm | không đạt mức nhanh | mất mức chậm | không đạt mức nhanh | 17.874 | 18.743 | -0.869 | cùng kết quả |
| s1i0 | còn mức chậm | không đạt mức nhanh | còn mức chậm | không đạt mức nhanh | 20.343 | 21.544 | -1.201 | cùng kết quả |
| s1i4 | mất mức chậm | không đạt mức nhanh | mất mức chậm | không đạt mức nhanh | 17.851 | 19.043 | -1.191 | cùng kết quả |

Uniform offset exception (same sign, each difference above 1.25 ms): False. All six cells give the same result: True. Cells that differ: none.
A (descriptive): primary còn mức chậm, fast không đạt mức nhanh.

Rule file written before the data (bench/onboard_e2e/rule.txt):

```
E2e rule, written 2026-10-31T00:43:11+07:00 before any E2e data exists. Final: do not edit after the run starts. Not from a source.
Approved by the user on 2026-10-31: the thresholds 0.05, 0.20, 0.95 (kept from E2d, applied to the new quantity eval_perf time), the uniform-offset exception with 1.25 ms (new number), 7 conditions x 2 blocks x 500 samples, the six cells below, and the root helper process on the board.
Board work done before this rule, no latency value recorded: (1) timing pilot over the LAN, bench/extra_tier1_lan_timing.txt (init_runtime 1.5 s, eval_perf 0.21 to 0.23 s per call); (2) interrupt count probe: 1 eval_perf call adds 63 interrupts on irq 36, 5 calls add 315, so one eval_perf is one inference (k = 1.000); (3) runtime check: /usr/lib/librknnrt.so and /usr/lib64/librknnrt.so have the same md5 a37ee1d5d664c79836bf6e35b7ef6289, rknn_server maps /usr/lib/librknnrt.so, rknn_server 2.3.2 and rknnrt 2.3.2 (429f97ae6b@2025-04-09T09:09:27), the same as the C bench line.
Question 1: on the eval_perf path (PC Toolkit2 -> rknn_server on the board), do the position cells give the same results as on the C API path (E2d)? Question 2: how far apart are eval_perf and rknn_run times for the same cell (descriptive). Position only, no mechanism.
Setup: irqbalance stopped, rknn_server pinned with taskset -a -p, NPU interrupts 36, 37, 38 set to the target CPU, each block is a new session (load_rknn, init_runtime with core_mask CORE_0, eval_perf with fix_freq True), 50 warm-up calls then 500 timed calls, calls about 0.22 s apart, 20 s rest between blocks.
Conditions (rknn_server CPU, interrupt CPU): A (no pin, default interrupts), s4i4, s0i0, s4i0, s0i4, s1i0, s1i4. Order: half 1 is the list A, s4i4, s0i0, s4i0, s0i4, s1i0, s1i4. Half 2 is the reverse. 14 blocks.
Mẫu: 'Total Time(us)' của một lần eval_perf, đổi sang ms. Một lần eval_perf là một lần suy luận (k = 1.000 theo đếm ngắt trước khi viết quy tắc). Mức chậm: từ 20.0 ms trở lên. Mức nhanh: dưới 17.0 ms. Cùng ngưỡng E2, E2b, E2c, E2d. Các ngưỡng này lấy từ đường C API và áp lên một đại lượng mới là thời gian eval_perf.
Luồng bận: trong mỗi khối, luồng rknn_server có thời gian CPU tăng nhiều nhất giữa đầu và cuối khối.
Khối thực hiện đúng: mọi luồng rknn_server có Cpus_allowed_list bằng CPU đã ghim ở cả hai lần đọc (A được miễn), luồng bận có psr cuối khối đúng CPU đó, và ít nhất 99% ngắt NPU rơi vào CPU đích (A: CPU0). Khối không thực hiện đúng bị loại và được báo.
Kết quả chính theo ô, chỉ từ khối thực hiện đúng và cần cả hai khối, nếu không là 'chưa kết luận (khối không thực hiện đúng)'.
  'mất mức chậm' khi tỉ lệ mức chậm không quá 0.05 ở cả hai khối.
  'còn mức chậm' khi tỉ lệ mức chậm từ 0.20 trở lên ở cả hai khối.
  'lẫn lộn' trong các trường hợp còn lại.
Kết quả nhanh theo ô: 'đạt mức nhanh' khi tỉ lệ mức nhanh từ 0.95 trở lên ở cả hai khối, ngược lại 'không đạt mức nhanh'.
Ngưỡng 0.05, 0.20, 0.95 giữ từ E2d.
Không có kiểm tra hợp lệ chặn kết quả. Ô (4, 4) là một trong sáu ô so sánh, không phải điều kiện. Điều kiện A chỉ để mô tả.
So sánh với E2d: với mỗi ô trong sáu ô, so kết quả chính và kết quả nhanh với cùng ô ở E2d (đường C API) và báo 'cùng kết quả' hoặc 'khác'.
  Ngoại lệ đặt trước cho lệch mức đồng đều: nếu hiệu của trung vị theo khối (eval_perf trừ C API, cùng ô, trung bình hai khối mỗi bên) có cùng dấu ở cả sáu ô và mỗi hiệu lớn hơn 1.25 ms (một nửa khoảng cách 2.5 ms giữa các mức), báo 'không so được (lệch mức đồng đều)' thay vì 'khác', vì ngưỡng 17.0 và 20.0 ms lấy từ đường C API. Con số 1.25 ms là mới và do người viết quy tắc chọn.
Phát biểu cho phép, chỉ nói vị trí, không nói cơ chế hay nguyên nhân, và không viết thành hiệu ứng của cụm CPU hay xung nhịp (CPU0 đến 3 chạy 1.8 GHz, CPU4 đến 5 chạy 2.256 GHz, CPU6 đến 7 chạy 2.304 GHz).
  Nếu cả sáu ô 'cùng kết quả' thì được viết 'trong sáu ô đã thử, đường eval_perf cho cùng kết quả vị trí với đường C API'. Nếu có ô khác thì liệt kê các ô khác, không giải thích. Nếu áp dụng ngoại lệ lệch mức thì chỉ viết 'không so được' và báo mô tả.
Mô tả, không phải kết quả: trung vị, độ lệch chuẩn, tỉ lệ ba mức theo khối, hiệu trung vị eval_perf trừ C API cho từng ô, và điều kiện A.
Không ghép khối.
Mỗi ô hai khối kề nhau theo thời gian, một board, một lần khởi động. Các lần gọi cách nhau khoảng 0.22 giây nên không so trực tiếp với đường chạy liên tiếp của C API về thời gian nghỉ của CPU.
Ngoài phạm vi: cơ chế, lý do một vị trí chậm, ảnh hưởng của cụm CPU hay xung nhịp, mối liên hệ giữa 'Total Time(us)' và thời gian rknn_run, mô hình khác, FP16, CORE_0_1_2.
```

Board helper log:

```
01:44:03 state: kernel 6.1.99-rockchip-rk3588 ; up 2 hours, 5 minutes ; rknn_server pid 10380 tasks 9 ; adbd pid 11257 ; port 5555 listeners 1
01:44:03 librknnrt mapped by rknn_server: /usr/lib/librknnrt.so  ; md5 a37ee1d5d664c79836bf6e35b7ef6289 ; model md5 9285f251d6870ab67d60d79f16bade69
01:44:03 frequencies: policy0: performance 1800000 kHz;policy4: performance 2256000 kHz;policy6: performance 2304000 kHz;npu: userspace 1000000000 Hz;ddr: userspace 2112000000 Hz;
01:44:03 saved irq affinities: 36=0-7 37=0-7 38=0-7  ; irqbalance was active
01:44:03 daemon ready
01:44:18 block 1 cond A pid 10380 readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0  rknn_server tasks allowed:       9 0-7  ; governors: performance performance performance userspace userspace 
01:44:23 block 1 inited: tasks 10, allowed      10 0-7 , readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
01:46:55 block 1 end: allowed      10 0-7 , readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
01:47:15 block 2 cond s4i4 pid 10380 readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4  rknn_server tasks allowed:       9 4  ; governors: performance performance performance userspace userspace 
01:47:18 block 2 inited: tasks 10, allowed      10 4 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
01:49:40 block 2 end: allowed      10 4 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
01:50:01 block 3 cond s0i0 pid 10380 readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0  rknn_server tasks allowed:       9 0  ; governors: performance performance performance userspace userspace 
01:50:03 block 3 inited: tasks 10, allowed      10 0 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
01:52:15 block 3 end: allowed      10 0 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
01:52:36 block 4 cond s4i0 pid 10380 readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0  rknn_server tasks allowed:       9 4  ; governors: performance performance performance userspace userspace 
01:52:38 block 4 inited: tasks 10, allowed      10 4 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
01:56:05 block 4 end: allowed      10 4 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
01:56:26 block 5 cond s0i4 pid 10380 readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4  rknn_server tasks allowed:       9 0  ; governors: performance performance performance userspace userspace 
01:56:30 block 5 inited: tasks 10, allowed      10 0 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
01:59:20 block 5 end: allowed      10 0 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
01:59:41 block 6 cond s1i0 pid 10380 readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0  rknn_server tasks allowed:       9 1  ; governors: performance performance performance userspace userspace 
01:59:44 block 6 inited: tasks 10, allowed      10 1 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
02:02:31 block 6 end: allowed      10 1 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
02:02:52 block 7 cond s1i4 pid 10380 readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4  rknn_server tasks allowed:       9 1  ; governors: performance performance performance userspace userspace 
02:02:58 block 7 inited: tasks 10, allowed      10 1 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
02:05:47 block 7 end: allowed      10 1 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
02:06:08 block 8 cond s1i4 pid 10380 readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4  rknn_server tasks allowed:       9 1  ; governors: performance performance performance userspace userspace 
02:06:11 block 8 inited: tasks 10, allowed      10 1 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
02:09:24 block 8 end: allowed      10 1 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
02:09:45 block 9 cond s1i0 pid 10380 readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0  rknn_server tasks allowed:       9 1  ; governors: performance performance performance userspace userspace 
02:09:49 block 9 inited: tasks 10, allowed      10 1 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
02:12:34 block 9 end: allowed      10 1 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
02:12:55 block 10 cond s0i4 pid 10380 readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4  rknn_server tasks allowed:       9 0  ; governors: performance performance performance userspace userspace 
02:12:57 block 10 inited: tasks 10, allowed      10 0 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
02:15:34 block 10 end: allowed      10 0 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
02:15:54 block 11 cond s4i0 pid 10380 readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0  rknn_server tasks allowed:       9 4  ; governors: performance performance performance userspace userspace 
02:15:57 block 11 inited: tasks 10, allowed      10 4 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
02:19:24 block 11 end: allowed      10 4 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
02:19:45 block 12 cond s0i0 pid 10380 readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0  rknn_server tasks allowed:       9 0  ; governors: performance performance performance userspace userspace 
02:19:48 block 12 inited: tasks 10, allowed      10 0 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
02:22:34 block 12 end: allowed      10 0 , readback: irq 36 smp=0 effective=0 irq 37 smp=0 effective=0 irq 38 smp=0 effective=0 
02:22:55 block 13 cond s4i4 pid 10380 readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4  rknn_server tasks allowed:       9 4  ; governors: performance performance performance userspace userspace 
02:22:58 block 13 inited: tasks 10, allowed      10 4 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
02:26:12 block 13 end: allowed      10 4 , readback: irq 36 smp=4 effective=4 irq 37 smp=4 effective=4 irq 38 smp=4 effective=4 
02:26:33 block 14 cond A pid 10380 readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0  rknn_server tasks allowed:       9 0-7  ; governors: performance performance performance userspace userspace 
02:26:35 block 14 inited: tasks 10, allowed      10 0-7 , readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
02:30:14 block 14 end: allowed      10 0-7 , readback: irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0 
02:30:35 stop file seen
02:30:35 exit: restored (read before irqbalance restarts): irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0  rknn_server tasks allowed:       9 0-7 
02:30:35 exit: irqbalance now active, irq 36 smp=0-7 effective=0 irq 37 smp=0-7 effective=0 irq 38 smp=0-7 effective=0
```
