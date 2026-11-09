# X2: interventions on cpu-sleep and the CPU4 clock, ppe5_yolov8n_baseline_100ep_s0

Rule: experiments/npu/latency/onboard_x2/rule.txt (written before the data).

| block | cond | half | realized (X1 test, manipulation) | T ms | C ms | N ms | D ms | P ms | app idle1 | irq idle1 | D median us | trace state1 shares |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a4i4 | 1 | True (True, True) | 16.407 | 0.899 | 15.367 | 0.141 | 0.000 | 0.00 | 0.00 | 2.0 | {} |
| 2 | a4i0 | 1 | True (True, True) | 19.199 | 0.913 | 17.960 | 0.325 | 0.000 | 0.00 | 1.00 | 5.0 | {} |
| 3 | a4i0s | 1 | True (True, True) | 16.679 | 0.903 | 15.514 | 0.262 | 0.000 | 0.00 | 0.00 | 4.0 | {'irq_cpu_state1': 0.0} |
| 4 | a1i4 | 1 | True (True, True) | 19.340 | 2.098 | 15.154 | 2.087 | 0.000 | 0.62 | 0.00 | 47.0 | {} |
| 5 | a1i4s | 1 | True (True, True) | 17.534 | 1.805 | 15.198 | 0.530 | 0.000 | 0.00 | 0.00 | 8.0 | {'app_cpu_state1': 0.0} |
| 6 | a4i4f | 1 | True (True, True) | 18.560 | 1.419 | 16.822 | 0.320 | 0.000 | 0.63 | 0.63 | 6.0 | {} |
| 7 | a1i0 | 1 | True (True, True) | 22.060 | 2.116 | 17.744 | 2.330 | 0.000 | 0.67 | 1.00 | 49.0 | {} |
| 8 | a1i0s | 1 | True (True, True) | 17.752 | 1.807 | 15.360 | 0.585 | 0.000 | 0.00 | 0.00 | 9.0 | {'irq_cpu_state1': 0.0, 'app_cpu_state1': 0.0} |
| 9 | a1i0s | 2 | True (True, True) | 17.745 | 1.804 | 15.356 | 0.584 | 0.000 | 0.00 | 0.00 | 9.0 | {'irq_cpu_state1': 0.0, 'app_cpu_state1': 0.0} |
| 10 | a1i0 | 2 | True (True, True) | 22.110 | 2.125 | 17.751 | 2.318 | 0.000 | 0.67 | 1.00 | 49.0 | {} |
| 11 | a4i4f | 2 | True (True, True) | 18.566 | 1.423 | 16.827 | 0.320 | 0.000 | 0.63 | 0.63 | 6.0 | {} |
| 12 | a1i4s | 2 | True (True, True) | 17.555 | 1.814 | 15.208 | 0.532 | 0.000 | 0.00 | 0.00 | 8.0 | {'app_cpu_state1': 0.0} |
| 13 | a1i4 | 2 | True (True, True) | 19.366 | 2.101 | 15.156 | 2.102 | 0.000 | 0.63 | 0.00 | 48.0 | {} |
| 14 | a4i0s | 2 | True (True, True) | 16.671 | 0.903 | 15.508 | 0.260 | 0.000 | 0.00 | 0.00 | 4.0 | {'irq_cpu_state1': 0.0} |
| 15 | a4i0 | 2 | True (True, True) | 19.212 | 0.905 | 17.984 | 0.323 | 0.000 | 0.00 | 1.00 | 5.0 | {} |
| 16 | a4i4 | 2 | True (True, True) | 16.407 | 0.899 | 15.370 | 0.138 | 0.000 | 0.00 | 0.00 | 2.0 | {} |

S1 (N, a4i0 vs a4i0s against a4i4): {'halves': [{'g': 2.5929999883373966, 'gs': 0.146999986100127}, {'g': 2.6140000099985627, 'gs': 0.13800000488117803}], 'verdict': 'làm mất'}

S2 (D, a1i4 vs a1i4s against a4i4): {'halves': [{'g': 1.9455000110610854, 'gs': 0.38900002800801303}, {'g': 1.9634999953268562, 'gs': 0.3939999733120203}], 'verdict': 'chưa kết luận'}

S3 (C clock): {'halves': [{'r': 1.5784204445485777, 'c1': 2.3337040760093783}, {'r': 1.5823137032669372, 'c1': 2.3370411662172033}], 'clock_ratio': 1.2533333333333332, 'verdict': 'chưa kết luận'}

S4 (T ms, half 1 and 2): {'a1i0s': [17.752, 17.745], 'a1i0': [22.06, 22.11], 'a4i4': [16.407, 16.407]}

Labels decided by the rule: {'U1b': 'CHƯA BIẾT', 'U3': 'CHƯA BIẾT'}
