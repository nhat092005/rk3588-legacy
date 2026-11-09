# FPS levers under P3

## E1 int8h5 against int8

- int8|T1: 54.69 FPS spread 0.07 realized True meets60 False
- int8h5|T1: 54.86 FPS spread 0.09 realized True meets60 False
- int8|T3: 56.17 FPS spread 0.06 realized True meets60 False
- int8h5|T3: 56.30 FPS spread 0.06 realized True meets60 False
- int8|T4: 101.90 FPS spread 0.57 realized True meets60 True
- int8h5|T4: 101.89 FPS spread 0.29 realized True meets60 True
- int8|T5: 141.13 FPS spread 0.45 realized True meets60 True
- int8h5|T5: 142.09 FPS spread 0.73 realized True meets60 True
- T1: {'a': 'int8h5|T1', 'b': 'int8|T1', 'diff_fps': 0.1700000000000017, 'diff_pct': 0.3108429328944994, 'larger_spread': 0.09000000000000341, 'verdict': 'không phân biệt được'}
- T3: {'a': 'int8h5|T3', 'b': 'int8|T3', 'diff_fps': 0.12999999999999545, 'diff_pct': 0.23144027060707753, 'larger_spread': 0.060000000000002274, 'verdict': 'không phân biệt được'}
- T4: {'a': 'int8h5|T4', 'b': 'int8|T4', 'diff_fps': -0.010000000000005116, 'diff_pct': -0.009813542688915716, 'larger_spread': 0.5699999999999932, 'verdict': 'không phân biệt được'}
- T5: {'a': 'int8h5|T5', 'b': 'int8|T5', 'diff_fps': 0.960000000000008, 'diff_pct': 0.6802239070360717, 'larger_spread': 0.7299999999999898, 'verdict': 'không phân biệt được'}

## E2 split head decode overlap

- A: 54.79 FPS spread 0.09 realized True meets60 False e2e_p50 None stage_sum 18.08
- B: 42.05 FPS spread 0.05 realized True meets60 False e2e_p50 None stage_sum 23.53
- C: 65.99 FPS spread 0.03 realized True meets60 True e2e_p50 132.559 stage_sum 27.55
- D: 65.74 FPS spread 0.21 realized True meets60 True e2e_p50 132.867 stage_sum 27.43
- E: 68.21 FPS spread 0.65 realized True meets60 True e2e_p50 128.653 stage_sum 27.36
- F: 56.62 FPS spread 0.21 realized True meets60 False e2e_p50 141.62 stage_sum 18.73
- B_vs_A: {'a': 'B', 'b': 'A', 'diff_fps': -12.740000000000002, 'diff_pct': -23.252418324511776, 'larger_spread': 0.0899999999999963, 'verdict': 'khác'}
- C_vs_A: {'a': 'C', 'b': 'A', 'diff_fps': 11.199999999999996, 'diff_pct': 20.44168643913122, 'larger_spread': 0.0899999999999963, 'verdict': 'khác'}
- C_vs_B: {'a': 'C', 'b': 'B', 'diff_fps': 23.939999999999998, 'diff_pct': 56.93222354340072, 'larger_spread': 0.05000000000000426, 'verdict': 'khác'}
- D_vs_C: {'a': 'D', 'b': 'C', 'diff_fps': -0.25, 'diff_pct': -0.3788452795878164, 'larger_spread': 0.21000000000000796, 'verdict': 'không phân biệt được'}
- E_vs_C: {'a': 'E', 'b': 'C', 'diff_fps': 2.219999999999999, 'diff_pct': 3.3641460827398078, 'larger_spread': 0.6499999999999915, 'verdict': 'không phân biệt được'}
- F_vs_A: {'a': 'F', 'b': 'A', 'diff_fps': 1.8299999999999983, 'diff_pct': 3.3400255521080457, 'larger_spread': 0.21000000000000085, 'verdict': 'không phân biệt được'}
- E3 triggered: False
