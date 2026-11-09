# split head

V-PC {'max_abs': 0.00018310546875}
V-board {'fp16_split': 49.94, 'fp16': 49.76, 'diff': 0.17999999999999972, 'ok': True}
S-1 {'split_minus_fp16': 0.0800000000000054, 'split_minus_int8h5': 0.18000000000000682, 'split_minus_int8': 13.090000000000003, 'meets': True}
S-4 {'split': {'kept_anchors': 14805, 'decode_ms_median': 8.261}, 'fp16_split': {'kept_anchors': 15719, 'decode_ms_median': 8.222}, 'int8': {'kept_anchors': 11255, 'decode_ms_median': None}, 'int8h5': {'kept_anchors': 11255, 'decode_ms_median': None}}
decode 8.02 ms p95 8.246
All realized: False

- cost int8h5|CORE_0: {'int8_ms': 16.52, 'ms': 16.4425, 'diff_ms': -0.07750000000000057, 'diff_pct': -0.4691283292978243, 'gap_ms': 0.015999999999998238, 'changes': 'chưa kết luận'}
- cost split|CORE_0: {'int8_ms': 16.52, 'ms': 13.07, 'diff_ms': -3.4499999999999993, 'diff_pct': -20.883777239709442, 'gap_ms': 0.018000000000000682, 'changes': 'chưa kết luận'}
- cost int8h5|CORE_0_1_2: {'int8_ms': 16.0395, 'ms': 15.928, 'diff_ms': -0.11149999999999949, 'diff_pct': -0.6951588266467127, 'gap_ms': 0.023999999999999133, 'changes': 'chưa kết luận'}
- cost split|CORE_0_1_2: {'int8_ms': 16.0395, 'ms': 12.557, 'diff_ms': -3.4825, 'diff_pct': -21.712023442127247, 'gap_ms': 0.012000000000000455, 'changes': 'chưa kết luận'}
- total CORE_0: {'split_npu_ms': 13.07, 'decode_ms': 8.02, 'split_total_ms': 21.09, 'int8h5_ms': 16.4425, 'diff_vs_int8h5_ms': 4.647500000000001}
- total CORE_0_1_2: {'split_npu_ms': 12.557, 'decode_ms': 8.02, 'split_total_ms': 20.576999999999998, 'int8h5_ms': 15.928, 'diff_vs_int8h5_ms': 4.648999999999997}
- coverage split {'per_session': [0.01822021406220938, 0.01822021406220938], 'build_total_cycles': 5532317, 'rows': 106, 'cpu_ops': [], 'dtypes': ['CPU|INT8', 'NPU|INT8']}
