# Phase 2e board results

A2 split: V-PC {'int8': 1.3639358087402798e-07, 'fp16': 1.3639358087402798e-07, 'ok': True}, V-board {'diff': -0.14, 'ok': True}, N2-1 {'minus_fp16_split': 0.77, 'minus_fp16': 0.63, 'minus_int8': 13.21, 'criterion': 47.98, 'meets': True}, layers {'rows': 106, 'counts': {'CPU|INT8': 3, 'NPU|INT8': 103}, 'cpu_ops': []}

QAT time: all realized True
- qat_vs_split|CORE_0: {'qat_ms': 13.84, 'other_ms': 13.02, 'diff_ms': 0.8200000000000003, 'diff_pct': 6.298003072196623, 'gap_ms': 0.016000000000000014, 'realized': True, 'verdict': 'khác'}
- qat_vs_int8h5|CORE_0: {'qat_ms': 13.84, 'other_ms': 16.382, 'diff_ms': -2.5420000000000016, 'diff_pct': -15.517030887559525, 'gap_ms': 0.012000000000000455, 'realized': True, 'verdict': 'khác'}
- qat_vs_split|CORE_0_1_2: {'qat_ms': 13.329, 'other_ms': 12.553999999999998, 'diff_ms': 0.7750000000000021, 'diff_pct': 6.173331209176376, 'gap_ms': 0.006000000000000227, 'realized': True, 'verdict': 'khác'}
- qat_vs_int8h5|CORE_0_1_2: {'qat_ms': 13.329, 'other_ms': 15.8975, 'diff_ms': -2.5685000000000002, 'diff_pct': -16.156628400691933, 'gap_ms': 0.017000000000001236, 'realized': True, 'verdict': 'khác'}
- total CORE_0: {'qat_npu_ms': 13.84, 'decode_ms': 8.02, 'qat_total_ms': 21.86, 'int8h5_ms': 16.382, 'diff_vs_int8h5_ms': 5.477999999999998}
- total CORE_0_1_2: {'qat_npu_ms': 13.329, 'decode_ms': 8.02, 'qat_total_ms': 21.349, 'int8h5_ms': 15.8975, 'diff_vs_int8h5_ms': 5.451499999999999}
- coverage [{'total_cycles': 5532317, 'build_total_cycles': 5532317, 'coverage': 0.034415959895284384}, {'total_cycles': 5532317, 'build_total_cycles': 5532317, 'coverage': 0.034415959895284384}]
- board table {'CPU|UINT8': 1, 'NPU|UINT8': 1, 'NPU|INT8': 102, 'NPU|FLOAT16': 8, 'CPU|INT8': 2}
m4: {"threads": 3, "calls_per_thread": 100, "calls_done": 300, "exceptions": 0, "exception_text": [], "outputs_with_diff_above_0": 300, "max_diff": 1.0411573648452759, "wall_s_threads": 5.001059610999619, "wall_s_300_sequential_incl_init": 6.255428628000118}
m4c: {"threads": 1, "calls_per_thread": 300, "calls_done": 300, "exceptions": 0, "exception_text": [], "outputs_with_diff_above_0": 0, "max_diff": 0.0, "wall_s_threads": 6.354986696999731, "wall_s_sequential_incl_init": 6.083469967999918}
m6: {"calls": [{"i": 0, "vs_same": 0.0, "vs_previous": null}, {"i": 1, "vs_same": 0.0, "vs_previous": 0.61075758934021}, {"i": 2, "vs_same": 0.0, "vs_previous": 0.38531017303466797}, {"i": 3, "vs_same": 0.0, "vs_previous": 0.43859776854515076}, {"i": 4, "vs_same": 0.0, "vs_previous": 0.7665213346481323}, {"i": 5, "vs_same": 0.0, "vs_previous": 0.7829174995422363}, {"i": 6, "vs_same": 0.0, "vs_previous": 0.422201544046402}, {"i": 7, "vs_same": 0.0, "vs_previous": 0.405805379152298}], "init_error": null}
m7: {"batch1_median_ms": 41.9027160000951, "batch4_output_shape": [4, 9, 8400], "max_abs_diff_per_image": [0.0, 0.0, 0.0, 0.0], "batch4_median_ms": 90.19063050027398, "batch4_ms_per_image": 22.547657625068496}
m16: {"contexts_created": 32, "stop": {"reason": "cap_reached", "at_context": 32}, "rows": [{"n": 1, "mem_available_mib": 3182, "rss_mib": 66.3, "max_diff": 0.0}, {"n": 2, "mem_available_mib": 3166, "rss_mib": 83.0, "max_diff": 0.0}, {"n": 3, "mem_available_mib": 3149, "rss_mib": 100.0, "max_diff": 0.0}, {"n": 4, "mem_available_mib": 3131, "rss_mib": 117.1, "max_diff": 0.0}, {"n": 5, "mem_available_mib": 3114, "rss_mib": 134.3, "max_diff": 0.0}, {"n": 6, "mem_available_mib": 3098, "rss_mib": 151.2, "max_diff": 0.0}, {"n": 7, "mem_available_mib": 3080, "rss_mib": 168.4, "max_diff": 0.0}, {"n": 8, "m
