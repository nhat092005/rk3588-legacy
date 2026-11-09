# int8h5 standard export

V-1 {'layers_equal': True, 'ap_diff_pp': 0.0, 'verdict': 'export reproduces h5', 'sha_export': '80be7f5d6adaa091b5b706f24052f97218dbbcda39e28d3ab2034e812cc24938', 'sha_build_recipe': '9658075e443e447408d1ee20da4d147e5df118242bee7b66337880e08ccb4dfa'}
V-2 {'sha_default_int8': 'a501f94ee391bab370c141bbc4853eb78c062635b0a01597fe3a9efb0d99e7dd', 'unchanged': True}

- silu_s0: int8h5 49.66, int8 36.75, fp16 49.76
- silu_s1: int8h5 50.79, int8 38.10, fp16 51.01
- silu_s2: int8h5 50.69, int8 37.74, fp16 51.00
- relu_s0: int8h5 46.98, int8 33.24, fp16 46.49
- range css_test AP50-95: {'int8h5': {'map50_95': 1.1300000000000026, 'NoHelmet': 3.700000000000003, 'Vest': 3.9799999999999898}, 'int8': {'map50_95': 1.3500000000000014, 'NoHelmet': 1.8499999999999979, 'Vest': 2.4000000000000057}, 'fp16': {'map50_95': 1.25, 'NoHelmet': 0.6199999999999974, 'Vest': 4.3600000000000065}}
