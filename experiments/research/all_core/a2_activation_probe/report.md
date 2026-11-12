# Static audit of candidate blocks (PC compile, no board)

| block | act | compiled ops | blockers (not measured multi-core) | partial ops | stable over sizes and precisions |
|---|---|---|---|---|---|
| act_pwl_slope25 | None | {'Conv': 2, 'BatchNormalization': 1, 'Clip': 1, 'Mul': 1} | {'Mul': 1, 'BatchNormalization': 1, 'Clip': 1} | Conv | yes |
| act_hs_noscale | None | {'Conv': 2, 'Add': 1, 'Clip': 1, 'Mul': 1} | {'Mul': 1, 'Clip': 1} | Conv | yes |
| act_relu_sq | None | {'ConvRelu': 1, 'Mul': 1, 'Conv': 1} | {'Mul': 1} | Conv | yes |
| act_relu_gate | None | {'Conv': 2, 'Clip': 1, 'Mul': 1} | {'Mul': 1, 'Clip': 1} | Conv | yes |
