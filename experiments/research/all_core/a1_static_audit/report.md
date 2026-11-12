# Static audit of candidate blocks (PC compile, no board)

| block | act | compiled ops | blockers (not measured multi-core) | partial ops | stable over sizes and precisions |
|---|---|---|---|---|---|
| conv_act | relu | {'ConvRelu': 1} | - | - | yes |
| conv_act | relu6 | {'ConvClip': 1} | - | - | yes |
| conv_act | leakyrelu | {'ConvLeakyRelu': 1} | - | - | yes |
| conv_act | silu | {'ConvExSwish': 1} | {'ConvExSwish': 1} | - | yes |
| conv_act | hardswish | {'ConvHardSwish': 1} | {'ConvHardSwish': 1} | - | yes |
| hswish_split | None | {'ConvHardSwish': 1} | {'ConvHardSwish': 1} | - | yes |
| dense3 | relu | {'ConvRelu': 2} | - | - | yes |
| dense3 | leakyrelu | {'ConvLeakyRelu': 2} | - | - | yes |
| res3 | relu | {'ConvRelu': 1, 'ConvAddRelu': 1} | {'ConvAddRelu': 1} | - | yes |
| res3 | leakyrelu | {'ConvLeakyRelu': 1, 'ConvAdd': 1, 'LeakyRelu': 1} | {'LeakyRelu': 1} | - | yes |
| mbconv_k3 | relu | {'ConvRelu': 2, 'ConvAdd': 1} | - | - | yes |
| mbconv_k3 | relu6 | {'ConvClip': 2, 'ConvAdd': 1} | - | - | yes |
| mbconv_k5 | relu6 | {'ConvClip': 2, 'ConvAdd': 1} | - | - | yes |
| uib_ib | relu | {'ConvRelu': 2, 'ConvAdd': 1} | - | - | yes |
| uib_convnext | relu | {'Conv': 1, 'ConvRelu': 1, 'ConvAdd': 1} | - | Conv | yes |
| uib_extradw | relu | {'Conv': 1, 'ConvRelu': 2, 'ConvAdd': 1} | - | Conv | yes |
| uib_ffn | relu | {'ConvRelu': 1, 'ConvAdd': 1} | - | - | yes |
| star | relu6 | {'Conv': 3, 'ConvClip': 1, 'ConvMul': 1, 'Add': 1} | {'ConvMul': 1} | Conv | yes |
| ghost | relu | {'ConvRelu': 2, 'Concat': 1} | - | Concat | yes |
| down_conv | relu | {'ConvRelu': 1} | - | - | yes |
| down_maxpool | relu | {'MaxPool': 1} | {'MaxPool': 1} | - | yes |
| neck_resize | relu | {'Resize': 1, 'Concat': 1, 'ConvRelu': 1} | {'Resize': 1} | Concat | yes |
| neck_convt | relu | {'ConvTranspose': 1, 'Concat': 1, 'ConvRelu': 1} | {'ConvTranspose': 1} | Concat | yes |
| neck_none | relu | {'ConvRelu': 1} | - | - | yes |
| head_v6 | leakyrelu | {'ConvLeakyRelu': 9, 'Conv': 6} | - | Conv | yes |
| head_v6 | silu | {'ConvExSwish': 9, 'Conv': 6} | {'ConvExSwish': 9} | Conv | yes |
| head_v6 | relu | {'ConvRelu': 9, 'Conv': 6} | - | Conv | yes |
