# a7: exact trainable graphs (PC compile, 640x640, nc=5)

| model | prec | layers | params | GFLOPs | cycles M | blockers | ops |
|---|---|---|---|---|---|---|---|
| allcore_dense | fp16 | 43 | 3379771 | 7.218 | 9.1 | - | {'ConvRelu': 35, 'Conv': 6, 'Concat': 2} |
| allcore_dense | int8 | 43 | 3379771 | 7.218 | 4.8 | - | {'ConvRelu': 35, 'Conv': 6, 'Concat': 2} |
| allcore_mbconv | fp16 | 47 | 2533371 | 5.231 | 8.5 | - | {'ConvRelu': 35, 'ConvAdd': 4, 'Conv': 6, 'Concat': 2} |
| allcore_mbconv | int8 | 47 | 2533371 | 5.231 | 5.2 | - | {'ConvRelu': 35, 'ConvAdd': 4, 'Conv': 6, 'Concat': 2} |
