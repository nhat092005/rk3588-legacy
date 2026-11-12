# a6: head audit (PC compile, dense bottom-up ReLU 640x640, nc=5)

| head | prec | layers | head params | params M | GFLOPs | cycles M | blockers | ops |
|---|---|---|---|---|---|---|---|---|
| v6 | fp16 | 34 | 1641051 | 4.90 | 9.784 | 11.4 | - | {'ConvRelu': 26, 'Conv': 6, 'Concat': 2} |
| v6 | int8 | 34 | 1641051 | 4.90 | 9.784 | 5.7 | - | {'ConvRelu': 26, 'Conv': 6, 'Concat': 2} |
| stock | fp16 | 43 | 121563 | 3.38 | 7.218 | 9.1 | - | {'ConvRelu': 35, 'Conv': 6, 'Concat': 2} |
| stock | int8 | 43 | 121563 | 3.38 | 7.218 | 4.8 | - | {'ConvRelu': 35, 'Conv': 6, 'Concat': 2} |
| stock_legacy | fp16 | 37 | 442203 | 3.70 | 8.48 | 10.1 | - | {'ConvRelu': 29, 'Conv': 6, 'Concat': 2} |
| stock_legacy | int8 | 37 | 442203 | 3.70 | 8.48 | 5.2 | - | {'ConvRelu': 29, 'Conv': 6, 'Concat': 2} |
