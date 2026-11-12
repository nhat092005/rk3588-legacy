# Static audit of whole detectors (PC compile, no board, 640x640, nc=5)

Params include BN. GFLOPs = 2 x MACs of Conv and ConvTranspose. Cycles are the compiler estimate, not latency.

| net | prec | layers | params M | GFLOPs | cycles M | blockers (layers per op) | partial ops |
|---|---|---|---|---|---|---|---|
| yolov8n (reference) | fp16 | 117 | 3.01 | 8.087 | 9.5 | {'ConvExSwish': 57, 'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Sub': 2, 'Mul': 2, 'Sigmoid': 1, 'Transpose': 1} | Concat, Conv |
| yolov8n (reference) | int8 | 117 | 3.01 | 8.087 | 5.7 | {'ConvExSwish': 57, 'Split': 9, 'MaxPool': 3, 'Resize': 2, 'Reshape': 7, 'exSoftmax13': 1, 'Sub': 2, 'Mul': 2, 'Sigmoid': 1, 'Transpose': 1} | Concat, Conv |
| yolo26n (reference) | fp16 | 180 | 2.51 | 5.626 | 7.3 | {'ConvExSwish': 87, 'Split': 12, 'MaxPool': 3, 'Reshape': 13, 'exSDPAttention': 2, 'Resize': 2, 'Sub': 2, 'Mul': 2, 'Sigmoid': 1} | Concat, Conv |
| yolo26n (reference) | int8 | 180 | 2.51 | 5.626 | 5.0 | {'ConvExSwish': 87, 'Split': 12, 'MaxPool': 3, 'Reshape': 13, 'exSDPAttention': 2, 'Resize': 2, 'Sub': 2, 'Mul': 2, 'Sigmoid': 1} | Concat, Conv |
| dense resize relu | fp16 | 40 | 5.45 | 12.615 | 14.1 | {'Resize': 2} | Concat, Conv |
| dense resize relu | int8 | 40 | 5.45 | 12.615 | 7.1 | {'Resize': 2} | Concat, Conv |
| dense resize leakyrelu | fp16 | 40 | 5.45 | 12.615 | 14.1 | {'Resize': 2} | Concat, Conv |
| dense resize leakyrelu | int8 | 40 | 5.45 | 12.615 | 7.1 | {'Resize': 2} | Concat, Conv |
| dense convt relu | fp16 | 40 | 5.78 | 13.035 | 15.8 | {'ConvTranspose': 2} | Concat, Conv |
| dense convt relu | int8 | 40 | 5.78 | 13.035 | 7.9 | {'ConvTranspose': 2} | Concat, Conv |
| dense convt leakyrelu | fp16 | 40 | 5.78 | 13.035 | 15.8 | {'ConvTranspose': 2} | Concat, Conv |
| dense convt leakyrelu | int8 | 40 | 5.78 | 13.035 | 7.9 | {'ConvTranspose': 2} | Concat, Conv |
| dense none relu | fp16 | 34 | 4.90 | 9.784 | 11.4 | - | Concat, Conv |
| dense none relu | int8 | 34 | 4.90 | 9.784 | 5.7 | - | Concat, Conv |
| dense none leakyrelu | fp16 | 34 | 4.90 | 9.784 | 11.4 | - | Concat, Conv |
| dense none leakyrelu | int8 | 34 | 4.90 | 9.784 | 5.7 | - | Concat, Conv |
| res resize relu | fp16 | 40 | 5.45 | 12.615 | 14.1 | {'Resize': 2} | Concat, Conv |
| res resize relu | int8 | 40 | 5.45 | 12.615 | 7.1 | {'Resize': 2} | Concat, Conv |
| res resize leakyrelu | fp16 | 40 | 5.45 | 12.615 | 14.1 | {'Resize': 2} | Concat, Conv |
| res resize leakyrelu | int8 | 40 | 5.45 | 12.615 | 7.1 | {'Resize': 2} | Concat, Conv |
| res convt relu | fp16 | 40 | 5.78 | 13.035 | 15.8 | {'ConvTranspose': 2} | Concat, Conv |
| res convt relu | int8 | 40 | 5.78 | 13.035 | 7.9 | {'ConvTranspose': 2} | Concat, Conv |
| res convt leakyrelu | fp16 | 40 | 5.78 | 13.035 | 15.8 | {'ConvTranspose': 2} | Concat, Conv |
| res convt leakyrelu | int8 | 40 | 5.78 | 13.035 | 7.9 | {'ConvTranspose': 2} | Concat, Conv |
| res none relu | fp16 | 34 | 4.90 | 9.784 | 11.4 | - | Concat, Conv |
| res none relu | int8 | 34 | 4.90 | 9.784 | 5.7 | - | Concat, Conv |
| res none leakyrelu | fp16 | 34 | 4.90 | 9.784 | 11.4 | - | Concat, Conv |
| res none leakyrelu | int8 | 34 | 4.90 | 9.784 | 5.7 | - | Concat, Conv |
| mbconv resize relu6 | fp16 | 44 | 4.61 | 10.629 | 13.6 | {'Resize': 2} | Concat, Conv |
| mbconv resize relu6 | int8 | 44 | 4.61 | 10.629 | 7.5 | {'Resize': 2} | Concat, Conv |
| mbconv resize relu | fp16 | 44 | 4.61 | 10.629 | 13.6 | {'Resize': 2} | Concat, Conv |
| mbconv resize relu | int8 | 44 | 4.61 | 10.629 | 7.5 | {'Resize': 2} | Concat, Conv |
| mbconv convt relu6 | fp16 | 44 | 4.93 | 11.048 | 15.2 | {'ConvTranspose': 2} | Concat, Conv |
| mbconv convt relu6 | int8 | 44 | 4.93 | 11.048 | 8.3 | {'ConvTranspose': 2} | Concat, Conv |
| mbconv convt relu | fp16 | 44 | 4.93 | 11.048 | 15.2 | {'ConvTranspose': 2} | Concat, Conv |
| mbconv convt relu | int8 | 44 | 4.93 | 11.048 | 8.3 | {'ConvTranspose': 2} | Concat, Conv |
| mbconv none relu6 | fp16 | 38 | 4.05 | 7.798 | 10.8 | - | Concat, Conv |
| mbconv none relu6 | int8 | 38 | 4.05 | 7.798 | 6.1 | - | Concat, Conv |
| mbconv none relu | fp16 | 38 | 4.05 | 7.798 | 10.8 | - | Concat, Conv |
| mbconv none relu | int8 | 38 | 4.05 | 7.798 | 6.1 | - | Concat, Conv |
| uib resize relu | fp16 | 48 | 4.61 | 10.657 | 13.9 | {'Resize': 2} | Concat, Conv |
| uib resize relu | int8 | 48 | 4.61 | 10.657 | 7.8 | {'Resize': 2} | Concat, Conv |
| uib convt relu | fp16 | 48 | 4.94 | 11.076 | 15.5 | {'ConvTranspose': 2} | Concat, Conv |
| uib convt relu | int8 | 48 | 4.94 | 11.076 | 8.6 | {'ConvTranspose': 2} | Concat, Conv |
| uib none relu | fp16 | 42 | 4.06 | 7.825 | 11.1 | - | Concat, Conv |
| uib none relu | int8 | 42 | 4.06 | 7.825 | 6.4 | - | Concat, Conv |
| star resize relu | fp16 | 56 | 4.72 | 11.029 | 16.6 | {'Resize': 2, 'ConvMul': 4} | Concat, Conv |
| star resize relu | int8 | 56 | 4.72 | 11.029 | 10.5 | {'Resize': 2, 'ConvMul': 4} | Concat, Conv |
| star convt relu | fp16 | 56 | 5.05 | 11.449 | 18.2 | {'ConvMul': 4, 'ConvTranspose': 2} | Concat, Conv |
| star convt relu | int8 | 56 | 5.05 | 11.449 | 11.3 | {'ConvMul': 4, 'ConvTranspose': 2} | Concat, Conv |
| star none relu | fp16 | 50 | 4.17 | 8.198 | 13.8 | {'ConvMul': 4} | Concat, Conv |
| star none relu | int8 | 50 | 4.17 | 8.198 | 9.1 | {'ConvMul': 4} | Concat, Conv |
