# X7: INT8 against FP32 accuracy, ppe5_yolov8n_baseline_100ep_s0

| set | images | metric | INT8 | FP32 | difference pp |
|---|---|---|---|---|---|
| css_test | 82 | precision | 0.7660 | 0.8950 | -12.90 |
| css_test | 82 | recall | 0.6713 | 0.7396 | -6.83 |
| css_test | 82 | map50 | 0.6965 | 0.8006 | -10.41 |
| css_test | 82 | map50_95 | 0.3675 | 0.4999 | -13.24 |
| ppe2286_ext | 180 | precision | 0.5234 | 0.6160 | -9.26 |
| ppe2286_ext | 180 | recall | 0.6092 | 0.6051 | +0.41 |
| ppe2286_ext | 180 | map50 | 0.5330 | 0.5736 | -4.06 |
| ppe2286_ext | 180 | map50_95 | 0.2313 | 0.2624 | -3.11 |

## X7b: FP16 on the same RKNNLite path

| set | FP32 mAP50-95 | FP16 RKNNLite | D16 pp | INT8 minus FP32 (D8) pp |
|---|---|---|---|---|
| css_test | 0.4999 | 0.4976 | -0.23 | -13.24 |
| ppe2286_ext | 0.2624 | 0.2635 | +0.11 | -3.11 |

Verdict on css_test: mức giảm của INT8 chủ yếu đi theo lượng tử hóa
