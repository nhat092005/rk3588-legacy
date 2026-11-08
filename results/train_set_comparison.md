# Training-set comparison (YOLOv8n, Ultralytics defaults, seeds 0 to 2)

Mean ± sample std over seeds. AP50-95 per class. PPE4 = mean of Helmet, NoHelmet, NoVest, Vest. `-` = the model has no such class, or the test set has no ground truth for it.

## Test set `css_test` (82 images, fingerprint 480dd0fabae6)

| Train set | Helmet | NoHelmet | NoVest | Vest | Person | PPE4 AP50-95 | PPE4 AP50 |
|---|---|---|---|---|---|---|---|
| ppe5 | 0.601 ± 0.014 | 0.312 ± 0.001 | 0.504 ± 0.011 | 0.584 ± 0.021 | 0.531 ± 0.011 | 0.500 ± 0.008 | 0.799 ± 0.009 |
| ppe5_plus | 0.469 ± 0.020 | 0.258 ± 0.009 | 0.482 ± 0.003 | 0.576 ± 0.012 | 0.552 ± 0.003 | 0.446 ± 0.004 | 0.762 ± 0.012 |
| ppe4 | 0.604 ± 0.016 | 0.360 ± 0.019 | 0.515 ± 0.012 | 0.592 ± 0.004 | - | 0.518 ± 0.005 | 0.817 ± 0.014 |
| ppe2286 | 0.063 ± 0.018 | 0.012 ± 0.003 | 0.079 ± 0.019 | 0.174 ± 0.057 | - | 0.082 ± 0.022 | 0.209 ± 0.052 |

## Test set `ppe2286_heldout` (407 images, fingerprint 642006bdec0a)

| Train set | Helmet | NoHelmet | NoVest | Vest | Person | PPE4 AP50-95 | PPE4 AP50 |
|---|---|---|---|---|---|---|---|
| ppe5 | 0.055 ± 0.003 | 0.004 ± 0.001 | 0.149 ± 0.009 | 0.231 ± 0.011 | - | 0.109 ± 0.003 | 0.257 ± 0.004 |
| ppe5_plus | 0.227 ± 0.003 | 0.342 ± 0.015 | 0.197 ± 0.006 | 0.320 ± 0.007 | - | 0.271 ± 0.002 | 0.443 ± 0.003 |
| ppe4 | 0.050 ± 0.002 | 0.004 ± 0.000 | 0.133 ± 0.014 | 0.266 ± 0.013 | - | 0.113 ± 0.005 | 0.258 ± 0.006 |
| ppe2286 | 0.405 ± 0.009 | 0.541 ± 0.004 | 0.308 ± 0.015 | 0.519 ± 0.012 | - | 0.443 ± 0.003 | 0.632 ± 0.007 |
