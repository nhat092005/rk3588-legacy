# ppe5_plus - Data Card

Derived from `ppe5`, same 5 classes: `0 Helmet, 1 NoHelmet, 2 NoVest, 3 Vest, 4 Person` (`classes.txt`).
Built by `scripts/data/prepare_ppe5_plus.py` (`make prepare-data DATASET=ppe5_plus`, needs `data/ppe5/processed`).
`processed/` holds symlinks to the same image and label files `ppe5` uses.

| Split | Source | Images |
|---|---|---|
| `train` | `ppe5` train (2,605) + `ppe5` test_ext (180) | 2,785 |
| `val` | `ppe5` val | 114 |
| `test` | `ppe5` test | 82 |

The only difference from `ppe5` is that the 180 ppe2286 images are in `train` instead of an external test set.
`val` and `test` are unchanged, so results on eval set `css_test` compare directly with `ppe5`. There is no external
test set: a model trained on `ppe5_plus` has seen the `ppe2286_ext` images and must not be evaluated on them.

Measured train instances (2026-10-26): Helmet 3,329, NoHelmet 2,381, NoVest 4,066, Vest 3,177, Person 9,779 (70 empty
label files). Person labels of the 180 added images are COCO `yolov8s.pt` pseudo-labels, see `data/ppe5/DATA_CARD.md`.
Other known limits are the same as `ppe5`.
