# ppe4 — Data Card

Derived dataset, 4 classes: `0 Helmet, 1 NoHelmet, 2 NoVest, 3 Vest` (`classes.txt`). It is `ppe5` without the Person
class: same images, same train/val/test split, same boxes for the four PPE classes.
Built by `scripts/data/prepare_ppe4.py` (`make prepare-data DATASET=ppe4`, needs `data/ppe5/processed`).
Images are symlinks to the ppe5 images. Label files are new copies of the ppe5 labels with the Person (class 4) lines
removed. `raw/` of the sources is never modified.

Purpose: isolate the effect of the Person class. `ppe5` and `ppe4` differ only by the Person boxes, so a difference
between them on the four PPE classes is caused by Person. The comparison `ppe5` vs `ppe2286` cannot do this because
the two also differ in image source, size and labeling style.

| Split | Source | Images | Notes |
|---|---|---|---|
| `train` | `ppe5` train | 2,605 | |
| `val` | `ppe5` val | 114 | |
| `test` | `ppe5` test | 82 | eval set `ppe4_test` (same images as `css_test`, no Person labels) |

`ppe5` `test_ext` is not carried over (its Person labels are pseudo-labels and the images are ppe2286 images).

## Measured counts (2026-10-26, instances per class)

| Split | Helmet | NoHelmet | NoVest | Vest | Person |
|---|---|---|---|---|---|
| train | 3,145 | 2,317 | 3,962 | 3,033 | 0 |
| val | 79 | 69 | 106 | 41 | 0 |
| test | 110 | 41 | 90 | 61 | 0 |

The four PPE columns equal the `ppe5` counts exactly.

Images without labels (background): train 153, val 33, test 27. In `ppe5` they are 70, 30, 22. The difference (83, 3,
5) are images whose only labels were Person. They are kept as background images so that the image set stays identical
to `ppe5`. Such an image may show an unlabeled person.

## Evaluation

A model trained on `ppe4` has no Person output. On a test set with Person ground truth (`css_test`), the Person AP is 0
and is excluded by reading only the "PPE4" columns of `results/train_set_comparison.md` (`make compare-train-sets`).
Common test sets for the comparison: `css_test` and `ppe2286_heldout` (see `data/ppe2286/DATA_CARD.md`).
