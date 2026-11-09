"""Prepare ppe5_plus: ppe5 with the 180 test_ext images moved into train.

train = ppe5 train + ppe5 test_ext, val and test are the ppe5 splits unchanged, so css_test stays comparable.
There is no external test set here. Requires data/ppe5/processed (run prepare_ppe5.py first).
Images and labels are symlinks to the same files ppe5 uses.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import symlink_pair, write_classes, write_dataset_yaml

DATA = Path(__file__).resolve().parents[2] / "data"
SRC = DATA / "ppe5"
ROOT = DATA / "ppe5_plus"
PROCESSED = ROOT / "processed"

# ppe5_plus split <- ppe5 splits that feed it
SPLITS = {"train": ["train", "test_ext"], "val": ["val"], "test": ["test"]}


def main() -> None:
    classes = (SRC / "classes.txt").read_text().split()
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)  # derived data only, rebuilt from ppe5/processed
    for split, sources in SPLITS.items():
        for source in sources:
            images = sorted((SRC / "processed" / "images" / source).iterdir())
            for image in images:
                label = SRC / "processed" / "labels" / source / f"{image.stem}.txt"
                symlink_pair(image, label, PROCESSED / "images" / split, PROCESSED / "labels" / split)
        print(f"{split}: {len(list((PROCESSED / 'images' / split).iterdir()))} images")
    write_classes(classes, ROOT / "classes.txt")
    write_dataset_yaml(
        PROCESSED / "dataset.yaml",
        PROCESSED,
        classes,
        {split: f"images/{split}" for split in SPLITS},
    )


if __name__ == "__main__":
    main()
