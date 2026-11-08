"""Prepare ppe2286: 4 classes (Helmet, NoHelmet, NoVest, Vest) exactly as published, no remapping, no filtering.

train and val are the authors' train and valid folders (data/ppe2286/raw). The authors give no test split
(raw/data.yaml has an empty `test:`), so `test` points at the same images as `val`. Numbers measured on `test`
are therefore validation numbers: best.pt is chosen on the same images. raw/ is never modified.

`heldout` is the authors' valid images that are not in ppe5 test_ext (457 - 50 = 407). Neither ppe5 nor ppe5_plus
trains on them, so it is a common test set for models trained on ppe5, ppe5_plus, ppe4 and ppe2286. Needs
data/ppe5/processed (run prepare_ppe5.py first). ppe2286 itself still selects best.pt on `val`, which contains them.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import symlink_pair, write_classes, write_dataset_yaml

DATA = Path(__file__).resolve().parents[2] / "data"
ROOT = DATA / "ppe2286"
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"

CLASSES = ["Helmet", "NoHelmet", "NoVest", "Vest"]
SPLITS = {"train": "train", "val": "valid"}  # processed split -> raw folder
PPE5_IMAGES = DATA / "ppe5" / "processed" / "images"


def main() -> None:
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)  # derived data only, rebuilt from raw/
    for split, raw_split in SPLITS.items():
        images = sorted(p for p in (RAW / raw_split / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for image in images:
            label = RAW / raw_split / "labels" / f"{image.stem}.txt"
            symlink_pair(image, label, PROCESSED / "images" / split, PROCESSED / "labels" / split)
        print(f"{split}: {len(images)} images")

    # heldout: valid images that ppe5 test_ext (and so ppe5_plus train) does not contain
    test_ext = {p.name for p in (PPE5_IMAGES / "test_ext").iterdir()}
    heldout = sorted(p for p in (RAW / "valid" / "images").iterdir() if p.name not in test_ext)
    for image in heldout:
        symlink_pair(image, RAW / "valid" / "labels" / f"{image.stem}.txt",
                     PROCESSED / "images" / "heldout", PROCESSED / "labels" / "heldout")
    ppe5_names = {p.name for split in ("train", "val", "test") for p in (PPE5_IMAGES / split).iterdir()}
    assert not ppe5_names & {p.name for p in heldout}, "heldout overlaps a ppe5 split"
    print(f"heldout: {len(heldout)} images (valid minus {len(test_ext)} in test_ext)")
    write_classes(CLASSES, ROOT / "classes.txt")
    write_dataset_yaml(
        PROCESSED / "dataset.yaml",
        PROCESSED,
        CLASSES,
        {"train": "images/train", "val": "images/val", "test": "images/val", "heldout": "images/heldout"},  # no author test split
    )


if __name__ == "__main__":
    main()
