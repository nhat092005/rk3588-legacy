"""Prepare ppe4: ppe5 without the Person class, same images and same splits.

Used to isolate the effect of the Person class: ppe4 and ppe5 differ only by the Person boxes. Images are
symlinks to the ppe5 images. Label files are new copies of the ppe5 labels with class 4 (Person) lines removed.
Images whose only labels were Person stay in the split as background images. test_ext is not carried over.
Requires data/ppe5/processed (run prepare_ppe5.py first).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import write_classes, write_dataset_yaml

DATA = Path(__file__).resolve().parents[2] / "data"
SRC = DATA / "ppe5"
ROOT = DATA / "ppe4"
PROCESSED = ROOT / "processed"

SPLITS = ["train", "val", "test"]
PERSON = 4


def main() -> None:
    classes = (SRC / "classes.txt").read_text().split()
    assert classes[PERSON] == "Person", classes
    classes = classes[:PERSON]
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)  # derived data only, rebuilt from ppe5/processed
    for split in SPLITS:
        images_dst = PROCESSED / "images" / split
        labels_dst = PROCESSED / "labels" / split
        images_dst.mkdir(parents=True)
        labels_dst.mkdir(parents=True)
        background = 0
        images = sorted((SRC / "processed" / "images" / split).iterdir())
        for image in images:
            (images_dst / image.name).symlink_to(image.resolve())
            lines = (SRC / "processed" / "labels" / split / f"{image.stem}.txt").read_text().splitlines()
            kept = [line for line in lines if line.strip() and int(line.split()[0]) != PERSON]
            (labels_dst / f"{image.stem}.txt").write_text("".join(line + "\n" for line in kept))
            background += not kept
        print(f"{split}: {len(images)} images, {background} without labels")
    write_classes(classes, ROOT / "classes.txt")
    write_dataset_yaml(
        PROCESSED / "dataset.yaml",
        PROCESSED,
        classes,
        {split: f"images/{split}" for split in SPLITS},
    )


if __name__ == "__main__":
    main()
