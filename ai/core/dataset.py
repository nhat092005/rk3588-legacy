"""Resolve dataset.yaml paths produced by scripts/data/prepare_<name>.py."""
from pathlib import Path

from ai.paths import REPO_ROOT


def dataset_yaml_path(name: str) -> Path:
    path = REPO_ROOT / "data" / name / "processed" / "dataset.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/data/prepare_{name}.py first."
        )
    return path
