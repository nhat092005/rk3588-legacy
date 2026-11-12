"""All-core detector, dense family, grid point dense_w150_d2 (ai/research/all_core/grid/scale_grid.py): trained from scratch, no pretrained weights exist for it."""
from pathlib import Path

import ai.research.all_core.ultra  # noqa: F401  registers AllCoreDetect and MBConvU in ultralytics.nn.tasks

WEIGHTS = str(Path(__file__).with_name("allcore_dense_w150_d2.yaml"))
