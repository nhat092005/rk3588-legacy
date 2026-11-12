"""All-core detector, mbconv family (ai/research/all_core): trained from scratch, no pretrained weights exist for it."""
from pathlib import Path

import ai.research.all_core.ultra  # noqa: F401  registers AllCoreDetect and MBConvU in ultralytics.nn.tasks

WEIGHTS = str(Path(__file__).with_name("allcore_mbconv.yaml"))
