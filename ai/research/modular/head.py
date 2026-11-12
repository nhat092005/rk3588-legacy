"""Head contract. No design choice here: `Head` is Ultralytics `Detect` unchanged, a new head subclasses it.

The head must subclass `Detect` and be the last layer of the yaml, because DetectionModel computes strides, the loss and the export from
`model.model[-1]`. It reads [N3, N4, N5] as three tensors.
"""
from __future__ import annotations

from ultralytics.nn.modules.head import Detect


class Head(Detect):
    pass
