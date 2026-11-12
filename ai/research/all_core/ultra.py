"""Hooks that let Ultralytics build and train the all-core detectors from a yaml (parse_model resolves module names in ultralytics.nn.tasks).

Importing this module, once per process:
  - registers `MBConvU` (yaml: [c, act, k, e], act is a name in blocks.ACTS) so parse_model keeps channels at ch[f],
  - replaces `ultralytics.nn.tasks.Detect` by `AllCoreDetect`: stock Detect with a non-legacy cls branch (DWConv3, Conv1) that parse_model cannot
    switch on for a yaml without C3k2 (it forces `Detect.legacy = True`). Same class name in the yaml, so reg_max, end2end and ch args are extended as usual.
A process that also builds stock YOLOv8 yaml files after this import would get the non-legacy head: train a control in its own process.
"""
from __future__ import annotations

import torch.nn as nn
from ultralytics.nn import tasks
from ultralytics.nn.modules.head import Detect

from ai.research.all_core.candidates.blocks import cb


class MBConvU(nn.Module):
    """blocks.MBConv (pw expand, dw, pw linear, Add) with the argument order of an Ultralytics yaml row: [c, act, k, e].
    The conv stack is `body`, not `f`: Ultralytics sets module.f (the 'from' index) on every layer."""

    def __init__(self, c, act, k=3, e=4):
        super().__init__()
        h = c * e
        self.body = nn.Sequential(cb(c, h, 1, act=act), cb(h, h, k, g=h, act=act), cb(h, c, 1))

    def forward(self, x):
        return x + self.body(x)


class _PinLegacy(type):
    """parse_model assigns `Detect.legacy = True` on the class before building. Ignore that assignment for AllCoreDetect."""

    def __setattr__(cls, name, value):
        if name == "legacy" and cls.__dict__.get("legacy") is False:
            return
        super().__setattr__(name, value)


class AllCoreDetect(Detect, metaclass=_PinLegacy):
    legacy = False


tasks.MBConvU = MBConvU
tasks.Detect = AllCoreDetect
