"""Makes parts usable by name in an Ultralytics yaml (parse_model resolves names in `ultralytics.nn.tasks`).

    register(MyBackbone, MyNeck, MyHead)

A `Detect` subclass replaces `tasks.Detect`: the yaml keeps writing `Detect` and parse_model passes it reg_max and the channel tuple.
Any other class is registered under its own name. The change lasts for the whole process, so build stock Ultralytics yaml files in another process.

Yaml wiring (Backbone and Neck return lists, `Index` picks one tensor so that Detect reads plain tensors, channel counts are written in the yaml):

    backbone:
      - [-1, 1, MyBackbone, []]            # 0 -> [P3, P4, P5]
    head:
      - [0, 1, MyNeck, [[c3, c4, c5]]]     # 1 -> [N3, N4, N5]
      - [1, 1, Index, [c3, 0]]             # 2 N3
      - [1, 1, Index, [c4, 1]]             # 3 N4
      - [1, 1, Index, [c5, 2]]             # 4 N5
      - [[2, 3, 4], 1, Detect, [nc]]       # 5
"""
from __future__ import annotations

from ultralytics.nn import tasks
from ultralytics.nn.modules.head import Detect


def register(*parts: type) -> None:
    for cls in parts:
        setattr(tasks, "Detect" if issubclass(cls, Detect) else cls.__name__, cls)
