"""ai/research/modular: the frame holds contracts only, so it is tested with stub parts defined here.
Runs in a subprocess because register() replaces Detect in ultralytics.nn.tasks for its whole process."""
import subprocess
import sys

import torch
import torch.nn as nn

from ai.paths import REPO_ROOT

YAML = """
nc: 5
backbone:
  - [-1, 1, StubBackbone, []]
head:
  - [0, 1, StubNeck, [[8, 16, 32]]]
  - [1, 1, Index, [8, 0]]
  - [1, 1, Index, [16, 1]]
  - [1, 1, Index, [32, 2]]
  - [[2, 3, 4], 1, Detect, [nc]]
"""


def _build_and_run() -> None:
    import tempfile
    from types import SimpleNamespace

    from ultralytics import YOLO

    from ai.research.modular.backbone import Backbone
    from ai.research.modular.head import Head
    from ai.research.modular.neck import Neck
    from ai.research.modular.register import register

    class StubBackbone(Backbone):
        def __init__(self):
            super().__init__()
            self.out_channels = (8, 16, 32)
            self.s3, self.s4, self.s5 = nn.Conv2d(3, 8, 8, 8), nn.Conv2d(8, 16, 2, 2), nn.Conv2d(16, 32, 2, 2)

        def forward(self, x):
            p3 = self.s3(x)
            p4 = self.s4(p3)
            return [p3, p4, self.s5(p4)]

    class StubNeck(Neck):
        def __init__(self, in_channels):
            super().__init__()
            self.out_channels = tuple(in_channels)

        def forward(self, feats):
            return list(feats)

    register(StubBackbone, StubNeck, Head)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        f.write(YAML)
        f.flush()
        model = YOLO(f.name).model
    assert type(model.model[-1]) is Head
    assert [int(s) for s in model.stride] == [8, 16, 32]

    model.train()
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    batch = {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [3.0]]),
        "bboxes": torch.tensor([[0.3, 0.4, 0.2, 0.2], [0.5, 0.5, 0.4, 0.4]]),
    }
    loss, _ = model.loss(batch)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)


def test_stub_parts_build_and_train_mechanically():
    p = _run("from tests.test_modular_frame import _build_and_run; _build_and_run()")
    assert p.returncode == 0, p.stderr[-2000:]


def test_modules_are_independent():
    """Each module imports alone: no other part of the frame and nothing from all_core."""
    parts = ("backbone", "neck", "head", "register")
    for name in parts:
        others = [f"ai.research.modular.{o}" for o in parts if o != name]
        code = (
            "import sys\n"
            f"import ai.research.modular.{name}\n"
            f"bad = [k for k in sys.modules if k.startswith('ai.research.all_core') or k in {others}]\n"
            "assert not bad, bad\n"
        )
        p = _run(code)
        assert p.returncode == 0, (name, p.stderr[-1000:])
