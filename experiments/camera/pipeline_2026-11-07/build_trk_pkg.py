"""Build a torch-free copy of the Ultralytics ByteTrack code (same source files) for timing on the board.

Usage: python3 build_trk_pkg.py <ultralytics_dir> <out_dir> <scripts/demo/video/track.py>
Copies byte_tracker.py, basetrack.py, kalman_filter.py, matching.py, stracks.py unchanged and writes small stubs for
ultralytics.utils (LOGGER), ultralytics.utils.ops (xywh2ltwh extracted from the real source, linear_sum_assignment stubbed because the default use_lap=True path never calls it)
and ultralytics.utils.metrics (bbox_ioa extracted, batch_probiou stubbed: only used for oriented boxes).
"""
import ast
import shutil
import sys
from pathlib import Path

src, out, track_py = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if out.exists():
    shutil.rmtree(out)
(out / "ultralytics/utils").mkdir(parents=True)
(out / "ultralytics/trackers/utils").mkdir(parents=True)
(out / "ultralytics/__init__.py").write_text('__version__ = "trackers-only copy of ultralytics"\n')
(out / "ultralytics/trackers/__init__.py").write_text("")
(out / "ultralytics/trackers/utils/__init__.py").write_text("")
for rel in ("basetrack.py", "byte_tracker.py", "utils/kalman_filter.py", "utils/matching.py", "utils/stracks.py"):
    shutil.copy(src / "trackers" / rel, out / "ultralytics/trackers" / rel)
(out / "ultralytics/utils/__init__.py").write_text("import logging\nLOGGER = logging.getLogger('ultralytics')\n")


def extract(path, names):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    parts = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            parts.append(ast.get_source_segment(text, node))
    assert len(parts) == len(names), (path, names, len(parts))
    return "\n\n\n".join(parts)


ops = extract(src / "utils/ops.py", {"xywh2ltwh"}) + "\n\n\ndef linear_sum_assignment(cost_matrix):\n    raise NotImplementedError('use_lap=True path (lap.lapjv) is the default and is what runs')"
(out / "ultralytics/utils/ops.py").write_text(
    "import numpy as np\n\n\nclass _NoTorch:\n    Tensor = ()  # isinstance(x, ()) is always False\n\n\ntorch = _NoTorch()\n\n\n" + ops + "\n")
met = extract(src / "utils/metrics.py", {"bbox_ioa"})
(out / "ultralytics/utils/metrics.py").write_text(
    "import numpy as np\n\n\n" + met + "\n\n\ndef batch_probiou(*a, **k):\n    raise NotImplementedError('oriented boxes not used')\n")
print("built", out)

# Detections and TrackDisplay copied unchanged from the repo demo (ast extraction), so the display layer is timed too.
text = track_py.read_text(encoding="utf-8")
tree = ast.parse(text)
classes = [ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("Detections", "TrackDisplay")]
assert len(classes) == 2
(out / "track_classes.py").write_text(
    "from collections import Counter, deque\n\nimport numpy as np\n\n\n" + "\n\n\n".join(classes) + "\n")
print("copied Detections and TrackDisplay from", track_py)
