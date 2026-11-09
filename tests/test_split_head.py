"""decode_split against Detect._inference (random head outputs of a real YOLOv8n ppe5 model), and the split model against the standard output.

Run: .venv/bin/python -m unittest tests.test_split_head
"""
import unittest
from pathlib import Path

import numpy as np

from ai.core.rknn.split_head import decode_split, split_model
from ai.paths import RUNS_DIR

WEIGHTS = RUNS_DIR / "ppe5_yolov8n_baseline_100ep_s0" / "weights" / "best.pt"


@unittest.skipUnless(WEIGHTS.exists(), "trained weights not present")
class SplitHead(unittest.TestCase):
    def test_decode_matches_detect_inference(self):
        import torch

        model = split_model(WEIGHTS)
        head = model.model[-1]
        rng = np.random.default_rng(0)
        boxes = rng.normal(0, 4, (1, 64, 8400)).astype(np.float32)
        scores = rng.normal(-3, 3, (1, 5, 8400)).astype(np.float32)
        feats = [torch.zeros(1, 1, 80, 80), torch.zeros(1, 1, 40, 40), torch.zeros(1, 1, 20, 20)]
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        head.shape = None
        ref = head._inference({"boxes": torch.from_numpy(boxes), "scores": torch.from_numpy(scores), "feats": feats}).numpy()
        got = decode_split(boxes, scores)
        self.assertEqual(ref.shape, got.shape)
        self.assertLess(float(np.abs(ref - got).max()), 1e-3)

    def test_split_model_and_decode_match_standard_model(self):
        import torch
        from ultralytics import YOLO

        torch.manual_seed(0)
        x = torch.rand(1, 3, 640, 640)
        std = YOLO(str(WEIGHTS)).model.float().eval()
        with torch.no_grad():
            ref = std(x)[0].numpy()
            boxes, scores = split_model(WEIGHTS)(x)
        got = decode_split(boxes.numpy(), scores.numpy())
        self.assertLess(float(np.abs(ref - got).max()), 1e-2)


class SplitHeadNoDfl(unittest.TestCase):
    """reg_max 1 (candidate A2): the head has no DFL, decode_split must use the box channels as ltrb distances. Untrained model, no weights needed."""

    def test_decode_matches_detect_inference_reg_max_1(self):
        import torch
        import yaml
        from ultralytics.nn.tasks import DetectionModel

        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "ai" / "models" / "yolov8n_nodfl.yaml").read_text())
        cfg["nc"] = 5
        head = DetectionModel(cfg, verbose=False).eval().model[-1]
        self.assertEqual(head.reg_max, 1)
        rng = np.random.default_rng(0)
        boxes = np.abs(rng.normal(0, 3, (1, 4, 8400))).astype(np.float32)
        scores = rng.normal(-3, 3, (1, 5, 8400)).astype(np.float32)
        feats = [torch.zeros(1, 1, 80, 80), torch.zeros(1, 1, 40, 40), torch.zeros(1, 1, 20, 20)]
        head.shape = None
        ref = head._inference({"boxes": torch.from_numpy(boxes), "scores": torch.from_numpy(scores), "feats": feats}).numpy()
        got = decode_split(boxes, scores)
        self.assertEqual(ref.shape, got.shape)
        self.assertLess(float(np.abs(ref - got).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
