"""Calibration sampler must return the same list in every process (it once depended on PYTHONHASHSEED).

Run: .venv/bin/python -m unittest tests.test_calibration
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CODE = """
import json, sys
from pathlib import Path
from ai.core.rknn.calibration import sample_calibration_images
print(json.dumps(sample_calibration_images(Path(sys.argv[1]), n=30, min_per_class=4, seed=0)))
"""


class CalibrationDeterminism(unittest.TestCase):
    def test_same_list_for_different_hash_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(80):
                (Path(tmp) / f"img{i:03d}.txt").write_text(f"{i % 5} 0.5 0.5 0.1 0.1\n{(i * 3) % 5} 0.2 0.2 0.1 0.1\n")
            lists = []
            for hash_seed in ("1", "2", "3"):
                out = subprocess.check_output([sys.executable, "-c", CODE, tmp], env={**os.environ, "PYTHONHASHSEED": hash_seed}, text=True)
                lists.append(json.loads(out))
        self.assertEqual(lists[0], lists[1])
        self.assertEqual(lists[0], lists[2])
        self.assertEqual(len(lists[0]), 30)


if __name__ == "__main__":
    unittest.main()
