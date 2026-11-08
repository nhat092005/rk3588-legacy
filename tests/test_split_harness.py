"""postprocess_raw of ai/board/common.py: the split-head path decodes two raw outputs in numpy and then post-processes like the standard path.

Run: .venv/bin/python -m unittest tests.test_split_harness
"""
import unittest

import numpy as np

from ai.board.common import postprocess, postprocess_raw
from ai.board.split_decode import decode_split


class PostprocessRaw(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.boxes = rng.normal(0, 3, (1, 64, 8400)).astype(np.float32)
        self.scores = rng.normal(-2, 3, (1, 5, 8400)).astype(np.float32)

    def _meta(self):
        from ai.board.common import preprocess

        _, meta = preprocess(np.zeros((480, 640, 3), np.uint8))
        return meta

    def test_split_path_equals_decode_then_postprocess(self):
        meta = self._meta()
        got = postprocess_raw([self.boxes, self.scores], meta, False, True)
        ref = postprocess(decode_split(self.boxes, self.scores), meta, False)
        self.assertEqual(got.shape, ref.shape)
        self.assertGreater(len(got), 0)
        self.assertTrue(np.array_equal(got, ref))

    def test_standard_path_uses_first_output(self):
        meta = self._meta()
        pred = decode_split(self.boxes, self.scores)
        got = postprocess_raw([pred], meta, False, False)
        self.assertTrue(np.array_equal(got, postprocess(pred, meta, False)))


if __name__ == "__main__":
    unittest.main()
