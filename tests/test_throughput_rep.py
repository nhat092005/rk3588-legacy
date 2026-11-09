"""Throughput harness logic (threads and pipeline modes) with fake NPU contexts, no board needed.

Run: .venv/bin/python -m unittest tests.test_throughput_rep
"""
import time
import unittest

import numpy as np

from ai.board.bench.bench_throughput_rep import run_pipeline, run_threads


class FakeRknn:
    def __init__(self, delay):
        self.delay = delay

    def inference(self, inputs):
        time.sleep(self.delay)
        return [np.zeros((1, 9, 8400), dtype=np.float32)]


FRAMES = [np.full((480, 640, 3), 100 + i, dtype=np.uint8) for i in range(5)]


class Throughput(unittest.TestCase):
    def test_threads_counts_every_frame_once(self):
        ctx = [(FakeRknn(0.004), False, False) for _ in range(3)]
        wall, pre, npu, post, per_ctx = run_threads(ctx, FRAMES, 60)
        self.assertEqual(sum(per_ctx), 60)
        self.assertEqual((len(pre), len(npu), len(post)), (60, 60, 60))
        self.assertGreater(wall, 0)

    def test_pipeline_counts_every_frame_once_and_stops(self):
        ctx = [(FakeRknn(0.004), False, False) for _ in range(3)]
        wall, pre, npu, post, e2e, per_npu = run_pipeline(ctx, FRAMES, 60, 2, 2, 6)
        self.assertEqual(sum(per_npu), 60)
        self.assertEqual((len(pre), len(npu), len(post), len(e2e)), (60, 60, 60, 60))


if __name__ == "__main__":
    unittest.main()
