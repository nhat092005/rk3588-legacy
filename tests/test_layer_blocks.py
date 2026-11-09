"""eval_perf per-layer parser and block assignment vs a real RKNN-Toolkit2 2.3.2 output (YOLOv8n nc=5 INT8, CORE_0, RK3588).

Run: .venv/bin/python -m unittest tests.test_layer_blocks
"""
import collections
import re
import unittest
from pathlib import Path

from ai.core.rknn.benchmark import LAYERS_TIME_RE, parse_layers
from ai.core.rknn.blocks import assign

TEXT = (Path(__file__).parent / "fixtures" / "eval_perf_debug_int8_core0.txt").read_text()
ROWS = parse_layers(TEXT)


class LayerBlocksTest(unittest.TestCase):
    def test_table_shape(self):
        self.assertEqual(len(ROWS), 119)
        self.assertEqual((ROWS[0]["id"], ROWS[-1]["id"]), (1, 119))
        self.assertEqual([r["op"] for r in ROWS if r["target"] == "CPU"], ["InputOperator", "Transpose", "OutputOperator"])

    def test_layer_times_add_up_to_total(self):
        total = float(LAYERS_TIME_RE.search(TEXT).group(1))
        self.assertEqual(sum(r["time_us"] for r in ROWS), total)

    def test_total_cycles_is_max_of_ddr_and_npu(self):
        self.assertTrue(all(r["total_cycles"] == max(r["ddr_cycles"], r["npu_cycles"]) for r in ROWS))

    def test_single_core_workload(self):
        bad = [r["id"] for r in ROWS if r["target"] == "NPU" and r["op"] != "Reshape" and r["time_us"] > 0 and r["workload"] != "100.0%/0.0%/0.0%"]
        self.assertEqual(bad, [])
        # Reshape only changes the shape: three NPU rows of 1 to 3 us report 0% on every core
        self.assertEqual({r["workload"] for r in ROWS if r["op"] == "Reshape" and r["time_us"] > 0 and r["time_us"] < 4}, {"0.0%/0.0%/0.0%"})

    def test_assign_covers_every_row(self):
        blocks = collections.Counter(assign(r)[0] for r in ROWS)
        self.assertEqual(blocks["io"], 2)
        self.assertEqual(blocks["post_detect"], 3)
        self.assertEqual(blocks["L22"], 37)
        self.assertEqual(sum(v for k, v in blocks.items() if re.fullmatch(r"L\d+", k)), 114)
        self.assertEqual(sorted(int(k[1:]) for k in blocks if k.startswith("L")), list(range(23)))
        self.assertFalse([r["id"] for r in ROWS if assign(r)[1].startswith("other")])

    def test_assign_examples(self):
        by_id = {r["id"]: r for r in ROWS}
        self.assertEqual(assign(by_id[2]), ("L0", "conv"))
        self.assertEqual(assign(by_id[5]), ("L2", "split"))
        self.assertEqual(assign(by_id[6]), ("L2", "bottleneck"))
        self.assertEqual(assign(by_id[8]), ("L2", "bottleneck"))  # Add of the Bottleneck


if __name__ == "__main__":
    unittest.main()
