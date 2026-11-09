"""acceptance.py must reproduce the A, B and C verdicts already stored in the published G4 results (no board needed).

A verdict whose B condition reads "chưa kết luận" (the pinning check failed) is skipped: that gate is about realization, not thresholds.

Run: .venv/bin/python -m unittest tests.test_acceptance_replay
"""
import json
import unittest

from ai.automation.acceptance import check_a, check_b, check_c, evaluate, load_profile
from ai.paths import OPERATOR_SUPPORT_RESULTS_DIR


def stored(name: str) -> dict:
    path = OPERATOR_SUPPORT_RESULTS_DIR / name
    if not path.exists():
        raise unittest.SkipTest(f"{path.name} not present")
    return json.loads(path.read_text())


class ReplayStoredVerdicts(unittest.TestCase):
    def setUp(self):
        self.p = load_profile("npu_g4")

    def test_c_every_stored_comparison(self):
        e = stored("eval_g4.json")
        n = 0
        for key, v in e["gate_c"].items():
            self.assertEqual(check_c(v["diff_pp"], self.p), v["passes"], key)
            n += 1
        for name, tags in e["variants"].items():
            for key, v in tags.items():
                self.assertEqual(check_c(v["diff_vs_silu_pp"], self.p), v["passes"], f"{name} {key}")
                n += 1
        for s in e["relu_seed_gate_c"]["per_seed"]:
            self.assertEqual(check_c(s["diff_pp"], self.p), s["passes"], f"relu seed {s['seed']}")
            n += 1
        self.assertGreaterEqual(n, 12)

    def test_a_and_b_every_stored_verdict(self):
        n = 0
        for name in ("g4_latency_workload.json", "g4v_latency_workload.json", "g4v_latency_workload_2.json"):
            g = stored(name)
            for model, v in g["gate_a"].items():
                self.assertEqual(check_a(v["coverage_per_session"], self.p), v["verdict"] == "đạt", f"{name} A {model}")
                n += 1
            for key, v in g["gate_b"].items():
                if v["verdict"].startswith("chưa kết luận"):
                    continue
                self.assertEqual(check_b(v["drop_pct"], v["drop_ms"], v["gap_ms"], self.p), v["verdict"] == "nhanh hơn", f"{name} B {key}")
                n += 1
        self.assertGreaterEqual(n, 15)

    def test_relu_seed0_is_rejected_by_c_only(self):
        """Published label of ReLU (seed 0): Loại, because C fails. A and B pass (docs 01-ket-qua-g4: A and B reach, C fails)."""
        e, g = stored("eval_g4.json"), stored("g4_latency_workload.json")
        relu_b = g["gate_b"]["relu|CORE_0_1_2"]
        r = evaluate(self.p, g["gate_a"]["relu"]["coverage_per_session"], relu_b["drop_pct"], relu_b["drop_ms"], relu_b["gap_ms"],
                     e["gate_c"]["css_test|int8"]["diff_pp"])
        self.assertEqual((r["A"], r["B"], r["C"], r["passes"]), (True, True, False, False))

    def test_all_three_relu_seeds_fail_c(self):
        e = stored("eval_g4.json")
        self.assertEqual(e["relu_seed_gate_c"]["passing"], 0)
        self.assertFalse(any(check_c(s["diff_pp"], self.p) for s in e["relu_seed_gate_c"]["per_seed"]))


if __name__ == "__main__":
    unittest.main()
