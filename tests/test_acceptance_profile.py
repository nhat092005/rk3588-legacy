"""profiles/npu_g4.yaml must agree with the thresholds hard-coded in the G4 analysis scripts that produced the published results.

Run: .venv/bin/python -m unittest tests.test_acceptance_profile
"""
import re
import unittest

from ai.automation.report.acceptance import load_profile
from ai.paths import REPO_ROOT

OP = REPO_ROOT / "ai" / "npu" / "operator_support"


def numbers(script: str, pattern: str) -> list[float]:
    return [float(m) for m in re.findall(pattern, (OP / script).read_text())]


class ProfileMatchesCode(unittest.TestCase):
    def setUp(self):
        self.p = load_profile("npu_g4")

    def expect(self, found: list[float], want: float, count: int, what: str):
        self.assertEqual(len(found), count, f"{what}: pattern found {len(found)} times, expected {count} (code changed, update the test)")
        self.assertTrue(all(v == want for v in found), f"{what}: code has {found}, profile has {want}")

    def test_a_coverage(self):
        for s in ("analyze_g4.py", "analyze_g4v.py"):
            self.expect(numbers(s, r"min\(cov\) >= ([0-9.]+)"), self.p["a_coverage_min"], 1, f"A in {s}")

    def test_b_drop(self):
        for s in ("analyze_g4.py", "analyze_g4v.py"):
            self.expect(numbers(s, r"drop_pct >= ([0-9.]+) and drop_ms > gap"), self.p["b_drop_pct_min"], 1, f"B in {s}")

    def test_c_drop(self):
        found = [float(m[0] or m[1]) for m in re.findall(r"all\(d >= -([0-9.]+) for d in diff|all\(x >= -([0-9.]+) for x in", (OP / "analyze_g4c.py").read_text())]
        self.expect(found, self.p["c_max_drop_pp"], 3, "C in analyze_g4c.py")

    def test_seed_noise_boundary(self):
        self.expect(numbers("analyze_g4c.py", r">= ([0-9.]+) else \"ngoài nhiễu seed\""), self.p["seed_noise_range_pp"], 1, "seed noise in analyze_g4c.py")

    def test_c_columns_match_code(self):
        m = re.search(r"COLS = \(([^)]*)\)", (OP / "analyze_g4c.py").read_text())
        self.assertEqual([c.strip(' "') for c in m.group(1).split(",") if c.strip()], self.p["c_columns"])


if __name__ == "__main__":
    unittest.main()
