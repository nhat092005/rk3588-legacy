"""log_experiment appends rows to the experiment index and writes the header once.

Run: .venv/bin/python -m unittest tests.test_experiment_index
"""
import csv
import tempfile
import unittest
from pathlib import Path

from ai.core.provenance import INDEX_FIELDS, log_experiment
from ai.paths import EXPERIMENTS_DIR


class ExperimentIndex(unittest.TestCase):
    def test_header_once_and_rows_appended(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "index.csv"
            for i in range(2):
                log_experiment(f"exp{i}", "npu", "latency", "run_a", "ai/npu/latency/x.py", "results/npu/latency/x.json", index_path=path)
            rows = list(csv.DictReader(path.open()))
            self.assertEqual([r["exp_id"] for r in rows], ["exp0", "exp1"])
            self.assertEqual(list(rows[0]), INDEX_FIELDS)
            self.assertEqual(path.read_text().count("exp_id,date"), 1)
            self.assertTrue(rows[0]["git_commit"])

    def test_backfilled_index_points_to_existing_files(self):
        path = EXPERIMENTS_DIR / "index.csv"
        if not path.exists():
            self.skipTest("experiments/index.csv not present")
        repo = EXPERIMENTS_DIR.parent
        for r in csv.DictReader(path.open()):
            for field in ("script", "rule_file", "result_file"):
                for rel in r[field].split("|"):
                    if rel and rel != "unknown":
                        self.assertTrue((repo / rel).exists(), f"{r['exp_id']} {field}: {rel}")


if __name__ == "__main__":
    unittest.main()
