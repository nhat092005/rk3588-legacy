# automation/

Official execution pipeline: training, evaluation, RKNN export, tier-1 NPU benchmark, tables.
Run everything through `make` (`make help`); the order of steps is the Runbook in the Guide,
`docs/quy-trinh/runbook.rst`.

## Main commands

```bash
make phase2 DATASET=ppe5   # train YOLOv8n/s x seed 0 + crosscheck + tables
make phase3 DATASET=ppe5   # export FP16/INT8, complexity, evaluate ONNX/RKNN + tables
make phase4 DATASET=ppe5   # tier 1 + tier 2 + throughput + GPU reference + tables
make tables DATASET=ppe5   # results/<dataset>/table_*.md, tables.csv, SOURCES.md
```

Phases skip steps whose result file exists and stop at the first error; re-run the same command to continue.

## Structure

- `run.py`, `pipeline.py`, `evaluate.py`, `check_evaluator.py`, `export_npu.py`, `complexity.py`, `benchmark_npu.py`, `bench_gpu.py`, `board_check.py`, `tables.py`, `compare_train_sets.py`: the pipeline steps behind the `make` targets.
- `configs/`: experiment configurations (model, dataset, hyperparameters, seed, optional `acceptance: <profile>`).
- `profiles/<name>.yaml`: acceptance profiles. `npu_g4.yaml` holds the thresholds of conditions A, B, C of the NPU operator-support gate.
- `acceptance.py`: evaluates a candidate against a profile. `tests/test_acceptance_profile.py` checks that the profile agrees with the thresholds in `ai/npu/operator_support/analyze_g4*.py`, and `tests/test_acceptance_replay.py` checks that it reproduces the stored published verdicts.

## Where the data is

Code and data are separate. Paths come from `ai/paths.py`.

- `experiments/runs/<run_id>/`: `config.yaml`, `metrics.csv`, `logs/`, `eval/`, `bench/`, `export_<precision>.json`, `complexity.json`, `weights/` (logs and weights are git-ignored).
- `experiments/leaderboards/<dataset>.csv`: protocol v1 FP32 result of every run, written by `make train`.
- `experiments/golden/`: logs of golden and crosscheck runs.
- `results/<dataset>/`: tables A to L written by `make tables`.
- NPU experiments, their analysis scripts and results are in `ai/npu/`, `experiments/npu/` and `results/npu/`.
