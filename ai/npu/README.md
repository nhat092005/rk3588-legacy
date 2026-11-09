# npu/

NPU-specific work for the RK3588: build RKNN variants, probe the hardware, and analyse what the board measured.
Scripts read measurements from `experiments/npu/<topic>/` and write `results/npu/<topic>/`.

## Topics

| Topic | Question | Scripts |
|---|---|---|
| `operator_support/` | Which operators and model variants the NPU runs well, and at what accuracy and speed cost: gate G4, activation variants, quantization recipes, split head, QAT, no-DFL | Analysis: `analyze_g4`, `analyze_g4c`, `analyze_g4v`, `analyze_int8h5`, `analyze_q2c_diag`, `analyze_q2c_recipes`, `analyze_q2e_board`, `analyze_q2e_pc`, `analyze_q2h`, `analyze_q2i`, `analyze_q2j`, `analyze_q2kl`, `analyze_x7`. Build and probe tools: `build_recipe`, `build_split`, `build_batch_probe`, `qat_split`, `diagnose_quant`, `micro_ops`, `probe_core_all` |
| `latency/` | Per-block latency, on-board C API measurement, interrupt and CPU placement | `aggregate_layers`, `analyze_layers`, `analyze_onboard`, `analyze_trace_x1`, `analyze_x3`, `analyze_x4`, `analyze_x5`, `analyze_x8`, `analyze_x9`, `analyze_x10` |
| `throughput/` | Throughput and FPS levers under protocol P3 | `analyze_thr`, `analyze_q2f` |

Run a script as `.venv/bin/python -m ai.npu.<topic>.<script> ...`. The docstring of each script gives its exact command, inputs and outputs.
The experiment codes in the names (`g4`, `q2*`, `x*`) are kept so the documentation can refer to them. `results/README.md` maps them to topics.

## Where related code lives

| Path | Content |
|---|---|
| `ai/core/rknn/` | Shared RKNN library (blocks, calibration, hybrid recipes, split head, QAT blocks), imported by the pipeline and by these scripts |
| `ai/board/` | Code that runs on the board (benchmarks, camera pipeline, runtime probes) |
| `ai/automation/` | Pipeline infrastructure (train, evaluate, export, tier-1 benchmark, tables) and acceptance profiles |
| `experiments/npu/<topic>/<exp_id>/` | Raw measurements, rule files written before the data, drivers, board logs |
| `results/npu/<topic>/` | Analysis output, one `.json` and one `.md` per analysis |

## Adding an experiment

1. Put the script in the topic folder it belongs to. Create a new topic folder only when none of the three fits.
2. Read data with `ai.paths.experiment_dir(run, "<exp_id>")` and write with the topic constant from `ai.paths` (for example `LATENCY_RESULTS_DIR`).
3. Register the experiment name in `_EXPERIMENTS` of `ai/paths.py`.
4. Log it with `ai.core.provenance.log_experiment(...)` so it appears in `experiments/index.csv`.
