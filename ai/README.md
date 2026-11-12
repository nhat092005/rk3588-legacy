# ai/

PPE detection model pipeline for RK3588.

- `core/`: Shared engine (dataset loaders, trainer, evaluation metrics, RKNN library in `core/rknn/`).
- `models/`: One folder per model (`<name>/<name>.py` and `<name>.yaml`) with a status per model. See `models/README.md`.
- `research/`: Draft architectures before they enter `models/`, and the rule for graduating them.
- `automation/`: Official training and evaluation pipeline (`pipeline/`, `bench/`, `report/`), configs and acceptance profiles.
- `npu/`: NPU build, probe and analysis scripts grouped by topic (`operator_support/`, `latency/`, `throughput/`).
- `board/`: Scripts that run on the Orange Pi 5 itself (RKNNLite, ONNX Runtime CPU), grouped in `app/`, `bench/`, `probe/`, `sh/`, with their own `requirements.txt`. See `board/README.md`.
- `paths.py`: Single source of truth for data and result locations (`experiments/`, `results/`).
- `program.md`: Autonomous architecture search protocol (inactive).

Data is not kept in `ai/`: runs, leaderboards and NPU measurements are in `experiments/` and analysis output is in `results/`.
The Python environment (`.venv`, `pyproject.toml`) resides at repository root.

## Two `bench` folders

| Folder | Runs on | Content |
|---|---|---|
| `automation/bench/` | PC, board linked through adb (tier 1) | `benchmark_npu.py` (RKNN-Toolkit2 `eval_perf`), `bench_gpu.py`, `board_check.py` |
| `board/bench/` | The board itself (tier 2) | `bench_npu.py` (RKNNLite), `bench_throughput*.py`, `bench_cpu_onnx.py`, `bench_decode.py`, `rknn_perf_bench.c` |

`board/` is copied whole to the board by `make board-sync`, so only code that runs there lives in it.

## `npu/` experiment codes

File names in `npu/` keep the experiment codes (`g4`, `q2*`, `x*`) because the documentation cites them. `results/README.md` maps each code to its topic. `npu/` analyses measurements already taken. It holds no model or training code.
