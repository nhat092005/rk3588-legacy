# ai/

PPE detection model pipeline for RK3588.

- `core/`: Shared engine (dataset loaders, trainer, evaluation metrics, RKNN library in `core/rknn/`).
- `models/`: Model architecture definitions (`nn.Module`) with a status per model.
- `research/`: Draft architectures before they enter `models/`, and the rule for graduating them.
- `automation/`: Official training and evaluation pipeline (`pipeline/`, `bench/`, `report/`), configs and acceptance profiles.
- `npu/`: NPU build, probe and analysis scripts grouped by topic (`operator_support/`, `latency/`, `throughput/`).
- `board/`: Scripts that run on the Orange Pi 5 itself (RKNNLite, ONNX Runtime CPU), grouped in `app/`, `bench/`, `probe/`, `sh/`, with their own `requirements.txt`. See `board/README.md`.
- `paths.py`: Single source of truth for data and result locations (`experiments/`, `results/`).
- `program.md`: Autonomous architecture search protocol (inactive).

Data is not kept in `ai/`: runs, leaderboards and NPU measurements are in `experiments/` and analysis output is in `results/`.
The Python environment (`.venv`, `pyproject.toml`) resides at repository root.
