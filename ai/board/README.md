# board/

Code that runs on the Orange Pi 5 itself (RKNNLite, ONNX Runtime CPU). `make board-sync` copies this whole directory to the board with `rsync --delete`, so everything the board needs must stay inside it.

Run from the repository root on the board: `.venv-board/bin/python -m ai.board.<group>.<module>`. The board venv is built by `sh/setup_board.sh` from `requirements.txt` (numpy, opencv, psutil, PyYAML, onnxruntime, RKNN-Toolkit-Lite2, no torch or ultralytics).

## Layout

| Path | Content |
|---|---|
| `common.py`, `split_decode.py` | Shared pre/post-processing and split-head box decode. Numpy only. The PC also imports them (`ai/core/rknn/split_head.py`, `ai/automation/bench/bench_gpu.py`, `ai/npu/operator_support/build_split.py`), so they stay at this level and must not import torch. |
| `app/` | `camera.py` (live OV13855 pipeline), `detect_images.py`, `detect_video.py` (one RKNN detection pass over images or a video). |
| `bench/` | `bench_npu.py` (tier-2 latency), `bench_cpu_onnx.py`, `bench_decode.py`, `bench_throughput.py`, `bench_throughput_rep.py` (multi-core study), `rknn_perf_bench.c` (C API latency, built on the board). |
| `probe/` | `probe_runtime.py` (RKNNLite functional probes), `infer_dump.py` (raw NPU outputs for the evaluator), `split_gate.py` (split-head correctness gate). |
| `sh/` | `setup_board.sh` (venv), `lock_freq.sh` (fix CPU, NPU and DDR frequency), `start_rknn_server.sh` (PC to board link for tier 1). Run with `bash ai/board/sh/<name>.sh`. |

Scripts import each other as `ai.board.<group>.<module>`, for example `bench/bench_npu.py` is imported by `app/detect_images.py`.
