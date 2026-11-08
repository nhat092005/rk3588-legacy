# RK3588 PPE Detection Pipeline

End-to-end YOLO training/deployment research pipeline for PPE (Personal
Protective Equipment) detection, targeting the Rockchip RK3588 NPU (6 TOPS).
Covers dataset preparation, YOLOv8/YOLOv11 training, INT8/FP16 quantization
via RKNN-Toolkit2, and real multi-core latency benchmarking on the board.

Current scope: 5 classes (`Helmet`, `NoHelmet`, `NoVest`, `Vest`, `Person`),
primary dataset `css` (Roboflow Construction Site Safety v28), secondary
`ppe2286` (Huang and Cheng 2025, 4 classes without `Person`). Both are in
`data/<name>/raw/` and documented in `data/<name>/DATA_CARD.md`; the 5-class
`processed/` split is not built yet.

## Quickstart

Prerequisites:
- Linux x86_64, CUDA GPU for training
- [uv](https://docs.astral.sh/uv/) for virtualenv management
- SSH access to the board (`BOARD` variable, see `Makefile`) for the
  on-board benchmark step

```bash
# 1. Set up environment and verify bindings (Torch, CUDA, RKNN)
make sync
make check

# 2. Train baseline (seed 0) + crosscheck evaluator
make phase2 DATASET=ppe5

# 3. Export FP16/INT8, measure complexity, evaluate on ONNX/RKNN
make phase3 DATASET=ppe5

# 4. Tier 1 (PC) + tier 2 (board) benchmark — needs protocol LOCK and a
#    board that has already been set up
make phase4 DATASET=ppe5

# 5. Aggregate result tables
make tables DATASET=ppe5
```

## Full command reference

```bash
make help
```

lists every target (evaluate, golden-test, board-setup/sync, bench-board*,
demo-video/images, archive/restore, ...) with a description — this is the
canonical reference; the README does not duplicate it to avoid drifting out
of sync when the Makefile changes.

## Key locations

- Training configs: `ai/automation/configs/*.yaml`
- Per-run results & weights: `ai/automation/runs/<run_id>/`
- Documentation site sources (git submodule `docs/`, repo `rk3588-docs`): start at `docs/guide.rst`
- Runbook and Phase 0-10: `docs/quy-trinh/`; evaluation protocol and result tables A-M: `docs/thi-nghiem/`
- Papers (one page per paper, PDFs in `docs/_assets/pdf/papers/`): `docs/bai-bao/`
