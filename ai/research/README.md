# research/

Research code for architectures that are not yet official models, and the rule for graduating them to `ai/models/`.

## Layout

- `ai/research/<topic>/`: one folder per research topic, with `notes.md` (hypothesis, method, dated entries) and the code that studies it.
- Experiment data goes to `experiments/research/<topic>/<exp_id>/`, never into this folder.
- `notebooks/`: exploratory analysis (empty today).
- A trained candidate is an ordinary run in `experiments/runs/<run_id>/` (`ai/core/trainer.py` through `ai.automation.pipeline.run`), so a run has exactly one place whether or not the candidate graduates.

### Topic `all_core/` (active)

Detector whose every op runs on all 3 NPU cores (A/B = 1). Details in `all_core/notes.md`.

| Folder | Content |
|---|---|
| top level | `ultra.py`: registers `AllCoreDetect` and `MBConvU` in Ultralytics. It stays here because trained checkpoints store its import path. `ai/models/allcore_*/allcore_*.py` import it |
| `candidates/` | `blocks.py` (candidate blocks), `nets.py` (whole detectors: backbone family x neck x activation) |
| `audit/` | PC-only static audits: compile with rknn-toolkit2 and classify ops (`compile_audit`, `net_audit`, `head_audit`, `trainable_audit`, `op_class`, child builders `_build`, `_build_ref`) |
| `board/` | `board_measure.py` (build, run, analyze) and `board_run.sh` (runs on the board) |
| `grid/` | `scale_grid.py` (width x depth grid), `raw_refs.py` (YOLOv8n cut at raw head tensors) |

Run as `.venv/bin/python -m ai.research.all_core.<folder>.<module>`. Keep every `__init__.py` empty: `grid/raw_refs.py` asserts that `ultra` is not imported.

### Topic `modular/` (empty frame)

Detector as three parts, so that a backbone, a neck or a head can be replaced alone. Contracts only, no architecture yet. Details in `modular/notes.md`.

| File | Content |
|---|---|
| `backbone.py`, `neck.py`, `head.py` | Contract of each part (Backbone returns [P3, P4, P5], Neck returns [N3, N4, N5], Head is Ultralytics `Detect` and must be the last layer). They import nothing from each other or from `all_core` |
| `register.py` | `register(*parts)`: makes parts usable by name in an Ultralytics yaml, wiring example in the docstring |

Tested with stub parts by `tests/test_modular_frame.py`. A new part is a subclass in the matching file plus one `register` call. Run as `.venv/bin/python -m ai.research.modular.<module>`. Keep `__init__.py` empty.

## Graduation to `ai/models/`

A draft moves to `ai/models/` with an official config in `ai/automation/configs/` when all of these hold:

1. The config names an acceptance profile, for example `acceptance: npu_g4` (`ai/automation/profiles/<name>.yaml`). A different kind of problem adds its own profile.
2. The model is trained and evaluated under protocol v1 on the same evaluation sets and seed as its baseline (`make train`, row in `experiments/leaderboards/<dataset>.csv`).
3. The profile conditions are computed with `ai/automation/report/acceptance.py`. For `npu_g4` these are A, B and C. Conditions G1 to G3 of the gate need a reading of the documentation, so passing A, B and C is necessary and not sufficient for the label "Giữ".
4. `experiments/index.csv` has a row (domain `research`) for the draft experiment and a row for the acceptance evaluation of the graduated model, whose `note` field says `origin=<research exp_id>`.

A candidate that fails stays in `ai/models/` so it can be trained again. Its status is written in `ai/models/README.md`.
