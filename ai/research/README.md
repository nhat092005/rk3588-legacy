# research/

Draft model architectures before they enter `ai/models/`. No draft exists yet.

## Layout

- `ai/research/<topic>/<name>/`: `notes.md` (hypothesis, rationale, preliminary metrics) and `draft_model.py` (draft architecture).
- `notebooks/`: exploratory analysis and ablations.
- Experiment data goes to `experiments/research/<topic>/<exp_id>/`, never into this folder.
- A draft is trained with `ai/core/trainer.py` through `ai.automation.pipeline.run`. Its run is an ordinary run in `experiments/runs/<run_id>/`, so a run has exactly one place whether or not the draft graduates.

## Graduation to `ai/models/`

A draft moves to `ai/models/` with an official config in `ai/automation/configs/` when all of these hold:

1. The config names an acceptance profile, for example `acceptance: npu_g4` (`ai/automation/profiles/<name>.yaml`). A different kind of problem adds its own profile.
2. The model is trained and evaluated under protocol v1 on the same evaluation sets and seed as its baseline (`make train`, row in `experiments/leaderboards/<dataset>.csv`).
3. The profile conditions are computed with `ai/automation/report/acceptance.py`. For `npu_g4` these are A, B and C. Conditions G1 to G3 of the gate need a reading of the documentation, so passing A, B and C is necessary and not sufficient for the label "Giữ".
4. `experiments/index.csv` has a row (domain `research`) for the draft experiment and a row for the acceptance evaluation of the graduated model, whose `note` field says `origin=<research exp_id>`.

A candidate that fails stays in `ai/models/` so it can be trained again. Its status is written in `ai/models/README.md`.
