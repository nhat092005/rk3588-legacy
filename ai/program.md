# Autonomous Architecture Research Protocol

**Status: INACTIVE.** Requires baseline results in `experiments/leaderboards/` before activation.

## Execution Loop

1. Read current best `map50` (eval set `css_test`) from `experiments/leaderboards/<dataset>.csv`.
2. Create or modify a draft architecture in `research/<topic>/<name>/draft_model.py` and document hypothesis in `notes.md` next to it. Experiment data goes to `experiments/research/<topic>/<exp_id>/`.
3. Run evaluation via `python -m ai.automation.run` with a fixed epoch/time budget.
4. Compare `map50` (eval set `css_test`) against the baseline and log decision in `notes.md`.
5. Repeat from step 2.

## Activation Prerequisites

- `core/trainer.py` verified functional via `automation/run.py`.
- At least one baseline run logged in `experiments/leaderboards/<dataset>.csv`.
- Defined epoch/time budget per iteration.
