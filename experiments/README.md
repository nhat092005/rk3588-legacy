# experiments/

All experiment data, kept apart from the code in `ai/`. Paths come from `ai/paths.py`, never typed by hand.

| Path | Content | Written by |
|---|---|---|
| `runs/<run_id>/` | One trained model: `config.yaml`, `metrics.csv`, `eval/`, `export_<precision>.json`, `complexity.json`, pipeline benchmark files in `bench/` (`tier1_*`, `tier2_*`, `gpu_pt_fp32.json`, `camera_*`), `weights/` and `logs/` (git-ignored) | `ai/automation/` |
| `leaderboards/<dataset>.csv` | Protocol v1 FP32 result of every run | `make train` |
| `golden/` | Logs of golden and crosscheck runs of the evaluator | `make golden-test`, `make crosscheck` |
| `npu/<topic>/<exp_id>/` | Board and analysis experiments: rule files written before the data, drivers, raw measurements, board logs. Topics: `operator_support`, `latency`, `throughput` | `ai/npu/`, board drivers |
| `camera/<exp_id>/` | Camera pipeline experiments on the board (frame rate runs, GC and pinning A/B, encoder and RGA probes, preprocessing evaluations). Path constant `CAMERA_DIR` in `ai/paths.py`. Not registered in `_EXPERIMENTS`, no script reads them through `experiment_dir` | board drivers, `ai/board/app/camera.py` |
| `research/<topic>/<exp_id>/` | Architecture experiments (none yet) | `ai/research/` scripts |
| `index.csv` | One row per experiment, see below | `log_experiment()` |

## Rules

- A run has one place, `runs/<run_id>/`. Experiments refer to it by `run_id` and never copy it, so a model that graduates from `ai/research/` to `ai/models/` keeps its run where it is.
- An experiment run on several models has one `<run_id>/` subfolder per model (`onboard_g4b`, `onboard_g4v`, `onboard_g4v2`, `onboard_q2h`, `onboard_q2i`). Any other experiment belongs to the baseline run `ppe5_yolov8n_baseline_100ep_s0` only, and `ai.paths.experiment_dir` raises for another run.
- Analysis output is in `results/`, not here.
- Docs and docstrings point to `<exp_id>/<run_id>/` when they mean the baseline copy of an experiment run on several models, and to `<exp_id>/` when they mean the experiment as a whole.

## meta.json and index.csv

The 30 backfilled experiment folders have a `meta.json` with the same facts as their index row plus `rule_first_line`, the first line of its rule file copied verbatim (`unknown` when there is no rule file). It is not a hypothesis written afterwards.


Columns: `exp_id, date, domain, topic, run_id, script, rule_file, result_file, git_commit, status, note`. Several values in one field are joined with `|`.
It is an append-only log. `ai.core.provenance.log_experiment(...)` adds a row with the current date and git commit.

The 30 rows present on 2026-11-09 were backfilled from existing files:
- `date` is the earliest timestamp in the rule file of the experiment, `unknown` when it has none (`onboard`, `onboard_e2`). A date is when the rule was written, which can be earlier than the measurement.
- `script` lists every script under `ai/npu/` that names the experiment. `result_file` lists the results written by those scripts, so an experiment read by several scripts lists all of their outputs.
- `git_commit` is `unknown`: it was not recorded then. `status` is `recorded`, which only means the files exist. Gate labels such as Giữ or Loại are in the documentation.

## Historical files

On 2026-11-09 the locations changed (`ai/automation/runs` to `experiments/runs`, `runs/<id>/bench/<exp>` to `experiments/npu/<topic>/<exp>`, `results/<file>` to `results/npu/<topic>/<file>`).
Path strings inside tracked text files (rule files, drivers, session records, logs of results) were updated to match. No number was changed.

On 2026-11-10 `ai/board/` and `ai/automation/` were grouped into subpackages (`ai/board/{app,bench,probe,sh}/`, `ai/automation/{pipeline,bench,report}/`). Module and file paths in text files were rewritten the same way (`ai.board.bench_npu` to `ai.board.bench.bench_npu`, `ai.automation.run` to `ai.automation.pipeline.run`), no number was changed. A `git_commit` or a command in a log recorded before that date still shows the old flat name.

## Moved from tmp/ on 2026-11-10

Nine experiments that the documentation cites as board or PC measurements were kept in the git-ignored `tmp/` and now live here with a `meta.json` and an `index.csv` row (`date` is the file modification time, `git_commit` is `unknown`):
`camera/{ab_runs,camera_runs,cam_fps_2026-11-08,enc_rga_2026-11-08,distort,stretch}`, `npu/throughput/input_384x640`, `npu/latency/submit_trace`, `npu/operator_support/c5_build`.
Scripts without a result file were moved next to them: `camera/pipeline_2026-11-07` (measurement scripts and README), `camera/rga_caps` (RGA capability probe), `camera/isp_record` (ISP recording scripts and field-of-view images).
Files that can be regenerated were deleted, the command is in `npu/throughput/input_384x640/README.md`. Path strings in moved scripts, YAML and JSON were updated (`tmp/...` and `ai/automation/runs` to the new locations). `dataset.txt` and `export_*.log` of `input_384x640` keep their old paths verbatim.

## Git history rewritten on 2026-11-10

The history of `main` and `dev/model-architecture` was rewritten with `git filter-repo` to drop blobs that are heavy and regenerable. Every commit hash changed, no commit was squashed or dropped (82 commits before and after).
Removed from all history: `docs/papers/` (the PDFs live in the `docs` submodule under `_assets/pdf/papers`), `runs/*/plots/sample_predictions/`, and seven `*.onnx` intermediates of `onboard_q2c` that no script reads. Kept on purpose: raw board traces (`*.gz`), measured `.rknn` builds and `onboard_q2c/diag/ppe5_yolov8n_baseline_100ep_s0/int8_input.onnx`, which `analyze_q2e_pc.py` reads.
The tree of the branch tip is byte-identical before and after (tree hash `6797779638372389f07550c7cf5f4e6b9afd6642` at the commit that records the nine experiments), so only history changed.
`commit_map.tsv` maps old hashes to new ones. The `git_commit` values stored inside result JSON files were not edited and are old hashes, so look them up in this map (for example `d84836b` is `44a0d80`). The 40 files that record `33212040b359cc514ecbc1aef3aae5ef4017ea3e` point to a commit that was already unreachable before this rewrite and cannot be resolved.
