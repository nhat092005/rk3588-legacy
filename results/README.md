# results/

Output of analysis scripts. Raw measurements are in `experiments/`.

| Path | Content | Written by |
|---|---|---|
| `ppe5/` | Tables A to L, `tables.csv`, `SOURCES.md` | `make tables DATASET=ppe5` |
| `train_set_comparison.md` | Training sets on common test sets | `make compare-train-sets` |
| `npu/operator_support/` | Gate G4, activation variants, quantization recipes, split head, QAT, no-DFL, micro models | `ai/npu/operator_support/` |
| `npu/latency/` | Per-block latency, layer analysis, on-board and interrupt experiments | `ai/npu/latency/` |
| `npu/throughput/` | Throughput and FPS levers under protocol P3 | `ai/npu/throughput/` |
| `notes/` | Design notes and source summaries of phase 2 (not measurements) | by hand |

File names keep the experiment code, because the documentation refers to them. Code to topic:

| Code in the name | Topic | Script |
|---|---|---|
| `eval_g4`, `g4_*`, `g4v_*` | operator_support | `analyze_g4c`, `analyze_g4`, `analyze_g4v` |
| `int8h5`, `q2c_diag`, `q2c_recipes` | operator_support | `analyze_int8h5`, `analyze_q2c_diag`, `analyze_q2c_recipes` |
| `q2e_board`, `q2e_pc`, `q2e_micro`, `q2e_core_all` | operator_support | `analyze_q2e_board`, `analyze_q2e_pc`, `micro_ops`, `probe_core_all` |
| `q2h_*`, `q2i_*`, `q2j_*`, `q2k_*`, `q2l_*` | operator_support | `analyze_q2h`, `analyze_q2i`, `analyze_q2j`, `analyze_q2kl` |
| `eval_x7_*` | operator_support | `analyze_x7` |
| `latency_per_block_*`, `layer_analysis_*`, `onboard_<run_id>_*` | latency | `aggregate_layers`, `analyze_layers`, `analyze_onboard` |
| `onboard_x1_*` to `onboard_x9_*`, `analysis_x10_*` | latency | `analyze_trace_x1` (x1, x2), `analyze_x3` to `analyze_x5`, `analyze_x8`, `analyze_x9`, `analyze_x10` |
| `thr_p3`, `q2f_fps` | throughput | `analyze_thr`, `analyze_q2f` |
