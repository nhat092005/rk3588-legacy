# all_core: detector whose every op runs on all 3 NPU cores

Module map (2026-11-12). The entries below were written when the files sat flat in this folder and keep their original names. `ultra.py` stays at the top level because trained checkpoints (`experiments/runs/ppe5_allcore_*/weights/*.pt`) store `ai.research.all_core.ultra.AllCoreDetect` and `MBConvU` by import path.

| Old name in the entries | Module now |
|---|---|
| `blocks.py`, `nets.py` | `candidates/blocks.py`, `candidates/nets.py` |
| `compile_audit.py`, `net_audit.py`, `head_audit.py`, `trainable_audit.py`, `op_class.py`, `_build.py`, `_build_ref.py` | `audit/<same name>` |
| `board_measure.py`, `board_run.sh` | `board/<same name>` |
| `scale_grid.py`, `raw_refs.py` | `grid/<same name>` |

Run as `.venv/bin/python -m ai.research.all_core.<folder>.<module>`, for example `...grid.scale_grid list`.

Models in `ai/models/` moved from `ai/models/<name>.py` and `.yaml` to `ai/models/<name>/<name>.py` and `.yaml` on 2026-11-12. Entries below use the old paths.

Hypothesis: a detector built only from ops that the compiler maps to multi-core NPU kernels (Conv fused with ReLU, ReLU6 or LeakyReLU, Add fused into Conv, Concat) can run every layer split over `CORE_0_1_2`. YOLO cannot because of `ConvExSwish` (SiLU), `Split`, `Reshape`, `MaxPool`, `Resize` and, in YOLOv8n, the DFL `Softmax` and `Transpose`.

## Method

`compile_audit.py` builds each block or network on the PC with rknn-toolkit2 for rk3588, reads the "Network Layer Information Table" of the compiler log and classifies every compiled op with the WorkLoad column measured on the board in this project (`op_class.py`). No board is used. INT8 uses random calibration data, so it shows op names and targets only.

Check of the method: the reference models YOLOv8n and YOLO26n (random weights, `nc=5`, Ultralytics RKNN export path, opset 19) compile to the same op counts as the board tables (117 and 180 layers without Input and Output, 119 and 182 with them). Results are in `experiments/research/all_core/a1_static_audit/reference.json`.

## Findings of a1 (blocks) and a2 (activation probes)

| Finding | Evidence |
|---|---|
| `ConvRelu`, `ConvClip`, `ConvLeakyRelu`, `ConvAdd` are measured multi-core and appear in every dense and depthwise block | `a1_static_audit/report.md`, measured rows in `results/npu/operator_support/g4v_latency_workload_2.json` |
| SiLU always becomes `ConvExSwish` (core 0 only) | block `conv_act` with `silu` |
| HardSwish written as Add, Clip, Mul is fused back into `ConvHardSwish` | block `hswish_split` |
| SiLU-like gates that avoid the HardSwish pattern leave standalone `Mul` and `Clip` (and a standalone `BatchNormalization` for slope 0.25) | `a2_activation_probe/report.md` |
| Residual Add should be last in the block. Post-Add ReLU gives `ConvAddRelu` (never measured), post-Add LeakyReLU gives a standalone `LeakyRelu` | blocks `res3`, `mbconv_k3`, `uib_*` |
| `MaxPool`, `Resize`, `ConvTranspose` are blockers. `down_conv` and `neck_none` have none | blocks `down_*`, `neck_*` |
| StarNet compiles to `ConvMul`, an op never measured here | block `star` |
| YOLOv6 N/S style head with ReLU or LeakyReLU has no blocker (raw outputs, no sigmoid, no decode) | block `head_v6` |

## Findings of a3 (whole nano detectors, 640x640, nc=5)

`nets.py` builds backbone family x neck mode x activation with a YOLOv6 N/S style head. `net_audit.py` compiles each on the PC in FP16 and INT8. Results are in `experiments/research/all_core/a3_network_audit/`.

| Finding | Evidence |
|---|---|
| With the bottom-up only neck (`none`), the families `dense`, `res`, `mbconv` and `uib` compile to 34 to 42 layers with no blocker, in ReLU, ReLU6 and LeakyReLU. YOLOv8n has 117 layers with 10 blocker op types and YOLO26n 180 layers with 9 | `a3_network_audit/report.md` |
| The `Resize` neck leaves 2 `Resize` blockers, the `ConvTranspose` neck leaves 2 `ConvTranspose` (never measured) | same |
| The `star` family leaves 4 `ConvMul` (never measured) even without upsample | same |
| Compiler cycle estimate (INT8) of the blocker-free nets is 5.7 to 6.4 M at 7.8 to 9.8 GFLOPs, against 5.7 M at 8.1 GFLOPs for YOLOv8n. Estimates do not predict latency | same |
| Two groups of ops count as `partial`, not blocker: `Concat` (15 of 17 rows split in YOLOv8n) and plain `Conv` (17 of 18 rows). The head prediction convs (64 to 5 and 64 to 4 channels) are plain `Conv` and may fall below the multi-core granularity | `op_class.py` table, UG232 p.75 |

Nets are not matched to the YOLOv8n budget (7.8 to 13 GFLOPs against 8.1). Scaling comes after the op audit.

## Findings of a4 (board, whole detectors, on-board C bench)

Method: `board_measure.py` builds the nets as INT8 and FP16 `.rknn` (random weights, random calibration), `board_run.sh` runs `ai/board/bench/rknn_perf_bench.c` on the board pinned to CPU 6 with the clock locked, 3 repetitions x 100 runs per core mask with the masks rotated, plus the per-layer WorkLoad table of 10 runs at `CORE_0` and `CORE_0_1_2`. The YOLOv8n SiLU and ReLU models of the project runs were measured in the same session. Raw files are in `experiments/research/all_core/a4_board_3core/board_out/`, tables in `report.md` and `results.json`. Repetition medians differ by 0.1 percent or less.

| Finding (INT8) | Evidence |
|---|---|
| Every layer of the 7 blocker-free nets splits over all 3 cores (34 to 50 layers, 0 unsplit). The head 1x1 prediction convs split too, so the granularity worry did not appear at 640x640 | `report.md`, per-layer WorkLoad |
| `dense` ReLU: 7.19 ms at `CORE_0`, 4.27 ms at `CORE_0_1`, 3.24 ms at `CORE_0_1_2` (1.68x and 2.22x). `mbconv` ReLU6 8.81 to 3.85 ms (2.29x), `uib` 9.34 to 4.04 ms (2.31x), `star` 11.75 to 5.43 ms (2.16x). FP16 reaches 2.4x for `dense` | same |
| ReLU and LeakyReLU nets have identical latency (7.191 ms), and `res` (Add last) equals `dense`. The activation of the ReLU family is chosen by accuracy only | `dense_none_relu`, `dense_none_leakyrelu`, `res_*` |
| `ConvMul` (never measured before) splits over 3 cores in `star` | `star_none_relu` WorkLoad |
| `ConvTranspose` runs on one core (2 layers unsplit), `x3` falls to 1.88. `Resize` runs on one core (2 layers), `x3` 2.01 | `dense_convt_relu`, `dense_resize_relu` |
| Reference in the same session: YOLOv8n ReLU 12.91 to 10.82 ms (1.19x) with 80 of 111 layers split, YOLOv8n SiLU 18.88 to 19.06 ms (0.99x) with 29 of 117 layers split | `yolov8n_relu`, `yolov8n_silu` |
| The one unsplit `Conv` of YOLOv8n ReLU is `model.22/dfl/conv/Conv` (WorkLoad 100.0/0.0/0.0), which accounts for the 2.4 percent missing from the 97.6 percent coverage | `results.json`, layers of `yolov8n_relu_int8` |
| Shares follow integer splits of the output rows: H=320 gives 33.4/33.4/33.1, 160 gives 33.8/33.1/33.1, 80 gives 33.8/33.8/32.5 (27/27/26 rows), 40 gives 35.0/32.5/32.5 (14/13/13), 20 gives 35.0/35.0/30.0 (7/7/6). The 3x3 convs with 256 output channels at 20x20 give 37.5/37.5/25.0, which fits 3/3/2 of eight blocks of 32 output channels (a reading of the arithmetic, the compiler is closed). The compiler picks the axis per layer, the 1x1 conv at 20x20 uses rows | per-layer WorkLoad and output shapes, 4 nets |
| The uneven shares cost little. Weighting each layer by the largest share gives a balance-limited speedup of 2.87 to 2.92, against a measured 1.91 to 1.94 (sum of per-layer times in debug mode). The main loss is elsewhere | `dense`, `mbconv`, `uib`, `star` |
| Per-layer speedup falls with memory traffic and with short layers. By DDR/NPU cycle ratio quartile the median layer speedup is 2.18, 1.69, 1.65 and 1.31 (Pearson -0.61, 152 layers). By single-core layer time it is 1.22 (60 to 143 us), 1.77 (151 to 289 us) and 2.20 (290 us and more). Correlation only: short layers also tend to be memory heavy, and shared DDR bandwidth versus fixed per-layer sync cost is not separated | same 4 nets, layers above 20 us |

Cautions:
- Weights are random, accuracy is untouched. The nets have 7.8 to 13 GFLOPs against 8.1 for YOLOv8n and are not budget matched.
- This method gives 1.19x for YOLOv8n ReLU at 3 cores, while the project tier-1 measurement (`eval_perf`, `01-ket-qua-g4.rst`) gives 1.51x (12.834 to 8.524 ms). The single-core values agree (12.91 and 12.83 ms). The 3-core difference is unexplained, so compare nets with the reference of the same method, not with the older table.
- Zero input and a fixed CPU pin are used. IRQ placement was not controlled as in the E2 protocol.

## Findings of a5 (even splits: input height and channels divisible by 3)

Variants of `dense` ReLU and `mbconv` ReLU6 (bottom-up neck): v0 input 640 with widths 32/64/128/256 (control), v1 input 640 with widths 96/96/192/384, v2 inputs 576 and 768 with the original widths, v3 input 576 with widths 96/96/192/384. Same setup as a4, INT8 and FP16. Tables in `experiments/research/all_core/a5_div3/report.md`.

| Finding | Evidence |
|---|---|
| The split axes hold. With 576 input and 96-multiple widths every layer has its largest share at 33.3 percent or less (34 of 34 for `dense`, 38 of 38 for `mbconv`) and the balance-limited speedup is 3.00. Rows alone (v2_576) give 27 of 34, channels alone (v1) give 28 of 34 | `report.md`, WorkLoad of `v3_576_w96` against `v0_640` (12 of 34) |
| Even shares did not raise the measured speedup. INT8 x3 of `dense`: v0 2.22, v1 2.16, v2_576 2.11, v2_768 2.24, v3 2.32. `mbconv`: v0 2.29, v1 2.01, v2_576 2.19, v2_768 2.34, v3 2.12. FP16 `dense` v3 is 2.21 against 2.42 for v0 | same |
| No variant passed 2.51 (best: FP16 `dense` at 768). INT8 x3 sits between 2.0 and 2.35 from 5.9 ms to 23.5 ms of single-core time, so longer layers do not lift it past about 2.3 | same |
| Variants differ in compute (single-core INT8 5.9 to 23.5 ms), so the comparison is confounded. Only the shares are a clean test | same |

Reading: the compiler splits rows or 32-channel blocks, and a divisible size makes shares exactly equal, but speedup is capped near 2.3 by something that is not share imbalance (DDR bandwidth or per-layer synchronisation are candidates, not separated here). Divisibility by 3 is a free tidy-up, not a speed lever. It is not worth changing the input size for.

## Findings of a6 (head: HeadV6 against Ultralytics Detect with reg_max=1)

Same dense bottom-up ReLU backbone and neck at 640x640, nc=5, three heads. PC compile in `experiments/research/all_core/a6_head_audit/`, board (a4 setup, board rebooted before the run) in `experiments/research/all_core/a6_head/`. The graph ends at the raw per-level cls and box tensors, so the view and cat inside stock `Detect.forward_head` (training only) are not in the exported graph.

| Finding | Evidence |
|---|---|
| All three heads compile with no blocker: only `ConvRelu`, `Conv` and 2 `Concat`. Layers: v6 34, stock non-legacy 43, stock legacy 37 | `a6_head_audit/report.md` |
| All layers of all three split over 3 cores on the board (34 of 34, 43 of 43, 37 of 37, minimum core share 25.0 percent). The 16-channel box convs of stock `Detect` and the 5 and 4 channel prediction convs split too | `a6_head/report.md`, WorkLoad |
| Same session, INT8: v6 7.19 to 3.24 ms (2.22x), stock non-legacy 6.40 to 2.94 ms (2.18x), stock legacy 6.59 to 3.01 ms (2.19x). FP16 x3: 2.42, 2.37, 2.38. The v6 row reproduces a4 (7.19 and 3.24 ms) | same |
| Head cost: v6 1.64M params and 9.78 GFLOPs for the whole net, stock non-legacy 0.12M and 7.22 GFLOPs, stock legacy 0.44M and 8.48 GFLOPs. Stock non-legacy is 9 percent faster at 3 cores for the whole net (2.94 against 3.24 ms) | `a6_head_audit/report.md`, `a6_head/report.md` |

Reading: the head choice does not touch A/B and the stock head is cheaper. Accuracy of the three heads is unmeasured, so the choice rests on compile, A/B and cost only. Random weights, one run per head.

## Findings of a7 (the exact trainable graphs)

`ai/models/allcore_dense/allcore_dense.yaml` and `allcore_mbconv.yaml` are built by Ultralytics 8.4.154 through `ai/research/all_core/ultra.py` (registers `MBConvU` and swaps `ultralytics.nn.tasks.Detect` for `AllCoreDetect`, because `parse_model` forces `Detect.legacy = True` and only C3k2, A2C2f and C2fCIB turn it off). `trainable_audit.py` compiles the graph that comes out of the yaml: backbone, neck and the per-level cls and box convs of the head. Compile in `a7_trainable_audit/`, board (a4 setup) in `a7_trainable/`.

| Finding | Evidence |
|---|---|
| `allcore_dense` equals the a6 stock non-legacy net: 3,379,771 params, 7.218 GFLOPs, 43 layers (35 `ConvRelu`, 6 `Conv`, 2 `Concat`), no blocker | `a7_trainable_audit/report.md`, `a6_head_audit/report.md` |
| `allcore_mbconv` (ReLU, expansion 4): 2,533,371 params, 5.231 GFLOPs, 47 layers (35 `ConvRelu`, 4 `ConvAdd`, 6 `Conv`, 2 `Concat`), no blocker | same |
| Board INT8: dense 6.40 to 2.94 ms (2.18x), mbconv 8.01 to 3.55 ms (2.26x). FP16 x3: 2.37 and 2.30. Every layer splits over 3 cores (43 of 43, 47 of 47). Dense reproduces a6 (6.40 and 2.94 ms) | `a7_trainable/report.md` |
| mbconv has 28 percent fewer GFLOPs than dense but is 21 percent slower at 3 cores (3.55 against 2.94 ms): depthwise layers are not cheaper in time here | same |
| A smoke training of one short epoch (3 percent of ppe5, batch 16) runs and validates for both yamls. A checkpoint loads in a fresh process without importing the study code first (the pickle imports `ai.research.all_core.ultra`) | session of 2026-11-11, not kept as a result file |

Limits: Ultralytics control models in the same process as `ultra` would get the non-legacy head, so each is trained in its own process (the pipeline does that). Export to RKNN through `ai/automation/export_npu.py` is not checked for these models: the exported graph there includes the Detect decode, unlike the raw-tensor graph audited here. Weights are random in the board runs. Accuracy is unmeasured.

## Findings of a8 (training from scratch, 200 epoch, seed 0)

Four runs under protocol v1 on ppe5, no pretrained weights, configs `ai/automation/configs/ppe5_{allcore_dense,allcore_mbconv,yolov8n_relu_scratch,yolov8n_scratch}_200ep.yaml`. Numbers are AP50-95 from `experiments/leaderboards/ppe5.csv` (PyTorch FP32, not INT8). Val curve from `experiments/runs/<run_id>/metrics.csv`, column `metrics/mAP50-95(B)`.

| Model | css_test | ppe2286_ext | val at epoch 100, 150, 200 |
|---|---|---|---|
| `allcore_dense` | 0.3007 | 0.1537 | 0.292, 0.302, 0.313 (best 0.324 at 165) |
| `allcore_mbconv` | 0.2660 | 0.1021 | 0.248, 0.263, 0.273 (best 0.278 at 185) |
| `yolov8n_relu_scratch` | 0.3744 | 0.1800 | 0.327, 0.374, 0.392 (best 0.394 at 196) |
| `yolov8n_scratch` (SiLU) | 0.4104 | 0.1692 | 0.363, 0.404, 0.436 (best 0.436 at 200) |
| Reference, pretrained `yolov8n_relu` 200 epoch | 0.4957 | 0.2416 | not read |

Reading:
- No run has plateaued. Val still rises between epoch 150 and 200 by about 0.011 (dense), 0.010 (mbconv), 0.018 (ReLU), 0.032 (SiLU). The controls rise faster than dense over that span, so the 200 epoch gap is not shown to shrink with more epochs. The converged gap is unknown. An earlier note that dense was "near plateau" was wrong.
- On css_test dense is 0.074 below ReLU scratch and 0.110 below SiLU scratch. mbconv is 0.035 below dense. On ext the order is ReLU 0.180, SiLU 0.169, dense 0.154, mbconv 0.102.
- One seed, so differences of 1 to 3 points are not interpretable.
- Equal-ms comparison is not done: dense runs at 2.94 ms and YOLOv8n ReLU at 10.82 ms in the a4 method.

## Decisions for the equal-ms study (2026-11-11, user approved in grill-me)

| Item | Decision |
|---|---|
| First budget point | ms of YOLOv8n ReLU measured in the same board session as the candidates (10.82 ms in a4), INT8, 3 cores, a4 method (on-board C bench, clock locked) |
| Opponents | YOLOv8n ReLU and YOLOv8n SiLU as two Pareto points, ms of both re-measured in the same board session as the final candidate |
| Order | Scale dense first (width and depth grid), measure ms on the board with random weights, then train. Neck and P3 context ablations later |
| Training | Search tier: 200 epoch, 1 seed, ranks dense candidates. Conclusion tier: train until val is flat, seeds 0, 1, 2, for the candidate and both opponents (existing 200 epoch seed 0 controls are not reused) |
| Win rule | Mean AP50-95 of the candidate above the opponent mean and the min to max seed ranges do not overlap. Report level 1 (against ReLU, 0.374 at 200 epoch) and level 2 (against SiLU, 0.410 at 200 epoch), both at a candidate of 10.82 ms or less |
| Invariants | A/B = 1 at every grid point, stock `Detect(reg_max=1)` non-legacy, protocol v1 and both eval sets, a4 ms method, no numeric speedup threshold, no HUNG branch, no commit without being told, no edits of historical rows |
| Open | Pretrained weights for dense, real GPU hours of the scaled candidate, how many epochs count as flat |

## Findings of a9 (width x depth grid, PC compile and board)

`scale_grid.py` writes yamls from the layout of `allcore_dense.yaml`: width multiplier w on every channel, depth d as the repeat count of the Conv3x3 row of each backbone stage (neck unchanged). The control w = 1.0, d = 2 reproduces `allcore_dense` exactly (3,379,771 params, 7.218 GFLOPs, 43 layers, assert in `list`). PC compile with `trainable_audit.audit(weights=...)`, INT8, 640x640, `a9_scale_grid/report.md`:

| w, d | layers | params M | GFLOPs | cycles M | blockers |
|---|---|---|---|---|---|
| 1.0, 2 (control) | 43 | 3.38 | 7.22 | 4.8 | none |
| 1.5, 2 / 3 / 4 | 43 / 47 / 51 | 7.59 / 9.36 / 11.12 | 16.1 / 20.3 / 24.6 | 11.4 / 13.9 / 16.4 | none |
| 2.0, 2 / 3 / 4 | 43 / 47 / 51 | 13.49 / 16.63 / 19.76 | 28.5 / 36.0 / 43.6 | 16.0 / 19.7 / 23.4 | none |
| 2.5, 2 / 3 | 43 / 47 | 21.07 / 25.97 | 44.3 / 56.1 | 27.4 / 33.8 | none |

Reading of the compile: no blocker op appears at any grid point. The compile table does not show the per-core split, so the board run below decides A/B.

Board (2026-11-11 14:05, INT8, a4 method, clock locked, random weights, `a9_scale_grid/report.md` and `results.json`, raw files in `board_out/`). Check of the bench: YOLOv8n ReLU project export 10.820 ms (a4: 10.82) with 111 layers, 80 split, SiLU 19.048 ms (a4: 19.06) with 117 and 29, dense (1.0, 2) 2.934 ms (a7: 2.94).

| Model | GFLOPs | ms CORE_0 | ms CORE_0_1_2 | x3 | split layers |
|---|---|---|---|---|---|
| dense 1.0, 2 | 7.22 | 6.395 | 2.934 | 2.18 | 43 of 43 |
| dense 1.5, 2 | 16.1 | 14.251 | 6.264 | 2.28 | 43 of 43 |
| dense 1.5, 3 | 20.3 | 16.945 | 7.350 | 2.31 | 47 of 47 |
| dense 1.5, 4 | 24.6 | 19.644 | 8.448 | 2.33 | 51 of 51 |
| dense 2.0, 2 | 28.5 | 20.592 | 8.338 | 2.47 | 43 of 43 |
| dense 2.0, 3 | 36.0 | 24.633 | 9.868 | 2.50 | 47 of 47 |
| dense 2.0, 4 | 43.6 | 28.675 | 11.391 | 2.52 | 51 of 51 |
| dense 2.5, 2 | 44.3 | 43.127 | 16.590 | 2.60 | 43 of 43 |
| dense 2.5, 3 | 56.1 | 50.573 | 19.315 | 2.62 | 47 of 47 |
| YOLOv8n ReLU, project export (DFL and decode) | 8.1 | 12.905 (a9b: 12.879) | 10.820 (a9b: 10.794) | 1.19 | 80 of 111 |
| YOLOv8n ReLU, raw head cut, nc=5 (a9b session) | 8.09 | 9.076 | 6.768 | 1.34 | 76 of 89 |
| YOLOv8n SiLU, project export | 8.1 | 18.884 | 19.048 | 0.99 | 29 of 117 |
| YOLOv8n SiLU, raw head cut, nc=5 (a9b session) | 8.09 | 14.856 | 14.729 | 1.01 | 25 of 95 |

Reading of the board:
- A/B = 1 at all 9 grid points (every layer splits, 0 unsplit). The raw head cut of YOLOv8n ReLU is not A/B = 1 (76 of 89 layers split), it keeps `Split`, `MaxPool`, `Resize`.
- The 3-core speedup is not capped at 2.3x. It rises with size, 2.18 at 7 GFLOPs to 2.62 at 56 GFLOPs. The earlier "cap near 2.3x" holds only at nano size. Cause unknown. It agrees with the a4 correlation that short layers speed up less, which is a correlation only.
- Latency per GFLOP is 0.41 ms at nano, 0.34 at width 1.5, 0.26 to 0.29 at width 2.0 and 0.37 at width 2.5. Width 2.5 (channels 80, 160, 320, 640) is worse per GFLOP than width 2.0 (64, 128, 256, 512). Cause unknown, not tested.
- The linear guess in the PC part was wrong: 10.82 ms is not at 26.6 GFLOPs. Candidates at or under 10.82 ms are (2.0, 3) at 9.868 ms (36.0 GFLOPs, 16.6M params) and everything smaller. (2.0, 4) at 11.391 ms is 5 percent over.
- The budget depends on the reference form. Against the project export (10.82 ms) the largest grid point inside is (2.0, 3). Against the raw head cut (6.768 ms, a9b) the largest inside is (1.5, 2) at 6.264 ms (16.1 GFLOPs), and (1.5, 3) at 7.350 ms is 9 percent over.
- The DFL and decode of the project export cost 29.5 percent at one core (12.879 to 9.076 ms) and 37.3 percent at three (10.794 to 6.768 ms), a9b session.
- Correction: the first raw cuts of a9 (`yolov8n_*_raw` rows in `a9_scale_grid/report.md`, 7.040 ms ReLU and 15.377 ms SiLU at 3 cores) were built with the stock yaml default nc=80 and are superseded by the nc=5 rows of `a9b_refs/report.md`. The a9b session also re-measured the project exports (ReLU 10.794, SiLU 19.025 ms) and dense (1.0, 2) at 2.933 ms, so drift between the two sessions is 0.2 percent or less.

The board bundle was staged in the session scratchpad (`/tmp`, lost on reboot), not in the repo, and copied to `~/a9` on the board. Recreate it with:
- `.venv/bin/python -m ai.research.all_core.scale_grid audit --precision int8 --rknn-dir DIR/models` (9 grid models, 20 to 30 minutes on the PC),
- `.venv/bin/python -m ai.research.all_core.raw_refs --rknn-dir DIR/models` (YOLOv8n SiLU and ReLU cut at the raw head tensors, fresh process without `ultra`),
- `cp experiments/runs/ppe5_yolov8n_baseline_200ep_s0/weights/rknn_int8/best-rk3588.rknn DIR/models/yolov8n_silu_int8.rknn` and the same for `ppe5_yolov8n_relu_200ep_s0` to `yolov8n_relu_int8.rknn` (project exports, DFL and decode included),
- `cp ai/research/all_core/board_run.sh ai/board/bench/rknn_perf_bench.c DIR/`.

Budget anchor: the budget is the same-session ms of YOLOv8n ReLU (invariant of the equal-ms study), not the stored 10.82 ms. The project exports contain the DFL `Conv` (unsplit) and the decode, the grid graphs end at the raw head tensors. Both reference forms are staged so the budget can be read either way. Which form is the budget is a decision of the user.

Training time of the 200 epoch runs (`metrics.csv`, last `time`): dense 3009 s, mbconv 3864 s, YOLOv8n ReLU 4138 s, YOLOv8n SiLU 4187 s (0.84 to 1.16 hour). Estimate only, not a measurement of the scaled nets: scaling the dense time linearly with GFLOPs gives an upper bound of about 1.9 h (1.5, 2), 2.9 h (1.5, 4), 3.3 h (2.0, 2), 4.2 h (2.0, 3), 5.2 h (2.5, 2) per 200 epoch run. Data loading may dominate at small size, so the true value lies between the nano time (about 1 h) and this bound.

## Findings of a9c and a9d (points near the raw-head budget, user chose the raw cut as budget)

Budget decided 2026-11-11: the YOLOv8n ReLU raw head cut (nc=5), 6.77 to 6.79 ms at 3 cores in three sessions (a9b 6.768, a9c 6.793, a9d 6.789). Same setup as a9, `a9c_fine/` and `a9d_deep/` (report.md, results.json, board_out/). PC compile of the five a9c points showed no blocker (the `compile_report_only.md` of a9c was overwritten by the a9d compile, its numbers are in the task log only: layers 47, 51, 39, 39, 59 for the points below in table order).

| Point (w, d) | GFLOPs | params M | ms CORE_0_1_2 | x3 | split layers | Inside 6.79 ms |
|---|---|---|---|---|---|---|
| 1.0, 6 | 14.8 | 6.52 | 4.475 | 2.34 | 59 of 59 | yes |
| 1.0, 10 | 22.3 | 9.65 | 6.008 | 2.43 | 75 of 75 | yes |
| 1.25, 3 | 14.2 | 6.50 | 6.647 | 2.22 | 46 of 47 | no, A/B < 1 |
| 1.25, 4 | 17.1 | 7.73 | 7.495 | 2.27 | 50 of 51 | no, A/B < 1 and over |
| 1.5, 1 | 11.8 | 5.83 | 5.175 | 2.24 | 39 of 39 | yes |
| 1.5, 2 | 16.1 | 7.59 | 6.265 to 6.268 | 2.27 | 43 of 43 | yes |
| 1.75, 1 | 16.1 | 7.93 | 5.934 | 2.41 | 39 of 39 | yes |

Reading:
- Width 1.25 breaks A/B = 1: `model.11/conv/Conv` (the neck 3x3 conv at P4, 160 output channels) runs 100/0/0 on one core in both (1.25, 3) and (1.25, 4). The multi-core rule for this layer shape is not understood. Invariant 1 removes both points.
- Depth is cheap at nano width. From (1.0, 2) 2.934 ms to (1.0, 6) 4.475 ms and (1.0, 10) 6.008 ms, about 0.385 ms per added repeat of the four backbone stages, against about 0.47 ms per GFLOP for width. Plain 3x3 stacks this deep (75 layers) may train worse. Not tested.
- Search tier (200 epoch, seed 0, from scratch, protocol v1): (1.5, 1), (1.0, 6), (1.75, 1), (1.5, 2), (1.0, 10), run in this order by `train_chain.sh` from the session scratchpad. The control (1.0, 2) is the a8 run (0.3007 css_test). Model files `ai/models/allcore_dense_w<w>_d<d>.{py,yaml}`, configs `ai/automation/configs/ppe5_allcore_dense_w<w>_d<d>_200ep.yaml`.

## Limits

- The compile table has op name, target, shapes and cycle estimate. It does not show the per-core split, the multi-core granularity rule (a small layer runs on one core, UG232 p.75) or depthwise MAC use.
- A class `multi` means every measured row of that op split over 2 or more cores in the YOLOv8n and YOLO26n tables. It does not prove a different layer size splits.
- Compiler cycle estimates do not predict latency (Conv ran 2.66 times slower than the estimate on the board).

## Open

- Neck upsample has no multi-core option yet (`Resize` single core, `ConvTranspose` unmeasured). A neck without top-down path avoids it at an unknown accuracy cost.
- Unmeasured ops that candidates need: `Mul` between two large tensors, standalone `Clip`, `ConvMul`, `ConvTranspose`, `ConvAddRelu`, `ConvHardSwish`.
- Accuracy under ReLU family activations is not solved here. It needs training under protocol v1 with `acceptance: npu_g4`.
