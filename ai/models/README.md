# models/

Model architecture definitions without training logic. One folder per model, named after the model, holding `<name>.py` (`WEIGHTS`, optional `PRETRAINED`) and, when the model has its own architecture, `<name>.yaml`. Models that use a stock Ultralytics checkpoint or yaml (`yolov8n_baseline`, `yolov8s_baseline`, `yolo26n_baseline`, `yolov8n_scratch`) have only the `.py`.
Every model here can be trained with an official config in `ai/automation/configs/` whose `model:` field is the folder name. `ai/automation/pipeline/run.py` imports `ai.models.<name>.<name>` and passes its `WEIGHTS` to the trainer. A new model needs the folder, an empty `__init__.py`, the `.py`, and the yaml if any.
Being in this folder does not mean a model passed acceptance: the status column says what the documentation concluded.

| Folder | Description | Status |
|---|---|---|
| `yolov8n_baseline/` | Baseline YOLOv8 nano (SiLU) | Reference of the gate G4 |
| `yolov8s_baseline/` | Baseline YOLOv8 small | Reference, no gate result recorded here |
| `yolo26n_baseline/` | Baseline YOLO26 nano | No gate result recorded here |
| `yolov8n_relu/` | YOLOv8 nano with ReLU, initialized from `yolov8n.pt` | **Loại** (condition C fails, 100 epoch seed 0 and all 3 seeds) |
| `yolov8n_leakyrelu/` | YOLOv8 nano with LeakyReLU 0.1 | **Loại** (condition C fails) |
| `yolov8n_relu6/` | YOLOv8 nano with ReLU6 | **Loại** (condition C fails) |
| `yolov8n_relu/` with config `ppe5_yolov8n_relu_200ep.yaml` | ReLU trained 200 epoch | **Chưa biết** (A, B, C pass, G1 unresolved) |
| `yolov8n_nodfl/` | YOLOv8 nano without DFL (reg_max 1), SiLU kept, candidate A2 of phase 2d | No gate label, see the QAT and architecture result page |
| `allcore_dense/`, `allcore_mbconv/` | All-core detectors (ai/research/all_core): Conv or MBConv backbone, bottom-up neck, stock Detect with reg_max 1. Trained from scratch (200 epoch configs) | **Chưa biết** (A/B = 1 on the board. Trained once, seed 0: css_test AP50-95 0.3007 dense and 0.2660 mbconv. G4 not run, equal-ms comparison not done, see `ai/research/all_core/notes.md` a8) |
| `yolov8n_scratch/`, `yolov8n_relu_scratch/` (the ReLU one reads `../yolov8n_relu/yolov8n_relu.yaml`) | YOLOv8 nano SiLU and ReLU from scratch, no pretrained weights. Controls for the all-core detectors | **Chưa biết** (trained once, seed 0, 200 epoch: css_test AP50-95 0.3744 ReLU and 0.4104 SiLU, still rising at epoch 200) |
| `allcore_dense_w150_d1/`, `allcore_dense_w175_d1/`, `allcore_dense_w150_d2/`, `allcore_dense_w100_d6/`, `allcore_dense_w100_d10/` | Grid points of the dense all-core detector (width x depth, `ai/research/all_core/grid/scale_grid.py`). Folder name `w150_d2` is width x1.5, depth 2. Same layout as `allcore_dense/`, scratch training, 200 epoch seed 0 | **Chưa biết** (search tier only, one seed. css_test / ppe2286_ext AP50-95: w150_d1 0.3168 / 0.1246, w175_d1 0.3448 / 0.1307, w100_d6 0.3633 / 0.2016, w100_d10 0.3424 / 0.2038, w150_d2 0.3828 / 0.1552. Source: `experiments/leaderboards/ppe5.csv`. Grid and board latency in `ai/research/all_core/notes.md` a9, the note for these AP results is not written yet) |

Status as read from `docs/contents/03-phuong-phap/04-tang-npu-ket-qua/01-ket-qua-g4.rst` (ReLU), `02-ket-qua-bien-the-activation.rst` (LeakyReLU, ReLU6, ReLU 200 epoch)
and `07-ket-qua-qat-kien-truc.rst` (no-DFL), on 2026-11-09. Acceptance numbers live in `ai/automation/profiles/npu_g4.yaml`.

Weights download automatically via Ultralytics into `~/.cache/ultralytics/weights/`.
