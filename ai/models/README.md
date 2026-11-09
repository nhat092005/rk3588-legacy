# models/

Model architecture definitions without training logic. Every file here can be trained with an official config in `ai/automation/configs/`.
Being in this folder does not mean a model passed acceptance: the status column says what the documentation concluded.

| File | Description | Status |
|---|---|---|
| `yolov8n_baseline.py` | Baseline YOLOv8 nano (SiLU) | Reference of the gate G4 |
| `yolov8s_baseline.py` | Baseline YOLOv8 small | Reference, no gate result recorded here |
| `yolo26n_baseline.py` | Baseline YOLO26 nano | No gate result recorded here |
| `yolov8n_relu.py` + `.yaml` | YOLOv8 nano with ReLU, initialized from `yolov8n.pt` | **Loại** (condition C fails, 100 epoch seed 0 and all 3 seeds) |
| `yolov8n_leakyrelu.py` + `.yaml` | YOLOv8 nano with LeakyReLU 0.1 | **Loại** (condition C fails) |
| `yolov8n_relu6.py` + `.yaml` | YOLOv8 nano with ReLU6 | **Loại** (condition C fails) |
| `yolov8n_relu.py` with config `ppe5_yolov8n_relu_200ep.yaml` | ReLU trained 200 epoch | **Chưa biết** (A, B, C pass, G1 unresolved) |
| `yolov8n_nodfl.py` + `.yaml` | YOLOv8 nano without DFL (reg_max 1), SiLU kept, candidate A2 of phase 2d | No gate label, see the QAT and architecture result page |

Status as read from `docs/contents/03-phuong-phap/04-tang-npu-ket-qua/01-ket-qua-g4.rst` (ReLU), `02-ket-qua-bien-the-activation.rst` (LeakyReLU, ReLU6, ReLU 200 epoch)
and `07-ket-qua-qat-kien-truc.rst` (no-DFL), on 2026-11-09. Acceptance numbers live in `ai/automation/profiles/npu_g4.yaml`.

Weights download automatically via Ultralytics into `~/.cache/ultralytics/weights/`.
