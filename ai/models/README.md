# models/

Model architecture definitions without training logic.

- `yolov8n_baseline.py`: Baseline YOLOv8 nano.
- `yolov8s_baseline.py`: Baseline YOLOv8 small.
- `yolo26n_baseline.py`: Baseline YOLO26 nano.
- `yolov8n_relu.py` + `yolov8n_relu.yaml`: YOLOv8 nano with ReLU, initialized from `yolov8n.pt`.

Weights download automatically via Ultralytics into `~/.cache/ultralytics/weights/`.
