"""YOLOv8 nano with ReLU from scratch: same yaml as yolov8n_relu, no pretrained weights. Control for the all-core detectors."""
from pathlib import Path

WEIGHTS = str(Path(__file__).parents[1] / "yolov8n_relu" / "yolov8n_relu.yaml")
