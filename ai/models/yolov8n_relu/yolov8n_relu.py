"""YOLOv8 nano with ReLU activations, initialized from the SiLU yolov8n.pt weights (no official ReLU checkpoint)."""
from pathlib import Path

WEIGHTS = str(Path(__file__).with_name("yolov8n_relu.yaml"))
PRETRAINED = "yolov8n.pt"
