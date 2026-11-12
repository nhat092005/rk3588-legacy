"""YOLOv8 nano with ReLU6 activations, initialized from the SiLU yolov8n.pt weights (no official checkpoint with this activation)."""
from pathlib import Path

WEIGHTS = str(Path(__file__).with_name("yolov8n_relu6.yaml"))
PRETRAINED = "yolov8n.pt"
