"""YOLOv8 nano without DFL (reg_max 1), initialized from the SiLU yolov8n.pt where the tensor shapes match (the box branch cv2 does not)."""
from pathlib import Path

WEIGHTS = str(Path(__file__).with_name("yolov8n_nodfl.yaml"))
PRETRAINED = "yolov8n.pt"
