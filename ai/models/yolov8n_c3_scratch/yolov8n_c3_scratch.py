"""YOLOv8 nano with every C2f replaced by C3 (same repeats, widths and shortcut flags), from scratch. Test arm of the C2f versus C3 comparison, control is yolov8n_scratch."""
from pathlib import Path

WEIGHTS = str(Path(__file__).with_name("yolov8n_c3_scratch.yaml"))
