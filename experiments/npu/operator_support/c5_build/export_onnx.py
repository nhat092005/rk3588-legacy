import sys
from ultralytics import YOLO
print(YOLO(sys.argv[1]).export(format="onnx", imgsz=640, opset=19))
