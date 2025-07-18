import os
from ultralytics import YOLO
from src.constants import *

model_pt_path = os.path.join(YOLO_RESULTS, f"200_epochs-2", "weights", "best.pt")
model = YOLO(model_pt_path)
data_yaml_path = os.path.join(PROJECT_ROOT, 'yolo-artifacts', 'yolov9-512-polyp-dataset', 'data.yaml')
model.export(
    format='engine',      # TensorRT format
    dynamic=True,         # Enable dynamic shapes (batch and image sizes)
    batch=12,              # Max batch size (adjust as needed for your GPU memory)
    imgsz=512,            # Base image size (can be dynamic around this)
    half=True,            # FP16 for speed
    workspace=8,          # Workspace in GiB for optimizations
    # int8=False,           # Optional INT8 (needs calibration data via 'data' arg)
    data=data_yaml_path   # For metadata/class names and INT8 calibration
)

# Load the trained YOLO model at startup
model_path = os.path.join(YOLO_RESULTS, f"200_epochs-2", "weights", "best.engine")
print(f"Exported to {model_path} with dynamic batch (1-16) and image sizes (320-640).")
model = YOLO(model_path)
print("Model loaded successfully.")
