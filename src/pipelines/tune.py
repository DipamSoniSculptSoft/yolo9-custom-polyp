import os
from ultralytics import YOLO 
from constants import CONFIG, OUTPUT_PATH, YOLO_RESULTS


class Trainer:
    
    def __init__(self, 
                 model_name : str = 'yolov9e-seg.pt',
                 epochs : int  = 10,
                 batch_size: int  = 6,
                 imgsz : int = 512,
                 patience = 0):
        
        self.model = YOLO(model_name)
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.patience = patience
        self.project = YOLO_RESULTS
        self.name = f"{self.epochs}_epochs-"
        
    def invoke(self):
        data_yaml = os.path.join(OUTPUT_PATH, 'data.yaml')
        self.model.train(
            data = data_yaml,
            project = self.project,
            name = self.name,
            epochs = self.epochs,
            batch = self.batch_size,
            imgsz = self.imgsz,
            patience = self.patience
        )
        
        
        
    