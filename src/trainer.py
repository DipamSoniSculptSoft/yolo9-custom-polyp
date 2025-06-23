import os
# 
from pipelines.dataset import DatasetPreparer
from pipelines.tune import Trainer
from pipelines.predict import Predictor
from constants import *
from helpers.seeds import set_seeds


def main():
    
    set_seeds()
    
    #* MODULE TRAINER
    
    # Prepare Dataset:
    data_preparer = DatasetPreparer(
        data_dir=DATASET_DIR,
        output_dir=OUTPUT_PATH
    )
    data_preparer.prepare()
    
    # YOLO: Fine-Tune
    trainer = Trainer(
        epochs = CONFIG.get('epochs', 100),
        batch_size=CONFIG.get('batch_size', 12),
        imgsz=CONFIG.get('imgsz', 512)
    )
    trainer.invoke()
    

if __name__ == '__main__':
    main()