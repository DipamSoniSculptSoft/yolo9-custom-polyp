import os
from pipelines.predict import Predictor
from constants import *
from helpers.seeds import set_seeds



def main():
    """Prediction Pipeline:::::::::"""

    set_seeds()
    
    #* PREDICTER:
    model_path = os.path.join(YOLO_RESULTS, f"{CONFIG.get('epochs', 90)}_epochs-", "weights", "best.pt")
    predictor = Predictor(model_path) 
    
    
    # input_image_dir = os.path.join(
    #     os.path.dirname(PROJECT_ROOT), 
    #     'data', 'hyper-kvasir-segmented-images', 'segmented-images', 'images'
    #     )
    
    input_image_dir = os.path.join(
        DATASET_DIR, 'polyp', 'images'
    )
    
    
    predictor.predict_batch(image_dir= input_image_dir,
                            output_dir=os.path.join(PROJECT_ROOT, 'output', 'predictions', 'train_val'))
    

if __name__ == '__main__':
    main()

