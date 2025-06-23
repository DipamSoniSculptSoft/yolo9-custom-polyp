import os 
import cv2
import numpy as np
from ultralytics import YOLO
from constants import CONFIG, PROJECT_ROOT



class Predictor:
    """
    """
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.imgsz = CONFIG.get('imgsz', 512)
        
    def predict_single_image(self, 
                             image_path,
                             threshold: float = 0.5,
                             iou : float = 0.5):
        
        image = cv2.imread(image_path)    
        image = cv2.resize(image, (self.imgsz, int(self.imgsz * 0.75)))
        masks, boxes, labels = self.get_outputs(image, threshold, iou)
        result = self.draw_segmentation_map(image, masks, boxes, labels)
        return result    
    
    def predict_batch(self, 
                      image_dir, 
                      output_dir=None,
                      threshold: float = 0.5,
                      iou : float = 0.5):
        
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, 'output', 'predictions')
        
        os.makedirs(output_dir, exist_ok=True)
        for img_file in os.listdir(image_dir):
            if img_file.endswith('.png') or img_file.endswith('.jpg'):
                img_path = os.path.join(image_dir, img_file)
                result = self.predict_single_image(img_path, threshold, iou)
                save_path = os.path.join(output_dir, f"{img_file}")
                cv2.imwrite(save_path, result)

    def get_outputs(self, image, threshold, iou):
        
        outputs = self.model.predict(image,
                                     imgsz = self.imgsz,
                                     conf=threshold
                                     )
        scores = outputs[0].boxes.conf.detach().cpu().numpy()
        thresholded_indices = [idx for idx, score in enumerate(scores) if score > threshold]
        
        if thresholded_indices:
            masks = [outputs[0].masks.xy[idx] for idx in thresholded_indices]
            boxes = outputs[0].boxes.xyxy.detach().cpu().numpy()[thresholded_indices]
            boxes = [[(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))] for box in boxes]
            labels = [outputs[0].names[int(outputs[0].boxes.cls[idx])] for idx in thresholded_indices]
            
        else:
            masks, boxes, labels = [],[],[]
            
        return masks, boxes, labels
    
    
    def draw_segmentation_map(self, image, masks, boxes, labels):
        
        alpha = 1.0
        beta = 0.5
        image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for mask, box, label in zip(masks, boxes, labels):
            
            color = (0,255,0) # Green
            segmentation_map = np.zeros_like(image)
            
            if mask is not None and len(mask) > 0:
                poly = np.array(mask, dtype = np.int32)
                cv2.fillPoly(segmentation_map, [poly], color)
                
            # cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            cv2.rectangle(image, box[0], box[1], color=(255, 0, 0), thickness=2)  # Red color for bounding box
            cv2.putText(image, label, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 2)
            
        return image
    