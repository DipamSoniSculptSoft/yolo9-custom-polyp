import os
import yaml
import shutil
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# 
from utils.data_load import get_image_mask_pairs, mask_to_polygons
from constants import OUTPUT_PATH, CLASS_NAMES, NUM_CLASSES



class DatasetPreparer:
    """
    Converts the Raw Data format into the COCO format and later into the YOLO format.
    """
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str = OUTPUT_PATH,
                 test_size :float = 0.2,
                 random_state: int = 42
                 ):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        
        # YOLO Dataset Format Dirs:
        self.train_images_dir = os.path.join(output_dir, 'train', 'images')
        self.val_images_dir = os.path.join(output_dir, 'val', 'images')
        self.train_labels_dir = os.path.join(output_dir, 'train', 'labels')
        self.val_labels_dir = os.path.join(output_dir, 'val', 'labels') 
    
        for d in [self.train_images_dir,
                  self.val_images_dir,
                  self.train_labels_dir,
                  self.val_labels_dir]:
            os.makedirs(d, exist_ok=True)
            
            
    def prepare(self):
        
        image_paths, mask_paths = get_image_mask_pairs(self.data_dir)
        
        # Splitting TRAIN & VAL Dataset:
        train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=self.test_size,random_state=self.random_state)
        
        # Train Data Process
        self.process_data(train_img_paths, 
                          train_mask_paths,
                          self.train_images_dir,
                          self.train_labels_dir)
        # Val Data Process
        self.process_data(val_img_paths,
                          val_mask_paths,
                          self.val_images_dir,
                          self.val_labels_dir)
        
        # Data YAML creation:
        self.create_yaml()


    def process_data(self,
                     image_paths,
                     mask_paths,
                     output_images_dir,
                     output_labels_dir):
        
        annotations = []
        images = []
        image_id = 0
        ann_id = 0
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            image_id += 1
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            
            # if mask.shape != img.shape[:2]:
            #     mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            shutil.copy(img_path, os.path.join(output_images_dir, os.path.basename(img_path)))
            
            
            images.append({
                'id': image_id,
                'file_name' : os.path.basename(img_path),
                'height': img.shape[0],
                'width' : img.shape[1]
            })

            unique_values = np.unique(mask)
            for value in unique_values:
                if value == 0:
                    continue
                
                object_mask = (mask == value).astype(np.uint8) * 255
                polygons = mask_to_polygons(object_mask)
                
                for poly in polygons:
                    ann_id += 1
                    annotations.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": [poly],
                    })
        
        coco_input = {
            "images": images,
            "annotations": annotations,
            "categories" : [{"id": 1, "name": "polyp"}]
        }
                    
        # Convert COCO-like dictionery to YOLO format:
        for img_info in coco_input['images']:
            img_id = img_info['id']
            img_ann = [ann for ann in coco_input['annotations'] if ann['image_id'] == img_id]
            img_w, img_h = img_info['width'], img_info['height']
            
            if img_ann:
                with open(os.path.join(output_labels_dir, os.path.splitext(img_info['file_name'])[0] + '.txt'), 'w') as file_object:
                    for ann in img_ann:
                        current_category = ann['category_id'] - 1
                        polygon = ann['segmentation'][0]
                        normalized_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(polygon)]
                        file_object.write(f"{current_category} " + " ".join(normalized_polygon) + "\n")
            
    def create_yaml(self,
                    train_images_dir : str = 'train/images', 
                    val_images_dir : str = 'val/images'):
        yaml_data = {
            'names' : CLASS_NAMES,
            'nc': NUM_CLASSES,
            'train': train_images_dir,
            'val': val_images_dir,
            'test': ' ',
        }        
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
            