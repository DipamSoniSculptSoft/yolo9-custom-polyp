import os
import cv2
import numpy as np
from typing import Tuple, List
from constants import DATASET_DIR

def get_image_mask_pairs(data_dir: str = DATASET_DIR) -> Tuple[List[str]]:
    
    image_paths = []
    mask_paths = []
    
    for root,_,files in os.walk(data_dir):
        if 'images' in root:
            for file in files:
                if file.endswith('.png'):
                  image_paths.append(os.path.join(root,file))
                  mask_paths.append(os.path.join(root.replace('images','masks'), file.replace('.png','.tif')))
    return image_paths, mask_paths


def mask_to_polygons(mask, epsilon=1.0):
    
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:
            poly = contour.reshape(-1).tolist()
            if len(poly) > 4:
                polygons.append(poly)

    return polygons
