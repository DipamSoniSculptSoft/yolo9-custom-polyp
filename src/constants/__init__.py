import os 
import torch
import yaml


DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_STR)


# PATH Configurations:
SCRIPT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)



CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, 'config.yaml')
CONFIG:dict = yaml.load(open(CONFIG_FILE_PATH), Loader = yaml.SafeLoader)

# Helper Function: Resolve Paths from config relative to SCRIPT_DIR
def resolve_config_path(config_path_value):
    return os.path.abspath(os.path.join(SCRIPT_DIR, config_path_value))

# Other Paths
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

# YOLO Configurations:
YOLO_VERSION = 'v9'
YOLO_MODEL = 'yolov9e-seg'

OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'yolo-artifacts')
OUTPUT_PATH = os.path.join(OUTPUT_BASE, f"yolo{YOLO_VERSION}-{CONFIG.get('imgsz', 512)}-polyp-dataset")
YOLO_RESULTS = f"yolo{YOLO_VERSION}-{CONFIG.get('imgsz', 512)}-polyp/results"
# OUTPUT_PATH = resolve_config_path(CONFIG['output_path'])
YOLO_RESULTS = os.path.join(OUTPUT_PATH, 'results')

#   
# os.makedirs(YOLO_RESULTS, exist_ok=True)

# Class Names:
CLASS_NAMES = os.listdir(DATASET_DIR) 
NUM_CLASSES = len(CLASS_NAMES)
