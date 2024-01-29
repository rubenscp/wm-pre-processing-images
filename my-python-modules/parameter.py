"""
Project: White Mold 
Description: Application parameters 
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 20/10/2023
Version: 1.0
"""
   
PROJECT_HOME_PATH = 'E:/Doctorate/White-Mold-Dataset/'   

META_FILE = 'meta.json'

# Parameters of image preparation according input annotations format 
INPUT_PATH = PROJECT_HOME_PATH + '01-input'   
INPUT_SUPERVISELY_FORMAT = 'supervisely-format'
INPUT_SUPERVISELY_FORMAT_IMAGES = 'img'
INPUT_SUPERVISELY_FORMAT_ANNOTATIONS = 'ann'

# Output folder with all images and annotations selected 
OUTPUT_ALL_IMAGES_AND_ANNOTATIONS = PROJECT_HOME_PATH + '02-all-images-and-annotations'

# Output folder for the image dataset spliting criteria 
OUTPUT_IMAGE_DATASET_SPLIT_CRITERIA_PATH = PROJECT_HOME_PATH
OUTPUT_IMAGE_DATASET_SPLIT_BY_IMAGES = '03-splitting_by_images/'
OUTPUT_IMAGE_DATASET_SPLIT_BY_BOUNDING_BOXES = '03-splitting_by_bounding_boxes/'

# Output folder for split dataset 
OUTPUT_SPLIT_DATASET_PATH =  '03.1-lists-to-split-train-valid-test'
OUTPUT_SPLIT_DATASET_PATH_ALL_LIST   = 'all-bbox-list.txt'
OUTPUT_SPLIT_DATASET_PATH_TRAIN_LIST = 'train-bbox-list.txt'
OUTPUT_SPLIT_DATASET_PATH_VALID_LIST = 'valid-bbox-list.txt'
OUTPUT_SPLIT_DATASET_PATH_TEST_LIST  = 'test-bbox-list.txt'
OUTPUT_SPLIT_DATASET_PATH_VIOLIN_PLOT = 'bounding-box-plot.png'

# Output folder for processing results
OUTPUT_MODEL_PATH =  '03.2-output-dataset'

# SSD model 
SSD_MODEL_FOLDER = 'ssd'
SSD_MODEL_TRAIN  = 'train'
SSD_MODEL_VALID  = 'valid'
SSD_MODEL_TEST   = 'test'
SSD_MODEL_BBOX   = 'bbox'
SSD_MODEL_IMAGES = 'images'
SSD_MODEL_ANNOTATION_FILENAME = 'label.csv'

# SSD model with Pascal VOC format
SSD_PASCAL_VOC_MODEL = 'ssd_pascal_voc'
SSD_PASCAL_VOC_MODEL_TRAIN = 'train'
SSD_PASCAL_VOC_MODEL_VALID = 'valid'
SSD_PASCAL_VOC_MODEL_TEST = 'test'
SSD_PASCAL_VOC_MODEL_BBOX = 'bbox'

# Faster R-CNN model
FASTER_RCNN_MODEL = 'faster_rcnn'
FASTER_RCNN_MODEL_TRAIN = 'train'
FASTER_RCNN_MODEL_VALID = 'valid'
FASTER_RCNN_MODEL_TEST = 'test'
FASTER_RCNN_MODEL_BBOX = 'bbox'

# YOLOv8  model
YOLO_V8_MODEL = 'yolov8'
YOLO_V8_MODEL_TRAIN = 'train'
YOLO_V8_MODEL_VALID = 'valid'
YOLO_V8_MODEL_TEST = 'test'
YOLO_V8_MODEL_IMAGES = 'images'
YOLO_V8_MODEL_LABELS = 'labels'
YOLO_V8_MODEL_BBOX = 'bbox'

# Zip output directory 
ZIP_PATH =  '03.3-zipped-dataset'
ZIP_FILENAME = 'white_mold_image_dataset'
ZIP_FILENAME_SPLIT_BY_IMAGES = '_splitting_by_images'
ZIP_FILENAME_SPLIT_BY_BOUNDING_BOXES = '_splitting_by_bounding_boxes'

# Supervisely annotation format 
# SUPERVISELY_FORMAT_FOLDER = 'supervisely-format/'
# SUPERVISELY_FORMAT_IMAGE = 'img/'
# SUPERVISELY_FORMAT_ANNOTATION = 'ann/'
# CROP_OUTPUT_PATH_SUPERVISELY = CROP_OUTPUT_PATH + SUPERVISELY_FORMAT_FOLDER
# CROP_OUTPUT_PATH_SUPERVISELY_IMAGES = CROP_OUTPUT_PATH + SUPERVISELY_FORMAT_FOLDER + SUPERVISELY_FORMAT_IMAGE
# CROP_OUTPUT_PATH_SUPERVISELY_ANNOTATIONS = CROP_OUTPUT_PATH + SUPERVISELY_FORMAT_FOLDER + SUPERVISELY_FORMAT_ANNOTATION

