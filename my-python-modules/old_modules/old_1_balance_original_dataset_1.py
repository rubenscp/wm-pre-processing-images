"""
Project: White Mold 
Description: Create lists of bouding boxes images to be cropped 
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 27/03/2024
Version: 1.0
"""

# Importing libraries
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing python modules
from common.manage_log import *
from common.utils import Utils
from common.entity.ImageAnnotation import ImageAnnotation
from random import randrange

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

# Fixing random state for reproducibility
np.random.seed(19680801)

# ###########################################
# Application Methods
# ###########################################

# ###########################################
# Methods of Level 1
# ###########################################

def copy_images_to_balanced_dataset_per_class(parameters, model):

    logging_info(f'Copy images to balanced dataset for model {model}')

    # initialize statistic dictionary 
    statistics = initialize_statistics(parameters)
    logging_info(f'statistics: {statistics}')

    # list of images for copy at all format and types of dataset (train/valid/test)
    balanced_image_dataset = {
        "train_images": [],
        "valid_images": [],
        "test_images": []
    }

    # processing annotations list 
    for item in parameters['input']['dimensions']:
        height = item['height']
        width  = item['width']





def copy_images_dataset_to_balanced_dataset_ssd_pascal_voc(parameters):

    logging_info(f'Copy images dataset for SSD Model')

    # initialize statistic dictionary 
    statistics = initialize_statistics(parameters)
    logging_info(f'statistics: {statistics}')

    # processing annoations list 
    for item in parameters['input']['dimensions']:
        height = item['height']
        width  = item['width']
        
        input_dataset_type = 'train'
        copy_dataset_by_input_dataset_type_ssd_pascal_voc(parameters, statistics, height, width, input_dataset_type)
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

        input_dataset_type = 'valid'
        copy_dataset_by_input_dataset_type_ssd_pascal_voc(parameters, statistics, height, width, input_dataset_type)
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

        input_dataset_type = 'test'
        copy_dataset_by_input_dataset_type_ssd_pascal_voc(parameters, statistics, height, width, input_dataset_type)
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

def copy_images_dataset_to_balanced_dataset_faster_rcnn(parameters):

    logging_info(f'Copy images dataset for Faster RCNN Model')

    # initialize statistic dictionary 
    statistics = initialize_statistics(parameters)
    logging_info(f'statistics: {statistics}')

    # processing annoations list 
    for item in parameters['input']['dimensions']:
        height = item['height']
        width  = item['width']
        
        input_dataset_type = 'train'
        copy_dataset_by_input_dataset_type_faster_rcnn(parameters, statistics, height, width, input_dataset_type)
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

        input_dataset_type = 'valid'
        copy_dataset_by_input_dataset_type_faster_rcnn(parameters, statistics, height, width, input_dataset_type)
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

        input_dataset_type = 'test'
        copy_dataset_by_input_dataset_type_faster_rcnn(parameters, statistics, height, width, input_dataset_type)
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

def copy_images_dataset_to_balanced_dataset_yolov8(parameters):

    logging_info(f'Copy images dataset for YOLOv8 Model')

    # initialize statistic dictionary 
    statistics = initialize_statistics(parameters)
    logging_info(f'statistics: {statistics}')

    # processing annoations list 
    for item in parameters['input']['dimensions']:
        height = item['height']
        width  = item['width']
        
        input_dataset_type = 'train'
        copy_dataset_by_input_dataset_type_yolov8(parameters, statistics, height, width, input_dataset_type)
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

        input_dataset_type = 'valid'
        copy_dataset_by_input_dataset_type_yolov8(parameters, statistics, height, width, input_dataset_type)
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

        input_dataset_type = 'test'
        copy_dataset_by_input_dataset_type_yolov8(parameters, statistics, height, width, input_dataset_type)
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')


# ###########################################
# Methods of Level 2
# ###########################################

def initialize_statistics(parameters):
    # creating statistic dictionary 
    statistics = {}

    # setting statistic dimensions
    statistics['train'] = {}
    statistics['valid'] = {}
    statistics['test'] = {}
    for class_name in parameters['neural_network_model']['classes']:
        if class_name[:5] != 'class' and class_name[:14] != '__background__':                
            statistics['train'][class_name] = {}
            statistics['train'][class_name]['number_of'] = 0
            statistics['valid'][class_name] = {}
            statistics['valid'][class_name]['number_of'] = 0
            statistics['test'][class_name] = {}
            statistics['test'][class_name]['number_of'] = 0

    # returning statistics dictionary 
    return statistics   







def copy_dataset_by_input_dataset_type_ssd_pascal_voc(parameters, statistics, height, width, input_dataset_type):

    # preparing input folder names
    input_main_folder = os.path.join(
        parameters['results']['output_dataset']['ssd_model_with_pascal_voc_format']['main_folder'],
        str(height) + 'x' + str(width)
    )
    if input_dataset_type == 'train': dataset_type_folder = 'train_folder'
    if input_dataset_type == 'valid': dataset_type_folder = 'valid_folder'
    if input_dataset_type == 'test': dataset_type_folder = 'test_folder'

    input_dataset_type_folder = os.path.join(
        input_main_folder,
        parameters['results']['output_dataset']['ssd_model_with_pascal_voc_format'][dataset_type_folder]
    )

    # preparing output folder names
    output_main_folder = os.path.join(
        parameters['results']['balanced_output_dataset']['ssd_model_with_pascal_voc_format']['main_folder'],
        str(height) + 'x' + str(width)
    )
    output_dataset_type_folder = os.path.join(
        output_main_folder,
        parameters['results']['balanced_output_dataset']['ssd_model_with_pascal_voc_format'][dataset_type_folder]
    )

    input_bbox_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['output_dataset']['ssd_model_with_pascal_voc_format']['bounding_box_folder']
    )
    output_bbox_folder = os.path.join(
        output_dataset_type_folder,
        parameters['results']['balanced_output_dataset']['ssd_model_with_pascal_voc_format']['bounding_box_folder']
    )        

    # getting list of images
    annotations = Utils.get_files_with_extensions(input_dataset_type_folder, 'xml')

    logging_info(f'len annotations: {len(annotations)}')
    logging_info(f'annotations: {annotations}')
    number_of_annotations = len(annotations)
    img = 0
    for _ in range(number_of_annotations):
        # getting an item randomly
        index = randrange(len(annotations))
        annotation_file = annotations[index]

        # read annotation file 
        path_and_filename_annotation = os.path.join(input_dataset_type_folder, annotation_file)
        image_annotation = ImageAnnotation()
        image_annotation.get_annotation_file_in_voc_pascal_format(path_and_filename_annotation)
        
        img += 1
        logging_info(f'img: {img} filename: {annotation_file}   bbox: {len(image_annotation.bounding_boxes)}')

        # processing bounding boxes 
        for bounding_box in image_annotation.bounding_boxes:          
            if statistics[input_dataset_type][bounding_box.class_title]['number_of'] < \
               parameters['processing']['balance_of_images_per_class'] \
                         ['number_of_images_for_'+input_dataset_type]:
                
                # getting filename and extension separeted 
                filename, extension = Utils.get_filename_and_extension(annotation_file)              

                # copying files 
                filename_to_copy = filename + '.jpg'
                Utils.copy_file_same_name(filename_to_copy, input_dataset_type_folder, output_dataset_type_folder)
                Utils.copy_file_same_name(annotation_file, input_dataset_type_folder, output_dataset_type_folder)
                if parameters['input']['draw_and_save_bounding_box']:
                    filename_to_copy = filename + '-drawn.jpg'
                    Utils.copy_file_same_name(filename_to_copy, input_bbox_folder, output_bbox_folder)

                # updating counter per class
                number_of = statistics[input_dataset_type][bounding_box.class_title]['number_of']
                statistics[input_dataset_type][bounding_box.class_title]['number_of'] = number_of + 1
                # logging_info(f'copiou imagem: {filename}  ' \
                #              f'  input_dataset_type: {input_dataset_type}  ' \
                #              f'  bounding_box.class_title: {bounding_box.class_title}  ' \
                #              f'  number_of: {number_of}'
                # )

        # removing item from list 
        annotations.pop(index)

def copy_dataset_by_input_dataset_type_faster_rcnn(parameters, statistics, height, width, input_dataset_type):

    # preparing input folder names
    input_main_folder = os.path.join(
        parameters['results']['output_dataset']['faster_rcnn_model']['main_folder'],
        str(height) + 'x' + str(width)
    )
    if input_dataset_type == 'train': dataset_type_folder = 'train_folder'
    if input_dataset_type == 'valid': dataset_type_folder = 'valid_folder'
    if input_dataset_type == 'test': dataset_type_folder = 'test_folder'

    input_dataset_type_folder = os.path.join(
        input_main_folder,
        parameters['results']['output_dataset']['faster_rcnn_model'][dataset_type_folder]
    )

    # preparing output folder names
    output_main_folder = os.path.join(
        parameters['results']['balanced_output_dataset']['faster_rcnn_model']['main_folder'],
        str(height) + 'x' + str(width)
    )
    output_dataset_type_folder = os.path.join(
        output_main_folder,
        parameters['results']['balanced_output_dataset']['faster_rcnn_model'][dataset_type_folder]
    )

    input_bbox_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['output_dataset']['faster_rcnn_model']['bounding_box_folder']
    )              
    output_bbox_folder = os.path.join(
        output_dataset_type_folder,
        parameters['results']['balanced_output_dataset']['faster_rcnn_model']['bounding_box_folder']
    )        

    # getting list of images
    annotations = Utils.get_files_with_extensions(input_dataset_type_folder, 'xml')

    logging_info(f'len annotations: {len(annotations)}')
    logging_info(f'annotations: {annotations}')
    number_of_annotations = len(annotations)
    img = 0
    for _ in range(number_of_annotations):
        # getting an item randomly
        index = randrange(len(annotations))
        annotation_file = annotations[index]

        # read annotation file 
        path_and_filename_annotation = os.path.join(input_dataset_type_folder, annotation_file)
        image_annotation = ImageAnnotation()
        image_annotation.get_annotation_file_in_voc_pascal_format(path_and_filename_annotation)
        
        img += 1
        logging_info(f'img: {img} filename: {annotation_file}   bbox: {len(image_annotation.bounding_boxes)}')

        # processing bounding boxes 
        for bounding_box in image_annotation.bounding_boxes:           
            if statistics[input_dataset_type][bounding_box.class_title]['number_of'] < \
               parameters['processing']['balance_of_images_per_class'] \
                         ['number_of_images_for_'+input_dataset_type]:
                
                # getting filename and extension separeted 
                filename, extension = Utils.get_filename_and_extension(annotation_file)                

                # copying files 
                filename_to_copy = filename + '.jpg'
                Utils.copy_file_same_name(filename_to_copy, input_dataset_type_folder, output_dataset_type_folder)
                Utils.copy_file_same_name(annotation_file, input_dataset_type_folder, output_dataset_type_folder)
                if parameters['input']['draw_and_save_bounding_box']:
                    filename_to_copy = filename + '-drawn.jpg'
                    Utils.copy_file_same_name(filename_to_copy, input_bbox_folder, output_bbox_folder)

                # updating counter per class
                number_of = statistics[input_dataset_type][bounding_box.class_title]['number_of']
                statistics[input_dataset_type][bounding_box.class_title]['number_of'] = number_of + 1
                # logging_info(f'copiou imagem: {filename}  ' \
                #              f'  input_dataset_type: {input_dataset_type}  ' \
                #              f'  bounding_box.class_title: {bounding_box.class_title}  ' \
                #              f'  number_of: {number_of}'
                # )

        # removing item from list 
        annotations.pop(index)

def copy_dataset_by_input_dataset_type_yolov8(parameters, statistics, height, width, input_dataset_type):

    # preparing input folder names
    input_main_folder = os.path.join(
        parameters['results']['output_dataset']['yolov8_model']['main_folder'],
        str(height) + 'x' + str(width)
    )
    if input_dataset_type == 'train': dataset_type_folder = 'train_folder'
    if input_dataset_type == 'valid': dataset_type_folder = 'valid_folder'
    if input_dataset_type == 'test': dataset_type_folder = 'test_folder'

    input_dataset_type_folder = os.path.join(
        input_main_folder,
        parameters['results']['output_dataset']['yolov8_model'][dataset_type_folder]
    )

    # preparing output folder names
    output_main_folder = os.path.join(
        parameters['results']['balanced_output_dataset']['yolov8_model']['main_folder'],
        str(height) + 'x' + str(width)
    )
    output_dataset_type_folder = os.path.join(
        output_main_folder,
        parameters['results']['balanced_output_dataset']['yolov8_model'][dataset_type_folder]
    )

    # preparing input folder names for labels, images and bbox
    input_images_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['output_dataset']['yolov8_model']['images_folder']
    )        
    input_labels_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['output_dataset']['yolov8_model']['labels_folder']
    )        
    input_bbox_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['output_dataset']['yolov8_model']['bounding_box_folder']
    )        

    # preparing output folder names for labels, images and bbox
    output_images_folder = os.path.join(
        output_dataset_type_folder,
        parameters['results']['balanced_output_dataset']['yolov8_model']['images_folder']
    )        
    output_labels_folder = os.path.join(
        output_dataset_type_folder,
        parameters['results']['balanced_output_dataset']['yolov8_model']['labels_folder']
    )        
    output_bbox_folder = os.path.join(
        output_dataset_type_folder,
        parameters['results']['balanced_output_dataset']['yolov8_model']['bounding_box_folder']
    )        

    # getting list of images
    annotations = Utils.get_files_with_extensions(input_labels_folder, 'txt')

    logging_info(f'len annotations: {len(annotations)}')
    logging_info(f'annotations: {annotations}')
    number_of_annotations = len(annotations)
    img = 0
    for _ in range(number_of_annotations):
        # getting an item randomly
        index = randrange(len(annotations))
        annotation_file = annotations[index]

        # read annotation file 
        path_and_filename_yolo_annotation = os.path.join(input_labels_folder, annotation_file)
        image_annotation = ImageAnnotation()
        image_annotation.get_annotation_file_in_yolo_v5_format(
            path_and_filename_yolo_annotation, 
            parameters['neural_network_model']['classes'],
            height, width)
        
        img += 1
        logging_info(f'img: {img} filename: {annotation_file}   bbox: {len(image_annotation.bounding_boxes)}')

        # processing bounding boxes 
        for bounding_box in image_annotation.bounding_boxes:           
            if statistics[input_dataset_type][bounding_box.class_title]['number_of'] < \
               parameters['processing']['balance_of_images_per_class'] \
                         ['number_of_images_for_'+input_dataset_type]:
                
                # getting filename and extension separeted 
                filename, extension = Utils.get_filename_and_extension(annotation_file)              

                # copying files 
                filename_to_copy = filename + '.jpg'
                Utils.copy_file_same_name(filename_to_copy, input_images_folder, output_images_folder)
                Utils.copy_file_same_name(annotation_file, input_labels_folder, output_labels_folder)
                if parameters['input']['draw_and_save_bounding_box']:
                    filename_to_copy = filename + '-drawn.jpg'
                    Utils.copy_file_same_name(filename_to_copy, input_bbox_folder, output_bbox_folder)

                # updating counter per class
                number_of = statistics[input_dataset_type][bounding_box.class_title]['number_of']
                statistics[input_dataset_type][bounding_box.class_title]['number_of'] = number_of + 1
                # logging_info(f'copiou imagem: {filename}  ' \
                #              f'  input_dataset_type: {input_dataset_type}  ' \
                #              f'  bounding_box.class_title: {bounding_box.class_title}  ' \
                #              f'  number_of: {number_of}'
                # )

        # removing item from list 
        annotations.pop(index)
