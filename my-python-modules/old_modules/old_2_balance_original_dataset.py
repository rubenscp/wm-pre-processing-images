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
    # logging_info(f'statistics: {statistics}')

    # list of images for copy at all format and types of dataset (train/valid/test)
    balanced_image_dataset = {}

    # processing annotations list 
    for item in parameters['input']['dimensions']:
        height = item['height']
        width  = item['width']
        dimension = str(height) + 'x' + str(width)

        balanced_image_dataset[dimension] = {}
        balanced_image_dataset[dimension]['train_images'] = []
        balanced_image_dataset[dimension]['valid_images'] = []
        balanced_image_dataset[dimension]['test_images'] = []
        
        input_dataset_type = 'train'
        train_images_list = get_balanced_dataset_per_dataset_type(parameters, statistics, height, width, input_dataset_type)
        balanced_image_dataset[dimension]['train_images'] = train_images_list

        input_dataset_type = 'valid'
        valid_images_list = get_balanced_dataset_per_dataset_type(parameters, statistics, height, width, input_dataset_type)
        balanced_image_dataset[dimension]['valid_images'] = valid_images_list

        input_dataset_type = 'test'
        test_images_list = get_balanced_dataset_per_dataset_type(parameters, statistics, height, width, input_dataset_type)
        balanced_image_dataset[dimension]['test_images'] = test_images_list

        logging.info(f'')
        logging.info(f'')
        logging.info(f"balanced_image_dataset - train: {len(balanced_image_dataset[dimension]['train_images'])}")
        logging.info(f"balanced_image_dataset - train: {balanced_image_dataset[dimension]['train_images']}")

        logging.info(f"balanced_image_dataset - valid: {len(balanced_image_dataset[dimension]['valid_images'])}")
        logging.info(f"balanced_image_dataset - valid: {balanced_image_dataset[dimension]['valid_images']}")

        logging.info(f"balanced_image_dataset - test: {len(balanced_image_dataset[dimension]['test_images'])}")
        logging.info(f"balanced_image_dataset - test: {balanced_image_dataset[dimension]['test_images']}")

        logging.info(f'balanced_image_dataset: {Utils.get_pretty_json(balanced_image_dataset)}')
        logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

        running_id = parameters['processing']['running_id']
        running_id_text = 'running-' + f'{running_id:04}'
        path_and_filename = os.path.join(
            parameters['results']['balanced_output_dataset']['balanced_output_dataset_folder'],            
            'balanced_image_dataset_' + running_id_text + '.xlsx'
        )
        save_sheet_balanced_image_dataset(path_and_filename, balanced_image_dataset, dimension)

        # copying files according by balanced image list 
        build_balanced_image_dataset_ssd_pascal_voc(parameters, dimension, balanced_image_dataset, 'train')
        build_balanced_image_dataset_ssd_pascal_voc(parameters, dimension, balanced_image_dataset, 'valid')
        build_balanced_image_dataset_ssd_pascal_voc(parameters, dimension, balanced_image_dataset, 'test')
                
        build_balanced_image_dataset_faster_rcnn(parameters, dimension, balanced_image_dataset, 'train')
        build_balanced_image_dataset_faster_rcnn(parameters, dimension, balanced_image_dataset, 'valid')
        build_balanced_image_dataset_faster_rcnn(parameters, dimension, balanced_image_dataset, 'test')
                

# def build_balanced_image_dataset_ssd_pascal_vocxxxxxxx(parameters, height, width,
#                                                 balanced_image_dataset):
#     logging_info(f'')
#     logging_info(f'Build balanced image dataset for SSD Model')

#     # processing annoations list 
#     for item in parameters['input']['dimensions']:
#         height = item['height']
#         width  = item['width']
        
#         input_dataset_type = 'train'
#         copy_dataset_by_input_dataset_type_ssd_pascal_voc(parameters, statistics, height, width, input_dataset_type)
#         logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

#         input_dataset_type = 'valid'
#         copy_dataset_by_input_dataset_type_ssd_pascal_voc(parameters, statistics, height, width, input_dataset_type)
#         logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

#         input_dataset_type = 'test'
#         copy_dataset_by_input_dataset_type_ssd_pascal_voc(parameters, statistics, height, width, input_dataset_type)
#         logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

# def copy_images_dataset_to_balanced_dataset_faster_rcnn(parameters):

#     logging_info(f'Copy images dataset for Faster RCNN Model')

#     # initialize statistic dictionary 
#     statistics = initialize_statistics(parameters)
#     logging_info(f'statistics: {statistics}')

#     # processing annoations list 
#     for item in parameters['input']['dimensions']:
#         height = item['height']
#         width  = item['width']
        
#         input_dataset_type = 'train'
#         copy_dataset_by_input_dataset_type_faster_rcnn(parameters, statistics, height, width, input_dataset_type)
#         logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

#         input_dataset_type = 'valid'
#         copy_dataset_by_input_dataset_type_faster_rcnn(parameters, statistics, height, width, input_dataset_type)
#         logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

#         input_dataset_type = 'test'
#         copy_dataset_by_input_dataset_type_faster_rcnn(parameters, statistics, height, width, input_dataset_type)
#         logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

# def copy_images_dataset_to_balanced_dataset_yolov8(parameters):

#     logging_info(f'Copy images dataset for YOLOv8 Model')

#     # initialize statistic dictionary 
#     statistics = initialize_statistics(parameters)
#     logging_info(f'statistics: {statistics}')

#     # processing annoations list 
#     for item in parameters['input']['dimensions']:
#         height = item['height']
#         width  = item['width']
        
#         input_dataset_type = 'train'
#         copy_dataset_by_input_dataset_type_yolov8(parameters, statistics, height, width, input_dataset_type)
#         logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

#         input_dataset_type = 'valid'
#         copy_dataset_by_input_dataset_type_yolov8(parameters, statistics, height, width, input_dataset_type)
#         logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')

#         input_dataset_type = 'test'
#         copy_dataset_by_input_dataset_type_yolov8(parameters, statistics, height, width, input_dataset_type)
#         logging.info(f'statistics: {Utils.get_pretty_json(statistics)}')


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


def get_balanced_dataset_per_dataset_type(parameters, statistics, height, width, input_dataset_type):

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

    # getting list of images
    annotations = Utils.get_files_with_extensions(input_dataset_type_folder, 'xml')
    # logging_info(f'len annotations: {len(annotations)}')
    # logging_info(f'annotations: {annotations}')
    number_of_annotations = len(annotations)

    # creating images list 
    images_list = []

    # img = 0
    for _ in range(number_of_annotations):
        # getting an item randomly
        index = randrange(len(annotations))
        annotation_file = annotations[index]

        # read annotation file 
        path_and_filename_annotation = os.path.join(input_dataset_type_folder, annotation_file)
        image_annotation = ImageAnnotation()
        image_annotation.get_annotation_file_in_voc_pascal_format(path_and_filename_annotation)
        
        # img += 1
        # logging_info(f'img: {img} filename: {annotation_file}   bbox: {len(image_annotation.bounding_boxes)}')

        # processing bounding boxes 
        for bounding_box in image_annotation.bounding_boxes:
            limit_value = parameters['input']['balance_of_images_per_class'][bounding_box.class_title] \
                         ['number_for_'+input_dataset_type]

            # evaluating the number of images selected and the limit value 
            number_of = statistics[input_dataset_type][bounding_box.class_title]['number_of']
            if number_of < limit_value:
                
                # getting filename and extension separeted 
                filename, extension = Utils.get_filename_and_extension(annotation_file)              

                # adding image name to list 
                images_list.append(filename)

                # updating counter per class
                statistics[input_dataset_type][bounding_box.class_title]['number_of'] = number_of + 1

        # removing item from list 
        annotations.pop(index)

        # check if finish tark  
        if is_task_finish(parameters, statistics, input_dataset_type):
            logging_info(f'Finish before pass into all images list - ')
            logging_info(f'input_dataset_type: {input_dataset_type}  images_list: {len(images_list)}')
            logging_info(f'')
            break
    
    logging_info(f'')
    logging_info(f'get_balanced_dataset_per_dataset_type')
    logging_info(f'input_dataset_type: {input_dataset_type}  images_list: {len(images_list)}')
    logging_info(f'images_list: {images_list}')    

    # returning images list 
    return images_list


def save_sheet_balanced_image_dataset(path_and_filename, balanced_image_dataset, dimension):

    # preparing list to save sheet 
    list = []
    for image_name in balanced_image_dataset[dimension]['train_images']:
        item = [dimension, 'train', image_name]
        list.append(item)

    for image_name in balanced_image_dataset[dimension]['valid_images']:
        item = [dimension, 'valid', image_name]
        list.append(item)

    for image_name in balanced_image_dataset[dimension]['test_images']:
        item = [dimension, 'test', image_name]
        list.append(item)

    # preparing columns name to list
    column_names = [
        'dimension', 
        'dataset type',
        'image name',
    ]

    # creating dataframe from list 
    df = pd.DataFrame(list, columns=column_names)

    # writing excel file from dataframe
    df.to_excel(path_and_filename, sheet_name='balanced_image_dataset', index=False)


def build_balanced_image_dataset_ssd_pascal_voc(
    parameters, dimension, balanced_image_dataset, input_dataset_type):

    logging_info(f'')
    logging_info(f'Build balanced image dataset for SSD Model')
    number_of = len(balanced_image_dataset[dimension][input_dataset_type + '_images'])
    logging_info(f'dimension: {dimension}  type: {input_dataset_type}  images: {number_of}')

    # preparing type of dataset 
    if input_dataset_type == 'train': dataset_type_folder = 'train_folder'
    if input_dataset_type == 'valid': dataset_type_folder = 'valid_folder'
    if input_dataset_type == 'test': dataset_type_folder = 'test_folder'

    # preparing input and output folder names
    input_dataset_type_folder = os.path.join(
        parameters['results']['output_dataset']['ssd_model_with_pascal_voc_format']['main_folder'],
        dimension,
        parameters['results']['output_dataset']['ssd_model_with_pascal_voc_format'][dataset_type_folder]
    )
    output_dataset_type_folder = os.path.join(
        parameters['results']['balanced_output_dataset']['ssd_model_with_pascal_voc_format']['main_folder'],
        dimension,
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

    # copying images, annotations and bbox images 
    images_names_list = balanced_image_dataset[dimension][input_dataset_type + '_images']
    train_images_list = [i + ".jpg" for i in images_names_list]
    copy_images(input_dataset_type_folder, output_dataset_type_folder, train_images_list)
    annotations_list = [i + ".xml" for i in images_names_list]
    copy_images(input_dataset_type_folder, output_dataset_type_folder, annotations_list)
    if parameters['input']['draw_and_save_bounding_box']:
        bbox_images_list = [i + "-drawn.jpg" for i in images_names_list]
        copy_images(input_bbox_folder, output_bbox_folder, bbox_images_list)

def build_balanced_image_dataset_faster_rcnn(
    parameters, dimension, balanced_image_dataset, input_dataset_type):

    logging_info(f'')
    logging_info(f'Build balanced image dataset for Faster RCNN')
    logging_info(f'dimension: {dimension}  type: {input_dataset_type}')
    logging_info(f'train: {len(balanced_image_dataset["train_images"])}')
    number_of = len(balanced_image_dataset[dimension][input_dataset_type + '_images'])
    logging_info(f'dimension: {dimension}  type: {input_dataset_type}  images: {number_of}')

    # preparing type of dataset 
    if input_dataset_type == 'train': dataset_type_folder = 'train_folder'
    if input_dataset_type == 'valid': dataset_type_folder = 'valid_folder'
    if input_dataset_type == 'test': dataset_type_folder = 'test_folder'

    # preparing input and output folder names
    input_dataset_type_folder = os.path.join(
        parameters['results']['output_dataset']['faster_rcnn_model']['main_folder'],
        dimension,
        parameters['results']['output_dataset']['faster_rcnn_model'][dataset_type_folder]
    )
    output_dataset_type_folder = os.path.join(
        parameters['results']['balanced_output_dataset']['faster_rcnn_model']['main_folder'],
        dimension,
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

    # copying images, annotations and bbox images 
    images_names_list = balanced_image_dataset[dimension][input_dataset_type + '_images']
    train_images_list = [i + ".jpg" for i in images_names_list]
    copy_images(input_dataset_type_folder, output_dataset_type_folder, train_images_list)
    annotations_list = [i + ".xml" for i in images_names_list]
    copy_images(input_dataset_type_folder, output_dataset_type_folder, annotations_list)
    if parameters['input']['draw_and_save_bounding_box']:
        bbox_images_list = [i + "-drawn.jpg" for i in images_names_list]
        copy_images(input_bbox_folder, output_bbox_folder, bbox_images_list)

def build_balanced_image_dataset_yolo(
    parameters, dimension, balanced_image_dataset, input_dataset_type):

    logging_info(f'')
    logging_info(f'Build balanced image dataset for YOLO v8')
    number_of = len(balanced_image_dataset[dimension][input_dataset_type + '_images'])
    logging_info(f'dimension: {dimension}  type: {input_dataset_type}  images: {number_of}')

    # preparing type of dataset 
    dataset_type_folder = input_dataset_type

    # preparing input and output folder names
    input_dataset_type_folder = os.path.join(
        parameters['results']['output_dataset']['yolov8_model']['main_folder'],
        dimension,
        parameters['results']['output_dataset']['yolov8_model'][dataset_type_folder],
    )
    output_dataset_type_folder = os.path.join(
        parameters['results']['balanced_output_dataset']['yolov8_model']['main_folder'],
        dimension,
        parameters['results']['balanced_output_dataset']['yolov8_model'][dataset_type_folder],
    )

    input_images_dataset_type_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['output_dataset']['yolov8_model']['images_folder'],
    )
    output_images_dataset_type_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['balanced_output_dataset']['yolov8_model']['images_folder'],
    )

    input_labels_dataset_type_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['output_dataset']['yolov8_model']['labels_folder'],
    )
    output_labels_dataset_type_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['balanced_output_dataset']['yolov8_model']['labels_folder'],
    )

    input_bboxes_dataset_type_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['output_dataset']['yolov8_model']['bounding_box_folder'],
    )
    output_bboxes_dataset_type_folder = os.path.join(
        input_dataset_type_folder,
        parameters['results']['balanced_output_dataset']['yolov8_model']['bounding_box_folder'],
    )

    # copying images, annotations and bbox images 
    images_names_list = balanced_image_dataset[dimension][input_dataset_type + '_images']
    train_images_list = [i + ".jpg" for i in images_names_list]
    copy_images(input_images_dataset_type_folder, output_images_dataset_type_folder, train_images_list)
    annotations_list = [i + ".xml" for i in images_names_list]
    copy_images(input_labels_dataset_type_folder, output_labels_dataset_type_folder, annotations_list)
    if parameters['input']['draw_and_save_bounding_box']:
        bbox_images_list = [i + "-drawn.jpg" for i in images_names_list]
        copy_images(input_bboxes_dataset_type_folder, output_bboxes_dataset_type_folder, bbox_images_list)



# ####################################################################################



# def copy_dataset_by_input_dataset_type_ssd_pascal_voc(parameters, statistics, height, width, input_dataset_type):

#     # preparing input folder names
#     input_main_folder = os.path.join(
#         parameters['results']['output_dataset']['ssd_model_with_pascal_voc_format']['main_folder'],
#         str(height) + 'x' + str(width)
#     )
#     if input_dataset_type == 'train': dataset_type_folder = 'train_folder'
#     if input_dataset_type == 'valid': dataset_type_folder = 'valid_folder'
#     if input_dataset_type == 'test': dataset_type_folder = 'test_folder'

#     input_dataset_type_folder = os.path.join(
#         input_main_folder,
#         parameters['results']['output_dataset']['ssd_model_with_pascal_voc_format'][dataset_type_folder]
#     )

#     # preparing output folder names
#     output_main_folder = os.path.join(
#         parameters['results']['balanced_output_dataset']['ssd_model_with_pascal_voc_format']['main_folder'],
#         str(height) + 'x' + str(width)
#     )
#     output_dataset_type_folder = os.path.join(
#         output_main_folder,
#         parameters['results']['balanced_output_dataset']['ssd_model_with_pascal_voc_format'][dataset_type_folder]
#     )

#     input_bbox_folder = os.path.join(
#         input_dataset_type_folder,
#         parameters['results']['output_dataset']['ssd_model_with_pascal_voc_format']['bounding_box_folder']
#     )
#     output_bbox_folder = os.path.join(
#         output_dataset_type_folder,
#         parameters['results']['balanced_output_dataset']['ssd_model_with_pascal_voc_format']['bounding_box_folder']
#     )        

#     # getting list of images
#     annotations = Utils.get_files_with_extensions(input_dataset_type_folder, 'xml')

#     logging_info(f'len annotations: {len(annotations)}')
#     logging_info(f'annotations: {annotations}')
#     number_of_annotations = len(annotations)
#     img = 0
#     for _ in range(number_of_annotations):
#         # getting an item randomly
#         index = randrange(len(annotations))
#         annotation_file = annotations[index]

#         # read annotation file 
#         path_and_filename_annotation = os.path.join(input_dataset_type_folder, annotation_file)
#         image_annotation = ImageAnnotation()
#         image_annotation.get_annotation_file_in_voc_pascal_format(path_and_filename_annotation)
        
#         img += 1
#         logging_info(f'img: {img} filename: {annotation_file}   bbox: {len(image_annotation.bounding_boxes)}')

#         # processing bounding boxes 
#         for bounding_box in image_annotation.bounding_boxes:          
#             if statistics[input_dataset_type][bounding_box.class_title]['number_of'] < \
#                parameters['processing']['balance_of_images_per_class'] \
#                          ['number_of_images_for_'+input_dataset_type]:
                
#                 # getting filename and extension separeted 
#                 filename, extension = Utils.get_filename_and_extension(annotation_file)              

#                 # copying files 
#                 filename_to_copy = filename + '.jpg'
#                 Utils.copy_file_same_name(filename_to_copy, input_dataset_type_folder, output_dataset_type_folder)
#                 Utils.copy_file_same_name(annotation_file, input_dataset_type_folder, output_dataset_type_folder)
#                 if parameters['input']['draw_and_save_bounding_box']:
#                     filename_to_copy = filename + '-drawn.jpg'
#                     Utils.copy_file_same_name(filename_to_copy, input_bbox_folder, output_bbox_folder)

#                 # updating counter per class
#                 number_of = statistics[input_dataset_type][bounding_box.class_title]['number_of']
#                 statistics[input_dataset_type][bounding_box.class_title]['number_of'] = number_of + 1
#                 # logging_info(f'copiou imagem: {filename}  ' \
#                 #              f'  input_dataset_type: {input_dataset_type}  ' \
#                 #              f'  bounding_box.class_title: {bounding_box.class_title}  ' \
#                 #              f'  number_of: {number_of}'
#                 # )

#         # removing item from list 
#         annotations.pop(index)

# def copy_dataset_by_input_dataset_type_faster_rcnn(parameters, statistics, height, width, input_dataset_type):

#     # preparing input folder names
#     input_main_folder = os.path.join(
#         parameters['results']['output_dataset']['faster_rcnn_model']['main_folder'],
#         str(height) + 'x' + str(width)
#     )
#     if input_dataset_type == 'train': dataset_type_folder = 'train_folder'
#     if input_dataset_type == 'valid': dataset_type_folder = 'valid_folder'
#     if input_dataset_type == 'test': dataset_type_folder = 'test_folder'

#     input_dataset_type_folder = os.path.join(
#         input_main_folder,
#         parameters['results']['output_dataset']['faster_rcnn_model'][dataset_type_folder]
#     )

#     # preparing output folder names
#     output_main_folder = os.path.join(
#         parameters['results']['balanced_output_dataset']['faster_rcnn_model']['main_folder'],
#         str(height) + 'x' + str(width)
#     )
#     output_dataset_type_folder = os.path.join(
#         output_main_folder,
#         parameters['results']['balanced_output_dataset']['faster_rcnn_model'][dataset_type_folder]
#     )

#     input_bbox_folder = os.path.join(
#         input_dataset_type_folder,
#         parameters['results']['output_dataset']['faster_rcnn_model']['bounding_box_folder']
#     )              
#     output_bbox_folder = os.path.join(
#         output_dataset_type_folder,
#         parameters['results']['balanced_output_dataset']['faster_rcnn_model']['bounding_box_folder']
#     )        

#     # getting list of images
#     annotations = Utils.get_files_with_extensions(input_dataset_type_folder, 'xml')

#     logging_info(f'len annotations: {len(annotations)}')
#     logging_info(f'annotations: {annotations}')
#     number_of_annotations = len(annotations)
#     img = 0
#     for _ in range(number_of_annotations):
#         # getting an item randomly
#         index = randrange(len(annotations))
#         annotation_file = annotations[index]

#         # read annotation file 
#         path_and_filename_annotation = os.path.join(input_dataset_type_folder, annotation_file)
#         image_annotation = ImageAnnotation()
#         image_annotation.get_annotation_file_in_voc_pascal_format(path_and_filename_annotation)
        
#         img += 1
#         logging_info(f'img: {img} filename: {annotation_file}   bbox: {len(image_annotation.bounding_boxes)}')

#         # processing bounding boxes 
#         for bounding_box in image_annotation.bounding_boxes:           
#             if statistics[input_dataset_type][bounding_box.class_title]['number_of'] < \
#                parameters['processing']['balance_of_images_per_class'] \
#                          ['number_of_images_for_'+input_dataset_type]:
                
#                 # getting filename and extension separeted 
#                 filename, extension = Utils.get_filename_and_extension(annotation_file)                

#                 # copying files 
#                 filename_to_copy = filename + '.jpg'
#                 Utils.copy_file_same_name(filename_to_copy, input_dataset_type_folder, output_dataset_type_folder)
#                 Utils.copy_file_same_name(annotation_file, input_dataset_type_folder, output_dataset_type_folder)
#                 if parameters['input']['draw_and_save_bounding_box']:
#                     filename_to_copy = filename + '-drawn.jpg'
#                     Utils.copy_file_same_name(filename_to_copy, input_bbox_folder, output_bbox_folder)

#                 # updating counter per class
#                 number_of = statistics[input_dataset_type][bounding_box.class_title]['number_of']
#                 statistics[input_dataset_type][bounding_box.class_title]['number_of'] = number_of + 1
#                 # logging_info(f'copiou imagem: {filename}  ' \
#                 #              f'  input_dataset_type: {input_dataset_type}  ' \
#                 #              f'  bounding_box.class_title: {bounding_box.class_title}  ' \
#                 #              f'  number_of: {number_of}'
#                 # )

#         # removing item from list 
#         annotations.pop(index)

# def copy_dataset_by_input_dataset_type_yolov8(parameters, statistics, height, width, input_dataset_type):

#     # preparing input folder names
#     input_main_folder = os.path.join(
#         parameters['results']['output_dataset']['yolov8_model']['main_folder'],
#         str(height) + 'x' + str(width)
#     )
#     if input_dataset_type == 'train': dataset_type_folder = 'train_folder'
#     if input_dataset_type == 'valid': dataset_type_folder = 'valid_folder'
#     if input_dataset_type == 'test': dataset_type_folder = 'test_folder'

#     input_dataset_type_folder = os.path.join(
#         input_main_folder,
#         parameters['results']['output_dataset']['yolov8_model'][dataset_type_folder]
#     )

#     # preparing output folder names
#     output_main_folder = os.path.join(
#         parameters['results']['balanced_output_dataset']['yolov8_model']['main_folder'],
#         str(height) + 'x' + str(width)
#     )
#     output_dataset_type_folder = os.path.join(
#         output_main_folder,
#         parameters['results']['balanced_output_dataset']['yolov8_model'][dataset_type_folder]
#     )

#     # preparing input folder names for labels, images and bbox
#     input_images_folder = os.path.join(
#         input_dataset_type_folder,
#         parameters['results']['output_dataset']['yolov8_model']['images_folder']
#     )        
#     input_labels_folder = os.path.join(
#         input_dataset_type_folder,
#         parameters['results']['output_dataset']['yolov8_model']['labels_folder']
#     )        
#     input_bbox_folder = os.path.join(
#         input_dataset_type_folder,
#         parameters['results']['output_dataset']['yolov8_model']['bounding_box_folder']
#     )        

#     # preparing output folder names for labels, images and bbox
#     output_images_folder = os.path.join(
#         output_dataset_type_folder,
#         parameters['results']['balanced_output_dataset']['yolov8_model']['images_folder']
#     )        
#     output_labels_folder = os.path.join(
#         output_dataset_type_folder,
#         parameters['results']['balanced_output_dataset']['yolov8_model']['labels_folder']
#     )        
#     output_bbox_folder = os.path.join(
#         output_dataset_type_folder,
#         parameters['results']['balanced_output_dataset']['yolov8_model']['bounding_box_folder']
#     )        

#     # getting list of images
#     annotations = Utils.get_files_with_extensions(input_labels_folder, 'txt')

#     logging_info(f'len annotations: {len(annotations)}')
#     logging_info(f'annotations: {annotations}')
#     number_of_annotations = len(annotations)
#     img = 0
#     for _ in range(number_of_annotations):
#         # getting an item randomly
#         index = randrange(len(annotations))
#         annotation_file = annotations[index]

#         # read annotation file 
#         path_and_filename_yolo_annotation = os.path.join(input_labels_folder, annotation_file)
#         image_annotation = ImageAnnotation()
#         image_annotation.get_annotation_file_in_yolo_v5_format(
#             path_and_filename_yolo_annotation, 
#             parameters['neural_network_model']['classes'],
#             height, width)
        
#         img += 1
#         logging_info(f'img: {img} filename: {annotation_file}   bbox: {len(image_annotation.bounding_boxes)}')

#         # processing bounding boxes 
#         for bounding_box in image_annotation.bounding_boxes:           
#             if statistics[input_dataset_type][bounding_box.class_title]['number_of'] < \
#                parameters['processing']['balance_of_images_per_class'] \
#                          ['number_of_images_for_'+input_dataset_type]:
                
#                 # getting filename and extension separeted 
#                 filename, extension = Utils.get_filename_and_extension(annotation_file)              

#                 # copying files 
#                 filename_to_copy = filename + '.jpg'
#                 Utils.copy_file_same_name(filename_to_copy, input_images_folder, output_images_folder)
#                 Utils.copy_file_same_name(annotation_file, input_labels_folder, output_labels_folder)
#                 if parameters['input']['draw_and_save_bounding_box']:
#                     filename_to_copy = filename + '-drawn.jpg'
#                     Utils.copy_file_same_name(filename_to_copy, input_bbox_folder, output_bbox_folder)

#                 # updating counter per class
#                 number_of = statistics[input_dataset_type][bounding_box.class_title]['number_of']
#                 statistics[input_dataset_type][bounding_box.class_title]['number_of'] = number_of + 1
#                 # logging_info(f'copiou imagem: {filename}  ' \
#                 #              f'  input_dataset_type: {input_dataset_type}  ' \
#                 #              f'  bounding_box.class_title: {bounding_box.class_title}  ' \
#                 #              f'  number_of: {number_of}'
#                 # )

#         # removing item from list 
#         annotations.pop(index)

# ###########################################
# Methods of Level 3
# ###########################################

def is_task_finish(parameters, statistics, input_dataset_type):

    is_finish = True

    classes = parameters['neural_network_model']['classes']
    for class_name in classes:
        if class_name == '__background__': continue
        
        selected_value = statistics[input_dataset_type][class_name]['number_of']        
        limit_value = parameters['input']['balance_of_images_per_class'][class_name] \
                                ['number_for_'+input_dataset_type]
        if selected_value < limit_value:
            is_finish = False
            break

    # returning result of evaluation 
    return is_finish

def copy_images(input_folder, outputfolder, images_list):
    for image_filename in images_list:
        Utils.copy_file_same_name(image_filename, input_folder, outputfolder)
