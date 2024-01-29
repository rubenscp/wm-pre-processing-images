"""
Project: White Mold 
Description: Crop input images in small size according by parameters and 
             split dataset by bounding boxes
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 27/11/2023
Version: 2.0
"""

# Importing libraries
import json
import os
from pathlib import Path

# Importing frameworks
# import torch

# Importing python modules
from manage_log import *
from utils import Utils
from random import randrange
import parameter as parameter 
from split_original_dataset import * 
from crop_bbox_ssd_model import * 
from crop_bbox_ssd_model_pascal_voc_format import * 
from crop_bbox_faster_rcnn_model import * 
from crop_bbox_yolo_v8_model import * 

# Importing entity classes
from entity.ImageAnnotation import ImageAnnotation

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

# ###########################################
# Application Methods
# ###########################################

# ###########################################
# Methods of Level 1
# ###########################################

def main():
    """
    Main method that processes all annotadoted images by cropping each bounding box to a 
    small size (height and width) positioning the object in the central position of the new 
    image.

    All values of the parameters used here are defined at the "parameter.py".

    """
    # creating log file 
    logging_create_log()

    logging_info('White Mold Project ')
    logging_info('Cropp original images from dataset' + LINE_FEED)
    print('White Mold Project ')
    print('Cropp original images from dataset' + LINE_FEED)
    
    # setting dictionary initial parameters for processing
    logging_info('1) Setting processing parameters' + LINE_FEED)
    print('1) Setting processing parameters' + LINE_FEED)
    processing_parameters = set_processing_parameters()
    processing_statistics = set_processing_statistics(processing_parameters)
    
    # creating all working folders 
    logging_info('2) Creating working folders' + LINE_FEED)
    print('2) Creating working folders' + LINE_FEED)
    create_working_folders()

    # getting object classes
    image_input_supervisely_path = os.path.join(parameter.INPUT_PATH, parameter.INPUT_SUPERVISELY_FORMAT)
    classes_dict, classes_statistics = get_object_classes(image_input_supervisely_path, parameter.META_FILE)    

    # getting the list of images for training, validaton and testing selected randomly
    # to use in all tasks of image cropping for all dimensions
    logging_info('3) Creating list of bounding boxes from original annotated images' + LINE_FEED)
    print('3) Creating list of bounding boxes from original annotated images' + LINE_FEED)
    train_bbox_list, valid_bbox_list, test_bbox_list, \
    train_bbox_df, valid_bbox_df, test_bbox_df = \
        create_bbox_list_from_original_dataset(processing_parameters, processing_statistics)

    # evalute models for processing 
    if 'ssd' in processing_parameters['models']:
        logging_info('')
        logging_info('4.1) Cropping bounding box image for SSD model' + LINE_FEED)
        print('4.1) Cropping bounding box image for SSD model' + LINE_FEED)
        crop_bbox_list_for_ssd_model(processing_parameters, 
                                     train_bbox_list,
                                     valid_bbox_list,
                                     test_bbox_list, 
                                     processing_statistics)
    
    if 'ssd_pascal_voc' in processing_parameters['models']:
        logging_info('')
        logging_info('4.2) Cropping bounding box image for SDD model with Pascal VOC format' + LINE_FEED)
        print('4.2) Cropping bounding box image for SDD model with Pascal VOC format' + LINE_FEED)
        crop_bbox_list_for_ssd_pascal_voc_model(processing_parameters,
                                                train_bbox_list, 
                                                valid_bbox_list, 
                                                test_bbox_list,
                                                processing_statistics)
                                        
    if 'faster_rcnn' in processing_parameters['models']:
        logging_info('')
        logging_info('4.3) Cropping bounding box image for Faster R-CNN model' + LINE_FEED)
        print('4.3) Cropping bounding box image for Faster R-CNN model' + LINE_FEED)
        crop_bbox_list_for_faster_rcnn_model(processing_parameters, 
                                             train_bbox_list, 
                                             valid_bbox_list, 
                                             test_bbox_list,
                                             processing_statistics)

    if 'yolov8' in processing_parameters['models']:
        logging_info('')
        logging_info('4.4) Cropping bounding box image for YOLOv8 model' + LINE_FEED)
        print('4.4) Cropping bounding box image for YOLOv8 model' + LINE_FEED)
        crop_bbox_list_for_yolo_v8_model(processing_parameters,
                                         train_bbox_list, 
                                         valid_bbox_list, 
                                         test_bbox_list,
                                         processing_statistics)
        
    # zipping ouput diretories 
    logging_info('')
    logging_info('5) Zipping output folders' + LINE_FEED)
    print('5) Zipping output folders' + LINE_FEED)
    zip_output_folders(processing_parameters)

    # logging processing statistics 
    logging_info('')
    logging_info('6) Processing statistics' + LINE_FEED)
    print('6) Processing statistics' + LINE_FEED)
    log_processing_statistics(processing_parameters,
                              processing_statistics)

    # end of processing   
    logging_info('')
    logging_info('End of processing')
    print('End of processing')


# ###########################################
# Methods of Level 2
# ###########################################

def set_processing_parameters():
    '''
    Set dictionary parameters for processing
    '''    
    processing_parameters = {}
    
    # setting models to prepare images 
    processing_parameters['models'] = ['ssd', 'ssd_pascal_voc', 'faster_rcnn', 'yolov8']
    # processing_parameters['models'] = ['ssd_pascal_voc']
    # processing_parameters['models'] = ['ssd']
    # processing_parameters['models'] = ['yolov8']
    # processing_parameters['models'] = []

    # setting the classes selected to crop annotations     
    # processing_parameters["classes"] = ['Coal', 'Mature Sclerotium', 'Imature Sclerotium', 'Disease White Mold', 'Apothecium', 'Petal', 'Mushroom']
    processing_parameters['classes'] = ['Apothecium']
    
    # setting dimensions for cropping original images 
    processing_parameters['dimensions'] = [(64,64), (128,128), (256,256), (512,512), (1024,1024)]
    # processing_parameters['dimensions'] = [(128,128)]

    # setting percent for split training, validation and testing datasets
    processing_parameters['split_dataset'] = {
        'train' : 80,
        'valid' : 15,
        'test' : 5
        }

    # setting percent for split training, validation and testing datasets
    processing_parameters['bounding_box'] = {
        'draw_and_save' : False,
        }

    # returning parameters 
    return processing_parameters

def set_processing_statistics(processing_parameters):
    '''
    Set dictionary for the processing statistics
    '''  

    # creating dicionary 
    processing_statistics = {}
    processing_statistics['models'] = {}
    processing_statistics['original_image_size'] = {}

    # buiding dicionary 
    for model in processing_parameters['models']:
        processing_statistics['models'][model] = {}
        for heigth, width in processing_parameters['dimensions']:
            processing_statistics['models'][model][heigth] = {}
            processing_statistics['models'][model][heigth]['train'] = {}
            processing_statistics['models'][model][heigth]['train']['success'] = 0
            processing_statistics['models'][model][heigth]['train']['error'] = 0
            processing_statistics['models'][model][heigth]['valid'] = {}
            processing_statistics['models'][model][heigth]['valid']['success'] = 0
            processing_statistics['models'][model][heigth]['valid']['error'] = 0
            processing_statistics['models'][model][heigth]['test'] = {}
            processing_statistics['models'][model][heigth]['test']['success'] = 0
            processing_statistics['models'][model][heigth]['test']['error'] = 0

    # returning processing statistics
    return processing_statistics

# Read classes of objects 
def get_object_classes(input_annotations_path, filename):

    # defining working objects    
    classes_statistics = {}

    # setting path and filename of class names 
    classes_path_and_filename = os.path.join(input_annotations_path, filename)

    # reading file with class names in json format
    with open(classes_path_and_filename) as file:
        classes_dict = json.load(file)

    # creating classes statistics
    for object_class in classes_dict["classes"]:        
        item = {object_class["title"]: 0}
        classes_statistics.update(item)

    # returning classes dictionaries
    return classes_dict, classes_statistics
    
# Create all working folders 
def create_working_folders():  

    # removing output folders 
    Utils.remove_directory(parameter.OUTPUT_SPLIT_DATASET_PATH)
    Utils.remove_directory(parameter.OUTPUT_MODEL_PATH)
    Utils.remove_directory(parameter.ZIP_PATH)
    
    # creating output folder for all images and annotations together 
    Utils.create_directory(parameter.OUTPUT_ALL_IMAGES_AND_ANNOTATIONS)
    
    # creating output folder for split dataset 
    Utils.create_directory(parameter.OUTPUT_SPLIT_DATASET_PATH)

    # Output folder for model processing results
    Utils.create_directory(parameter.OUTPUT_MODEL_PATH)

    # Zip folder for model processing results
    Utils.create_directory(parameter.ZIP_PATH)

def zip_output_folders(processing_parameters):
    # create zip files from output directories

    for model in processing_parameters['models']:
        for height, width in processing_parameters['dimensions']:

            logging_info(f'Creating zipfile of image dataset ...')

            source_directory = os.path.join(parameter.OUTPUT_MODEL_PATH,
                                            model, str(height) + 'x' + str(width))            
            output_filename = os.path.join(parameter.ZIP_PATH,
                                           parameter.ZIP_FILENAME + 
                                           '_' + model + '_' + str(height) + 'x' + str(width))
            result, full_output_filename = Utils.zip_directory(source_directory, output_filename)
            if result:
                print(f'Zipfile of image dataset')
                print()
                print(f'Source directory: {source_directory}')
                print(f'Output filename : {full_output_filename}')

                logging_info(f'Zipfile of image dataset')
                logging_info(f'Source directory: {source_directory}')
                logging_info(f'Output filename : {full_output_filename}')
            else:
                logging_error(f'Error in creating of the zipfile of image dataset!')

def log_processing_statistics(processing_parameters, 
                              processing_statistics):

    # inicitializing counters 
    # total_of_success = 0
    # total_of_error = 0

    # logging processing statistics 
    for model in processing_parameters['models']:          
        for heigth, width in processing_parameters['dimensions']:
            model_total_of_success = 0
            model_total_of_error = 0

            number_of_success = processing_statistics['models'][model][heigth]['train']['success']
            number_of_error   = processing_statistics['models'][model][heigth]['train']['error'] 
            model_total_of_success  += number_of_success
            model_total_of_error    += number_of_error
            logging_info(f'{model} - {heigth} - train - success: {number_of_success}  error : {number_of_error}')

            number_of_success = processing_statistics['models'][model][heigth]['valid']['success']
            number_of_error   = processing_statistics['models'][model][heigth]['valid']['error'] 
            model_total_of_success  += number_of_success
            model_total_of_error    += number_of_error
            logging_info(f'{model} - {heigth} - valid - success: {number_of_success}  error : {number_of_error}')

            number_of_success = processing_statistics['models'][model][heigth]['test']['success']
            number_of_error   = processing_statistics['models'][model][heigth]['test']['error'] 
            model_total_of_success  += number_of_success
            model_total_of_error    += number_of_error
            logging_info(f'{model} - {heigth} - test  - success: {number_of_success}  error : {number_of_error}')
            logging_info(f'{model} - {heigth} - total - success: {model_total_of_success}  error : {model_total_of_error}')
            logging_info(f'')

        # logging_info(f'')
        # logging_info(f'{model} - total - success: {model_total_of_success}  error : {model_total_of_error}')
        # total_of_success += model_total_of_success
        # total_of_error   += model_total_of_error

    # logging total 
    # logging_info(f'')
    # logging_info(f'Total - success: {total_of_success}  error : {total_of_error}')

    logging_info(f'')
    logging_info(f'Statistics of original image size (heigth and width)')
    logging_info(f'')
    total_of_original_images = 0
    for key in processing_statistics['original_image_size']:
        logging_info(f"{key}: {str(processing_statistics['original_image_size'].get(key))}")
        total_of_original_images += processing_statistics['original_image_size'].get(key)

    logging_info(f'Total of original images: {total_of_original_images}')

# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
