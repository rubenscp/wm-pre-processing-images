"""
Project: White Mold 
Description: Crop input images in small size according by parameters and 
             split dataset by images.
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 15/12/2023
Version: 2.0
"""

# Importing libraries
import json
import os
from pathlib import Path

# Importing python modules
from common.manage_log import *
from common.utils import Utils
from common.entity.ImageAnnotation import ImageAnnotation
from common.tasks import Tasks

from random import randrange
from split_original_dataset import * 
from crop_bbox_ssd_model import * 
from crop_bbox_ssd_model_pascal_voc_format import * 
from crop_bbox_faster_rcnn_model import * 
from crop_bbox_yolo_v8_model import *
from balance_original_dataset import * 


# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'
NEW_FILE = True

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

    # creating Tasks object 
    processing_tasks = Tasks()

    # setting dictionary initial parameters for processing
    full_path_project = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-pre-processing-images'

    # getting application parameters
    processing_tasks.start_task('Getting application parameters')
    parameters_filename = 'wm_pre_processing_images_parameters.json'
    parameters = get_parameters(full_path_project, parameters_filename)
    processing_tasks.finish_task('Getting application parameters')

    # getting last running id
    processing_tasks.start_task('Getting running id')
    running_id = get_running_id(parameters)
    processing_tasks.finish_task('Getting running id')

    # setting output folder results
    processing_tasks.start_task('Setting result folders')
    set_results_folder(parameters)
    processing_tasks.finish_task('Setting result folders')

    # creating log file 
    processing_tasks.start_task('Creating log file')
    logging_create_log(
        parameters['results']['log_folder'], 
        parameters['results']['log_filename']
    )
    processing_tasks.finish_task('Creating log file')

    logging_info('White Mold Research')
    logging_info('Pre-processing Original Images' + LINE_FEED)
    logging_info('Crop original images from dataset splitting by images' + LINE_FEED)

    logging_info(f'')
    logging_info(f'>> Get running id')
    logging_info(f'running id: {str(running_id)}')   
    logging_info(f'')
    logging_info(f'>> Set result folders')

    # creating new instance of parameters file related to current running
    processing_tasks.start_task('Saving processing parameters')
    save_processing_parameters(parameters_filename, parameters)
    processing_tasks.finish_task('Saving processing parameters')

    # creating processing statistics
    processing_tasks.start_task('Creating processing statistics')
    processing_statistics = set_processing_statistics(parameters)
    processing_tasks.finish_task('Creating processing statistics')

    # getting object classes
    image_input_supervisely_path = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['input']['main_dataset_folder'],
        parameters['input']['input_dataset_folder'],
        parameters['input']['supervisely']['original_images_folder'],
    )
    classes_dict, classes_statistics = get_object_classes(
        image_input_supervisely_path,
        parameters['input']['supervisely']['meta_file'],
    )    
    logging.info(f'Classes: {Utils.get_pretty_json(classes_dict)}')

    # getting the list of images for training, validaton and testing selected randomly
    # to use in all tasks of image cropping for all dimensions
    processing_tasks.start_task('Creating list of bboxes from original annotated images')
    logging_info('Creating list of bounding boxes from original annotated images' + LINE_FEED)
    train_bbox_list, valid_bbox_list, test_bbox_list, \
    train_bbox_df, valid_bbox_df, test_bbox_df = \
        create_bbox_list_from_original_dataset(parameters, processing_statistics)
    processing_tasks.finish_task('Creating list of bboxes from original annotated images')

    # evalute models for processing 
    for model in parameters['input']['models']:
    
        if model == 'ssd':
            processing_tasks.start_task('Cropping bounding box image for SSD model')
            logging_info('')
            logging_info('4.1) Cropping bounding box image for SSD model' + LINE_FEED)
            crop_bbox_list_for_ssd_model(parameters, 
                                        train_bbox_list,
                                        valid_bbox_list,
                                        test_bbox_list, 
                                        processing_statistics)
            processing_tasks.finish_task('Cropping bounding box image for SSD model')
        
        if model == 'ssd_pascal_voc':
            processing_tasks.start_task('Cropping bounding box image for SSD model with Pascal VOC')
            logging_info('')
            logging_info('4.2) Cropping bounding box image for SDD model with Pascal VOC format' + LINE_FEED)
            crop_bbox_list_for_ssd_pascal_voc_model(parameters,
                                                    train_bbox_list, 
                                                    valid_bbox_list, 
                                                    test_bbox_list,
                                                    processing_statistics)
            processing_tasks.finish_task('Cropping bounding box image for SSD model with Pascal VOC')
                                            
        if model == 'faster_rcnn':
            processing_tasks.start_task('Cropping bounding box image for Faster R-CNN model')
            logging_info('')
            logging_info('4.3) Cropping bounding box image for Faster R-CNN model' + LINE_FEED)
            crop_bbox_list_for_faster_rcnn_model(parameters, 
                                                train_bbox_list, 
                                                valid_bbox_list, 
                                                test_bbox_list,
                                                processing_statistics)
            processing_tasks.finish_task('Cropping bounding box image for Faster R-CNN model')

        if model == 'yolov8':
            processing_tasks.start_task('Cropping bounding box image for YOLOv8 model')
            logging_info('')
            logging_info('4.4) Cropping bounding box image for YOLOv8 model' + LINE_FEED)
            crop_bbox_list_for_yolo_v8_model(parameters,
                                            train_bbox_list, 
                                            valid_bbox_list, 
                                            test_bbox_list,
                                            processing_statistics)
            processing_tasks.finish_task('Cropping bounding box image for YOLOv8 model')
        
    # zipping ouput diretories 
    if parameters['input']['create_zipfile']:    
        processing_tasks.start_task('Zipping outpu directories')
        logging_info('')
        logging_info('5) Zipping output folders' + LINE_FEED)
        zip_output_folders(parameters)
        processing_tasks.finish_task('Zipping outpu directories')

    # logging processing statistics 
    logging_info('')
    logging_info('6) Processing statistics' + LINE_FEED)
    log_processing_statistics(parameters,
                              processing_statistics)

    # create new image dataset with balanced number of images
    processing_tasks.start_task('Creating balanced image dataset')
    create_dataset_with_balanced_number_of_images(parameters)
    processing_tasks.finish_task('Creating balanced image dataset')

    # # getting statistics of input dataset
    # processing_tasks.start_task('Getting statistics of input dataset')
    # annotation_statistics = get_input_dataset_statistics(parameters)
    # show_input_dataset_statistics(parameters, annotation_statistics)
    # processing_tasks.finish_task('Getting statistics of input dataset')

    # end of processing   
    logging_info('')
    logging_info('Finished the cropping bounding boxes by images' + LINE_FEED)

    # printing tasks summary 
    processing_tasks.finish_processing()
    logging_info(processing_tasks.to_string())

# ###########################################
# Methods of Level 2
# ###########################################

def get_parameters(full_path_project, parameters_filename):
    '''
    Get dictionary parameters for processing
    '''    
    # getting parameters 
    path_and_parameters_filename = os.path.join(full_path_project, parameters_filename)
    parameters = Utils.read_json_parameters(path_and_parameters_filename)
    
    # returning parameters 
    return parameters

def get_running_id(parameters):
    '''
    Get last running id to calculate the current id
    '''    
    # setting control filename 
    running_control_filename = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['processing']['project_name_folder'],
        parameters['processing']['running_control_filename'],
    )

    # getting control info 
    running_control = Utils.read_json_parameters(running_control_filename)

    # calculating the current running id 
    running_control['last_running_id'] = int(running_control['last_running_id']) + 1

    # updating running control file 
    running_id = int(running_control['last_running_id'])

    # saving file 
    Utils.save_text_file(running_control_filename, \
                         Utils.get_pretty_json(running_control), 
                         NEW_FILE)

    # updating running id in the processing parameters 
    parameters['processing']['running_id'] = running_id
 
    # returning running id 
    return running_id

def set_results_folder(parameters):
    '''
    Set folder name of output results
    '''
    # setting runnig id text 
    running_id = parameters['processing']['running_id']
    running_id_text = 'running-' + f'{running_id:04}'
    parameters['results']['running_folder'] = running_id_text

    # creating results folders 
    running_folder = os.path.join(
        parameters['processing']['research_root_folder'],     
        parameters['input']['main_dataset_folder'],
        parameters['results']['main_folder'],
        parameters['results']['running_folder'],
    )
    parameters['results']['running_folder'] = running_folder
    Utils.create_directory(running_folder)

    # setting and creating log folder 
    log_folder = os.path.join(
        running_folder,
        parameters['results']['log_folder'],
    )
    parameters['results']['log_folder'] = log_folder
    Utils.create_directory(log_folder)

    # setting and creating parameter folder 
    processing_parameters_folder = os.path.join(
        running_folder,
        parameters['results']['processing_parameters_folder'],
    )
    parameters['results']['processing_parameters_folder'] = processing_parameters_folder
    Utils.create_directory(processing_parameters_folder)

    # setting and creating all images folder 
    all_images = os.path.join(
        running_folder,
        parameters['results']['all_images'],
    )
    parameters['results']['all_images'] = all_images
    Utils.create_directory(all_images)

    # setting and creating splitting dataset folder
    if  parameters['input']['image_dataset_spliting_criteria'] == 'images':
        criteria_splitting = parameters['results']['criteria_splitting']['images']
    else:
        criteria_splitting = parameters['results']['criteria_splitting']['bounding_boxes']
    splitting_dataset_folder = os.path.join(
        running_folder,
        criteria_splitting,
    )
    parameters['results']['splitting_dataset']['splitting_dataset_folder'] = splitting_dataset_folder
    Utils.create_directory(splitting_dataset_folder)   

    list_folder = os.path.join(
        splitting_dataset_folder,
        parameters['results']['splitting_dataset']['list_folder'],
    )
    parameters['results']['splitting_dataset']['list_folder'] = list_folder
    Utils.create_directory(list_folder)

    output_dataset_folder = os.path.join(
        splitting_dataset_folder,
        parameters['results']['output_dataset']['output_dataset_folder'],
    )
    parameters['results']['output_dataset']['output_dataset_folder'] = output_dataset_folder
    Utils.create_directory(output_dataset_folder)

    balanced_output_dataset_folder = os.path.join(
        splitting_dataset_folder,
        parameters['results']['balanced_output_dataset']['balanced_output_dataset_folder'],
    )
    parameters['results']['balanced_output_dataset']['balanced_output_dataset_folder'] = balanced_output_dataset_folder
    Utils.create_directory(balanced_output_dataset_folder)

    # setting and creating models and image size folder folder
    for model in parameters['input']['models']:
        if model == 'ssd':
            output_balanced_dataset = 'output_dataset'
            output_balanced_folder = 'output_dataset_folder'
            set_ssd_model_folders(parameters, output_balanced_dataset, output_balanced_folder)

        if model == 'ssd_pascal_voc':
            output_balanced_dataset = 'output_dataset'
            output_balanced_folder = 'output_dataset_folder'
            set_ssd_model_with_pascal_voc_format_folders(parameters, output_balanced_dataset, output_balanced_folder)
            output_balanced_dataset = 'balanced_output_dataset'
            output_balanced_folder = 'balanced_output_dataset_folder'
            set_ssd_model_with_pascal_voc_format_folders(parameters, output_balanced_dataset, output_balanced_folder)

        if model == 'faster_rcnn':
            output_balanced_dataset = 'output_dataset'
            output_balanced_folder = 'output_dataset_folder'
            set_faster_rcnn_model_folders(parameters, output_balanced_dataset, output_balanced_folder)
            output_balanced_dataset = 'balanced_output_dataset'
            output_balanced_folder = 'balanced_output_dataset_folder'
            set_faster_rcnn_model_folders(parameters, output_balanced_dataset, output_balanced_folder)

        if model == 'yolov8':
            output_balanced_dataset = 'output_dataset'
            output_balanced_folder = 'output_dataset_folder'
            set_yolov8_model_folders(parameters, output_balanced_dataset, output_balanced_folder)
            output_balanced_dataset = 'balanced_output_dataset'
            output_balanced_folder = 'balanced_output_dataset_folder'
            set_yolov8_model_folders(parameters, output_balanced_dataset, output_balanced_folder)

    # setting and creating folder of zipped files
    if parameters['input']['create_zipfile']:    
        zip_main_folder = os.path.join(
            splitting_dataset_folder,
            parameters['results']['output_dataset']['zip']['main_folder'],
        )
        parameters['results']['output_dataset']['zip']['main_folder'] = zip_main_folder
        Utils.create_directory(zip_main_folder)

def save_processing_parameters(parameters_filename, parameters):
    '''
    Update parameters file of the processing
    '''    
    # setting full path and log folder  to write parameters file 
    path_and_parameters_filename = os.path.join(
        parameters['results']['processing_parameters_folder'], 
        parameters_filename)

    # saving current processing parameters in the log folder 
    Utils.save_text_file(path_and_parameters_filename, \
                        Utils.get_pretty_json(parameters), 
                        NEW_FILE)

def set_processing_statistics(parameters):
    '''
    Set dictionary for the processing statistics
    '''  

    # creating dicionary 
    processing_statistics = {}
    processing_statistics['models'] = {}
    processing_statistics['original_image_size'] = {}

    # buiding dicionary 
    for model in parameters['input']['models']:
        processing_statistics['models'][model] = {}
        for item in parameters['input']['dimensions']:
            height = item['height']
            width  = item['width']
            processing_statistics['models'][model][height] = {}
            processing_statistics['models'][model][height]['train'] = {}
            processing_statistics['models'][model][height]['train']['success'] = 0
            processing_statistics['models'][model][height]['train']['error'] = 0
            processing_statistics['models'][model][height]['valid'] = {}
            processing_statistics['models'][model][height]['valid']['success'] = 0
            processing_statistics['models'][model][height]['valid']['error'] = 0
            processing_statistics['models'][model][height]['test'] = {}
            processing_statistics['models'][model][height]['test']['success'] = 0
            processing_statistics['models'][model][height]['test']['error'] = 0

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
def set_folder_by_split_criteria(processing_parameters, parameter):

    if processing_parameters['image_dataset_spliting_criteria'] == 'images':
        parameter.OUTPUT_IMAGE_DATASET_SPLIT_CRITERIA_PATH += \
            parameter.OUTPUT_IMAGE_DATASET_SPLIT_BY_IMAGES
        parameter.ZIP_FILENAME += parameter.ZIP_FILENAME_SPLIT_BY_IMAGES

    if processing_parameters['image_dataset_spliting_criteria'] == 'bounding_boxes':
        parameter.OUTPUT_IMAGE_DATASET_SPLIT_CRITERIA_PATH += \
            parameter.OUTPUT_IMAGE_DATASET_SPLIT_BY_BOUNDING_BOXES
        parameter.ZIP_FILENAME += parameter.ZIP_FILENAME_SPLIT_BY_BOUNDING_BOXES     
        
    parameter.OUTPUT_SPLIT_DATASET_PATH = parameter.OUTPUT_IMAGE_DATASET_SPLIT_CRITERIA_PATH + \
        parameter.OUTPUT_SPLIT_DATASET_PATH
    parameter.OUTPUT_MODEL_PATH = parameter.OUTPUT_IMAGE_DATASET_SPLIT_CRITERIA_PATH + \
        parameter.OUTPUT_MODEL_PATH    
    parameter.ZIP_PATH = parameter.OUTPUT_IMAGE_DATASET_SPLIT_CRITERIA_PATH + \
        parameter.ZIP_PATH
    

# Create all working folders 
def create_working_folders(processing_parameters):  

    # removing output folders 
    Utils.remove_directory(parameter.OUTPUT_ALL_IMAGES_AND_ANNOTATIONS)
    Utils.remove_directory(parameter.OUTPUT_IMAGE_DATASET_SPLIT_CRITERIA_PATH)
    Utils.remove_directory(parameter.OUTPUT_MODEL_PATH)
    Utils.remove_directory(parameter.ZIP_PATH)
    
    # creating output folder for all images and annotations together 
    Utils.create_directory(parameter.OUTPUT_ALL_IMAGES_AND_ANNOTATIONS)
    
    # creating output folder according by splitting criteria
    Utils.create_directory(parameter.OUTPUT_IMAGE_DATASET_SPLIT_CRITERIA_PATH)

    # creating output folder for split dataset 
    Utils.create_directory(parameter.OUTPUT_SPLIT_DATASET_PATH)

    # Output folder for model processing results
    Utils.create_directory(parameter.OUTPUT_MODEL_PATH)

    # Zip folder for model processing results
    Utils.create_directory(parameter.ZIP_PATH)

def zip_output_folders(parameters):
    # create zip files from output directories

    for model in parameters['input']['models']:        
        for item in parameters['input']['dimensions']:
            height = item['height']
            width  = item['width']

            logging_info(f'Creating zipfile of image dataset ...')

            if model == 'ssd': output_dataset_model_key = 'ssd_model'
            if model == 'ssd_pascal_voc': output_dataset_model_key = 'ssd_model_with_pascal_voc_format'
            if model == 'faster_rcnn': output_dataset_model_key = 'faster_rcnn_model'
            if model == 'yolov8': output_dataset_model_key = 'yolov8_model'
            
            source_directory = os.path.join(
                parameters['results']['output_dataset'][output_dataset_model_key]['main_folder'],
                str(height) + 'x' + str(width),
            )
            # source_directory = os.path.join(parameter.OUTPUT_MODEL_PATH,
            #                                 model, str(height) + 'x' + str(width))            
            output_filename = os.path.join(
                parameters['results']['output_dataset']['zip']['main_folder'],
                parameters['results']['output_dataset']['zip']['filename'] + '_' + model + '_' + str(height) + 'x' + str(width)
            )
            # output_filename = os.path.join(parameter.ZIP_PATH,
            #                                parameter.ZIP_FILENAME + 
            #                                '_' + model + '_' + str(height) + 'x' + str(width))
            result, full_output_filename = Utils.zip_directory(source_directory, output_filename)
            if result:
                logging_info(f'Zipfile of image dataset')
                logging_info(f'Source directory: {source_directory}')
                logging_info(f'Output filename : {full_output_filename}')
            else:
                logging_error(f'Error in creating of the zipfile of image dataset!')

def create_dataset_with_balanced_number_of_images(parameters):

    # checking creating balanced dataset activated
    if not parameters['input']['balance_of_images_per_class']['active']:
        logging_info(f'Creation of dataset with balanced number of images is INACTIVE')
        return 

    # create balanced image dataset according by parameters 
    create_balanced_image_dataset(parameters)
   

def log_processing_statistics(parameters, 
                              processing_statistics):

    # inicitializing counters 
    # total_of_success = 0
    # total_of_error = 0

    # logging processing statistics 
    for model in parameters['input']['models']:        
        for item in parameters['input']['dimensions']:
            height = item['height']
            width  = item['width']

            model_total_of_success = 0
            model_total_of_error = 0

            number_of_success = processing_statistics['models'][model][height]['train']['success']
            number_of_error   = processing_statistics['models'][model][height]['train']['error'] 
            model_total_of_success  += number_of_success
            model_total_of_error    += number_of_error
            logging_info(f'{model} - {height} - train - success: {number_of_success}  error : {number_of_error}')

            number_of_success = processing_statistics['models'][model][height]['valid']['success']
            number_of_error   = processing_statistics['models'][model][height]['valid']['error'] 
            model_total_of_success  += number_of_success
            model_total_of_error    += number_of_error
            logging_info(f'{model} - {height} - valid - success: {number_of_success}  error : {number_of_error}')

            number_of_success = processing_statistics['models'][model][height]['test']['success']
            number_of_error   = processing_statistics['models'][model][height]['test']['error'] 
            model_total_of_success  += number_of_success
            model_total_of_error    += number_of_error
            logging_info(f'{model} - {height} - test  - success: {number_of_success}  error : {number_of_error}')
            logging_info(f'{model} - {height} - total - success: {model_total_of_success}  error : {model_total_of_error}')
            logging_info(f'')

        # logging_info(f'')
        # logging_info(f'{model} - total - success: {model_total_of_success}  error : {model_total_of_error}')
        # total_of_success += model_total_of_success
        # total_of_error   += model_total_of_error

    # logging total 
    # logging_info(f'')
    # logging_info(f'Total - success: {total_of_success}  error : {total_of_error}')

    logging_info(f'')
    logging_info(f'Statistics of original image size (height and width)')
    logging_info(f'')
    total_of_original_images = 0
    for key in processing_statistics['original_image_size']:
        logging_info(f"{key}: {str(processing_statistics['original_image_size'].get(key))}")
        total_of_original_images += processing_statistics['original_image_size'].get(key)

    # logging_info(f'Total of original images: {total_of_original_images}')


# # getting statistics of input dataset 
# def get_input_dataset_statistics(parameters):
    
#     annotation_statistics = AnnotationsStatistic()
#     steps = ['train', 'valid', 'test'] 
#     annotation_statistics.processing_statistics(parameters, steps)
#     return annotation_statistics

# def show_input_dataset_statistics(parameters, annotation_statistics):

#     logging_info(f'Input dataset statistic')
#     logging_info(annotation_statistics.to_string())
#     path_and_filename = os.path.join(
#         parameters['training_results']['metrics_folder'],
#         parameters['neural_network_model']['model_name'] + '_annotations_statistics.xlsx',
#     )
#     annotation_format = parameters['input']['input_dataset']['annotation_format']
#     input_image_size = parameters['input']['input_dataset']['input_image_size']
#     classes = (parameters['neural_network_model']['classes'])[1:5]
#     annotation_statistics.save_annotations_statistics(
#         path_and_filename,
#         annotation_format,
#         input_image_size,
#         classes
#     )

# ###########################################
# Methods of Level 3
# ###########################################

def set_ssd_model_folders(parameters, output_balanced_dataset, output_balanced_folder):
    '''
    Create folder for results of SSD model 
    '''    

    model_folder = os.path.join(
        parameters['results'][output_balanced_dataset][output_balanced_folder],
        parameters['results'][output_balanced_dataset]['ssd_model']['main_folder'],
    )
    parameters['results'][output_balanced_dataset]['ssd_model']['main_folder'] = model_folder
    Utils.create_directory(model_folder)

    for item in parameters['input']['dimensions']:            
        height = item['height']
        width  = item['width']

        # train folder 
        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['ssd_model']['train_folder']
        )
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['ssd_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['ssd_model']['valid_folder']
        )   
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['ssd_model']['bounding_box_folder']
            ) 
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['ssd_model']['test_folder']
        )   
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['ssd_model']['bounding_box_folder']
            )
            Utils.create_directory(bbox_folder)


def set_ssd_model_with_pascal_voc_format_folders(parameters, output_balanced_dataset, output_balanced_folder):
    '''
    Create folder for results of SSD model 
    '''    

    model_folder = os.path.join(
        parameters['results'][output_balanced_dataset][output_balanced_folder],
        parameters['results'][output_balanced_dataset]['ssd_model_with_pascal_voc_format']['main_folder'],
    )
    parameters['results'][output_balanced_dataset]['ssd_model_with_pascal_voc_format']['main_folder'] = model_folder
    Utils.create_directory(model_folder)

    for item in parameters['input']['dimensions']:            
        height = item['height']
        width  = item['width']

        # train folder 
        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['ssd_model_with_pascal_voc_format']['train_folder']
        )
    
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['ssd_model_with_pascal_voc_format']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['ssd_model_with_pascal_voc_format']['valid_folder']
        )   
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['ssd_model_with_pascal_voc_format']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['ssd_model_with_pascal_voc_format']['test_folder']
        )   
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['ssd_model_with_pascal_voc_format']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

def set_faster_rcnn_model_folders(parameters, output_balanced_dataset, output_balanced_folder):
    '''
    Create folder for results of Faster RCNN model 
    '''    

    model_folder = os.path.join(
        parameters['results'][output_balanced_dataset][output_balanced_folder],
        parameters['results'][output_balanced_dataset]['faster_rcnn_model']['main_folder'],
    )
    parameters['results'][output_balanced_dataset]['faster_rcnn_model']['main_folder'] = model_folder
    Utils.create_directory(model_folder)

    for item in parameters['input']['dimensions']:            
        height = item['height']
        width  = item['width']

        # train folder 
        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['faster_rcnn_model']['train_folder']
        )
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['faster_rcnn_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['faster_rcnn_model']['valid_folder']
        )   
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['faster_rcnn_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['faster_rcnn_model']['test_folder']
        )   
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['faster_rcnn_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)


def set_yolov8_model_folders(parameters, output_balanced_dataset, output_balanced_folder):
    '''
    Create folder for results of Faster RCNN model 
    '''    

    model_folder = os.path.join(
        parameters['results'][output_balanced_dataset][output_balanced_folder],
        parameters['results'][output_balanced_dataset]['yolov8_model']['main_folder'],
    )
    parameters['results'][output_balanced_dataset]['yolov8_model']['main_folder'] = model_folder
    Utils.create_directory(model_folder)

    for item in parameters['input']['dimensions']:            
        height = item['height']
        width  = item['width']

        # train folder 
        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['yolov8_model']['train_folder']
        )
        Utils.create_directory(folder)
        images_folder = os.path.join(
            folder, 
            parameters['results'][output_balanced_dataset]['yolov8_model']['images_folder'],
        )
        Utils.create_directory(images_folder)
        labels_folder = os.path.join(
            folder, 
            parameters['results'][output_balanced_dataset]['yolov8_model']['labels_folder'],
        )
        Utils.create_directory(labels_folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['yolov8_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['yolov8_model']['valid_folder']
        )   
        Utils.create_directory(folder)
        images_folder = os.path.join(
            folder, 
            parameters['results'][output_balanced_dataset]['yolov8_model']['images_folder'],
        )
        Utils.create_directory(images_folder)
        labels_folder = os.path.join(
            folder, 
            parameters['results'][output_balanced_dataset]['yolov8_model']['labels_folder'],
        )
        Utils.create_directory(labels_folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['yolov8_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            model_folder,
            str(height) + 'x' + str(width),
            parameters['results'][output_balanced_dataset]['yolov8_model']['test_folder']
        )   
        Utils.create_directory(folder)
        images_folder = os.path.join(
            folder, 
            parameters['results'][output_balanced_dataset]['yolov8_model']['images_folder'],
        )
        Utils.create_directory(images_folder)
        labels_folder = os.path.join(
            folder, 
            parameters['results'][output_balanced_dataset]['yolov8_model']['labels_folder'],
        )
        Utils.create_directory(labels_folder)   
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results'][output_balanced_dataset]['yolov8_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)


# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    main()
