"""
Project: White Mold 
Description: Crop bbox images and save for SSD Model
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 27/11/2023
Version: 1.0
"""

# Importing libraries
import os

# Importing python modules
from manage_log import *
from utils import Utils
from image_utils import ImageUtils
from random import randrange

# Importing entity classes
from entity.ImageAnnotation import ImageAnnotation

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

def crop_bbox_list_for_ssd_model(
        parameters, 
        train_bbox_list, valid_bbox_list, test_bbox_list, 
        processing_statistics):
    
    # creating working folders 
    create_working_folders(parameters)

    # cropping bbox images from train, valid and test lists    
    for item in parameters['input']['dimensions']:
        height = item['height']
        width  = item['width']

        logging_info('')
        logging_info(f'SSD Model - processing cropping window (HxW): ({height},{width})')
        logging_info(f'-'*50 + LINE_FEED)

        target_folder = os.path.join(
            parameters['results']['output_dataset']['ssd_model']['main_folder'],
            str(height) + 'x' + str(width),
            parameters['results']['output_dataset']['ssd_model']['train_folder']
        )
        number_of_sucess, number_of_errors = \
            crop_bbox_images_from_list(parameters, train_bbox_list, 
                                       height, width, target_folder)
        processing_statistics['models']['ssd'][height]['train']['success'] = number_of_sucess
        processing_statistics['models']['ssd'][height]['train']['error']   = number_of_errors
        
        target_folder = os.path.join(
            parameters['results']['output_dataset']['ssd_model']['main_folder'],
            str(height) + 'x' + str(width),
            parameters['results']['output_dataset']['ssd_model']['valid_folder']
        )      
        number_of_sucess, number_of_errors = \
            crop_bbox_images_from_list(parameters, valid_bbox_list, 
                                       height, width, target_folder)
        processing_statistics['models']['ssd'][height]['valid']['success'] = number_of_sucess
        processing_statistics['models']['ssd'][height]['valid']['error']   = number_of_errors

        target_folder = os.path.join(
            parameters['results']['output_dataset']['ssd_model']['main_folder'],
            str(height) + 'x' + str(width),
            parameters['results']['output_dataset']['ssd_model']['test_folder']
        )
        number_of_sucess, number_of_errors = \
            crop_bbox_images_from_list(parameters, test_bbox_list, 
                                       height, width, target_folder)
        processing_statistics['models']['ssd'][height]['test']['success'] = number_of_sucess
        processing_statistics['models']['ssd'][height]['test']['error']   = number_of_errors

# ###########################################
# Methods of Level 2
# ###########################################

# Create all working folders 
# Create all working folders 
def create_working_folders(parameters):

    # creating output folders
    for item in parameters['input']['dimensions']:
        height = item['height']
        width  = item['width']
        folder = os.path.join(
            parameters['results']['output_dataset']['output_dataset_folder'],
            parameters['results']['output_dataset']['ssd_model']['main_folder'],
            str(height) + 'x' + str(width),
            parameters['results']['output_dataset']['ssd_model']['train_folder']
        )   
      
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results']['output_dataset']['ssd_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            parameters['results']['output_dataset']['output_dataset_folder'],
            parameters['results']['output_dataset']['ssd_model']['main_folder'],
            str(height) + 'x' + str(width),
            parameters['results']['output_dataset']['ssd_model']['valid_folder']
        )         
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results']['output_dataset']['ssd_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)

        folder = os.path.join(
            parameters['results']['output_dataset']['output_dataset_folder'],
            parameters['results']['output_dataset']['ssd_model']['main_folder'],
            str(height) + 'x' + str(width),
            parameters['results']['output_dataset']['ssd_model']['test_folder']
        )         
        Utils.create_directory(folder)
        if parameters['input']['draw_and_save_bounding_box']:
            bbox_folder = os.path.join(
                folder,
                parameters['results']['output_dataset']['ssd_model']['bounding_box_folder']
            )             
            Utils.create_directory(bbox_folder)
            
# Crop bbox images 
def crop_bbox_images_from_list(parameters, bbox_list, 
                               crop_height, crop_width, target_folder):
    
    logging_info(f'crop_bbox_images_from_list ' + 
                 f'crop_height: {crop_height} ' + 
                 f'crop_width: {crop_width} ' + 
                 f'target_folder: {target_folder}')                     

    # setting auxiliary variables 
    number_of_sucess = 0
    number_of_errors = 0    
    
    # setting image and annotation folders 
    # image_target_folder = os.path.join(target_folder, parameter.SSD_MODEL_IMAGES)
    image_target_folder = target_folder
    annotation_target_folder = target_folder 
    bbox_target_folder = os.path.join(
        target_folder, 
        parameters['results']['output_dataset']['ssd_model']['bounding_box_folder']
    )

    # processing all bounding boxes 
    for bbox_item in bbox_list:

        # setting objects 
        bounding_box = bbox_item[0]
        image_annotation = bbox_item[1]

        # evaluating bounding box size related to cropped window size
        eval_bbox_result, eval_bbox_status = \
            bounding_box.evaluate_bbox_size_at_cropping_window(crop_height, crop_width)
        if not eval_bbox_result:
            logging_error(f'Img {image_annotation.image_name_with_extension} bbox {bounding_box.id}' + 
                          f' size ({bounding_box.get_height()},{bounding_box.get_width()})' + 
                          f' greater than' + 
                          f' cropping window size ({crop_height},{crop_width})' +
                          f' class {bounding_box.class_title}' +
                          f' status error: {eval_bbox_status}' +
                          f' original image folder: {image_annotation.original_image_folder}')            
            number_of_errors += 1
            continue

        # reading the original image 
        image_name = image_annotation.image_name_with_extension
        input_folder = os.path.join(
            parameters['results']['all_images']
        )
        image = ImageUtils.read_image(image_annotation.image_name_with_extension, input_folder)

        # calculating the new coordinates of the new cropped image 
        result, linP1, colP1, linP2, colP2 = \
            ImageUtils.calculate_coordinates_new_cropped_image(
                image_name, image, bounding_box, crop_height, crop_width)

        # evaluating if is possible create the cropped bounding box
        if not result:
            number_of_errors += 1
            continue 

        if (linP1 < 0 or colP1 < 0 or linP2 < 0 or colP2 < 0):
            logging_error(f'It\'s not possible to create a new cropped image from {image_name} {bounding_box.id} at ({linP1},{colP1}) or ({linP2},{colP2})')
            number_of_errors += 1
            continue
        
        # getting new cropped image 
        new_image = image[linP1:linP2, colP1:colP2]
        new_image_height = new_image.shape[0]
        new_image_width  = new_image.shape[1]

        # creating new annotation to new cropped image
        cropped_image_supervisely_annotation = ImageAnnotation()
        cropped_image_supervisely_annotation.set_annotation_of_cropped_image(
            image_annotation.image_name,
            image_annotation.image_name_with_extension,
            image_target_folder, 
            new_image_height, 
            new_image_width, 
            bounding_box)
        cropped_image_supervisely_annotation.update_coordinates_of_bounding_box(linP1, colP1)

        # checking the size of cropped image 
        if cropped_image_supervisely_annotation.height != crop_height  or  \
           cropped_image_supervisely_annotation.width  != crop_width:
            text = f'Cropped image with different size (HxW) of ({crop_height},{crop_width}): '
            text += f'{cropped_image_supervisely_annotation.image_name_with_extension}' + \
                    f' - bbox: {bounding_box.id}' + \
                    f' ({cropped_image_supervisely_annotation.height},{cropped_image_supervisely_annotation.width})'
            logging_error(text)
            number_of_errors += 1
            continue

        # checking consistency of bounding box coordinates
        result = cropped_image_supervisely_annotation.bounding_boxes[0]. \
                    check_consistency_of_coordinates(
                        image_annotation.image_name_with_extension,
                        crop_height, crop_width)
        
        if result != '':
            logging_error(f'Bounding box with inconsistent coordinates: {result}')
            number_of_errors += 1
            continue 

        # -------------------------------------
        # SSD Model
        # -------------------------------------

        # saving new cropped image in Supervisely images folder 
        filename_cropped_image_SSD = image_annotation.image_name + '-bbox-' + f'{bounding_box.id}' + '.jpg'
        path_and_filename_cropped_image_SSD = os.path.join(image_target_folder, filename_cropped_image_SSD)
        ImageUtils.save_image(path_and_filename_cropped_image_SSD, new_image)
        
        # save annoatations file for SSD implementation 
        # path_and_filename_annotation_SSD = os.path.join(annotation_target_folder, parameter.SSD_MODEL_ANNOTATION_FILENAME)
        path_and_filename_annotation_SSD = os.path.join(
            annotation_target_folder,
            parameters['results']['output_dataset']['ssd_model']['annotation_file']
        )
        save_annotation_file_SSD(path_and_filename_annotation_SSD, 
                                 filename_cropped_image_SSD,
                                 bounding_box.get_id_class_SSD(),
                                 cropped_image_supervisely_annotation.bounding_boxes[0].col_point1,
                                 cropped_image_supervisely_annotation.bounding_boxes[0].lin_point1,
                                 cropped_image_supervisely_annotation.bounding_boxes[0].col_point2,
                                 cropped_image_supervisely_annotation.bounding_boxes[0].lin_point2
                                 )

        # adding to number of cropped images create with sucess 
        number_of_sucess += 1

        # drawing and saving new images with bounding box
        if parameters['input']['draw_and_save_bounding_box']:
            # setting path and image filename
            new_image_filename_bbox_drawed = \
                image_annotation.image_name + '-bbox-' + f'{bounding_box.id}' + '-drawn' + '.jpg'
            path_and_filename_new_image = os.path.join(
                bbox_target_folder, new_image_filename_bbox_drawed)
            
            # drawing bounding box in new image
            draw_and_save_bounding_box(
                new_image, 
                new_image_height, new_image_width, 
                str(bounding_box.get_id_class_SSD()),
                cropped_image_supervisely_annotation.bounding_boxes[0].lin_point1,
                cropped_image_supervisely_annotation.bounding_boxes[0].col_point1,
                cropped_image_supervisely_annotation.bounding_boxes[0].lin_point2,
                cropped_image_supervisely_annotation.bounding_boxes[0].col_point2,
                path_and_filename_new_image
            )
        
    return number_of_sucess, number_of_errors

def save_annotation_file_SSD(path_and_filename_annotation_SSD, 
                             cropped_image_name,
                             id_class,
                             col_point1, lin_point1, 
                             col_point2, lin_point2 
                             ):
    ''' 
    Save one annotation according by used in SSD implementation.
    '''

    # cheking if exists file and create it with header
    if not os.path.exists(path_and_filename_annotation_SSD):
        create_annotation_file_SSD_with_header(path_and_filename_annotation_SSD)
        
    # opening annotation file 
    annotation_file = open(path_and_filename_annotation_SSD, 'a+')

    # setting line of bounding box 
    line =  LINE_FEED \
            + cropped_image_name + ',' \
            + str(id_class) + ',' \
            + str(col_point1) + ',' \
            + str(lin_point1) + ',' \
            + str(col_point2) + ',' \
            + str(lin_point2)

    # write line
    annotation_file.write(line)

    # closing file
    annotation_file.close()

# Save one annotation according by used in SSD implementation
def create_annotation_file_SSD_with_header(path_and_filename_annotation_SSD):

    # cheking if exists file and create it with header
    if os.path.exists(path_and_filename_annotation_SSD):
        return True

    # opening annotation file 
    annotation_file = open(path_and_filename_annotation_SSD, 'a+')

    # setting line with header
    line = 'img_name,label,xmin,ymin,xmax,ymax'

    # write line
    annotation_file.write(line)

    # closing file
    annotation_file.close()

# Draw and save bounding box at image 
def draw_and_save_bounding_box(image, 
                               image_height, image_width,
                               bbox_label,
                               lin_point1, col_point1,
                               lin_point2, col_point2, 
                               path_and_filename_image
                               ):   

    # creating new image to check the new coordinates of bounding box
    bgrBoxColor = [0, 0, 255]  # red color
    thickness = 1
    label = bbox_label
    new_image = image.copy()
    new_image_with_bbox_drawn = ImageUtils.draw_bounding_box(
        new_image, 
        lin_point1, 
        col_point1, 
        lin_point2, 
        col_point2, 
        bgrBoxColor, 
        thickness,
        label)
    ImageUtils.save_image(path_and_filename_image, new_image_with_bbox_drawn)
