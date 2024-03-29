"""
Project: White Mold 
Description: Create lists of bouding boxes images to be cropped 
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 27/11/2023
Version: 1.0
"""

# Importing libraries
import json
import os
import pandas as pd
# import openpyxl
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

def create_bbox_list_from_original_dataset(parameters, processing_statistics):
    '''
    Create list of bounding box from original annotated images and 
    split list for training, validation and testing according parameters.
    '''

    # creating list of images with bounding boxes selected
    images_with_annotations = create_lists_image_original_dataset(parameters, processing_statistics)

    # creating list of bounding box images splitted into training, validation and 
    # testing according by parameters 

    # selecting criteria for split all the images dataset 
    if parameters['input']['image_dataset_spliting_criteria'] == 'images':
        logging_info('')
        logging_info('Splitting original image dataset according by the images criteria')
        logging_info('')
        all_bbox_list, train_bbox_list, valid_bbox_list, test_bbox_list = \
            create_lists_splitting_by_images(parameters, images_with_annotations)

    elif parameters['input']['image_dataset_spliting_criteria'] == 'bounding_boxes':
        logging_info('')
        logging_info('Splitting original image dataset according by the bounding boxes criteria')
        logging_info('')
        all_bbox_list, train_bbox_list, valid_bbox_list, test_bbox_list = \
            create_lists_splitting_by_bounding_boxes(parameters, images_with_annotations)

    else:   
        logging_info('')
        logging_info('No valid criteria for splitting original image dataset' + LINE_FEED)
        logging_info('')
        return None, None, None, None, None, None, 

    # draw violin plot of bounding boxes
    violin_plot_filename = os.path.join(
        parameters['results']['splitting_dataset']['splitting_dataset_folder'],
        parameters['results']['splitting_dataset']['violin_plot_file']
    )
    draw_violin_plot_of_bounding_boxes(all_bbox_list, violin_plot_filename)

    # saving lists 
    train_filename = os.path.join(
        parameters['results']['splitting_dataset']['list_folder'],
        parameters['results']['splitting_dataset']['train_list_file']
    )
    save_bbox_lists(train_filename, train_bbox_list)
    train_bbox_df = evaluate_bounding_boxes_size(train_bbox_list)
    save_bbox_list_excel(train_filename, train_bbox_df)

    valid_filename = os.path.join(
        parameters['results']['splitting_dataset']['list_folder'],
        parameters['results']['splitting_dataset']['valid_list_file']
    )
    save_bbox_lists(valid_filename, valid_bbox_list)
    valid_bbox_df = evaluate_bounding_boxes_size(valid_bbox_list)
    save_bbox_list_excel(valid_filename, valid_bbox_df)

    test_filename = os.path.join(
        parameters['results']['splitting_dataset']['list_folder'],
        parameters['results']['splitting_dataset']['test_list_file']
    )
    save_bbox_lists(test_filename, test_bbox_list)
    test_bbox_df = evaluate_bounding_boxes_size(test_bbox_list)
    save_bbox_list_excel(test_filename, test_bbox_df)

    # returning list of bounding boxes splitted to be cropped for each model and dimensions 
    return train_bbox_list, valid_bbox_list, test_bbox_list, \
            train_bbox_df, valid_bbox_df, test_bbox_df

# ###########################################
# Methods of Level 2
# ###########################################

def create_lists_image_original_dataset(parameters, processing_statistics):
    '''
    Create lists of the images to be cropped according by format and dimensions 
    specified in the processing parameters.
    '''

    # creating working lists
    images_with_annotations = []

    # reading the original images in all datasets from Supervisely platform

    # setting input supervisely path 
    input_supervisely_path = os.path.join(
        parameters['processing']['research_root_folder'],
        parameters['input']['main_dataset_folder'],
        parameters['input']['input_dataset_folder'],
        parameters['input']['supervisely']['original_images_folder']
    )
     
    # getting list of image folders
    dataset_folders = Utils.get_folders(input_supervisely_path)

    count = 0
    for dataset_folder in dataset_folders:

        # setting annotation folder name 
        input_supervisely_image_path = os.path.join(
            input_supervisely_path,
            dataset_folder,
            parameters['input']['supervisely']['images']
        )

        input_supervisely_annotation_path = os.path.join(
            input_supervisely_path, 
            dataset_folder,
            parameters['input']['supervisely']['annotations']
        )  

        # getting annotation files 
        supervisely_annotation_files =  Utils.get_files_with_extensions(
            input_supervisely_annotation_path, '.json')

        count += 1
        logging_info(f'Processing dataset folder #{count}: {dataset_folder} - {len(supervisely_annotation_files)} images')

        # processing one annotation file
        for supervisely_annotation_file in supervisely_annotation_files:

            # getting suprvisely annotation of one image
            path_and_filename_image_annotation = os.path.join(input_supervisely_annotation_path, 
                                                              supervisely_annotation_file)

            # reading file with class names in json format
            with open(path_and_filename_image_annotation) as file:

                # getting json annotation
                annotation_json = json.load(file)

                # creating image annotation object 
                image_annotation = ImageAnnotation()

                # setting annotation fields
                image_filename = supervisely_annotation_file.replace('.json', '').replace('.jpg', '').replace('.jpeg', '')
                image_filename_with_extension = supervisely_annotation_file.replace('.json', '')
                image_annotation.set_annotation_fields_in_supervisely_format(
                    image_filename,
                    image_filename_with_extension,
                    supervisely_annotation_file,                   
                    annotation_json,
                    parameters['input']['classes'], 
                    dataset_folder)
    
            # adding image annotations if exist
            if (len(image_annotation.bounding_boxes) == 0):
                # number_of_images_with_no_annotations += 1
                continue
            else:   
                # adding annotation              
                images_with_annotations.append(image_annotation)

            # copying image and annotation files
            Utils.copy_file(image_filename_with_extension,
                            input_supervisely_image_path,
                            image_filename_with_extension,
                            parameters['results']['all_images'])            
            Utils.copy_file(supervisely_annotation_file,
                            input_supervisely_annotation_path,
                            supervisely_annotation_file,
                            parameters['results']['all_images'])            

            # updating numbe of images at same size (height and width)
            # update_statistics_of_image_size(image_annotation, processing_statistics)        

    # returning list of images with bounding boxes selected according criterias
    return images_with_annotations

def create_lists_splitting_by_images(parameters, image_annotations):
    '''
    Create lists of the bbox images to be split according by processing parameters.
    '''

    # creating working list 
    all_bbox_list = []
    train_bbox_list = []
    valid_bbox_list = []
    test_bbox_list  = []

    all_image_annotations_list = []
    train_image_annotations_list = []
    valid_image_annotations_list = []
    test_image_annotations_list = []

    # copying all images to a new list 
    all_image_annotations_list = image_annotations.copy()

    # creating list of all bounding boxes of all image datasets
    all_bbox_list = create_bbox_list_from_image_annotations(image_annotations)

    # setting splitting percentages to train, validating and testing
    train_percent = parameters['input']['split_dataset']['train']
    valid_percent = parameters['input']['split_dataset']['valid']
    test_percent  = parameters['input']['split_dataset']['test']
    total_percent = train_percent + valid_percent + test_percent

    # calcuting amount of images for training, validation and tes dataset 
    number_of_train_image = int(len(all_image_annotations_list) * train_percent / 100)
    number_of_valid_image = int(len(all_image_annotations_list) * valid_percent / 100)
    number_of_test_image  = len(all_image_annotations_list) - number_of_train_image - number_of_valid_image

    # spliting image dataset for test set 
    test_image_annotations_list = split_image_list_randomly(all_image_annotations_list, number_of_test_image)
    test_bbox_list = create_bbox_list_from_image_annotations(test_image_annotations_list)
    
    # spliting image dataset for validation set 
    valid_image_annotations_list = split_image_list_randomly(all_image_annotations_list, number_of_valid_image)
    valid_bbox_list = create_bbox_list_from_image_annotations(valid_image_annotations_list)
    
    # spliting image dataset for train set 
    train_image_annotations_list = split_image_list_randomly(all_image_annotations_list, number_of_train_image)    
    train_bbox_list = create_bbox_list_from_image_annotations(train_image_annotations_list)

    logging_info(f'Images estimated spliting of the Original Image Dataset:')
    logging_info(f'Training  : {parameters["input"]["split_dataset"]["train"]} % - ' + \
                 f'{number_of_train_image} original images ' + \
                 f'with {len(train_bbox_list)} bbox images ')
    logging_info(f'Validation: {parameters["input"]["split_dataset"]["valid"]} % - ' + \
                 f'{number_of_valid_image} original images ' + \
                 f'with {len(valid_bbox_list)} bbox images ')
    logging_info(f'Test      : {parameters["input"]["split_dataset"]["test"]} % - ' + \
                 f'{number_of_test_image} original images ' + \
                 f'with {len(test_bbox_list)} bbox images ')
    logging_info(f'Total     : {total_percent} % - {len(image_annotations)} original images ' + \
                 f'with {len(train_bbox_list) + len(valid_bbox_list) + len(test_bbox_list)} bbox images ' + LINE_FEED)

    # returning list of bbox 
    return all_bbox_list, train_bbox_list, valid_bbox_list, test_bbox_list

def create_lists_splitting_by_bounding_boxes(parameters, image_annotations):
    '''
    Create lists of the bbox images to be split according by processing parameters.
    '''

    # creating working list 
    all_bbox_list = []
    train_bbox_list = []
    valid_bbox_list = []
    test_bbox_list  = []

    # creating list of all bounding boxes of all image datasets
    all_bbox_list = create_bbox_list_from_image_annotations(image_annotations)
  
    # copying all bbox list to return 
    all_bbox_list_to_return = all_bbox_list.copy()

    # saving list file
    all_filename = os.path.join(parameter.OUTPUT_SPLIT_DATASET_PATH, 
                                parameter.OUTPUT_SPLIT_DATASET_PATH_ALL_LIST)   
    save_bbox_lists(all_filename, all_bbox_list)
    all_bbox_df = evaluate_bounding_boxes_size(all_bbox_list)
    save_bbox_list_excel(all_filename, all_bbox_df)

    train_percent = parameters["split_dataset"]["train"]
    valid_percent = parameters["split_dataset"]["valid"]
    test_percent  = parameters["split_dataset"]["test"]
    total_percent = train_percent + valid_percent + test_percent

    # calcuting amount of images for training, validation and tes dataset 
    number_of_train_bbox = int(len(all_bbox_list) * train_percent / 100)
    number_of_valid_bbox = int(len(all_bbox_list) * valid_percent / 100)
    number_of_test_bbox  = len(all_bbox_list) - number_of_train_bbox - number_of_valid_bbox

    logging_info('')
    logging_info(f'Spliting estimated of Bounding Boxes of Original Image Dataset:')
    logging_info(f'Training  : {parameters["split_dataset"]["train"]}% >> {number_of_train_bbox} cropped images')
    logging_info(f'Validation: {parameters["split_dataset"]["valid"]}% >> {number_of_valid_bbox} cropped images')
    logging_info(f'Test      : {parameters["split_dataset"]["test"]}% >> {number_of_test_bbox} cropped images')
    logging_info(f'Total     : {str(total_percent)} % >> {len(all_bbox_list)} images' + LINE_FEED)

    # spliting bbox for test 
    test_bbox_list = split_bbox_list_randomly(all_bbox_list, number_of_test_bbox)
    
    # spliting image dataset for validation 
    valid_bbox_list = split_bbox_list_randomly(all_bbox_list, number_of_valid_bbox)
    
    # spliting image dataset for train 
    train_bbox_list = split_bbox_list_randomly(all_bbox_list, number_of_train_bbox)    

    # returning list of bbox 
    return all_bbox_list_to_return, train_bbox_list, valid_bbox_list, test_bbox_list

# ###########################################
# Methods of Level 3
# ###########################################

def update_statistics_of_image_size(image_annotation, processing_statistics):
    key = '(' + str(image_annotation.height) + ',' + str(image_annotation.width) + ')'
    current_counting = processing_statistics['original_image_size'].get(key)
    if current_counting != None:
        processing_statistics['original_image_size'][key] += 1
    else:
        processing_statistics['original_image_size'][key] = 1

# Splits list of images for specific set 
def split_image_list_randomly(all_image_annotations, number_of_images):

    # setting image filenames list 
    image_annotations_list = []
    
    for _ in range(number_of_images):
        # getting the list index to select the annotation and imge 
        index = randrange(len(all_image_annotations))
        item = all_image_annotations[index]

        # adding image filename to list 
        image_annotations_list.append(item)
        
        # removing item from list 
        all_image_annotations.pop(index)

    # returning images filename list 
    return image_annotations_list

# Splits list of bounding boxes for specific set 
def split_bbox_list_randomly(all_bbox_and_image_annotations, number_of_images):

    # setting image filenames list 
    bbox_and_image_annotation_list = []
    
    for _ in range(number_of_images):
        # getting the list index to select the annotation and imge 
        index = randrange(len(all_bbox_and_image_annotations))
        item = all_bbox_and_image_annotations[index]

        # adding image filename to list 
        bbox_and_image_annotation_list.append(item)
        
        # removing item from list 
        all_bbox_and_image_annotations.pop(index)

    # returning images filename list 
    return bbox_and_image_annotation_list

def create_bbox_list_from_image_annotations(image_annotations):
    # initializing list of bounding boxes
    all_bbox_list = []

    # creating list of all bounding boxes of all image datasets
    for image_annotation in image_annotations:
        for bounding_box in image_annotation.bounding_boxes:
            all_bbox_list.append([bounding_box, image_annotation])
    
    # returning list of bounding boxes
    return all_bbox_list

def save_bbox_lists(path_and_list_filename, bbox_list):
    ''' 
    Save bbox lists.
    '''
    
    # setting auxiliary variable 
    first_item = True        

    # opening annotation file 
    bbox_list_file = open(path_and_list_filename, 'a+')

    # saving all bbox list 
    for bbox_item in bbox_list:        

        # setting line of bounding box
        line = ('' if first_item else LINE_FEED) + \
                bbox_item[1].image_name + ',' + f'{bbox_item[0].id}'        

        # write line
        bbox_list_file.write(line)

        # setting second item and so on 
        first_item = False

    # closing file
    bbox_list_file.close()


def evaluate_bounding_boxes_size(bbox_list):

    ''' 
    Evaluate bouding box size related to the window cropping  
    '''

    # preparing columns name to list
    column_names = [
        'image_name_with_extension',
        'height',
        'width',
        'annotation_name',
        'id',
        'class_id',
        'class_title',
        'geometry_type',
        'lin_point1',
        'col_point1',
        'lin_point2',
        'bbox_height',
        'bbox_width',   
        'bbox_area',       
        'cropping_64x64',
        'cropping_128x128',
        'cropping_256x256',
        'cropping_512x512',
        'cropping_1024x1024',
        'labeler_login',
        'created_at',
        'updated_at',
    ]

    # preparing list
    excel_list = []
    for bbox_item in bbox_list:
        eval_bbox_result, cropping_64x64 = bbox_item[0].evaluate_bbox_size_at_cropping_window(64, 64)
        eval_bbox_result, cropping_128x128 = bbox_item[0].evaluate_bbox_size_at_cropping_window(128, 128)
        eval_bbox_result, cropping_256x256 = bbox_item[0].evaluate_bbox_size_at_cropping_window(256, 256)
        eval_bbox_result, cropping_512x512 = bbox_item[0].evaluate_bbox_size_at_cropping_window(512, 512)
        eval_bbox_result, cropping_1024x1024 = bbox_item[0].evaluate_bbox_size_at_cropping_window(1024, 1024)

        excel_item = [
            bbox_item[1].image_name_with_extension,
            bbox_item[1].height,
            bbox_item[1].width,
            bbox_item[1].annotation_name,
            bbox_item[0].id,
            bbox_item[0].class_id,
            bbox_item[0].class_title,
            bbox_item[0].geometry_type,
            bbox_item[0].lin_point1,
            bbox_item[0].col_point1,
            bbox_item[0].lin_point2,
            bbox_item[0].get_height(),
            bbox_item[0].get_width(),
            bbox_item[0].get_area(),
            cropping_64x64,
            cropping_128x128,
            cropping_256x256,
            cropping_512x512,
            cropping_1024x1024,
            bbox_item[0].labeler_login,
            bbox_item[0].created_at,
            bbox_item[0].updated_at,
        ]
        excel_list.append(excel_item)

    # creating dataframe from list 
    df = pd.DataFrame(excel_list, columns=column_names)

    # returning dataframe of bounding boxes 
    return df
   
def save_bbox_list_excel(path_and_list_filename, bbox_df):
    ''' 
    Save bbox lists.
    '''

    # adjusting filename 
    excel_filename = path_and_list_filename.replace('.txt', '.xlsx')

    # writing excel file 
    bbox_df.to_excel(excel_filename, sheet_name='bbox_image', index=False)

def draw_violin_plot_of_bounding_boxes(bbox_list, violin_plot_filename):
    ''' 
    Draw violin plot for bounding boxes.

    '''
    # setting bounding boxes area to plot 
    # for bbox in bbox_list:
    #     x = bbox[0].get_area()
    #     y = 0
    bbox_area_list = []
    bbox_area_list.append([bbox[0].get_area() for bbox in bbox_list])
    # bbox_area_list.append([bbox[0].get_area() for bbox in bbox_list])
    min_value = min(bbox_area_list[0])
    max_value = max(bbox_area_list[0])

    fs = 10  # fontsize
    # pos = [1, 2]
    # xlabels = ['Apothecium', 'teste']
    pos = [1]
    xlabels = ['Apothecium']

    fig, axs = plt.subplots()
    
    axs.violinplot(bbox_area_list, pos, points=100, widths=0.7,
                   showmeans=False, showextrema=True, showmedians=False,
                   bw_method=1.0, vert=False)
    axs.set_ylabel(xlabels[0], fontsize=fs)
    axs.set_xlabel(f'bbox area (H*W) - min: {str(min_value)} | max: {max_value}')
    # axs.set_title('Bounding Box Distribution of Original Images', fontsize=fs)
   
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    
    # axs[0, 0].violinplot(bbox_area_list, pos, points=60, widths=0.7,
    #                     showmeans=True, showextrema=True, showmedians=True,
    #                     bw_method=0.5)
    # axs[0, 0].set_title('Bounding Box Distribution of Original Images', fontsize=fs)

    # for ax in axs.flat:
    #     ax.set_yticklabels([])

    fig.suptitle('Bounding Box Distribution of Original Images')
    fig.subplots_adjust(hspace=0.4)
    # plt.ylabel('bbox area (H * W)')
    # plt.xlabel(xlabels)

    # saving bounding box violin plot
    Utils.remove_file(violin_plot_filename)
    plt.plot(1)
    plt.savefig(violin_plot_filename)
    plt.close()

    # show plot
    # plt.show()

    # close plot 
    # plt.close()
