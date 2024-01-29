# ImageSuperviselyAnnotation.py Class

# Importing libraries
import json
import xml.etree.ElementTree as ET
from lxml import etree
from dict2xml import dict2xml

from entity.BoundingBox import BoundingBox

class ImageAnnotation:
    def __init__(self, image_name='', image_name_with_extension='', annotation_name = '', 
                 height=None, width=None, original_image_folder='', bounding_boxes=None):
        self.image_name = image_name
        self.image_name_with_extension = image_name_with_extension
        self.annotation_name = annotation_name
        self.height = height
        self.width = width
        self.original_image_folder = ''
        self.bounding_boxes = bounding_boxes

    def toString(self):
        text = 'Image: ' + self.image_name + \
               ' height: ' + str(self.height) + \
               ' width: ' + str(self.width) + \
               ' bouding boxes: ' + self.bounding_boxes
        return text

    def set_annotation_fields_in_supervisely_format(self, image_name, image_name_with_extension,
                                                    annotation_name, annotation_json,
                                                    selected_classes, original_image_folder):
        self.image_name = image_name
        self.image_name_with_extension = image_name_with_extension
        self.annotation_name = annotation_name
        self.height = annotation_json["size"]["height"]
        self.width = annotation_json["size"]["width"]
        self.original_image_folder = original_image_folder

        self.bounding_boxes = []

        for object in annotation_json["objects"]:
            # creating new bounding box object 
            bounding_box = BoundingBox()

            # setting fields
            bounding_box.id = object["id"]
            bounding_box.class_id = object["classId"] 
            bounding_box.class_title = object["classTitle"]
            bounding_box.geometry_type = object["geometryType"]
            bounding_box.labeler_login = object["labelerLogin"]
            bounding_box.created_at = object["createdAt"]
            bounding_box.updated_at = object["updatedAt"]
            bounding_box.lin_point1 = object["points"]["exterior"][0][1]
            bounding_box.col_point1 = object["points"]["exterior"][0][0]
            bounding_box.lin_point2 = object["points"]["exterior"][1][1]
            bounding_box.col_point2 = object["points"]["exterior"][1][0]

            # selecting bounding box of seleted classes
            if bounding_box.class_title in selected_classes:
                # adding bounding box to list 
                self.bounding_boxes.append(bounding_box)
            
    def set_annotation_of_cropped_image(self, image_name, image_name_with_extension, annotation_name, height, width, original_bounding_box):

        self.image_name = image_name
        self.image_name_with_extension = image_name_with_extension
        self.annotation_name = annotation_name
        self.height = height
        self.width = width
        self.bounding_boxes = []

        # creating new bounding box object 
        bounding_box = BoundingBox(original_bounding_box)

        # setting fields
        bounding_box.id = original_bounding_box.id 
        bounding_box.class_id = original_bounding_box.class_id 
        bounding_box.class_title = original_bounding_box.class_title
        bounding_box.geometry_type = original_bounding_box.geometry_type
        bounding_box.labeler_login = original_bounding_box.labeler_login
        bounding_box.created_at = original_bounding_box.created_at
        bounding_box.updated_at = original_bounding_box.updated_at        
        bounding_box.lin_point1 = original_bounding_box.lin_point1
        bounding_box.col_point1 = original_bounding_box.col_point1
        bounding_box.lin_point2 = original_bounding_box.lin_point2
        bounding_box.col_point2 = original_bounding_box.col_point2

        # adding bounding box to list 
        self.bounding_boxes.append(bounding_box)

    def update_coordinates_of_bounding_box(self, linP1, colP1):
        self.bounding_boxes[0].lin_point1 = self.bounding_boxes[0].lin_point1 - linP1
        self.bounding_boxes[0].col_point1 = self.bounding_boxes[0].col_point1 - colP1
        self.bounding_boxes[0].lin_point2 = self.bounding_boxes[0].lin_point2 - linP1
        self.bounding_boxes[0].col_point2 = self.bounding_boxes[0].col_point2 - colP1

    def get_annotation_in_supervisely_format(self):

        # transforming annotation attributes into json format annotation
        dictionary = {
            "description": "",
            "tags": [],
            "size": {
                "height": self.height,
                "width": self.width,
            },
            "objects": []
        }   

        for bounding_box in self.bounding_boxes:
            # setting one bounding box
            object = {
                "id": bounding_box.id,
                "classId": bounding_box.class_id,
                "description": "",
                "geometryType": "rectangle",
                "labelerLogin": bounding_box.labeler_login,
                "createdAt": bounding_box.created_at,
                "updatedAt": bounding_box.updated_at,
                "tags": [],
                "classTitle": bounding_box.class_title,
                "points": {
                    "exterior": [
                        [
                            bounding_box.col_point1,
                            bounding_box.lin_point1
                        ],
                        [
                            bounding_box.col_point2,
                            bounding_box.lin_point2
                        ]
                    ],
                    "interior": []
                }
            }

            # adding one bounding box to list  
            objects = dictionary['objects']     
            objects.append(object)
    
        # getting json format annotation 
        json_format_string = json.dumps(dictionary, indent = 4)

        # returning json format 
        return json_format_string
    

    def get_annotation_in_voc_pascal_format(self):

        # creating the root element 
        annotation = ET.Element("annotation")

        # creating image tags
        folder = ET.SubElement(annotation, "folder")
        filename = ET.SubElement(annotation, "filename")
        filename.text = self.image_name

        source = ET.SubElement(annotation, "source")
        source_database = ET.SubElement(source, "database")
        source_annotation = ET.SubElement(source, "annotation")
        source_image = ET.SubElement(source, "image")

        size = ET.SubElement(annotation, "size")
        size_width = ET.SubElement(size, "width")
        size_width.text = str(self.width)
        size_height = ET.SubElement(size, "height")
        size_height.text =str(self.height)
        size_depth = ET.SubElement(size, "depth")
        size_depth.text = str(3)

        # creating bounding box tags
        for bounding_box in self.bounding_boxes:
            # setting one bounding box
            object = ET.SubElement(annotation, "object")
            name = ET.SubElement(object, "name")
            name.text = bounding_box.class_title
            pose = ET.SubElement(object, "pose")
            pose.text = 'Unspecified'
            truncated = ET.SubElement(object, "truncated")
            truncated.text = str(0)
            difficult = ET.SubElement(object, "difficult")
            difficult = str(0)

            bndbox = ET.SubElement(object, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(bounding_box.col_point1)
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(bounding_box.lin_point1)
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(bounding_box.col_point2)
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(bounding_box.lin_point2)
           
        # creating xml annotation object in pretty string format 
        xml_annotation_str = ET.tostring(annotation, encoding='unicode')
        xml_annotation = etree.fromstring(xml_annotation_str)
        xml_annotation_pretty_string = etree.tostring(xml_annotation, pretty_print=True).decode()

        # returning xml annotation 
        return xml_annotation_pretty_string
    
    
    def get_annotation_in_yolo_v5_pytorch_format(self, image_height, image_width):
        # creating working variables 
        annotation_text = ''
      
        bbox_class_id = 0 
        bbox_center_x_col = 0.0
        bbox_center_y_lin = 0.0
        bbox_height = 0.0
        bbox_width = 0.0 

        # processing all bounding boxes of image 
        for bounding_box in self.bounding_boxes:

            # annotation_text += bounding_box.get_id_class_SSD()
            bbox_class_id= 0

            # calculating the central point of bounding box 
            bbox_center_x_col = bounding_box.col_point1 + (bounding_box.col_point2 - bounding_box.col_point1) / 2.0
            bbox_center_y_lin = bounding_box.lin_point1 + (bounding_box.lin_point2 - bounding_box.lin_point1) / 2.0

            # calculating the height and width of bounding box 
            bbox_height = bounding_box.lin_point2 - bounding_box.lin_point1
            bbox_width  = bounding_box.col_point2 - bounding_box.col_point1

            # normalizing all values above according image size
            bbox_center_x_col = bbox_center_x_col / image_width
            bbox_center_y_lin = bbox_center_y_lin / image_height

            # calculating the height and width of bounding box 
            bbox_height = bbox_height / image_height
            bbox_width  = bbox_width / image_width

            # concatenating bounding box annotation 
            annotation_text += str(bbox_class_id) + ' ' + \
                               str(bbox_center_x_col) + ' ' + \
                               str(bbox_center_y_lin) + ' ' + \
                               str(bbox_width) + ' ' + \
                               str(bbox_height) + '\n'

        # returning the annoation text for all bounding boxes
        return annotation_text, \
               bbox_class_id, bbox_center_x_col, bbox_center_y_lin, bbox_height, bbox_width
