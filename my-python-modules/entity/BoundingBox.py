# BoundingBox Class
# from Entity.Pixel import Pixel

class BoundingBox:
    def __init__(self, id=None, class_id=None, class_title='', geometry_type='',
                 labeler_login='', created_at=None, updated_at=None, 
                 lin_point1=None, col_point1=None, lin_point2=None, col_point2=None,
                 confidence=0, annotation_filename=''):
        self.id = id
        self.class_id = class_id
        self.class_title = class_title
        self.geometry_type = geometry_type
        self.labeler_login = labeler_login
        self.created_at = created_at
        self.updated_at = updated_at
        self.lin_point1 = lin_point1
        self.col_point1 = col_point1
        self.lin_point2 = lin_point2
        self.col_point2 = col_point2
        self.confidence = confidence
        self.annotation_filename = annotation_filename

    def toString(self):
        text = 'Class: ' + self.class_name + ' P1: (' + str(self.lin_point1) + ',' + str(
            self.col_point1) + ')  P2: (' + str(self.lin_point2) + ',' + str(
            self.col_point2) + ')' + '  confidence: ' + str(self.confidence)
        return text

    def get_height(self):
        return self.lin_point2 - self.lin_point1

    def get_width(self):
        return self.col_point2 - self.col_point1
    
    def get_area(self):
        return (self.get_height() * self.get_width())
    
    # def isBelongs(self, pixel: Pixel):
    #     # checking if pixel is in the bounding box
    #     if all([
    #         pixel.lin >= self.linPoint1,
    #         pixel.lin <= self.linPoint2,
    #         pixel.col >= self.colPoint1,
    #         pixel.col <= self.colPoint2
    #     ]):
    #         return True

    #     # if pixel.lin >= self.linPoint1 and pixel.lin <= self.linPoint2 and pixel.col >= self.colPoint1 and pixel.col <= self.colPoint2:

    #     # obtaining the new bounding box coordinates using neighbor of one pixel
    #     newLinPoint1 = self.linPoint1 - 1
    #     newColPoint1 = self.colPoint1 - 1
    #     newLinPoint2 = self.linPoint2 + 1
    #     newColPoint2 = self.colPoint2 + 1

    #     # checking if pixel is neighbor of 1 pixel of the bounding box
    #     if pixel.lin >= newLinPoint1 and pixel.lin <= newLinPoint2 and \
    #             pixel.col >= newColPoint1 and pixel.col <= newColPoint2:
    #         return True

    #     # the current pixel does not belong to none bounding box
    #     return False

    # def getYoloAnnotation(self, width, height):
    #     heightOfCentrePoint = self.linPoint2 - self.linPoint1
    #     widthOfCentrePoint = self.colPoint2 - self.colPoint1
    #     linOfCentrePoint = (self.linPoint1 + (heightOfCentrePoint / 2.0)) / height
    #     colOfCentrePoint = (self.colPoint1 + (widthOfCentrePoint / 2.0)) / width
    #     heightOfCentrePoint /= height
    #     widthOfCentrePoint /= width
    #     return linOfCentrePoint, colOfCentrePoint, widthOfCentrePoint, heightOfCentrePoint

    # def setYoloAnnotation(self, imageHeight, imageWidth, colOfCentrePoint, linOfCentrePoint, heightOfCentrePoint,
    #                       widthOfCentrePoint, idBoundingBox, idClass):
    #     # setting class name
    #     self.setClassName(idClass)

    #     # calculates the coordinates of the two points

    #     # unnormalizing values
    #     heightOfBoundingBox = heightOfCentrePoint * imageHeight
    #     widthOfBoundingBox = widthOfCentrePoint * imageWidth
    #     lin = linOfCentrePoint * imageHeight
    #     col = colOfCentrePoint * imageWidth

    #     # getting the points coordiniates
    #     self.linPoint1 = int(lin - (heightOfBoundingBox / 2.0))
    #     self.colPoint1 = int(col - (widthOfBoundingBox / 2.0))
    #     self.linPoint2 = int(lin + (heightOfBoundingBox / 2.0))
    #     self.colPoint2 = int(col + (widthOfBoundingBox / 2.0))

    #     # print('Yolo annotations: ', imageWidth, imageHeight, linOfCentrePoint, colOfCentrePoint, widthOfCentrePoint,
    #     #       heightOfCentrePoint)
    #     # print('BB:', idBoundingBox, 'Two points coordinates: ', self.linPoint1, self.colPoint1, self.linPoint2,
    #     #       self.colPoint2, self.className, )
    #     return

    # def expandBoudingBox(self, expandedPixels):
    #     self.linPoint1 -= expandedPixels
    #     self.colPoint1 -= expandedPixels
    #     self.linPoint2 += expandedPixels
    #     self.colPoint2 += expandedPixels

    def get_id_class_SSD(self):
        if self.class_title == 'Mature Sclerotium':
            return 1
        elif self.class_title == 'Imature Sclerotium':
            return 2
        elif self.class_title == 'Apothecium':
            return 3
        elif self.class_title == 'Mushroom':
            return 4
        elif self.class_title == 'Petal':
            return 5
        elif self.class_title == 'Coal':
            return 6
        elif self.class_title == 'Disease White Mold':
            return 7
        else:
            return -1
        
    def evaluate_bbox_size_at_cropping_window(self, height_crop_window, width_crop_window):
        if self.get_height() <= height_crop_window and self.get_width() <= width_crop_window:
            return True, 'OK'
        if self.get_height() > height_crop_window  and self.get_width() > width_crop_window:
            return False, 'HW >'
        if self.get_height() > height_crop_window:
            return False, 'H >'
        if self.get_width() > width_crop_window:
            return False, 'W >'
            
    def check_consistency_of_coordinates(self, image_name, height, width):
        # initializing variable 
        text = ''

        # checking values of bounding box coordinates 
        if self.lin_point1 < 0:
            text +=  ' lin_point1: ' + str(self.lin_point1)
        if self.lin_point1 > height:
            text +=  ' lin_point1: ' + str(self.lin_point1)

        if self.col_point1 < 0:
            text +=  ' col_point1: ' + str(self.col_point1)
        if self.col_point1 > width:
            text +=  ' col_point1: ' + str(self.col_point1)

        if self.lin_point2 < 0:
            text +=  ' lin_point2: ' + str(self.lin_point2)
        if self.lin_point2 > height:
            text +=  ' lin_point2: ' + str(self.lin_point2)

        if self.col_point2 < 0:
            text +=  ' col_point2: ' + str(self.col_point2)
        if self.col_point2 > width:
            text +=  ' col_point2: ' + str(self.col_point2)

        # returning with correct consistency
        if text != '':
            # adding image name and bounding box id
            text = image_name + '-bbox-' + str(self.id) + ' ' + text 

        # returning result
        return text
        