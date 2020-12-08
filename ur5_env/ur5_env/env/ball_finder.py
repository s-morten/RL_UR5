from cv2 import cv2
import numpy as np
from PIL import Image

class Ball_Finder():
    '''
    very bad way of finding position of ball, hough circles was failing misarably, ml to find ball in future?
    '''

    def find_circle(self, image_array):
        # get image
        img = image_array
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # import like this turns red ball to blue? I dont know why and doesnt care...
        # search for blue ball instead of red

        low_blue = np.array([94, 80, 2])
        high_blue = np.array([126, 255, 255])
        blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
        blue = cv2.bitwise_and(img, img, mask=blue_mask)
        
        i, j = self.get_circle_coordinates(blue)
        return i, j

    def get_circle_coordinates(self, img, width=600, height=300):
        for i in range(height):
            for j in range(width):
                if img[i][j].any() != 0:
                    return i, j

        return -1, -1