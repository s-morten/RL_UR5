from cv2 import cv2
import numpy as np
from PIL import Image

class Ball_Finder():
    '''
    very bad way of finding position of ball, hough circles was failing misarably, ml to find ball in future?
    '''

    def find_circle(self, image_array):
        # get image
        img = Image.fromarray(image_array)
        img.save(f"/home/morten/Documents/code/RL_husky/ur5_env/ur5_env/obs_space.png")
        img = cv2.imread(f"/home/morten/Documents/code/RL_husky/ur5_env/ur5_env/obs_space.png")
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # lower mask (0-10)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # upper mask (170-180)
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
        # join my masks
        mask = mask0+mask1
        # set my output img to zero everywhere except my mask
        output_img = img.copy()
        output_img[np.where(mask==0)] = 0

        i, j = self.get_circle_coordinates(output_img)
        return i, j

    def get_circle_coordinates(self, img, width=600, height=300):
        for i in range(height):
            for j in range(width):
                if img[i][j].any() != 0:
                    return i, j

        return -1, -1