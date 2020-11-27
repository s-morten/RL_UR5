# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import cv2
import PIL.Image as PIL
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())
# # load the image, clone it for output, and then convert it to grayscale
# image = cv2.imread(args["image"])
img = cv2.imread("/home/morten/Documents/code/RL_husky/ur5_env/ur5_env/0.png")
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0



cv2.imshow('mask', output_img)
cv2.imshow('result', output_hsv)
cv2.waitKey()

print(output_img.shape)
print(output_img[135][300])
# edges = cv2.Canny(output, 100, 200)
gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
# detect circles in the image
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=1000)
cv2.imshow('result', gray)
cv2.waitKey()

print(output_img.shape[::1])

for i in range(300):
    for j in range(600):
        if output_img[i][j].any() != 0:
            print(i, j)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 1, maxRadius = 40)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
    print(f"{circles[0][0]}, {circles[0][1]}")
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # show the output image
    cv2.imshow("output", np.hstack([img, img]))
    cv2.waitKey(0)

else:
    print('Circle is none :o')
