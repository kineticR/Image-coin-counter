# Rhys Dunn - 2015
# Learning image vision/OpenCV


# import libraries
import numpy as np
import cv2


# load image
img = cv2.imread("money.jpg")


# prep image - blur and convert to grey scale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey, (17, 17), 0)


# show blurred image and grey scaled image
cv2.imshow("grey scale", grey)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)


# canny edge detector
outline = cv2.Canny(blurred, 30, 150)


# show canny edge detector
cv2.imshow("The edges", outline)
cv2.waitKey(0)


# find the contours
(cnts, _) = cv2.findContours(outline, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)


# draw contours: -1 will draw all contours
cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
cv2.imshow("Result", img)
cv2.waitKey(0)


# Print how many coins we found
print("I found %i coins" % len(cnts))