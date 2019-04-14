import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

from Calibrate import calibrate_camera
from Threshhold import binary_threshhold
from Perspective import perspective_transform_road_image

print('Computing camera calibration matrix')
mtx, dist, rvecs, tvecs = calibrate_camera()

# undistort all test_images and dump into output_images/
undistored_imgs = []
print('Undistorting test images')
for fname in os.listdir('test_images/'):
    img_rpath = 'test_images/' + fname
    img = cv2.imread(img_rpath)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    undistored_imgs.append((undistorted, fname))
    cv2.imwrite('output_images/' + 'undistorted_' + fname, undistorted)

# dump binary threshhold images to output_images/
binary_threshhold_imgs = []
print('Generating binary threshhold images')
for (u,fname) in undistored_imgs:
    binthresh = binary_threshhold(u)
    binary_threshhold_imgs.append((binthresh, fname))
    cv2.imwrite('output_images/' + 'binary_threshholded_' + fname, binthresh)

# dump perspective transform images
print('Generating perspective transform images')
count = 0
for (b,fname) in binary_threshhold_imgs:
    p = perspective_transform_road_image(b)
    cv2.imwrite('output_images/' + 'perspective_transform_' + fname, p)