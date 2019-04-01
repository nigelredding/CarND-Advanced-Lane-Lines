
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def hls_thresh(image, chan='h', thresh_min=90, thresh_max=255):
    if chan == 'h':
        ch = 0
    elif chan == 'l':
        ch = 1
    elif chan == 's':
        ch = 2
    else:
        raise ValueError('channel must be h, l or s')
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    chan_img = image[:,:,ch]
    thresh_img = np.zeros_like(chan_img)
    thresh_img[(chan_img > thresh_min) & (chan_img <= thresh_max)] = 1
    return thresh_img

img = cv2.imread('test_images/test3.jpg')

sChannel_thresh = hls_thresh(img, chan='s')
lChannel_thresh = hls_thresh(img, chan='l')
sobelx_thresh = abs_sobel_thresh(img, 'x', 20, 100)

combined = np.zeros_like(sobelx_thresh)
combined[(sChannel_thresh == 1) & (lChannel_thresh == 1) & (sobelx_thresh == 1)] = 1


plt.imshow(sChannel_thresh, cmap='gray')
plt.show()

plt.imshow(lChannel_thresh, cmap='gray')
plt.show()

plt.imshow(sobelx_thresh, cmap='gray')
plt.show()

plt.imshow(combined, cmap='gray')
plt.show()