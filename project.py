import numpy as np
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os

'''
    Data for images
'''
IMG_ROWS = 720
IMG_COLS = 1280

transform_src = np.array([
    [205, 720],
    [1120, 720],
    [745, 480],
    [550, 480]], np.float32)

transform_dst = np.array([
    [205, 720],
    [1120, 720],
    [1120, 0],
    [205, 0]], np.float32)

'''
    Camera Calibration
'''
board_dims = {
	1: (9, 5),
	2: (9, 6),
	3: (9, 6),
	4: (6, 5),
	5: (7, 6),
	6: (9, 6),
	7: (9, 6),
	8: (9, 6),
	9: (9, 6),
	10: (9, 6),
	11: (9, 6),
	12: (9, 6),
	13: (9, 6),
	14: (9, 6),
	15: (9, 6),
	16: (9, 6),
	17: (9, 6),
	18: (9, 6),
	19: (9, 6),
	20: (9, 6),
}

def calibrate_camera():
	objpoints = []
	imgpoints = []

	for i in range(1,21):
		# load the image, and convert to grayscale
		img = cv2.imread('camera_cal/calibration' + str(i) + '.jpg')
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, board_dims[i], None)
		if ret == False:
			print('Warning: ' + 'camera_cal/calibration' + str(i) + '.jpg has the wrong dimensions.')
		else:
			# prepare object points
			nx, ny = board_dims[i]
			objp = np.zeros((ny*nx,3), np.float32)
			objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
			# add both object points and image points to buffers
			objpoints.append(objp)
			imgpoints.append(corners)

	# compute the camera calibration
	imsize = (1280, 720)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imsize, None, None)
	return mtx, dist, rvecs, tvecs

def undistort(img, mtx, dist):
	return cv2.undistort(img, mtx, dist, None, mtx)


'''
    Perspective transform
'''
def perspective_transform_road_image(img):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(transform_src, transform_dst)
    warped = np.copy(cv2.warpPerspective(img, M, (IMG_COLS, IMG_ROWS)))
    Minv = cv2.getPerspectiveTransform(transform_dst, transform_src)
    return Minv, warped

'''
    Threshholding
'''
# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=10, thresh_max=100, sobel_kernel=3):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
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

def combined_images(sChannel_thresh, lChannel_thresh, sobelx_thresh):
    combined = np.zeros_like(sobelx_thresh)
    combined[(sChannel_thresh == 1) & (lChannel_thresh == 1) & (sobelx_thresh == 1)] = 1
    return combined

def binary_threshhold(img):
    sChannel_thresh = hls_thresh(img, chan='s')
    lChannel_thresh = hls_thresh(img, chan='l')
    sobelx_thresh = abs_sobel_thresh(img, 'x')
    return combined_images(sChannel_thresh, lChannel_thresh, sobelx_thresh)

'''
    Finding lane lines
'''
# global state for last polynomials
last_left_poly = []
last_right_poly = []

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        win_xleft_low = leftx_current - margin
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_xleft_high = leftx_current + margin
        win_y_high = binary_warped.shape[0] - window*window_height
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def draw_on_warped(binary_warped, leftx, lefty, rightx, righty, margin):
    global last_left_poly
    global last_right_poly

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = last_left_poly[0]*ploty**2 + last_left_poly[1]*ploty + last_left_poly[2]
    right_fitx = last_right_poly[0]*ploty**2 + last_right_poly[1]*ploty + last_right_poly[2]

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result

def search_around_poly(binary_warped):
    global last_left_poly
    global last_right_poly

    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (last_left_poly[0]*(nonzeroy**2) + last_left_poly[1]*nonzeroy + 
                    last_left_poly[2] - margin)) & (nonzerox < (last_left_poly[0]*(nonzeroy**2) + 
                    last_left_poly[1]*nonzeroy + last_left_poly[2] + margin)))
    right_lane_inds = ((nonzerox > (last_right_poly[0]*(nonzeroy**2) + last_right_poly[1]*nonzeroy + 
                    last_right_poly[2] - margin)) & (nonzerox < (last_right_poly[0]*(nonzeroy**2) + 
                    last_right_poly[1]*nonzeroy + last_right_poly[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def curvature(A,B,C,y):
    return (1+(2*A*y + B)**2)**(3/2)/abs(2*A)

def curve_and_distance(binary_warped, leftx, lefty, rightx, righty):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    ymax = np.max(ploty)

    ym_per_pix = 30/720
    xm_per_pix = 3.7/900

    # get the 'real world' polynomials, in x,y space
    left_fit_poly = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_poly = np.polyfit(rightx * ym_per_pix, righty * xm_per_pix, 2)

    # get radii of curvature
    left_curverad = ((1 + (2*left_fit_poly[0]*ymax*ym_per_pix + left_fit_poly[1])**2)**1.5) / np.absolute(2*left_fit_poly[0])
    right_curverad =  ((1 + (2*right_fit_poly[0]*ymax*ym_per_pix + right_fit_poly[1])**2)**1.5) / np.absolute(2*right_fit_poly[0])

    # get distance from center
    center_index = binary_warped.shape[1]//2
    lanes_center = (min(leftx) + max(rightx))//2
    
    dist_from_center = np.abs(center_index - lanes_center)*xm_per_pix

    return left_curverad, right_curverad, dist_from_center

## draw lane area
def fill_lane(img, bin_img, left_fit_poly, right_fit_poly, inverse_matrix):
    ## create copy of image
    img_copy = np.copy(cv2.resize(img, (1280, 720)))
    
    ## define range
    ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0])
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ## fit lines
    left_fitx = left_fit_poly[0]*ploty**2 + left_fit_poly[1]*ploty + left_fit_poly[2]
    right_fitx = right_fit_poly[0]*ploty**2 + right_fit_poly[1]*ploty + right_fit_poly[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, inverse_matrix, (img_copy.shape[1], img_copy.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img_copy, 1, newwarp, 0.3, 0)
    
    return result

def fit_polynomial_helper(binary_warped):
    '''
    This function sets last_left_poly and last_right_poly, which
    are the polys fit to the lane lines in the binary warped image.
    No need to return anything.
    '''
    global last_left_poly
    global last_right_poly
    
    if last_left_poly == [] or last_right_poly == []:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    else:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped)

    if len(leftx) != 0 and len(lefty) != 0:
        last_left_poly = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0 and len(righty) != 0:
        last_right_poly = np.polyfit(righty, rightx, 2)

    return leftx, lefty, rightx, righty

'''
    Pipeline
'''
# picture pipeline
def pic_pipeline(img, mtx, dist):
    '''
    Input a color image taken from a car camera.
    Output an image with lane lines drawn.
    '''
    global last_left_poly
    global last_right_poly

    # undistort the image
    undistorted = undistort(img, mtx, dist)
    # get the binary threshholded image
    binary_thresh = binary_threshhold(undistorted)
    # warp to the proper perspective
    Minv, binary_warped = perspective_transform_road_image(binary_thresh)
    # get lanes, prepare for drawing
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    # get the lane polynomials
    if len(leftx) != 0 and len(lefty) != 0:
        last_left_poly = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0 and len(righty) != 0:
        last_right_poly = np.polyfit(righty, rightx, 2)
    # warped image with lanes drawn
    warped_w_lanes = draw_on_warped(binary_warped, leftx, lefty, rightx, righty, margin=80)
    # get curvature and distance data
    left_curverad, right_curverad, dist_from_ct = curve_and_distance(binary_warped, leftx, lefty, rightx, righty)
    # draw lanes on original image
    orig_w_lanes = fill_lane(img, binary_warped, last_left_poly, last_right_poly, Minv)
    # draw the stats on the image
    draw_stats_on_img(orig_w_lanes, left_curverad, right_curverad, dist_from_ct)
    
    return binary_thresh, binary_warped, warped_w_lanes, orig_w_lanes

def draw_stats_on_img(img, left_curverad, right_curverad, dist_from_ct):
    curve_stats = 'Left curve radius: ' + str(left_curverad) + '  Right curve radius: ' + str(right_curverad)
    dist_stat = 'Distance from center: ' + str(dist_from_ct)
    cv2.putText(img, curve_stats, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.putText(img, dist_stat, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), lineType=cv2.LINE_AA) 

# video pipeline
def vid_pipeline(img, mtx, dist):
    '''
    Input a color image taken from a car camera.
    Output an image with lane lines drawn.
    '''
    # undistort the image
    undistorted = undistort(img, mtx, dist)
    # get the binary threshholded image
    binary_thresh = binary_threshhold(undistorted)
    # warp to the proper perspective
    Minv, binary_warped = perspective_transform_road_image(binary_thresh)
    # fit the polynomial to the output image
    leftx, lefty, rightx, righty = fit_polynomial_helper(binary_warped)
    # get curvature and distance data
    left_curverad, right_curverad, dist_from_ct = curve_and_distance(binary_warped, leftx, lefty, rightx, righty)
    # draw the lane lines on the original image
    out = fill_lane(img, binary_warped, last_left_poly, last_right_poly, Minv)
    # draw the stats on the image
    draw_stats_on_img(out, left_curverad, right_curverad, dist_from_ct)

    return out

'''
DEMO
'''
# get the camera calibration
mtx, dist, _, _ = calibrate_camera()

# undistort calibration3.jpg
print('Undistorting camera_cal/calibration3.jpg')
cal3 = cv2.imread('camera_cal/calibration3.jpg')
cal3_undistorted = undistort(cal3, mtx, dist)
cv2.imwrite('output_images/undistorted_calibration3.jpg', cal3_undistorted)

# warp color image
print('Warping test_images/test3.jpg')
img = cv2.imread('test_images/test3.jpg')
_, warped = perspective_transform_road_image(img)
cv2.imwrite('output_images/perspective_color_test3.jpg', warped)

# run pipeline on all images
print('Running pipeline on test_images/')
for fname in os.listdir('test_images/'):
    print('Processing ' + fname)
    img = cv2.imread('test_images/' + fname)
    binary_thresh, binary_warped, warped_w_lanes, orig_w_lanes = pic_pipeline(img, mtx, dist)
    cv2.imwrite('output_images/' + 'binary_thresh_' + fname, binary_thresh.astype('uint8') * 255)
    cv2.imwrite('output_images/' + 'warped_w_lanes_' + fname, warped_w_lanes)
    cv2.imwrite('output_images/' + 'orig_w_lanes_' + fname, orig_w_lanes)

print('Processing video')
clip1 = VideoFileClip('project_video.mp4')
white_clip = clip1.fl_image(lambda img: vid_pipeline(img, mtx, dist))
white_clip.write_videofile('output_project_video.mp4')
