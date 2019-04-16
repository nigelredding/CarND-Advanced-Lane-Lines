## Advanced Lane Finding Project

[//]: # (Image References)

[image1]: ./output_images/undistorted_calibration3.jpg "Undistorted"
[image2]: ./test_images/test3.jpg "Road Transformed"
[image3]: ./output_images/binary_thresh_test3.jpg "Binary Example"
[image4]: ./output_images/perspective_color_test3.jpg "Warp Example"
[image5]: ./output_images/warped_w_lanes_test3.jpg "Fit Visual"
[image6]: ./output_images/orig_w_lanes_test3.jpg "Output"
[video1]: ./output_project_video.mp4 "Video"

---

### Writeup

All code references are to `project.py`.

### Camera Calibration

The code for this step is contained in lines 52 to 75 of project.py. We have 20 calibration chessboard images. I set up two arrays, objpoints and imgpoints, indexed by the chessboard images. The entries in objpoints are the "real-world" points corresponding to the corners of the given chessboard. The entries in imgpoints are the actual coordinates of the corners on the chessboard image.

We use these two arrays to calculate the camera matrix and distortion coefficients, using the `cv2.calibrateCamera()` function. We use this matrix
to undistort the image `camera_cal/calibration3.jpg`. The result is below

![alt text][image1]

### Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In order to create my binary threshholded images, I used a combination of a Sobel transform (gradient in the x direction), and threshholding for the 'l' and 's' channels of the image. The code for this is in lines 96-140 of project.py.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My perspective transform code is included in a function called `perspective_transform_road_image` (lines 84-89 of `project.py`).
I used `transform_src` and `transform_dst` to warp our perspective from the camera perspective to a bird's eye perspective.

I used the following set of points for my transformation function

transform_src = np.array([
    [205, 720],
    [1120, 720],
    [745, 480],
    [550, 480]], np.float32)

transform_dst = np.array([
    [205, 720],
    [1120, 720],
    [1120, 0],
    [205, 0]], np.float32).

We use `cv2.getPerspectiveTransform(transform_src, transform_dst)` to get a perspective transform matrix. We use this matrix, along with the `cv2.warpPerspective` function to warp our original image to a bird's eye image, pictured below.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Once I have the warped binary image (last picture), I find the lane line points. Our strategy is as follows. If we already have polynomials for the left and right lanes of the previous frame, we apply `search_around_poly` (lines 258-289). This function searches around the left and right lane polynomials for lane points  (nonzero values). If we do not have any previous polynomials, we apply the sliding window search, with the function `find_lane_pixels` (lines 149-222), to obtain the left and right lane points.

Below we see the result of fitting out lanes with the resulting polynomials. This is the result of `draw_on_warped`, contained in lines 224-256 of `project.py`.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 292 through 316 in my code in `project.py`. I first fit the polynomials for the left and right lines in "x,y real world space". First we set 

`ym_per_pix = 30/720`

`xm_per_pix = 3.7/900`

and use these constants to calculate the second order polynomials for the left and right lanes, respectively,

`left_fit_poly = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)`

`right_fit_poly = np.polyfit(rightx * ym_per_pix, righty * xm_per_pix, 2)`

From these polynomials we calculate the left and right curve radii from the formulae

`left_curverad = ((1 + (2*left_fit_poly[0]*ymax*ym_per_pix + left_fit_poly[1])**2)**1.5) / np.absolute(2*left_fit_poly[0])`

`right_curverad =  ((1 + (2*right_fit_poly[0]*ymax*ym_per_pix + right_fit_poly[1])**2)**1.5) / np.absolute(2*right_fit_poly[0])`.

We calculate the position of the vehicle with respect to the center by calculating the difference between the center of the images and the center point between the two lines. Our full calculation is as follows:

`center_index = binary_warped.shape[1]//2`

`lanes_center = (min(leftx) + max(rightx))//2`
    
`dist_from_center = np.abs(center_index - lanes_center)*xm_per_pix`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 319 through 347 in my code in `project.py` in the function `fill_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Final Video Output

Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I faced was when the colour of the road changed dramatically, as in  0:23 of the video. The main problem is that the yellow line is barely visible when the vehicle is approaching a sunny region. Threfore, I should modify my pipeline so that white lines and yellow lines are always detected, even when half of the image is in the sunlight and the other half is in a shaddow.
