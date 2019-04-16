## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_calibration3.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_thresh_test1.jpg "Binary Example"
[image4]: ./output_images/perspective_color_test1.jpg "Warp Example"
[image5]: ./output_images/warped_w_lanes_test1.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 52 to 75 of project.py. We have 20 calibration chessboard images. I set up two arrays, objpoints and imgpoints, indexed by the chessboard images. The entries in objpoints are the "real-world" points corresponding to the corners of the given chessboard. The entries in imgpoints are the actual coordinates of the corners on the chessboard image.

We use these two arrays to calculate the camera matrix and distortion coefficients, using the `cv2.calibrateCamera()` function. We use this matrix
to undistort the image `camera_cal/calibration3.jpg`. The result is below

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

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

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
