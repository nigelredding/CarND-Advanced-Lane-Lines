
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob 

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

objpoints = []
imgpoints = []

for i in range(1,21):
    # load the image, and convert to grayscale
    img = cv2.imread('camera_cal/calibration' + str(i) + '.jpg')
    print(img.shape)
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

distored = cv2.imread('camera_cal/calibration1.jpg')
undistored = cv2.undistort(distored, mtx, dist, None, mtx)
plt.imshow(undistored)
plt.show()

