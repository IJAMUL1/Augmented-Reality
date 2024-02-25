import numpy as np
import cv2
import glob

# Termination criteria for subpixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7 * 9,3), np.float32)
objp[:, :2] = np.mgrid[0:9,0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D point in real-world space
imgpoints = []  # 2D points in the image plane.

images = glob.glob('Calibration_Images/*.jpg')  # Replace with your image directory

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,7), None)

    # If found, add object points and image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine corner positions for better accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Camera Matrix
print(mtx)

#Calculate the reprojection error

# Calculate reprojection error
mean_error = 0  # Initialize mean error
for i in range(len(objpoints)):
    # Project the object points onto the image plane using calibration parameters
    image_points_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # Calculate the absolute error between the detected image points and the reprojected points
    error = cv2.norm(imgpoints[i], image_points_reprojected, cv2.NORM_L2) / len(image_points_reprojected)

    # Accumulate the errors for all images
    mean_error += error

# Calculate the mean reprojection error across all images
mean_error /= len(objpoints)

print(f"Mean Reprojection Error: {mean_error}")

# Save calibration results
np.savez('calibration_result.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

