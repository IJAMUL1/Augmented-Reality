import numpy as np
import cv2
from pupil_apriltags import Detector
import os

# Load the calibration results from the .npz file
calibration_data = np.load('calibration_result.npz')

# Access the saved variables (K matrix, distortion coefficients, etc.) by their names
K_matrix = calibration_data['mtx']
distortion_coefficients = calibration_data['dist'].reshape((-1,1))
rotation_vectors = calibration_data['rvecs']
translation_vectors = calibration_data['tvecs']

# extract image center point and focal length variables from camera matrix
fx = K_matrix[0][0]
fy = K_matrix[1][1]
cx = K_matrix[0][2]
cy = K_matrix[1][2]

os.add_dll_directory(r"C:\Users\ifeda\anaconda3\envs\robot_perception\Lib\site-packages\pupil_apriltags.libs")

# Initialize the AprilTag detector
at_detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

# Create VideoWriter object to save video
output_filename = "output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30
frame_size = (640, 480)
out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

# Connect to the webcam
cap = cv2.VideoCapture(0)
image_count = 0

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the grayscale frame
    tags = at_detector.detect(img=gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=0.034)
    if len(tags) > 0:
        for tag in tags:
            # Get the tag's pose (translation and rotation)
            pose_trans = tag.pose_t.squeeze()
            pose_rotat = tag.pose_R

            # Define cube vertices in the tag's local coordinate system
            tag_size = 0.034
            half_size = tag_size / 2
            cube_vertices = np.array([
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0],
                [-half_size, -half_size, -tag_size],
                [half_size, -half_size, -tag_size],
                [half_size, half_size, -tag_size],
                [-half_size, half_size, -tag_size]
            ])

            # Project the 3D cube vertices to 2D image coordinates
            cube_2d, _ = cv2.projectPoints(cube_vertices, pose_rotat, pose_trans, K_matrix, distortion_coefficients)
            cube_2d = cube_2d.reshape(-1, 2)

            # Draw lines connecting the cube vertices on the image
            for i in range(4):
                cv2.line(frame, tuple(map(int, cube_2d[i])), tuple(map(int, cube_2d[(i + 1) % 4])), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, cube_2d[i + 4])), tuple(map(int, cube_2d[(i + 1) % 4 + 4])), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, cube_2d[i])), tuple(map(int, cube_2d[i + 4])), (0, 255, 0), 2)
    
    # Write frame to the video
    out.write(frame)

    # Display the frame with detected tags and the rendered 3D cubes
    cv2.imshow('AR Cube on AprilTag', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
out.release()
cv2.destroyAllWindows()
