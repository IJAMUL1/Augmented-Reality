# Augmented Reality Cube on AprilTag Detection

This project demonstrates the implementation of augmented reality (AR) using a webcam and AprilTag detection. It involves the following steps:

![Untitled video - Made with Clipchamp](https://github.com/IJAMUL1/Augmented-Reality/assets/60096099/f72a6714-7ba3-4999-a765-297f73c7af26)


## Overview
1. **Calibration**: Camera calibration is performed using a set of images to obtain intrinsic parameters such as the camera matrix and distortion coefficients. See full instructions here: https://github.com/IJAMUL1/Camera-Callibration.git
2. **AprilTag Detection**: AprilTags, which are fiducial markers, are detected in real-time using the `pupil_apriltags` library. These tags provide a reference point for anchoring virtual objects in the real world.
3. **AR Cube Rendering**: Upon detecting an AprilTag, a 3D cube is rendered on top of it. The cube's vertices are projected onto the 2D image plane using camera calibration parameters and drawn onto the frame.
4. **User Interaction**: The program continuously captures frames from the webcam and renders the AR cube whenever an AprilTag is detected. The rendering is displayed in real-time.
5. **Saving Output**: Optionally, the program saves the frames with rendered AR cubes as image files for later analysis or visualization.

## Dependencies
- `numpy` for numerical computations.
- `cv2` (OpenCV) for image processing and camera interaction.
- `pupil_apriltags` for AprilTag detection.
  
## Usage
1. Ensure the camera is properly calibrated and the calibration data is saved.
2. Run the script *augmented_reality_save_vid.py*.
3. Point the webcam towards AprilTags, and the AR cube will be rendered on detection.
4. Press 'q' to quit the program.

## References
- `pupil_apriltags`: [https://github.com/pupil-labs/apriltags](https://github.com/pupil-labs/apriltags)

## Contributing
Contributions and feedback are welcome. Feel free to submit issues or pull requests.
