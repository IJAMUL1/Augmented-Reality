import cv2
import os

def capture_calibration_images(save_directory, num_images_to_capture=12):
    # Create a VideoCapture object for your webcam (usually 0 for the default webcam)
    cap = cv2.VideoCapture(0)

    # Create a window to display the live webcam feed
    cv2.namedWindow("Calibration Image Capture")

    # Create a counter to track the number of calibration images captured
    image_counter = 0

    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    try:
        while image_counter < num_images_to_capture:
            # Capture a frame from the webcam
            ret, frame = cap.read()

            # Display the live feed
            cv2.imshow("Calibration Image Capture", frame)

            # Check for user input to capture an image
            key = cv2.waitKey(1)

            if key == ord('c'):
                # Increment the image counter
                image_counter += 1

                # Generate a filename for the captured image
                image_filename = os.path.join(save_directory, f'calibration_image_{image_counter}.jpg')

                # Save the captured image
                cv2.imwrite(image_filename, frame)
                print(f"Saved: {image_filename}")

                # Display the number of images captured
                print(f"Images captured: {image_counter}/{num_images_to_capture}")

            # Check for user input to exit the capture process
            elif key == ord('q'):
                break

    finally:
        # Release the webcam and close the OpenCV window
        cap.release()
        cv2.destroyAllWindows()

        print("Image capture completed.")

if __name__ == "__main__":
    # Directory where calibration images will be saved
    save_directory = r'Calibration_Images'  # Change this to your desired directory

    # Number of calibration images to capture (optional, defaults to 12)
    num_images_to_capture = 10

    capture_calibration_images(save_directory, num_images_to_capture)