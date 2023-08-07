import cv2
from scipy.cluster.vq import kmeans, vq
import numpy as np

def detect_color(image):
    if image.shape[0] == 0:
        print("Warning: Empty image encountered.")
        return None # or a default value

    print(f"Original shape: {image.shape}")  # Check original shape
    pixels = image.reshape(-1, 3)
    print(f"Shape after reshaping: {pixels.shape}")  # Check shape after reshaping
    pixels = np.float32(pixels)
    print(f"Shape after conversion to float: {pixels.shape}")  # Check shape after conversion
    # Reshape the image to be a list of pixels

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    center = np.uint8(centers)

    # Map the labels to the centers
    segmented_image = center[labels.flatten()]

    # Reshape back to the original image
    segmented_image = segmented_image.reshape(image.shape)

    # Get the most frequent color
    center_counts = np.bincount(labels.flatten())
    most_frequent_color = center[np.argmax(center_counts)]

    return most_frequent_color

def detect_yellow_coordinates(image, frame_number, fps):
    if image.shape[0] == 0:
        print("Warning: Empty image encountered.")
        return [], None

    # 현재 프레임의 시간을 계산
    current_time_seconds = frame_number / fps
    minutes, seconds = divmod(current_time_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    detection_time = f"{int(hours)}:{int(minutes)}:{int(seconds)}"

    # Convert to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a color range for yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a binary mask where white pixels represent the presence of yellow
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Find the coordinates of yellow pixels
    yellow_coordinates = np.column_stack(np.where(mask > 0))

    return yellow_coordinates, detection_time


    

    