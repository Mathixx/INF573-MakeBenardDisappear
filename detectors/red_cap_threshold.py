import cv2
import numpy as np
from detectors import Detector

class RedCapDetector(Detector):
    def detect(self, frame: np.ndarray, hue_range=(0, 5), alt_hue_range=(170, 180), min_circularity=0.325, ratio=0.0002, min_saturation=100, min_value=100) -> list:
        """
        Detect the region where the red cap is most likely located.
        Returns bounding boxes around regions that match the red cap color and shape criteria.

        Parameters:
        - frame: The input image (video frame) as a numpy array (in BGR format).
        - hue_range: A tuple specifying the primary hue range for red detection (0-10).
        - alt_hue_range: A tuple specifying the secondary hue range for red detection (170-180).
        - min_circularity: Minimum circularity to filter out non-cap objects (between 0 and 1).
        - ratio: A scaling factor for minimum contour area, adjusting dynamically based on image size (default is 0.001).

        Returns:
        - A list of bounding boxes in the form (x, y, w, h) for detected red caps.
          If no cap is found, returns an empty list.
        """

        # Convert the frame from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask manually based on the specified hue ranges
        mask1 = (
            (hsv_frame[:, :, 0] >= hue_range[0]) & (hsv_frame[:, :, 0] <= hue_range[1]) &
            (hsv_frame[:, :, 1] >= min_saturation) & (hsv_frame[:, :, 2] >= min_value)
        )
        mask2 = (
            (hsv_frame[:, :, 0] >= alt_hue_range[0]) & (hsv_frame[:, :, 0] <= alt_hue_range[1]) &
            (hsv_frame[:, :, 1] >= min_saturation) & (hsv_frame[:, :, 2] >= min_value)
        )

        # Combine the masks for both hue ranges
        combined_mask = (mask1 | mask2).astype(np.uint8) * 255

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours are found, return an empty list
        if not contours:
            return []

        # --- Dynamic Size Thresholding using ratio parameter ---
        # Adjust min_size dynamically based on image resolution and the ratio parameter
        height, width = frame.shape[:2]
        min_size = int((height * width) * ratio)

        # Filter contours based on shape criteria: area, aspect ratio, circularity
        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_size:
                continue  # Filter small objects

            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Compute circularity: (4 * π * Area) / (Perimeter^2)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            # Add additional checks: circularity, aspect ratio
            if circularity >= min_circularity and 0.5 <= aspect_ratio <= 2.0:
                bounding_boxes.append((x, y, w, h))

        return bounding_boxes