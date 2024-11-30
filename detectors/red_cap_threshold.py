import cv2
import numpy as np
from detectors import Detector
from trackers import OpenCVTracker

def do_boxes_intersect(box1, box2):
    """
    Check if two bounding boxes intersect.

    Parameters:
    - box1: The first bounding box (x, y, w, h).
    - box2: The second bounding box (x, y, w, h).

    Returns:
    - bool: True if the boxes intersect, False otherwise.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the intersection rectangle
    x_inter_left = max(x1, x2)
    y_inter_top = max(y1, y2)
    x_inter_right = min(x1 + w1, x2 + w2)
    y_inter_bottom = min(y1 + h1, y2 + h2)

    # Check if there is an intersection
    if x_inter_left < x_inter_right and y_inter_top < y_inter_bottom:
        return True  # There is an intersection
    return False  # No intersection



class RedCapDetector(Detector):
    def __init__(self):
        self.tracker = OpenCVTracker()
        self.tracking = False

    def detect(self, frame: np.ndarray, hue_range=(160, 180), min_circularity=0.3, ratio=0.0002, min_saturation=100, min_value=100) -> list:
        """
        Detect the region where the red cap is most likely located.
        """
        # Convert the frame from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask based on hue ranges
        mask2 = cv2.inRange(hsv_frame, (hue_range[0], min_saturation, min_value), (hue_range[1], 255, 255))
        combined_mask = mask2

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return []

        # Adjust min_size dynamically based on frame resolution
        height, width = frame.shape[:2]
        min_size = max(10, int((height * width) * ratio))

        # Filter contours based on area, aspect ratio, and circularity
        bounding_boxes = []
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_size:
                continue

            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Compute circularity: (4 * π * Area) / (Perimeter^2)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

            # Add checks: circularity, aspect ratio
            if circularity >= min_circularity and aspect_ratio >= 0.7 and aspect_ratio <= 5:
                # Find the contour with the largest area
                if area > max_area:
                    max_area = area
                    bounding_boxes.insert(0,(x, y, w, h))
                else:
                    bounding_boxes.append((x, y, w, h))
                    
        return bounding_boxes
    
    def detect_and_track(self, frame: np.ndarray):
        print(f"Processing frame, frame shape: {frame.shape}, dtype: {frame.dtype}")

        # Detect red caps
        red_cap_boxes = self.detect(frame)
        print(f"Detected red caps: {red_cap_boxes}")

        if red_cap_boxes:
            bbox = red_cap_boxes[0]
            self.reinitialize_tracker(frame, bbox)
            return red_cap_boxes

        print("No red caps detected. Using tracker fallback.")
        if not self.tracker.initialized:
            print("Tracker is not initialized. Cannot fallback.")
            return []

        try:
            tracked_bbox = self.tracker.update(frame)
            if tracked_bbox:
                print(f"Tracker successfully updated: {tracked_bbox}")
                return [tracked_bbox]
        except Exception as e:
            print(f"Error updating tracker: {e}")
            self.tracker.initialized = False

        return []
    
    def reinitialize_tracker(self, frame, bbox):
        """
        Reinitialize the tracker safely if necessary.
        """
        try:
            self.tracker = OpenCVTracker(tracker_type="KCF")  # Reset tracker
            self.tracker.init(frame, bbox)
            self.initialized = True
            print(f"Tracker reinitialized with bbox: {bbox}")
        except Exception as e:
            print(f"Failed to reinitialize tracker: {e}")
            self.initialized = False


