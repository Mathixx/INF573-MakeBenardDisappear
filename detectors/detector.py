from abc import ABC, abstractmethod
import cv2
import numpy as np

class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Abstract method to detect objects in a frame.

        Parameters:
        - frame: The input image (video frame) as a numpy array.

        Returns:
        - An array representing the detection result (e.g., bounding box, mask, etc.).
          If no cap is found, returns None.
        """
        pass
    
    def draw_boxes(self, frame: np.ndarray, bounding_boxes: np.ndarray, human_boxes=None, red_cap_boxes=None, debugging_frame_level=None):
        """
        Draw the bounding boxes on the frame.

        Parameters:
        - frame: The input image (video frame) as a numpy array.
        - bounding_boxes: The bounding boxes to draw on the frame.
        - human_boxes: Bounding boxes for human detection (optional).
        - red_cap_boxes: Bounding boxes for red cap detection (optional).
        - debugging_frame_level: Debugging level, if applicable.

        Returns:
        - If human_boxes and red_cap_boxes are None:
            - A single frame (working_frame) with `bounding_boxes` drawn.
        - If human_boxes and red_cap_boxes are not None:
            - A tuple of three frames:
                1. The frame with `bounding_boxes` drawn.
                2. The frame with `human_boxes` drawn.
                3. The frame with `red_cap_boxes` drawn.
        """
        working_frame = frame.copy()

        # Draw bounding boxes on the working frame
        for box in bounding_boxes:
            x, y, w, h = box
            cv2.rectangle(working_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # If human_boxes and red_cap_boxes are None, return only the working frame
        if human_boxes is None and red_cap_boxes is None:
            return working_frame

        # Otherwise, create additional frames
        human_boxed_frame = frame.copy()
        red_cap_boxed_frame = frame.copy()

        if debugging_frame_level == 'complete_detector':
            # Draw human boxes
            if human_boxes:
                for box in human_boxes:
                    x, y, w, h = box
                    cv2.rectangle(human_boxed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Draw red cap boxes
            if red_cap_boxes:
                for box in red_cap_boxes:
                    x, y, w, h = box
                    cv2.rectangle(red_cap_boxed_frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

        # Return all three frames if human_boxes and red_cap_boxes are not None
        return working_frame, human_boxed_frame, red_cap_boxed_frame

    

