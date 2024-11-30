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
    
    def draw_boxes(self, frame: np.ndarray, bounding_boxes: np.ndarray, human_boxes, red_cap_boxes, debugging_frame_level) -> np.ndarray:
        """
        Draw the bounding boxes on the frame.

        Parameters:
        - frame: The input image (video frame) as a numpy array.
        - bounding_boxes: The bounding boxes to draw on the frame.

        Returns:
        - The frame with the bounding boxes drawn.
        """
        working_frame = frame.copy()
        human_boxed_frame = frame.copy()
        red_cap_boxed_frame = frame.copy()
        for box in bounding_boxes:
            x, y, w, h = box
            cv2.rectangle(working_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if debugging_frame_level == 'complete_detector':
            for box in human_boxes:
                x, y, w, h = box
                cv2.rectangle(human_boxed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            for box in red_cap_boxes:
                x, y, w, h = box
                cv2.rectangle(red_cap_boxed_frame, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
            
        return working_frame, human_boxed_frame, red_cap_boxed_frame

