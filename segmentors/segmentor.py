from abc import ABC, abstractmethod
import numpy as np
import cv2

class Segmentor(ABC):
    """
    Abstract base class for all segmentors.
    Each segmentor must implement the 'segment' method.
    """

    @abstractmethod
    def segment(self, frame: np.ndarray, bounding_boxes: list) -> np.ndarray:
        """
        Abstract method to perform segmentation.
        Parameters:
        - frame: The input image (as a numpy array).
        - bounding_boxes: A list of bounding boxes where the object is supposed to be located.
        
        Returns:
        - Segmentation mask (numpy array) where segmented regions are highlighted.
        """
        pass

    def draw_masks(self, frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Draw the masks on the frame.
        Parameters:
        - frame: The input image (as a numpy array).
        - masks: The masks to draw on the frame.
        
        Returns:
        - The frame with the masks drawn.
        """
        working_frame = frame.copy()
        # Add the masks to the frame by adding a transparent red layer
        for mask in masks:
            # Resize mask if necessary to match the frame size
            if mask.shape[:2] != working_frame.shape[:2]:
                mask = cv2.resize(mask, (working_frame.shape[1], working_frame.shape[0]))

            # If the mask is single-channel, convert it to a 3-channel (RGB) mask
            if len(mask.shape) == 2:  # Grayscale mask
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Blend the mask with the frame, using a transparency factor
            working_frame = cv2.addWeighted(working_frame, 1, mask, 0.5, 0)
        return working_frame
