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

    def draw_masks(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
        if mask is not None:
            # Resize mask if necessary to match the frame size
            if mask.shape[:2] != working_frame.shape[:2]:
                raise Exception("Mask and frame sizes do not match.")

            red_overlay = np.zeros_like(working_frame)
            red_overlay[:, :, 2] = mask
            working_frame = cv2.addWeighted(working_frame, 1, red_overlay, 0.5, 0)
        return working_frame
