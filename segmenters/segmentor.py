from abc import ABC, abstractmethod
import numpy as np

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
