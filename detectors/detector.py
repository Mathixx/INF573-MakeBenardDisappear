from abc import ABC, abstractmethod
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
