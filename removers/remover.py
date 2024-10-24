from abc import ABC, abstractmethod
import numpy as np

class Remover(ABC):
    """
    Abstract base class for all removers.
    Each remover must implement the 'remove' method.
    """

    @abstractmethod
    def remove(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Abstract method to perform object removal.
        
        Parameters:
        - frame: The input image (as a numpy array).
        - mask: A binary mask where the object to be removed is highlighted (non-zero pixels).
        
        Returns:
        - The modified image (numpy array) with the object removed (blurred in this case).
        """
        pass