import numpy as np
import cv2
from removers import Remover

class OpenCvInpaintingRemover(Remover):
    """
    Concrete implementation of Remover that uses OpenCV's inpainting function.
    """

    def __init__(self, method: str = 'telea', inpaint_radius: int = 3):
        """
        Initialize the remover with specified inpainting method and radius.
        
        Parameters:
        - method: The inpainting method, either 'telea' or 'ns' (Navier-Stokes).
        - inpaint_radius: Radius of a circular neighborhood of each point inpainted.
        """
        if method == 'telea':
            self.inpaint_method = cv2.INPAINT_TELEA
        elif method == 'ns':
            self.inpaint_method = cv2.INPAINT_NS
        else:
            raise ValueError("Method must be 'telea' or 'ns'.")
        self.inpaint_radius = inpaint_radius

    def remove(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform object removal using OpenCV inpainting.
        
        Parameters:
        - frame: The input image (as a numpy array).
        - mask: A binary mask where the object to be removed is highlighted (non-zero pixels).
        
        Returns:
        - The modified image with the object removed.
        """
        # Convert the mask to a binary format if not already
        mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        
        # Perform inpainting
        result = cv2.inpaint(frame, mask_binary, self.inpaint_radius, self.inpaint_method)
        return result
