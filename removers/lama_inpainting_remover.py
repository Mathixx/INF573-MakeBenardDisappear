import numpy as np
from removers import Remover
from simple_lama_inpainting import SimpleLama

class LamaInpaintingRemover(Remover):
    """
    Concrete implementation of Remover that uses Simple Lama Inpainting Model.
    Based of : https://github.com/enesmsahin/simple-lama-inpainting
    """

    def __init__(self):
        """
        Initialize the remover with specified device.
        
        Parameters:
        - method: The inpainting method, either 'telea' or 'ns' (Navier-Stokes).
        - inpaint_radius: Radius of a circular neighborhood of each point inpainted.
        """
        self.remover = SimpleLama()

    def remove(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform object removal using SimpleLama Inpainting Model.
        
        Parameters:
        - frame: The input image (as a 3 Channel numpy array).
        - mask: A binary mask where the object to be removed is highlighted (non-zero pixels).
        
        Returns:
        - The modified image with the object removed.
        """
        # Convert the mask to a binary format if not already
        mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        
        # Perform inpainting
        result = self.remover(frame, mask_binary)
        
        # Convert the result to a numpy array
        return np.array(result)
