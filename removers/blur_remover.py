import cv2
import numpy as np
from removers import Remover

class BlurringRemover(Remover):
    """
    A simple remover that blurs the regions of the image specified by the mask.
    """

    def __init__(self, blur_kernel_size=(21, 21)):
        """
        Initialize the blurring remover.
        Parameters:
        - blur_kernel_size: The size of the kernel used for blurring (must be odd).
        """
        self.blur_kernel_size = blur_kernel_size

    def remove(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform object removal by blurring the region specified by the mask.
        
        Parameters:
        - frame: The input image (as a numpy array).
        - mask: A binary mask where the object to be removed is highlighted (non-zero pixels).
        
        Returns:
        - The modified image with the object removed (blurred).
        """
        # Create a blurred version of the entire image
        blurred_image = cv2.GaussianBlur(frame, self.blur_kernel_size, 0)

        # Create a mask that highlights the area to be replaced with the blurred version
        mask_3channel = cv2.merge([mask, mask, mask])  # Convert mask to 3 channels if it's grayscale

        # Blend the original image and the blurred image based on the mask
        result = np.where(mask_3channel > 0, blurred_image, frame)

        return result

