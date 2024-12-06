import numpy as np
from PIL import Image
import cv2
import logging
import warnings
from removers import Remover
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Suppress logs from `modelscope`
logging.getLogger("modelscope").setLevel(logging.ERROR)
# Suppress all warnings
warnings.filterwarnings("ignore")

class BetterLamaInpaintingRemover(Remover):
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
        self.remover = pipeline(Tasks.image_inpainting, model='damo/cv_fft_inpainting_lama', refine=True)

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

        input_image = Image.fromarray(frame)
        input_mask = Image.fromarray(mask)
        input = {
            'img':input_image,
            'mask':input_mask
        }

        result = self.remover(input)
        vis_img = result[OutputKeys.OUTPUT_IMG]
        vis_img = cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)
        
        return vis_img
