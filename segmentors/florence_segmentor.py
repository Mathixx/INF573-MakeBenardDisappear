import numpy as np
from ikomia.dataprocess.workflow import Workflow
from segmentors import Segmentor
import cv2

class FlorenceSegmentor(Segmentor):
    """
    Florence-based segmentor using Ikomia.
    This class segments objects using Flornce and generates a segmentation mask.
    """

    def __init__(self, workflow=None, device='cpu'):
        # Initialize an Ikomia workflow and add the YOLO detection task
        if workflow is not None:
            self.workflow = workflow
        else:
            self.workflow = Workflow()

        self.segmentor= self.workflow.add_task(name="infer_florence_2_segmentation", auto_connect=True)

        self.remover.set_parameters({
            "model_name":"microsoft/Florence-2-large",
            "task_prompt":"REFERRING_EXPRESSION_SEGMENTATION",
            "prompt":"A man wearing a red cap",
            "max_new_tokens":"1024",
            "num_beams":"3",
            "do_sample":"False",
            "early_stopping":"False",
            "cuda":"True"
        })

    def segment(self, frame: np.ndarray, bounding_boxes: list, conf_thres=0.5, iou_thres=0.5) -> np.ndarray:
        """
        Perform segmentation using Florence to detect objects and generate a mask.
        Parameters:
        - frame: The input image (as a numpy array).
        - bounding_boxes: A list of bounding boxes where the object is supposed to be located in format [(x, y, w, h)].
        
        Returns:
        - Segmentation mask (numpy array).
        """
        self.remover.set_parameters({
            "model_name":"microsoft/Florence-2-large",
            "task_prompt":"REFERRING_EXPRESSION_SEGMENTATION",
            "prompt":"a green car",
            "max_new_tokens":"1024",
            "num_beams":"3",
            "do_sample":"False",
            "early_stopping":"False",
            "cuda":"True"
        })

        # Convert the frame (numpy array) to a format Ikomia can process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        masked_image = image.copy()  # Start with a copy of the original image

        # Set everything in the image to black initially
        masked_image[:, :] = 0

        # Loop through each bounding box and fill in only those areas
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            # Copy the pixels within the bounding box from the original image to the masked image
            masked_image[y:y+h, x:x+w] = image[y:y+h, x:x+w]

        # Run the workflow on the final masked image
        self.workflow.run_on(masked_image)

        # Retrieve results from the segmentor
        masks = []
        res = self.segmentor.get_results()
        for obj in res.get_objects():
            if obj.label == 'person':
                mask = obj.mask
                masks.append(mask)

        # Combine the masks
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for m in masks:
            mask = cv2.bitwise_or(mask, m)

        return mask

