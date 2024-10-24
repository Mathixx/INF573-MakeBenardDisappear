import numpy as np
from ikomia.dataprocess.workflow import Workflow
from segmentors import Segmentor
import cv2

class YoloSegmentor(Segmentor):
    """
    YOLO-based segmentor using Ikomia.
    This class detects objects using YOLO and generates a segmentation mask.
    """

    def __init__(self, workflow=None, device='cpu'):
        # Initialize an Ikomia workflow and add the YOLO detection task
        if workflow is not None:
            self.workflow = workflow
        else:
            self.workflow = Workflow()

        self.segmentor= self.workflow.add_task(name="infer_yolo_v8_seg", auto_connect=True)

        self.segmentor.set_parameters({
            "cuda": str(device == 'cuda')
        })

    def segment(self, frame: np.ndarray, bounding_boxes: list, conf_thres=0.5, iou_thres=0.5) -> np.ndarray:
        """
        Perform segmentation using YOLO to detect objects and generate a mask.
        Parameters:
        - frame: The input image (as a numpy array).
        - bounding_boxes: A list of bounding boxes where the object is supposed to be located in format [(x, y, w, h)].
        
        Returns:
        - Segmentation mask (numpy array).
        """
        self.segmentor.set_parameters({
            "conf_thres": str(conf_thres),
            "iou_thres": str(iou_thres)
        })

        # Convert the frame (numpy array) to a format Ikomia can process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # we have to run the workflow on the input image but only in the parts specified by the bounding boxes
        masks = []
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            # Crop the frame to the bounding box area
            cropped_frame = frame_rgb.copy()
            # Fill everything outside the bounding box with black
            cropped_frame[:y, :] = 0
            cropped_frame[y+h:, :] = 0
            cropped_frame[:, :x] = 0
            cropped_frame[:, x+w:] = 0

            # Run the workflow on the cropped frame
            self.workflow.run_on(cropped_frame)
            res = self.segmentor.get_results()
            for obj in res.get_objects():
                mask = obj.mask
                masks.append(mask)

        # Combine the masks
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for m in masks:
            mask = cv2.bitwise_or(mask, m)

        return mask

