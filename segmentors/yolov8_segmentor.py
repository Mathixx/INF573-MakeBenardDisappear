import numpy as np
from ikomia.dataprocess.workflow import Workflow
from segmentors import Segmentor
import cv2

def extract_shadow_mask(person_mask):
    # Check if mask is empty
    if np.sum(person_mask) == 0:
        return person_mask

    # Dilate the person mask to find areas near the person that might contain the shadow
    kernel = np.ones((4, 4), np.uint8)
    dilated_person_mask = cv2.dilate(person_mask, kernel, iterations=5)

    # Dilate the person mask to find areas near the person that might contain the shadow
    kernel1 = np.ones((8, 8), np.uint8)
    dilated_person_mask1 = cv2.dilate(person_mask, kernel1, iterations=5)

    # Find the bounding box of the person in the mask
    y_indices, x_indices = np.where(person_mask > 0)
    top_y, bottom_y = min(y_indices), max(y_indices)
    middle_y = (top_y + bottom_y) // 2  # Middle y-coordinate

    # Only keep the shadow mask below the middle y-coordinate
    dilated_person_mask1[:middle_y, :] = 0
    dilated_person_mask[middle_y:, :] = 0

    # Apply this mask to keep only the shadow below the middle y-coordinate
    shadow_mask = cv2.bitwise_or(dilated_person_mask1, dilated_person_mask)

    return shadow_mask

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

        # Initialize an empty mask of the same size as the original frame
        final_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Loop through each bounding box
        for bbox in bounding_boxes:
            x, y, w, h = bbox

            # Crop the region of interest
            cropped_region = frame_rgb[y:y+h, x:x+w]

            # Run the workflow on the cropped region
            self.workflow.run_on(cropped_region)

            # Retrieve results from the segmentor for this cropped region
            res = self.segmentor.get_results()
        
            # Find the human (label: 'person') with the highest confidence in this region
            highest_confidence = 0
            best_mask = None

            for obj in res.get_objects():
                if obj.label == 'person' and obj.confidence > highest_confidence:
                    highest_confidence = obj.confidence
                    best_mask = obj.mask

            # If we have found a mask with the highest confidence, resize and place it in the final mask
            if best_mask is not None:
                # Resize the mask to fit the bounding box size
                best_mask_resized = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # Place the resized mask back onto the original mask coordinates in final_mask
                final_mask[y:y+h, x:x+w] = cv2.bitwise_or(final_mask[y:y+h, x:x+w], best_mask_resized)

        return extract_shadow_mask(final_mask)