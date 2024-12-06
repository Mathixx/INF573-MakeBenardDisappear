import cv2 
import numpy as np
#from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg
#from detectron2 import model_zoo
from segmentors import Segmentor

class MaskRCNNSegmentor(Segmentor):
    """
    Segmentor implementation using Mask R-CNN (via Detectron2).
    Only segments humans (class 0 in the COCO dataset).
    """
    def __init__(self):
        # Initialize the Mask R-CNN model using Detectron2
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(self.cfg)

    def segment(self, frame: np.ndarray, bounding_boxes: list) -> np.ndarray:
        """
        Perform segmentation using Mask R-CNN on the input frame, restricted to the bounding boxes.
        Only humans (class 0) are segmented.
        """
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Loop over bounding boxes
        for (x1, y1, w, h) in bounding_boxes:
            # Check for valid bounding box dimensions
            if w > 0 and h > 0:
                # Crop the region within the bounding box
                cropped_region = frame[y1:y1+h, x1:x1+w, :]

                # Perform segmentation within the bounding box region
                outputs = self.predictor(cropped_region)

                # Get predicted classes and masks
                classes = outputs["instances"].pred_classes.to("cpu").numpy()  # Class labels
                masks = outputs["instances"].pred_masks.to("cpu").numpy()      # Segmentation masks

                # Filter only the masks corresponding to the 'person' class (class index 0)
                person_masks = masks[classes == 0]

                # Insert the person masks back into the full-frame mask
                # Insert the person masks back into the full-frame mask
                for mask in person_masks:
                    mask = mask.astype(np.uint8)  # Convert boolean mask to uint8
                    mask_resized = cv2.resize(mask, (w, h))  # Resize mask to fit bounding box
                    combined_mask[y1:y1+h, x1:x1+w] = np.maximum(combined_mask[y1:y1+h, x1:x1+w], mask_resized * 255)
            else:
                print(f"Invalid bounding box: {(x1, y1, w, h)}")  # Log invalid boxes

        return combined_mask

