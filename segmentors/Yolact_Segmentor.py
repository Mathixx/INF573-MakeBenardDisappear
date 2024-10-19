import torch
import cv2
import numpy as np
from segmentors import Segmentor

# Import YOLACT from the yolact package
import sys
sys.path.append('yolact')


# Import YOLACT from the yolact package
from utils.augmentations import FastBaseTransform
from yolact import Yolact
from data import cfg, set_cfg
from layers.output_utils import postprocess


class YOLACTSegmentor(Segmentor):
    """
    Segmentor implementation using YOLACT.
    Only segments humans (class 0 in the COCO dataset).
    """
    def __init__(self, model_path='yolact_plus_resnet50_54_800000.pth'):
        # Set YOLACT configuration (based on ResNet50 in this case)
        set_cfg('yolact_plus_resnet50_config')

        # Initialize the YOLACT model
        self.model = Yolact()
        self.model.load_weights(model_path)
        self.model.eval()
        
        # Use the device available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def segment(self, frame: np.ndarray, bounding_boxes: list) -> np.ndarray:
        """
        Perform segmentation using YOLACT on the input frame, restricted to the bounding boxes.
        Only humans (class 0) are segmented.
        """
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Convert the frame to tensor and prepare it for YOLACT
        frame_tensor = torch.from_numpy(frame).float().to(self.device).unsqueeze(0)
        frame_tensor = FastBaseTransform()(frame_tensor)

        # Perform inference using YOLACT
        with torch.no_grad():
            preds = self.model(frame_tensor)

        # Post-process the results to get masks, classes, and scores
        h, w, _ = frame.shape
        dets_out = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0.3)

        # Extract masks, classes, and bounding boxes from postprocessed results
        masks = dets_out[3].permute(1, 2, 0).cpu().numpy()
        classes = dets_out[0].cpu().numpy()

        # Loop over bounding boxes to filter out human instances and apply segmentation
        for (x1, y1, w, h) in bounding_boxes:
            if w > 0 and h > 0:
                # Extract masks corresponding to the 'person' class (class index 0 in COCO)
                person_mask = np.zeros((h, w), dtype=np.uint8)
                for i in range(masks.shape[-1]):
                    if classes[i] == 0:  # Class 0 corresponds to 'person' in COCO
                        mask = masks[:, :, i].astype(np.uint8) * 255
                        mask_resized = cv2.resize(mask, (w, h))
                        combined_mask[y1:y1+h, x1:x1+w] = np.maximum(combined_mask[y1:y1+h, x1:x1+w], mask_resized)

        return combined_mask
