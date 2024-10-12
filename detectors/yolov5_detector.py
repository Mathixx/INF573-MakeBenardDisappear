import torch
import numpy as np
import cv2
import logging
import warnings
import contextlib
import os
from detector import Detector

# Set the logging level to ERROR to suppress detailed YOLOv5 logs
logging.getLogger().setLevel(logging.ERROR)

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class YOLOv5Detector(Detector):
    def __init__(self, model_path='yolov5s.pt', device='cpu'):
        """
        Initialize the YOLOv5 model for human detection.
        Parameters:
        - model_path: The path to the YOLOv5 model (default is 'yolov5s.pt').
        - device: 'cpu' or 'cuda' for running on the GPU.
        """
        self.device = device
        # Temporarily suppress stdout and stderr to hide model loading messages
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            # Load the YOLOv5 model quietly
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, verbose=False)
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode

    def detect(self, frame: np.ndarray, conf_threshold = 0.6, iou_threshold = 0.3) -> list:
        """
        Detect humans in the frame using the YOLOv5 model.
        Parameters:
        - frame: The input image (video frame) as a numpy array (in BGR format).
        - conf_threshold: Confidence threshold for filtering detections (default is 0.6).
        - iou_threshold: IoU threshold for non-maximum suppression (default is 0.3).
        Returns:
        - A list of bounding boxes [(x1, y1, x2, y2)] for detected humans.
        """
        # Convert the frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = self.model(frame)

        # Extract detection results in xyxy format
        filtered_results = results.xyxy[0]

        # Prepare bounding boxes, confidences, and class IDs for NMS
        boxes = []
        confidences = []
        class_ids = []

        for *box, conf, cls in filtered_results:
            if int(cls) == 0 and conf >= conf_threshold:  # Class index 0 is "person"
                x1, y1, x2, y2 = map(int, box)
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to (x, y, width, height)
                confidences.append(float(conf))
                class_ids.append(int(cls))

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

        print("indices", indices)

        # Filter the bounding boxes using NMS results
        detected_boxes = []

        if len(indices) > 0:  # Check if any indices were returned
            for i in indices.flatten():  # Use .flatten() to ensure compatibility with both cases
                box = boxes[i]
                detected_boxes.append(box)
        
        return detected_boxes


def find_intersecting_human(red_cap_boxes, human_boxes):
    """
    Find the human box that intersects with the red cap box.
    Parameters:
    - red_cap_box: Bounding box for the red cap (x, y, w, h).
    - human_boxes: List of human bounding boxes [(x1, y1, x2, y2)].
    Returns:
    - The human boxes that intersects with a red cap box, or None if no intersection is found.

    POTENTIAL IMPROVEMENT : KEEP THE BOX WITH THE LARGEST INTERSECTION
    """
    intersections = []
    for cap in red_cap_boxes:
        for human in human_boxes:
            cap_x, cap_y, cap_w, cap_h = cap
            human_x, human_y, human_w, human_h = human
            if cap_x < human_x + human_w and cap_x + cap_w > human_x and cap_y < human_y + human_h and cap_y + cap_h > human_y:
                intersections.append(human)

    return intersections if intersections else None
