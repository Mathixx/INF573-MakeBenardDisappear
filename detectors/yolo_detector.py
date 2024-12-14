import os
from detectors import Detector
from detectors import RedCapDetector
from ikomia.dataprocess.workflow import Workflow
import numpy as np
import cv2
import logging
import warnings

# # Set the logging level to ERROR to suppress detailed YOLOv5 logs
logging.getLogger().setLevel(logging.ERROR)

# # Ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class YOLODetector(Detector):
    def __init__(self, workflow=None, device='cpu',conf_threshold=0.6, iou_threshold=0.3):
        """
        Initialize the YOLOv5 model for human detection in Ikomia.
        Parameters:
        - model_version: The version of the YOLOv5 model to load (default is 'yolov5s').
        - device: 'cpu' or 'cuda' for running on the GPU (currently CPU is default in Ikomia).
        """
        # Initialize Ikomia workflow
        if workflow is None:
            self.workflow = Workflow()
        else:
            self.workflow = workflow

        # Add the YOL task (using Ikomia's built-in YOLOv7 model task)
        self.detector = self.workflow.add_task(name="infer_yolo_v7", auto_connect=True)
        self.red_cap_detector = RedCapDetector()

        self.detector.set_parameters({
            "conf_thres": str(conf_threshold),
            "iou_thres": str(iou_threshold),
            "cuda": str(device == 'cuda')
        })

    def human_detect(self, frame: np.ndarray) -> list:
        """
        Detect humans in the frame using the YOLOv5 model via Ikomia.
        Parameters:
        - frame: The input image (video frame) as a numpy array (in BGR format).
        - conf_threshold: Confidence threshold for filtering detections (default is 0.6).
        - iou_threshold: IoU threshold for non-maximum suppression (default is 0.3).
        Returns:
        - A list of bounding boxes [(x1, y1, w, h)] for detected humans.
        """

        # Convert BGR to RGB (if required)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the workflow to perform detection
        self.workflow.run_on(frame_rgb)

        # Extract detection results
        detection_output = self.detector.get_results()

        # Filter and process the results
        boxes = []
        for obj in detection_output.get_objects():
            if obj.label == "person":
                x1, y1, w, h = obj.box
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                boxes.append((x1, y1, w, h))

        return boxes

    def red_cap_detect(self, frame: np.ndarray) -> list:
        """
        Detect the region where the red cap is most likely located.
        Returns bounding boxes around regions that match the red cap color and shape criteria.
        Parameters:
        - frame: The input image (video frame) as a numpy array (in BGR format).
        Returns:
        - A list of bounding boxes in the form (x, y, w, h) for detected red caps.
          If no cap is found, returns an empty list.
        """

        # Detect the red cap in the frame
        red_cap_boxes = self.red_cap_detector.detect_and_track(frame)

        return red_cap_boxes
    
    def detect(self, frame: np.ndarray) -> list:
        """
        Parameters:
        - frame: The input image (video frame) as a numpy array.

        Returns:
        - An array representing the detection result (e.g., bounding box, mask, etc.).
          If no human with red cap is found, returns None.
        """
        # Detect humans in the frame
        human_Boxes = self.human_detect(frame)
        logging.debug(f"detected {len(human_Boxes)} humans")

        # Detect red caps in the frame
        red_cap_Boxes = self.red_cap_detect(frame)
        logging.debug(f"detected red caps : {red_cap_Boxes}")

        def find_intersecting_human(red_cap_boxes, human_boxes):
            """
            Find the human boxes that intersect the most with each red cap box.
    
             Parameters:
                - red_cap_boxes: List of bounding boxes for the red cap [(x, y, w, h)].
                - human_boxes: List of human bounding boxes [(x1, y1, w, h)].
    
            Returns:
                - A list of the human boxes that intersect the most with each red cap box.
            """
            def intersection_area(box1, box2):
                # Calculate the (x, y, w, h) of the intersection rectangle
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[0] + box1[2], box2[0] + box2[2])
                y2 = min(box1[1] + box1[3], box2[1] + box2[3])

                # Calculate intersection area
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)

                return width * height

            most_intersecting_humans = []

            # For each red cap, find the human box that intersects the most
            for cap in red_cap_boxes:
                max_intersection_area = 0
                best_human_box = None

                for human in human_boxes:
                    # Calculate the intersection area between the red cap and the human box
                    area = intersection_area(cap, human)

                    # Keep track of the human box with the largest intersection
                    if area > max_intersection_area:
                        max_intersection_area = area
                        best_human_box = human

                if best_human_box:
                    most_intersecting_humans.append(best_human_box)

            return most_intersecting_humans

        # Find the human boxes that intersect the most with each red cap box
        intersecting_humans = find_intersecting_human(red_cap_Boxes, human_Boxes)
        logging.debug(f"intersecting humans : {intersecting_humans}")
        
        # Now I noticed that when the interesting human is found near the bottom of the frame, the bounding stops before the bottom of the frame and lets some part of the legs outisde the bounding box.
        # I will try to fix this by extending the bounding box to the bottom of the frame.
        # Extend bounding boxes to the bottom of the frame if close enough
        for i, box in enumerate(intersecting_humans):
            x, y, w, h = box
            # Have the box slighly extend to the bottom of the shoe
            h += min(frame.shape[0] - y - h, 10)
            
            bottom_edge = y + h  # Current bottom of the box
            threshold = h / 20  # Threshold as 1/10th of the height of the box

            if frame.shape[0] - bottom_edge <= threshold:  # If within threshold from the bottom
                h = frame.shape[0] - y  # Adjust the height to extend to the bottom
                intersecting_humans[i] = (x, y, w, h)

        return intersecting_humans, human_Boxes, red_cap_Boxes

    def log_tracking_stats(self, frame_count: int):
        """
        Log the tracking statistics for the red cap detector.
        """
        logging.info(f"Red cap detected without tracking count: {self.red_cap_detector.red_cap_detected_count}")
        logging.info(f"Additional red cap detected with tracking count: {self.red_cap_detector.red_cap_detected_with_tracker_count}")
        logging.info(f"Total red cap detected count: {self.red_cap_detector.red_cap_detected_count + self.red_cap_detector.red_cap_detected_with_tracker_count}")
        logging.info(f"Total frame count: {frame_count}")
        logging.info(f"Red cap detection rate without tracker: {(self.red_cap_detector.red_cap_detected_count) / frame_count * 100:.2f}%")
        logging.info(f"Red cap detection rate with tracker: {(self.red_cap_detector.red_cap_detected_count + self.red_cap_detector.red_cap_detected_with_tracker_count) / frame_count * 100:.2f}%")
        logging.info(f"Percent increase in detection rate with tracker: {(self.red_cap_detector.red_cap_detected_with_tracker_count) / self.red_cap_detector.red_cap_detected_count * 100:.2f}%")
        logging.info("")

    def log_tracking_time(self, frame_count: int):
        """
        Log the mean time spent on initializing and updating the tracker.
        """
        logging.info(f"Mean time spent on initializing tracker: {self.red_cap_detector.total_time_init_tracker / frame_count:.4f} seconds")
        logging.info(f"Mean time spent on updating tracker: {self.red_cap_detector.total_time_update_tracker / self.red_cap_detector.red_cap_detected_with_tracker_count if self.red_cap_detector.red_cap_detected_with_tracker_count != 0 else 0:.4f} seconds")