from .tracker import BaseTracker
import cv2
import logging

class OpenCVTracker(BaseTracker):
    def __init__(self, tracker_type="KCF"):
        self.tracker = None
        self.initialized = False

        trackers = {
            "MOSSE": lambda: cv2.legacy.TrackerMOSSE_create(),
            "KCF": lambda: cv2.legacy.TrackerKCF_create(),
        }

        if tracker_type not in trackers:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")

        self.tracker = trackers[tracker_type]()
        if self.tracker is None:
            raise ValueError(f"Tracker type {tracker_type} is not available in OpenCV.")

    def init(self, frame, bbox):
        """
        Initialize the tracker with a frame and bounding box.

        Parameters:
        - frame: The video frame (numpy array).
        - bbox: The bounding box (x, y, w, h).
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided for tracker initialization.")
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"Invalid bounding box: {bbox}")
        logging.debug(f"Initializing tracker with bbox: {bbox} and frame shape: {frame.shape}")

        self.tracker.init(frame, bbox)
        self.initialized = True

    def update(self, frame):
        """
        Update the tracker with the current frame.

        Parameters:
        - frame: The video frame (numpy array).

        Returns:
        - bbox: The updated bounding box (x, y, w, h) or None if tracking fails.
        """
        if not self.initialized:
            raise RuntimeError("Tracker update called without initialization.")
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame provided for tracker update.")

        logging.debug(f"Updating tracker with frame shape: {frame.shape}")
        success, bbox = self.tracker.update(frame)
        if not success:
            logging.info("Tracking failed.")
            return None
        return bbox
