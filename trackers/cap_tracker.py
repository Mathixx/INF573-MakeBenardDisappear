import cv2
from mmtrack.apis import init_model, inference_sot
from .tracker import BaseTracker

class RedCapTracker(BaseTracker):
    def __init__(self, device='cuda:0'):
        """
        Initialize the RedCapTracker.

        Parameters:
        - device: Device to run the tracker on ('cuda:0' or 'cpu').
        """
        self.tracker = init_model('trackers/stark_st2_r50_50e_lasot.py')
        self.tracking = False
        self.bbox = None

    def init(self, frame, bbox):
        """
        Initialize the tracker with a new frame and bounding box.

        Parameters:
        - frame: The initial video frame.
        - bbox: The bounding box of the object to track (x, y, w, h).
        """
        self.tracker.initialize(frame, bbox)
        self.tracking = True
        self.detector_found_object = True
        self.bbox = bbox
        self.count = 0

    def update(self, frame):
        """
        Perform tracking on the current frame.

        Parameters:
        - frame: The current video frame.

        Returns:
        - bbox: The tracked bounding box (x, y, w, h) or None if tracking failed.
        """
        # Try detecting the object first
        detected_bbox = self.detector_found_object

        if detected_bbox:
            # If the detector finds the object, update the tracker and return the bbox
            self.init(frame, detected_bbox)
            return detected_bbox

        # If detection fails, use the tracker
        if self.tracking:
            result = inference_sot(self.tracker, frame, None)
            self.bbox = result.get('track_results', None)
            count += 1

            # Stop tracking if the tracker fails
            if self.bbox is None or count > 15:
                self.tracking = False

        return self.bbox
