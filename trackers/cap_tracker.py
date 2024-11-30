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
        self.tracker = init_model('trackers/stark_st2_r50_50e_lasot.py', device=device)
        self.tracking = False
        self.bbox = None
        self.count = 0

    def init(self, frame, bbox):
        """
        Initialize the tracker with a new frame and bounding box.

        Parameters:
        - frame: The initial video frame.
        - bbox: The bounding box of the object to track (x, y, w, h).
        """
        self.tracker.initialize(frame, bbox)
        self.tracking = True
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
        if self.tracking:
            result = inference_sot(self.tracker, frame, None)
            self.bbox = result.get('track_results', None)
            self.count += 1

            if self.bbox is None or self.count > 15:
                self.tracking = False

        return self.bbox
