from .tracker import BaseTracker
import cv2

class OpenCVTracker(BaseTracker):
    def __init__(self, tracker_type="CSRT"):
        super().__init__()
        trackers = {
            "CSRT": lambda: cv2.legacy.TrackerCSRT_create(),
            "KCF": lambda: cv2.legacy.TrackerKCF_create(),
            "MOSSE": lambda: cv2.legacy.TrackerMOSSE_create(),
        }
        if tracker_type not in trackers:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
        self.tracker = trackers[tracker_type]()

    def init(self, frame, bbox):
        self.tracker.init(frame, bbox)

    def update(self, frame):
        success, bbox = self.tracker.update(frame)
        return bbox if success else None
