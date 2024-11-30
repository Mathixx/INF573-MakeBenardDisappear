# trackers/multi_object_tracker.py
from .open_cv_trackers import OpenCVTracker

class MultiObjectTracker:
    def __init__(self, tracker_type="CSRT"):
        self.trackers = []  # List of trackers for multiple objects
        self.tracker_type = tracker_type

    def add_tracker(self, frame, bbox):
        tracker = OpenCVTracker(tracker_type=self.tracker_type)
        tracker.init(frame, bbox)
        self.trackers.append(tracker)

    def update(self, frame):
        """
        Updates all trackers with the current frame.
        Returns:
        - A list of bounding boxes [(x, y, w, h)] for tracked objects.
        """
        updated_boxes = []
        for tracker in self.trackers:
            bbox = tracker.update(frame)
            if bbox:
                updated_boxes.append(bbox)

        return updated_boxes
