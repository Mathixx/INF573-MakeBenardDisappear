import cv2

class BaseTracker:
    def __init__(self):
        self.tracker = None 
    
    def init(self, frame, bbox):
        """
        Initialize the tracker with the frame and bounding box.
        Parameters:
        - frame: Initial frame (numpy array).
        - bbox: Initial bounding box (x, y, w, h).
        """
        raise NotImplementedError("Subclasses must implement this method")

    def update(self, frame):
        """
        Update the tracker with the new frame.
        Returns the updated bounding box (x, y, w, h) or None if tracking fails.
        """
        raise NotImplementedError("Subclasses must implement this method")
