import numpy as np
import time

class DetectionFrequency():
    
    def __init__(self) -> None:
        pass

    def select_next(self, actual_frame: int) -> int:
        """
        Select the next frame to process based on the actual frame number.
        Parameters:
        - actual_frame: The actual frame number.
        
        Returns:
        - The next frame number to process.
        """
        pass

class FixedFrequency(DetectionFrequency):
    
    def __init__(self, frequency: int) -> None:
        self.last = -1
        self.frequency = frequency

    def select_next(self, actual_frame: int) -> int:
        if actual_frame - self.last >= self.frequency:
            self.last = actual_frame
            return True
        return False

class TimeFrequency(DetectionFrequency):
    
    def __init__(self, fps=30.0) -> None:
        self.start_time = time.time()
        self.fps = fps

    def select_next(self, actual_frame: int) -> int:
        actual_time = time.time()
        if actual_time - self.start_time >= actual_frame/self.fps:
            self.start_time = actual_time
            return True
        return False