import numpy as np

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
        self.frequency = frequency

    def select_next(self, actual_frame: int) -> int:
        return actual_frame + self.frequency
    

class TimeFrequency(DetectionFrequency):
    
    def __init__(self, time: int) -> None:
        self.time = time

    def select_next(self, actual_frame: int) -> int:
        return actual_frame + self.time