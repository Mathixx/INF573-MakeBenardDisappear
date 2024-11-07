import cv2
import numpy as np
from detection_frequency import DetectionFrequency, FixedFrequency, TimeFrequency
from detectors import Detector, YOLODetector, RedCapDetector
from segmentors import Segmentor, MaskRCNNSegmentor, YoloSegmentor
from removers import Remover, LamaInpaintingRemover, OpenCvInpaintingRemover, BlurringRemover

class VideoProcessor():
    def __init__(self, detector:Detector, segmentor:Segmentor, remover:Remover, frequency: DetectionFrequency) -> None:
        self.detector = detector
        self.segmentor = segmentor
        self.remover = remover
        self.frequency = frequency

    def process_video(self, input_video: str, output_video: str) -> None:
        """
        Process a video to remove objects detected by the detector.
        Parameters:
        - input_video: The path to the input video file.
        - output_video: The path to save the output video.
        """
        video = cv2.VideoCapture(input_video)
        if not video.isOpened():
            raise FileNotFoundError("Could not open video file")
        else :
            frame_count = 0
            ret, frame = video.read()

            while ret:
                if self.frequency.select_next(frame_count) == frame_count:
                    # Detect objects in the frame
                    final_frame = self.process_image(frame)

                    # Save the frame to the output video
                    cv2.imwrite(output_video, final_frame)

                frame_count += 1
                ret, frame = video.read()

    def process_image(self, frame:np.array) -> np.array:
        """
        Process an image to remove objects detected by the detector.
        Parameters:
        - input_image: The path to the input image file.
        Returns:
        - The processed image with objects removed.
        """
        # Detect objects in the frame
        bounding_boxes = self.detector.detect(frame)

        # Segment the objects in the frame
        mask = self.segmentor.segment(frame, bounding_boxes)

        # Remove the objects from the frame
        final_frame = self.remover.remove(frame, mask)

        return final_frame