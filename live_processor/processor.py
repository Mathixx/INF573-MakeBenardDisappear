import cv2
import os
import logging
import numpy as np
from detectors import Detector
from segmentors import Segmentor

import sys
sys.path.append('video_processor')
from video_processor.detection_frequency import DetectionFrequency, FixedFrequency, TimeFrequency

class LiveBenardSupressor():
    def __init__(self, detector:Detector, segmentor:Segmentor, background_sub:str, frequency: DetectionFrequency) -> None:
        self.detector = detector
        self.segmentor = segmentor
        if background_sub == 'MOG2':
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        elif background_sub == 'KNN':
            self.backSub = cv2.createBackgroundSubtractorKNN()
        else:
            raise ValueError("Background subtraction method must be 'MOG2' or 'KNN'")
        self.frequency = frequency

    def process_live(self, output_folder: str, debugging_frames_level='None', debugging_video_level='None') -> None:
        """
        Process a video to remove objects detected by the detector.
        Parameters:
        - input_path: The path to the input video.
        - output_folder: The folder where the output video will be saved.
        - debugging_frames_level: The level of debugging for the frames ('None', 'Detector', 'Segmentor', 'Remover', 'All').
        - debugging_video_level: The level of debugging for the video ('None', 'Detector', 'Segmentor', 'All').
        """
        # Open the default camera (webcam)
        video = cv2.VideoCapture(0)  # 0 is the default camera. Adjust if using a different camera.
        if not video.isOpened():
            logging.warning("Frame capture failed. Exiting.")
            raise RuntimeError("Cannot open live camera feed")

        original_frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        logging.debug(f"Original video properties: {original_frame_width}x{original_frame_height} @ {fps} FPS")

    
        #input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_folder = output_folder + f"live_output/"
        #output_path = output_folder + f"{input_filename}_processed.mp4"
        debugging_folder = output_folder + 'debugging/'
        if not os.path.exists(debugging_folder):
            os.makedirs(debugging_folder)

        output_video = None
        frame_count = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Feed the frame to backSub to update the background model
            _ = self.backSub.apply(frame)  # Update background model

            bounding_boxes, mask, final_frame = self.process_image(frame, debugging_folder, frame_count, debugging_frames_level)

            logging.debug(f"Processed frame {frame_count}")

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        video.release()
        cv2.destroyAllWindows()

    def process_image(self, frame:np.array, output_folder, frame_count:int=None, debugging_frames_level='None') -> np.array:
        """
        Process an image to remove objects detected by the detector.
        Parameters:
        - frame: the frame that will be processed.
        - output_file: The path to save the output image.
        Returns:
        - The processed image with objects removed.
        """
        # Detect objects in the frame
        bounding_boxes, human_Boxes, red_cap_Boxes = self.detector.detect(frame)

        boxed_frame = self.detector.draw_boxes(frame, bounding_boxes)

        # Segment the objects in the frame
        mask = self.segmentor.segment(frame, bounding_boxes)

        masked_frame = self.segmentor.draw_masks(frame, mask)
        
        # Replace the target individual's region with the background
        background = self.backSub.getBackgroundImage()
        if background is None:
            logging.error("Background model is not initialized.")
            return bounding_boxes, mask, frame

        # Apply the mask to the background
        background_cropped = cv2.bitwise_and(background, background, mask=mask)

        # Apply the inverse mask to the original frame
        frame_cropped = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

        # Combine the results
        final_frame = cv2.add(frame_cropped, background_cropped)

        # Display the combined frame
        cv2.imshow('Initial Video', frame)
        cv2.imshow('Boxed Video', boxed_frame)
        cv2.imshow('Masked Video', masked_frame)
        cv2.imshow('Final Video', final_frame)
        #cv2.resizeWindow('All Videos', int(1920/2), int(1080/2))

        return bounding_boxes, mask, final_frame

    def stack_images(self, frame: np.array, boxed_frame: np.array, masked_frame: np.array, final_frame: np.array) -> np.array:
        """
        Stack the input images in a 2 by 2 shape, adding labels.
    
        Parameters:
        - frame: The original frame.
        - boxed_frame: The frame with bounding boxes drawn.
        - masked_frame: The frame with masks drawn.
        - final_frame: The final frame with objects removed.
    
        Returns:
        - The stacked image with labels.
        """
        # Find the maximum width and height among all frames
        height = final_frame.shape[0]
        width = final_frame.shape[1]

        # Resize each frame to the maximum dimensions
        resized_frame = cv2.resize(frame, (width, height))
        resized_boxed_frame = cv2.resize(boxed_frame, (width, height))
        resized_masked_frame = cv2.resize(masked_frame, (width, height))
        resized_final_frame = cv2.resize(final_frame, (width, height))

        # Add labels to each frame
        labels = ["Original Frame", "Boxed Frame", "Masked Frame", "Final Frame"]
        self.add_label(resized_frame, labels[0])
        self.add_label(resized_boxed_frame, labels[1])
        self.add_label(resized_masked_frame, labels[2])
        self.add_label(resized_final_frame, labels[3])

        # Stack frames in a 2x2 grid
        top_row = np.hstack([resized_frame, resized_boxed_frame])
        bottom_row = np.hstack([resized_masked_frame, resized_final_frame])
        return np.vstack([top_row, bottom_row])


    @staticmethod
    def add_label(image, text, font_scale=1, thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)  # White color for text
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Place text at the top center of the image
        text_x = (image.shape[1] - text_width) // 2
        text_y = text_height + 10  # Slight padding from the top
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

