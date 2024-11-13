import cv2
import os
import numpy as np
from .detection_frequency import DetectionFrequency, FixedFrequency, TimeFrequency
from detectors import Detector, YOLODetector, RedCapDetector
from segmentors import Segmentor, MaskRCNNSegmentor, YoloSegmentor
from removers import Remover, LamaInpaintingRemover, OpenCvInpaintingRemover, BlurringRemover

class BenardSupressor():
    def __init__(self, detector:Detector, segmentor:Segmentor, remover:Remover, frequency: DetectionFrequency) -> None:
        self.detector = detector
        self.segmentor = segmentor
        self.remover = remover
        self.frequency = frequency

    def process_video(self, input_path: str, output_folder: str, debugging_frames_level='None', debugging_video_level='None') -> None:
        """
        Process a video to remove objects detected by the detector.
        Parameters:
        - input_path: The path to the input video.
        - output_folder: The folder where the output video will be saved.
        - debugging_frames_level: The level of debugging for the frames ('None', 'Detector', 'Segmentor', 'Remover', 'All').
        - debugging_video_level: The level of debugging for the video ('None', 'Detector', 'Segmentor', 'All').
        """
        video = cv2.VideoCapture(input_path)
        if not video.isOpened():
            raise FileNotFoundError("Could not open video file")
        original_frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        print(f"Original video properties: {original_frame_width}x{original_frame_height} @ {fps} FPS")

    
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_folder = output_folder + f"{input_filename}_output/"
        output_path = output_folder + f"{input_filename}_processed.mp4"
        debugging_folder = output_folder + 'debugging/'
        if not os.path.exists(debugging_folder):
            os.makedirs(debugging_folder)

        output_video = None
        frame_count = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            if self.frequency.select_next(frame_count):
                
                bounding_boxes, mask, final_frame = self.process_image(frame, debugging_folder, frame_count, debugging_frames_level)
                frame_height, frame_width = frame.shape[:2]

                if output_video is None:
                    processed_frame_height, processed_frame_width = final_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_video = cv2.VideoWriter(output_path, fourcc, fps, (processed_frame_width, processed_frame_height))
                    print(f"Output video properties: {processed_frame_width}x{processed_frame_height} @ {fps} FPS")
                    if debugging_video_level == 'Detector' or debugging_video_level == 'All':
                        detector_video = cv2.VideoWriter(debugging_folder + 'detector.mp4', fourcc, fps, (frame_width, frame_height))
                    
                    if debugging_video_level == 'Segmentor' or debugging_video_level == 'All':
                        segmentor_video = cv2.VideoWriter(debugging_folder + 'segmentor.mp4', fourcc, fps, (frame_width, frame_height))
                    
                    if debugging_video_level == 'All':
                        all_video = cv2.VideoWriter(debugging_folder + 'all.mp4', fourcc, fps, (processed_frame_width*2, processed_frame_height*2))

                print(f"Processed frame {frame_count}")

                if debugging_video_level == 'Detector' or debugging_video_level == 'All':
                    boxed_frame = self.detector.draw_boxes(frame, bounding_boxes)
                    detector_video.write(boxed_frame)

                if debugging_video_level == 'Segmentor' or debugging_video_level == 'All':
                    masked_frame = self.segmentor.draw_masks(frame, mask)
                    segmentor_video.write(masked_frame)

                if debugging_video_level == 'All':
                    output_image = self.stack_images(frame, boxed_frame, masked_frame, final_frame)
                    all_video.write(output_image)

                # Save the frame to the output video
                output_video.write(final_frame)

            frame_count += 1

        video.release()
        output_video.release()
        if debugging_video_level == 'Detector' or debugging_video_level == 'All':
            detector_video.release()
        if debugging_video_level == 'Segmentor' or debugging_video_level == 'All':
            segmentor_video.release()

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
        bounding_boxes = self.detector.detect(frame)

        if debugging_frames_level == 'Detector' or debugging_frames_level == 'All':
            if not os.path.exists(output_folder + 'detector/'):
                os.makedirs(output_folder + 'detector/')

            boxed_frame = self.detector.draw_boxes(frame, bounding_boxes)
            cv2.imwrite(output_folder + f"detector/frame_{frame_count}.jpg", boxed_frame)

        # Segment the objects in the frame
        mask = self.segmentor.segment(frame, bounding_boxes)

        if debugging_frames_level == 'Segmentor' or debugging_frames_level == 'All':
            if not os.path.exists(output_folder + 'segmentor/'):
                os.makedirs(output_folder + 'segmentor/')

            masked_frame = self.segmentor.draw_masks(frame, mask)
            cv2.imwrite(output_folder + f"segmentor/frame_{frame_count if frame_count is not None else 0}.jpg", masked_frame)

        # Remove the objects from the frame
        final_frame = self.remover.remove(frame, mask)

        if debugging_frames_level == 'Remover' or debugging_frames_level == 'All':
            if not os.path.exists(output_folder + 'remover/'):
                os.makedirs(output_folder + 'remover/')

            cv2.imwrite(output_folder + f"remover/frame_{frame_count if frame_count is not None else 0}.jpg", final_frame)

        if debugging_frames_level == 'All':
            if not os.path.exists(output_folder + 'all/'):
                os.makedirs(output_folder + 'all/')

            output_image = self.stack_images(frame, boxed_frame, masked_frame, final_frame)
            cv2.imwrite(output_folder + f"all/frame_{frame_count if frame_count is not None else 0}.jpg", output_image)
   
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

