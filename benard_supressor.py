import cv2
import numpy as np

class BenardSupressor:
    def __init__(self, detector, segmenter, remover):
        self.detector = detector
        self.segmenter = segmenter
        self.remover = remover

    def process_video(self, input_video_path, output_video_path):
        video_capture = cv2.VideoCapture(input_video_path)

        print(f"Processing video: {input_video_path}")
    
        # Get video properties
        original_frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        print(f"Original video properties: {original_frame_width}x{original_frame_height} @ {fps} FPS")

        # Initialize variables
        video_writer = None
        frame_count = 0
    
        # Process each frame
        while video_capture.isOpened():
            print(f"Processing frame {frame_count}")
            ret, frame = video_capture.read()
            if not ret:
                break  # End of video stream
        
            # Process the frame with the process_image method
            processed_frame = self.process_image(frame)

            # Convert to uint8 if not already
            if processed_frame.dtype != 'uint8':
                print("Converting to uint8")
                processed_frame = processed_frame.astype('uint8')

            # Check for and handle RGBA
            if processed_frame.ndim == 3 and processed_frame.shape[2] == 4:
                print("Converting RGBA to RGB")
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2RGB)

            # Check if a pixel has value > 255
            if (processed_frame > 255).any():
                print("Clipping values to 255")
                processed_frame = np.clip(processed_frame, 0, 255)
            

            # Set up the output video writer based on the processed frame's dimensions on the first frame
            if video_writer is None:
                processed_frame_height, processed_frame_width = processed_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (processed_frame_width, processed_frame_height))
        
            # Write the processed frame to the output video
            video_writer.write(processed_frame)

            # if frame_count % 10 == 0:
            #     # save the frame
            #     cv2.imwrite(f"frame_{frame_count}.png", processed_frame)

            frame_count += 1
            # if frame_count > 100:
            #     break  # Limit processing to the first 100 frames as specified

        # Release resources
        video_capture.release()
        if video_writer is not None:
            video_writer.release()


    def process_image(self, input_image):
        # Step 1: Detect the object (e.g., person with red cap)
        detection_boxes = self.detector.detect(input_image)

        # Step 2: Segment the object
        segmented_mask = self.segmenter.segment(input_image, detection_boxes)

        # Step 3: Remove the object
        final_image = self.remover.remove(input_image, segmented_mask)

        return final_image
