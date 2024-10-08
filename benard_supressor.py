class BenardSupressor:
    def __init__(self, detector, segmenter, remover):
        self.detector = detector
        self.segmenter = segmenter
        self.remover = remover

    def process_video(self, input_path, output_path):
        # Open video input (live feed or file)
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Step 1: Detect the object (e.g., person with red cap)
            detection = self.detector.detect(frame)

            # Step 2: Segment the object
            segmented = self.segmenter.segment(detection)

            # Step 3: Remove the object
            final_frame = self.remover.remove(segmented, frame)

            # Write processed frame to output video
            out.write(final_frame)

        cap.release()
        out.release()
