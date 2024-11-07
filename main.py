# Import necessary modules and classes
from video_processor.detection_frequency import DetectionFrequency, FixedFrequency, TimeFrequency
from video_processor.processor import VideoProcessor
from detectors import Detector, YOLODetector, RedCapDetector
from segmentors import Segmentor, MaskRCNNSegmentor, YoloSegmentor
from removers import Remover, LamaInpaintingRemover, OpenCvInpaintingRemover, BlurringRemover#from detectors.facial_detector import FaceDetector
# et le reste ...


#from benard_suppressor import BenardSupressor

# Function to initialize different components
def initialize_components():
    # Initialize detectors (start with RedCapDetector, but can be swapped with FaceDetector)
    detector = RedCapDetector()  # or FaceDetector() later

    # Initialize segmenter (Mask R-CNN or any other segmentation method)
    segmenter = MaskRCNNSegmentor()

    # Initialize remover (image inpainting)
    remover = OpenCvInpaintingRemover()

    # Initialize Frequency
    frequency = FixedFrequency(1)

    return detector, segmenter, remover, frequency

# Main function to run the video processing
def main():
    # Initialize components (detector, segmenter, and remover)
    detector, segmenter, remover, frequency = initialize_components()

    # Initialize the video processor with the chosen components
    video_processor = VideoProcessor(detector, segmenter, remover, frequency)

    # Start video feed processing (could be from a camera or a video file)
    input_video = 'path/to/video/file.mp4'  # Replace with live feed if necessary
    output_video = 'path/to/output/video.mp4'
    
    # Process the video
    video_processor.process_video(input_video, output_video)

if __name__ == "__main__":
    main()
