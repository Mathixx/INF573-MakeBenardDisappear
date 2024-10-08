# Import necessary modules and classes
from detectors.red_cap_detector import RedCapDetector
from detectors.facial_detector import FaceDetector
# et le reste ...


from benard_suppressor import BenardSupressor

# Function to initialize different components
def initialize_components():
    # Initialize detectors (start with RedCapDetector, but can be swapped with FaceDetector)
    detector = RedCapDetector()  # or FaceDetector() later

    # Initialize segmenter (Mask R-CNN or any other segmentation method)
    segmenter = MaskRCNNSegmenter()

    # Initialize remover (image inpainting)
    remover = InpaintingRemover()

    return detector, segmenter, remover

# Main function to run the video processing
def main():
    # Initialize components (detector, segmenter, and remover)
    detector, segmenter, remover = initialize_components()

    # Initialize the video processor with the chosen components
    video_processor = VideoProcessor(detector, segmenter, remover)

    # Start video feed processing (could be from a camera or a video file)
    input_video = 'path/to/video/file.mp4'  # Replace with live feed if necessary
    output_video = 'path/to/output/video.mp4'
    
    # Process the video
    video_processor.process_video(input_video, output_video)

if __name__ == "__main__":
    main()
