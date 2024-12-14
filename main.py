import time
import argparse
import cv2
import logging
from video_processor.detection_frequency import DetectionFrequency, FixedFrequency, TimeFrequency
from video_processor.processor import BenardSupressor
from live_processor.processor import LiveBenardSupressor
from detectors import Detector, YOLODetector, RedCapDetector
from segmentors import Segmentor, MaskRCNNSegmentor, YoloSegmentor
from removers import Remover, LamaInpaintingRemover, OpenCvInpaintingRemover, BlurringRemover, BetterLamaInpaintingRemover

########################################################
######### Define the input and output paths ############
########################################################
# These are used as defaults; you can override them via arguments.
default_input = "_test_data/test_4.mp4"
default_output_folder = "_test_data/"
########################################################

########################################################
################ Debugging options #####################
########################################################
default_debugging_frames_level = 'All'  # 'None', 'Detector', 'Segmentor', 'Remover', 'All', 'complete_detector'
default_debugging_video_level = 'All'  # 'None', 'Detector', 'Segmentor', 'All'
########################################################

########################################################
######## To complete to choose the components ##########
########################################################
def initialize_components(type:str):
    # To select between YOLOv5 and RedCap detectors
    detector = YOLODetector(device='cuda')

    # To select between YOLOv5 and Mask R-CNN segmentor
    segmenter = YoloSegmentor(device='cuda')

    if type == 'video' or type == 'photo':
        # To select between LAMA, Blurring and OpenCV inpainting remover
        remover = LamaInpaintingRemover()
    else:
        remover = None

    # To select between FixedFrequency and TimeFrequency
    frequency = FixedFrequency(1)

    return detector, segmenter, remover, frequency
########################################################
########################################################
########################################################

logging.basicConfig(
    filename='main.log',          # Name of the log file
    filemode='w',
    level=logging.DEBUG,          # Minimum level to log
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    datefmt='%Y-%m-%d %H:%M:%S'  # Timestamp format
)

def main():
    parser = argparse.ArgumentParser(description="Process video, photo, or live feed with Benard Supressor.")
    parser.add_argument(
        '--type', choices=['video', 'photo', 'live'], required=True,
        help="Type of input to process: 'video', 'photo', or 'live'."
    )
    parser.add_argument('--input', type=str, default=default_input,
                        help="Input path for video or photo. Ignored if type is 'live'.")
    parser.add_argument('--output', type=str, default=default_output_folder,
                        help="Output folder for processed files.")
    parser.add_argument('--debug_frames', type=str, choices=['None', 'Detector', 'Segmentor', 'Remover', 'All', 'complete_detector'],
                        default=default_debugging_frames_level,
                        help="Level of debugging for individual frames.")
    parser.add_argument('--debug_video', type=str, choices=['None', 'Detector', 'Segmentor', 'All'],
                        default=default_debugging_video_level,
                        help="Level of debugging for videos.")

    args = parser.parse_args()

    logging.info("Starting Benard Supressor...")
    logging.info(f"Selected type: {args.type}")
    logging.info(f"Input path: {args.input}")
    logging.info(f"Output folder: {args.output}")

    logging.info("Initializing components...\n")
    detector, segmenter, remover, frequency = initialize_components(type=args.type)
    
    if args.type == 'video':
        benard_supressor = BenardSupressor(detector, segmenter, remover, frequency)
        logging.debug("Initialization done")

        start = time.time()

        logging.info("Processing video...")
        benard_supressor.process_video(args.input, args.output, args.debug_frames, args.debug_video)
    elif args.type == 'photo':
        benard_supressor = BenardSupressor(detector, segmenter, remover, frequency)
        logging.debug("Initialization done")

        start = time.time()

        logging.debug("Processing photo...")
        photo = cv2.imread(args.input)
        benard_supressor.process_image(photo, args.output, debugging_frames_level=args.debug_frames)
    elif args.type == 'live':
        logging.debug("Processing live feed...")
        live_benard_supressor = LiveBenardSupressor(detector, segmenter, 'MOG2', frequency)

        start = time.time()

        live_benard_supressor.process_live(args.output, args.debug_frames, args.debug_video)
    else:
        raise ValueError("Invalid type. Choose between 'video', 'photo', or 'live'.")
    
    end = time.time()
    logging.info(f"Processing time: {end - start}")

if __name__ == "__main__":
    main()

########################################################
################ Command examples ######################
########################################################
# Process a video :
# python main.py --type video --input path/to/video.mp4 --output output_folder/
# Process a photo :
# python main.py --type photo --input path/to/photo.jpg --output output_folder/
# Process a live feed :
# python main.py --type live --output output_folder/
########################################################