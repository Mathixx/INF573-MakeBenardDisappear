import time
from video_processor.detection_frequency import DetectionFrequency, FixedFrequency, TimeFrequency
from video_processor.processor import BenardSupressor
from detectors import Detector, YOLODetector, RedCapDetector
from segmentors import Segmentor, MaskRCNNSegmentor, YoloSegmentor
from removers import Remover, LamaInpaintingRemover, OpenCvInpaintingRemover, BlurringRemover
#from benard_suppressor import BenardSupressor


########################################################
######### Define the input and output paths ############
########################################################
type = "video" # "video" or "photo"
input = "_test_data/test_3.mp4"
output_folder = "_test_data/"
########################################################
 
########################################################
################ Debugging options #####################
########################################################
debugging_frames_level = 'All' #'None', 'Detector', 'Segmentor', 'Remover', 'All'
debugging_video_level = 'All' #'None', 'Detector', 'Segmentor', 'All'
########################################################

########################################################
######## To complete to choose the components ##########
########################################################
def initialize_components():
    # To select between YOLOv5 and RedCap detectors
    detector = YOLODetector(device='cuda')

    # To select between YOLOv5 and Mask R-CNN segment
    segmenter = YoloSegmentor(device='cuda')

    # To select between LAMA, Bluring and OpenCV inpainting remover
    remover = LamaInpaintingRemover()

    # To select between FixedFrequency and TimeFrequency
    frequency = FixedFrequency(1)

    return detector, segmenter, remover, frequency
########################################################
########################################################
########################################################



def main():
    print("Initialising components..."+'\n')
    detector, segmenter, remover, frequency = initialize_components()

    benard_supressor = BenardSupressor(detector, segmenter, remover, frequency)
    print("Initialisation done")
    
    start = time.time()
    if type == 'video':
        print("Processing video...")
        benard_supressor.process_video(input, output_folder, debugging_frames_level, debugging_video_level)
    elif type == 'photo':
        print("Processing photo...")
        benard_supressor.process_image(input, output_folder, debugging_frames_level='All')
    else:
        raise ValueError("Invalid type. Choose between 'video' and 'photo'.")
    end = time.time()
    print("Processing time: ", end-start)

if __name__ == "__main__":
    main()
