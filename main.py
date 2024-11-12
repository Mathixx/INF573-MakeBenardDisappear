from video_processor.detection_frequency import DetectionFrequency, FixedFrequency, TimeFrequency
from video_processor.processor import BenardSupressor
from detectors import Detector, YOLODetector, RedCapDetector
from segmentors import Segmentor, MaskRCNNSegmentor, YoloSegmentor
from removers import Remover, LamaInpaintingRemover, OpenCvInpaintingRemover, BlurringRemover
#from detectors.facial_detector import FaceDetector
#from benard_suppressor import BenardSupressor


########################################################
######### Define the input and output paths ############
########################################################
type = "photo" # "video" or "photo"
input = "_test_data/test_1.jpg"
output_folder = "_test_data/"
########################################################
 
########################################################
################ Debugging options #####################
########################################################
debugging_frames_level = 'All' #'None', 'Detector', 'Segmentor', 'Remover', 'All'
debugging_video_level = 'None' #'None', 'Detector', 'Segmentor', 'All'
########################################################

########################################################
######## To complete to choose the components ##########
########################################################
def initialize_components():
    # To select between YOLOv5 and RedCap detectors
    detector = YOLODetector()

    # To select between YOLOv5 and Mask R-CNN segment
    segmenter = YoloSegmentor()

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
    
    if type == 'video':
        print("Processing video...")
        benard_supressor.process_video(input, output_folder, debugging_frames_level, debugging_video_level)
    elif type == 'photo':
        print("Processing photo...")
        benard_supressor.process_image(input, output_folder, debugging_frames_level='All')
    else:
        raise ValueError("Invalid type. Choose between 'video' and 'photo'.")

if __name__ == "__main__":
    main()
