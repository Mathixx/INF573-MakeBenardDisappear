import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

class DeepLabSegmentor(Segmentor):
    """
    Segmentor implementation using DeepLab (from torchvision).
    Only segments humans (class 15 in the COCO dataset).
    """
    def __init__(self):
        # Load the pre-trained DeepLabV3 model from torchvision
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def segment(self, frame: np.ndarray, bounding_boxes: list) -> np.ndarray:
        """
        Perform segmentation using DeepLab on the input frame, restricted to the bounding boxes.
        Only humans (class 15) are segmented.
        """
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Loop over bounding boxes
        for (x1, y1, x2, y2) in bounding_boxes:
            # Crop the region within the bounding box
            cropped_region = frame[y1:y2, x1:x2]
            image = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))

            # Preprocess and perform segmentation
            input_tensor = self.preprocess(image).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]

            # Get the predicted class labels for each pixel
            output_predictions = output.argmax(0).byte().cpu().numpy()

            # Only keep the pixels that belong to the 'person' class (class index 15)
            person_mask = (output_predictions == 15).astype(np.uint8) * 255  # Binary mask for 'person'

            # Resize the mask to match the bounding box and insert into the full-frame mask
            mask_resized = cv2.resize(person_mask, (x2 - x1, y2 - y1))
            combined_mask[y1:y2, x1:x2] = np.maximum(combined_mask[y1:y2, x1:x2], mask_resized)

        return combined_mask
