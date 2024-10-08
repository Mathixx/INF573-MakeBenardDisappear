# INF573-MakeBenardDisappear
Computer vision project : Detect a specific individual in a live recording and make him disappear form the screen !

The project is easly divided in many (sometimes independant) parts.
Here is a proposed plan that will be reflected in the project's architecture.

## 1. Project Setup and Data Collection
Live Video Feed Acquisition: Set up a way to acquire a live video feed (from a camera or pre-recorded video) for processing.
Dataset Preparation: In the initial phase, gather images/videos with people wearing red caps or other distinctive markers to train a model.
Preprocessing: Resize, normalize, and augment the images to improve model generalization.

Point de situation : PAS COMMENCER

## 2. Object Detection (Initial Phase)
Color-Based Detection: Start by detecting the individual based on color features like the red cap. This can be done using:
    - Color Thresholding: Detect specific colors in a video frame and apply masks.
    - Simple Blob Detection: For detecting and tracking regions of the image where the red cap appears.
    - Region Proposal Networks (RPN): This can evolve into a more advanced object detection model like YOLO or Faster R-CNN.

Point de situation : PAS COMMENCER

## 3. Segmentation and Removal (Initial Phase)
Object Segmentation: Once the individual is located, use image segmentation (e.g., Mask R-CNN) to segment the individual.

Inpainting Techniques: For removal, explore inpainting algorithms like:
    - PatchMatch (OpenCV): Remove the individual by filling the space with neighboring pixels.
    - Deep Learning Inpainting Models: For a more realistic fill, deep learning models like DeepFill v2 can be used for image inpainting.

Point de situation : PAS COMMENCER

## 4. Tracking and Real-Time Processing
Object Tracking: If the video is live, you’ll need real-time tracking. 
Techniques like:
    - Kalman Filters or Optical Flow: To track the person’s position between frames.
    - DeepSORT: An advanced object tracker that combines deep learning for accurate tracking.
    - Real-Time Constraints: Ensure the model is optimized for real-time performance, using lightweight models or leveraging GPU acceleration.

Point de situation : PAS COMMENCER

## 5. Integration of Face/Body Recognition (Later Phase)
Face Detection: When you want to locate based on face features, models like MTCNN or RetinaFace can detect and recognize the individual’s face.
Body Recognition: If the person is recognized based on body structure, use pose estimation models such as OpenPose or HRNet.

Point de situation : PAS COMMENCER

## 6. Testing and Evaluation
Evaluation Metrics: Assess the system’s accuracy in detecting, segmenting, and removing the individual.
Performance Optimization: Ensure smooth operation under real-time constraints by reducing model size, optimizing the code, and using hardware acceleration.

Point de situation : PAS COMMENCER

## 7. Final Deployment
Once satisfied with the results, deploy the model for live video feeds or as a post-processing system depending on the application (real-time vs. batch processing).
Feasibility
Short-Term: With color-based detection and inpainting, the project should be quite feasible. Many pre-trained models (like for object detection) can simplify the initial stages.
Long-Term: As you progress to more advanced techniques (e.g., face or body recognition), complexity will increase, requiring more computation, better datasets, and refined models.


Project Architecture : 

The project is based on OOP as it fits well with the idea of puzzle where you add each pieces.


LIEN DE LA PAGE secrete a ne pas regarder : https://chatgpt.com/share/6702cffd-0724-8009-9980-2ede68a49b92

