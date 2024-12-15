# INF573 - Make Benard Disappear

This project applies advanced computer vision techniques to remove a specific individual **wearing a red cap**, whom we will refer to as Benard, from visual media such as photos, videos, and live recordings. The modular pipeline integrates state-of-the-art algorithms for detection, segmentation, tracking, and inpainting.

## Table of Contents

- [Overview](#overview)
- [Pipeline Workflow](#pipeline-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

This project explores various applications of computer vision, such as:
- **Detection**: Identifying the individual using YOLO and a red cap heuristic.
- **Segmentation**: Creating pixel-level masks of the individual for further processing.
- **Tracking**: Ensuring continuous detection across video frames.
- **Inpainting**: Reconstructing the background after removing the individual.

## Pipeline Workflow

The pipeline consists of the following steps:

1. **Detection**:
   - YOLO identifies all individuals in the frame.
   - A heuristic-based red cap detection isolates areas potentially containing a red cap.
   - Results are combined to locate the individual wearing the red cap.

2. **Segmentation**:
   - Generates pixel-level masks for the detected individual.

3. **Tracking**:
   - KCF/MOSSE trackers ensure continuity in detection during temporary detection failures.

4. **Inpainting**:
   - Removes the identified individual and fills the void with reconstructed backgrounds using methods like LAMA or OpenCV-based techniques.

---

## Installation

### Prerequisites
- Python [3.9, 3.10]
- Dependencies listed in `requirements.txt`

### Clone the Repository
```bash
git clone https://github.com/Mathixx/INF573-MakeBenardDisappear.git
cd INF573-MakeBenardDisappear
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

1. **Prepare Input Media**:
   - Place your images or videos in the `input_media/` directory.

2. **Run the Pipeline**:
    It is mandatory to specify the type of media being processed using the --type option (video, photo, or live).

    Example for video processing:
    ```bash
    python main.py --type video --input input_media/example_video.mp4 --output output_media/
    ```

    Alternatively, you can modify the default_input and default_output_folder variables directly in main.py.

3. **Additional Options**:  
    You can specify debugging levels and other options via command-line arguments:

    ```bash
    python main.py --type video --input input_media/example_video.mp4 --output output_media/ --debug_frames Detector --debug_video All
    ``` 

    Available Options:

        --input: Path to the input file (ignored for --type live).
        --output: Folder for processed output files.
        --debug_frames: Debugging level for individual frames. Options: None, Detector, Segmentor, Remover, All, complete_detector.
        --debug_video: Debugging level for video outputs. Options: None, Detector, Segmentor, All.

4. **Visualize Results**:
   - Output media will be saved in the `output_media/` directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For more details, refer to the [project repository](https://github.com/Mathixx/INF573-MakeBenardDisappear).
