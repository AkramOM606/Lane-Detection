# Lane Detection for ETS2 using PyTorch LaneNet & YOLOPv2

This project is designed for lane detection in [Euro Truck Simulator 2 (ETS2)](https://store.steampowered.com/app/227300/Euro_Truck_Simulator_2/) using deep learning models implemented with PyTorch. It combines two distinct approaches for robust real-time scene understanding:
- **Lane Detection using LaneNet (with an ENet backbone) and Object Detection with YOLO11n**
- **Object Detection and Vehicle Detection using YOLOPv2**

### YOLOPv2 Preview
<p align="center">
   <video src="https://github.com/user-attachments/assets/0c8163ea-4d15-4092-aec0-a8da361e1189" controls></video>
</p>

### LaneNet (ENet) + YOLO11n Preview
<p align="center">
   <video src="https://github.com/user-attachments/assets/be79d4ce-5875-4474-8a18-65122f50bbc6" controls></video>
</p>

## Features

- **Lane Detection:** A PyTorch-based implementation using LaneNet (with an ENet backbone) for accurate lane marking detection, or an alternative approach using YOLOPv2.
- **Object Detection:** Integration of YOLOPv2 to identify key objects in the driving scene, with an alternative version utilizing YOLO11n.
- **Real-Time Processing:** Capturing game screen data from ETS2 and processing it in real time.
- **Modular Codebase:** A clearly separated structure for lane detection, object detection, and utility modules.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AkramOM606/Lane-Detection.git
   cd Lane-Detection
2. **Set Up a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

> [!IMPORTANT]
> Make sure ETS2 is running at 1280x720 in Windowed mode when you execute this script.

### Run Lane Detection (LaneNet with ENet):

```bash
python main_Lanenet_ENet.py
```

### Run Object Detection (YOLOPv2):

```bash
python main_yolopv2.py
```

## Repository Structure
```
Lane-Detection/
├── lanenet/                # Model definitions and configurations for LaneNet
├── utils/                  # Utility scripts and helper functions
├── weights/                # Pre-trained model weights for both detection methods
├── game_capture.py         # Script to capture game screen data from ETS2
├── lane_detector.py        # Lane detection module
├── main_Lanenet_ENet.py    # Entry point for running LaneNet with ENet
├── main_yolopv2.py         # Entry point for running YOLOPv2-based object detection
├── object_detection.py     # Core object detection functionality
├── yolopv2_detector.py     # YOLOPv2 detection helper
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Configuration
### Pre-trained Weights: Ensure that the required pre-trained model weights are placed in the weights/ directory.
### Customization: Adjust configuration parameters (e.g., file paths, model hyperparameters) in the scripts as needed for your local setup.

## Contributing
Contributions are greatly appreciated! To contribute:

1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear messages.
4. Submit a pull request for review.

Please adhere to the project's coding standards and update the documentation as needed.

## License
This project is licensed under the MIT License.

## Acknowledgements
A huge thanks to these awesome projects for providing their incredible models:

- [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) by CAIC-AD – for its advanced multi-task perception model.
- [LaneNet-Lane-Detection-PyTorch](https://github.com/IrohXu/lanenet-lane-detection-pytorch) by IrohXu – for the efficient lane detection model.
- And also the ETS2 community for inspiration.

Their contributions have been invaluable in advancing this project!

## Disclaimer
This project is a work in progress and is provided "as is" without any warranties. Use at your own risk.
