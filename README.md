# Eklipse_test
This repository contains the source code of May 16th Eklipse test
## Exercise 1: Hero detection


## Requirements
- Python 3.6 or above
- PyTorch
- OpenCV
- scikit-learn
- tqdm
- YOLOv5 weight

## Installation

### 1. Clone the YOLOv5 repository (Move to next step if you already have the yolov5 folder):

```shell
git clone https://github.com/ultralytics/yolov5.git
```

### 2. Change to the YOLOv5 directory:
```shell
cd yolov5
```
### 3. Install the required dependencies of YOLOv5:
```shell
pip install -r requirements.txt
```
### 4. Change to the main directory
```shell
cd ..
```
### 5. Install the required dependencies of exercise1_main.py:
```shell
pip install -r requirements.txt
```
## Usage

### 1. Prepare the input images, reference hero images and YOLOv5 weights:
- Create a folder containing the input images to be processed.
- Create a folder containing the reference hero images. These hero images can be crawled with datacrawling.py script
- Download the YOLOv5 weights from https://drive.google.com/drive/folders/1157Un5tRwyx50aFf3bgZsdL8MG2qvm-Y?usp=share_link
- Ensure that the input and reference image names follow the correct naming convention.

### 2. Execution
Run the exercise1_main.py script with the following command-line arguments:

Optional arguments:
- --weights: Model path or triton URL (default: ROOT/yolov5/runs/train/yolov5s_results8/weights/best.pt)
- --source: Path to the folder containing images (default: ROOT/custom_test/test_custom/images)
- --reference-heroes: Path to the folder containing reference heroes (default: ROOT/crawled_images)
- --data: Path to the dataset configuration file (default: yolov5/data/coco128.yaml)
- --imgsz: Inference size (height, width) in pixels (default: 416x416)
- --conf-thres: Confidence threshold for object detection (default: 0.4)
- --iou-thres: NMS IOU threshold for object detection (default: 0.45)
- --max-det: Maximum detections per image (default: 1000)
- --device: Device for inference (cuda device number or 'cpu')
- --project: Directory to save the results (default: runs/detect_hero)
- --name: Prefix for the result folder (default: result_)
- --exist-ok: Allow overwriting existing result folder (default: False)
- --line-thickness: Bounding box thickness for visualization (default: 3)
- --hide-labels: Hide labels in the visualizations (default: False)
- --hide-conf: Hide confidences in the visualizations (default: False)
- --dnn: Use OpenCV DNN for ONNX inference (default: False)

### 3. Example

```shell
python exercise1_main.py --weights <path_to_weights> --source <path_to_input_images>
```

path_to_weights: the trained YOLOv5 to be used is 'yolov5\runs\train\yolov5s_results8\weights\best.pt'

The script will process the input images, detect heroes, match them with reference heroes, and generate a report with the detected hero names.

The report and visualized images will be saved in "runs\detect_hero\result_\output.txt".

# Author
Ho Quang Minh - Calvin

