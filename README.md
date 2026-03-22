# Hard Hat Detection System with YOLOv8

A computer vision project that detects workers with and without helmets in industrial environments, demonstrating how AI can enhance workplace safety compliance.

## Project Overview

Workplace safety in industrial environments requires constant monitoring to ensure workers wear proper protective equipment. This project implements a YOLOv8 object detection model to automatically identify safety violations by detecting workers with helmets, workers without helmets (exposed heads), and head locations in real-time.

## Goal

Train a YOLO object detection model to identify three classes in industrial settings:
* **Helmet**: Workers wearing hard hats
* **Head**: Workers without helmets (safety violation)
* **Person**: Worker detection (original dataset class)

## Dataset

**Source:** [Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) - Kaggle

* Total Images: 5,000
* Format: PASCAL VOC (XML annotations)
* Classes: Helmet, Head, Person
* Splits: 70% Training, 15% Validation, 15% Test

## Technologies & Libraries

* Python 3.x
* YOLOv8 / Ultralytics - Object detection framework
* PyTorch - Deep learning backend
* OpenCV - Image processing
* NumPy - Numerical operations
* Matplotlib - Visualization
* scikit-learn - Data splitting and metrics

## Project Structure
```
├── Data Exploration
│   ├── Label verification
│   ├── Class distribution analysis
│   └── Bounding box visualization
│
├── Data Preprocessing
│   ├── PASCAL VOC to YOLO format conversion
│   ├── Train/val/test split (70/15/15)
│   └── YAML configuration file creation
│
├── Model Training
│   ├── YOLOv8n (nano) pre-trained weights
│   ├── Custom training configuration
│   └── Early stopping implementation
│
└── Evaluation
    ├── mAP, Precision, Recall metrics
    ├── Confusion Matrix
    ├── PR curves
    └── Prediction visualization (Ground Truth vs Predictions)
```

## Model Architecture

**YOLOv8n (Nano)**
* Base Model: COCO pre-trained YOLOv8n
* Input: 640x640 images
* Output: Multi-class object detection (helmet, head)
* Training: 30 epochs with early stopping (patience=10)
* Batch Size: 50

## Key Features

* **Format Conversion**: Automated PASCAL VOC XML to YOLO format conversion
* **Class Filtering**: Reduced from 3 classes (helmet, head, person) to 2 classes (helmet, head) for focused safety detection
* **Data Splitting**: Stratified 70/15/15 split with reproducible random state
* **Early Stopping**: Prevents overfitting by monitoring validation loss
* **Comprehensive Evaluation**: mAP50, mAP50-95, precision, recall metrics
* **Visual Analysis**: Confusion matrices, PR curves, and side-by-side ground truth vs prediction visualization

## Evaluation Metrics

The model is evaluated using:
* **mAP50**: Mean Average Precision at 50% IoU threshold
* **mAP50-95**: Mean Average Precision across IoU thresholds 50-95%
* **Precision**: Accuracy of positive predictions
* **Recall**: Ability to detect all positive cases
* **Confusion Matrix**: Per-class detection performance

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/hard-hat-detection-yolov8.git
cd hard-hat-detection-yolov8
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python numpy matplotlib scikit-learn
```

### 3. Download Dataset
```python
import kagglehub
path = kagglehub.dataset_download("andrewmvd/hard-hat-detection")
```

### 4. Run the Notebook
Execute cells sequentially to:
- Convert annotations to YOLO format
- Split dataset and create YAML config
- Train the model
- Evaluate and visualize results

## Results

The model provides:
* Test set mAP50 and mAP50-95 scores
* Per-class precision and recall
* Normalized and unnormalized confusion matrices
* Visual comparison of ground truth vs predictions on test images
* Training history plots (loss and metric curves)


**Note:** This project is for educational and demonstration purposes.
