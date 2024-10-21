# YOLO11-Instance-Segmentation-for-Brain-Tumor-Detection
This project leverages a customized YOLO11 neural network model for instance segmentation to detect and segment brain tumors from medical images. The model is fine-tuned to accurately identify the boundaries of brain tumors, helping in medical image analysis and potentially aiding in faster diagnosis of brain-related conditions.
# Features
Instance Segmentation: The project performs precise segmentation of brain tumors in medical images, detecting the exact shape and size of the tumor.
Custom YOLO11 Model: A modified version of YOLO is used for segmentation, specifically tailored for the medical image dataset.
Trainable on Custom Data: You can train the model on your custom dataset by configuring the YAML file for data input.
CPU Support: The project runs on CPU if GPU is unavailable, making it versatile for different systems.
Image Prediction: Once trained, the model can predict and segment tumors from test images with high accuracy.
# Project Structure
├── data.yaml                    # Dataset configuration file
├── yolo11n-seg.pt               # Pre-trained YOLO model for segmentation
├── runs/                        # Folder containing training results and weights
├── test_images/                 # Folder for testing images
└── script.py                    # Main script for training and prediction
# Usage
# 1. Install Required Libraries
Make sure you have the necessary libraries installed:
pip install ultralytics
2. Train the Model
Use your own dataset or modify the data.yaml file. Train the model using the following command:
from ultralytics import YOLO

# Load the model
model = YOLO("path_to_model/yolo11n-seg.pt")

# Train the model
model.train(
    data="path_to_data/data.yaml",
    epochs=10,
    imgsz=640,
    device="cpu"  # Set to "gpu" if available
)
3. Predict Tumors in New Images
After training, use the model to predict and segment tumors in test images:
# Load the trained model
model = YOLO('runs/segment/train/weights/best.pt')

# Predict and save results
results = model("test_images/image.jpg", save=True)
results[0].show()  # Display the image with tumor segmentation
4. Batch Prediction
You can also run predictions on a folder containing multiple test images:
# Load the trained model
model = YOLO('runs/segment/train/weights/best.pt')

# Predict and save results for a folder of images
results = model("test_images", save=True)
# Dataset
https://universe.roboflow.com/iotseecs/brain-tumor-yzzav/dataset/1
The dataset used for this project consists of brain tumor images annotated with segmentation masks.
You can use any dataset by adjusting the data.yaml file to point to your dataset's images and labels.
# Applications
Medical Image Analysis: This project can assist in analyzing medical images, particularly in detecting and segmenting brain tumors.
AI-Assisted Diagnosis: It provides a faster and more accurate way to help medical professionals diagnose brain conditions through automated tumor detection.
# Contributing
Feel free to contribute by improving the model, adding more features, or testing with different datasets. Pull requests are welcome.

# License
This project is licensed under the MIT License.


