#!/usr/bin/env python
# coding: utf-8

# Script Author: Usama Yasir Khan

"""
This script handles preprocessing, augmentation, and preparation of the YOLO dataset
for object detection using the ChestX-Det10 dataset. The script also performs training
and evaluation of YOLO models, including data visualization and metrics computation.
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from ultralytics import YOLO

# Global Variables and Paths
dataset_path = '/kaggle/input/chestxdet10dataset'
yolo_dataset_dir = '/kaggle/working/yolo_dataset'
output_images_dir = os.path.join(yolo_dataset_dir, 'images')
output_labels_dir = os.path.join(yolo_dataset_dir, 'labels')
category_to_index = {
    'Consolidation': 0, 'Pneumothorax': 1, 'Emphysema': 2, 'Calcification': 3,
    'Nodule': 4, 'Mass': 5, 'Fracture': 6, 'Effusion': 7, 'Atelectasis': 8, 'Fibrosis': 9
}

# Utility Functions

def load_annotations(annotation_file):
    """Load annotations from JSON."""
    with open(annotation_file, 'r') as f:
        return json.load(f)

def convert_to_yolo_format(annotation, image_width, image_height):
    """Convert bounding boxes to YOLO format."""
    yolo_annotations = []
    for box, sym in zip(annotation['boxes'], annotation['syms']):
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        class_id = category_to_index[sym]
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_annotations

def create_directory_structure():
    """Create required directory structure."""
    os.makedirs(f"{output_images_dir}/train", exist_ok=True)
    os.makedirs(f"{output_images_dir}/test", exist_ok=True)
    os.makedirs(f"{output_labels_dir}/train", exist_ok=True)
    os.makedirs(f"{output_labels_dir}/test", exist_ok=True)

# Data Preparation
annotations = load_annotations(os.path.join(dataset_path, 'train.json'))

def convert_annotations_to_yolo():
    """Convert dataset annotations to YOLO format."""
    for ann in annotations:
        image_name = ann['file_name']
        image_path = os.path.join(dataset_path, 'train_data', 'train-old', image_name)
        label_path = os.path.join(output_labels_dir, image_name.replace('.png', '.txt'))

        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size

            if ann['boxes'] and ann['syms']:
                yolo_labels = convert_to_yolo_format(ann, image_width, image_height)
                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_labels))
        except FileNotFoundError:
            print(f"Image {image_name} not found. Skipping.")

convert_annotations_to_yolo()
print("Annotations successfully converted to YOLO format.")

# Data Augmentation
def augment_images(image_path, label_path, augmentations=5):
    """Perform image augmentation."""
    image = Image.open(image_path)
    for i in range(augmentations):
        augmented_image = ImageOps.flip(image) if i % 2 == 0 else ImageOps.mirror(image)
        output_image_name = f"aug_{i}_{os.path.basename(image_path)}"
        augmented_image.save(os.path.join(output_images_dir, output_image_name))
        label_dest = os.path.join(output_labels_dir, output_image_name.replace('.png', '.txt'))
        with open(label_path, 'r') as f_in, open(label_dest, 'w') as f_out:
            f_out.writelines(f_in.readlines())

# Oversampling
class_counts = Counter([sym for ann in annotations for sym in ann['syms']])
print("Class Counts:", class_counts)

# Visualization
def visualize_annotations(image_path, label_path):
    """Visualize YOLO annotations on an image."""
    try:
        img = Image.open(image_path)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)

        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x_center *= img.width
            y_center *= img.height
            width *= img.width
            height *= img.height
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, str(int(class_id)), color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error visualizing annotations: {e}")

# Example Usage
example_image = os.path.join(output_images_dir, 'train', 'example.png')
example_label = os.path.join(output_labels_dir, 'train', 'example.txt')
visualize_annotations(example_image, example_label)

# Train/Test Split
def split_dataset(split_ratio=0.8):
    """Split dataset into train and test sets."""
    random.seed(42)
    for ann in annotations:
        image_name = ann['file_name']
        label_name = image_name.replace('.png', '.txt')
        if random.random() < split_ratio:
            shutil.copy(os.path.join(output_images_dir, image_name), os.path.join(output_images_dir, 'train'))
            shutil.copy(os.path.join(output_labels_dir, label_name), os.path.join(output_labels_dir, 'train'))
        else:
            shutil.copy(os.path.join(output_images_dir, image_name), os.path.join(output_images_dir, 'test'))
            shutil.copy(os.path.join(output_labels_dir, label_name), os.path.join(output_labels_dir, 'test'))

split_dataset()
print("Dataset successfully split into train and test sets.")

# Training
def train_model():
    """Train YOLO model."""
    model = YOLO('yolov8s.pt')
    model.train(
        data=f"{yolo_dataset_dir}/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project=f"{yolo_dataset_dir}/training_results",
        name="yolo_experiment"
    )
    print("Model training complete.")

# Evaluation
def evaluate_model():
    """Evaluate YOLO model on test dataset."""
    model = YOLO(f"{yolo_dataset_dir}/training_results/yolo_experiment/weights/best.pt")
    metrics = model.val(data=f"{yolo_dataset_dir}/dataset.yaml", batch=16)
    print(f"Precision: {metrics.box.p.mean():.4f}")
    print(f"Recall: {metrics.box.r.mean():.4f}")
    print(f"mAP@0.5: {metrics.box.map50.mean():.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map.mean():.4f}")

# Main Execution
if __name__ == "__main__":
    create_directory_structure()
    train_model()
    evaluate_model()
