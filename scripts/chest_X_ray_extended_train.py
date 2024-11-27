#!/usr/bin/env python
# coding: utf-8
"""
Comprehensive script for processing ChestX-Det10 Dataset:
- Visualization of labeled images
- Data augmentation for model training
- Implementing CBAM (Convolutional Block Attention Module)
- Running predictions and evaluation
Author: Usama Yasir Khan
"""

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, ColorJitter, RandomRotation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch import nn
import numpy as np

# === Define Paths ===
dataset_path = '/kaggle/input/chestxdet10dataset'
train_image_folder = os.path.join(dataset_path, 'train_data', 'train-old')
test_image_folder = os.path.join(dataset_path, 'test_data', 'test_data')
train_annotation_file = os.path.join(dataset_path, 'train.json')
test_annotation_file = os.path.join(dataset_path, 'test.json')
predictions_dir = '/kaggle/working/predictions'

os.makedirs(predictions_dir, exist_ok=True)

# === Load Annotations ===
with open(train_annotation_file, 'r') as f:
    train_annotations = json.load(f)
with open(test_annotation_file, 'r') as f:
    test_annotations = json.load(f)

# === Visualization of Labeled Images ===
def visualize_labeled_images(annotations, image_folder, num_images=5):
    """Visualize labeled images with bounding boxes and labels."""
    for annotation in annotations[:num_images]:
        file_name = annotation['file_name']
        labels = annotation['syms']
        boxes = annotation['boxes']
        
        image_path = os.path.join(image_folder, file_name)
        image = Image.open(image_path)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap="gray")
        
        for label, box in zip(labels, boxes):
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, label, color='red', fontsize=12, backgroundcolor='white')
        
        plt.show()

print("Visualizing labeled images:")
visualize_labeled_images(train_annotations, train_image_folder)

# === Data Augmentation ===
augmentation_transforms = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomRotation(10),
    ColorJitter(brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Dataset Definition ===
class ChestXRayDataset(Dataset):
    def __init__(self, annotations, image_folder, transform=None):
        self.annotations = [ann for ann in annotations if ann['boxes']]
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = os.path.join(self.image_folder, ann['file_name'])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.tensor([category_to_index[sym] for sym in ann['syms']], dtype=torch.long)
        return image, {"boxes": boxes, "labels": labels, "file_name": ann['file_name']}

# Instantiate dataset with augmentation
train_dataset = ChestXRayDataset(train_annotations, train_image_folder, transform=augmentation_transforms)

# === Attention Mechanism (CBAM) ===
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        avg_out = self.channel_attention(x)
        x = x * avg_out
        # Spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.spatial_attention(x)
        return x

# Modify the ResNet backbone to include CBAM
backbone = resnet_fpn_backbone('resnet50', pretrained=True)
cbam = CBAM(channels=2048)  # Adding CBAM after backbone output
backbone.body.add_module('cbam', cbam)

# === Model Definition ===
model = FasterRCNN(backbone, num_classes=11)
model.to(device)

# === Evaluation Metrics ===
def calculate_metrics(y_true, y_pred):
    """Calculate Precision, Recall, F1, Specificity, NPV, Accuracy, and DSC."""
    cm = confusion_matrix(y_true, y_pred, labels=list(set(y_true + y_pred)))

    # True Positives, False Positives, False Negatives, True Negatives
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)

    # Specificity: TN / (TN + FP)
    specificity = np.mean(tn / (tn + fp + 1e-6))

    # Negative Predictive Value (NPV): TN / (TN + FN)
    npv = np.mean(tn / (tn + fn + 1e-6))

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Dice Similarity Coefficient (DSC): 2 * TP / (2 * TP + FP + FN)
    dice_coeff = np.mean(2 * tp / (2 * tp + fp + fn + 1e-6))

    # Precision, Recall, and F1 Score
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return precision, recall, f1, specificity, npv, accuracy, dice_coeff

# === Run Evaluation ===
def evaluate_metrics(ground_truth, predictions):
    """Evaluate metrics for predictions compared to ground truth."""
    all_true_labels, all_pred_labels = [], []

    for gt in ground_truth:
        gt_labels = gt['syms']
        gt_boxes = gt['boxes']
        pred = predictions.get(gt['file_name'], {"labels": [], "boxes": []})
        pred_labels = pred['labels']

        all_true_labels.extend(gt_labels)
        all_pred_labels.extend(pred_labels)

    # Calculate metrics
    precision, recall, f1, specificity, npv, accuracy, dice_coeff = calculate_metrics(all_true_labels, all_pred_labels)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Negative Predictive Value (NPV): {npv:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Dice Similarity Coefficient: {dice_coeff:.4f}")
    return precision, recall, f1, specificity, npv, accuracy, dice_coeff

