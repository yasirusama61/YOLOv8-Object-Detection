#!/usr/bin/env python
# coding: utf-8
# Author: Usama Yasir Khan
# Description: This script implements a deep learning pipeline for detecting thoracic abnormalities in chest X-ray images 
# using a Faster R-CNN with a ResNet-50 backbone. The model trains, tests, and evaluates for abnormalities like pneumonia 
# and fractures, using bounding boxes and labels for each category.

import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import precision_score, recall_score, f1_score

# Set paths for dataset
dataset_path = '/kaggle/input/chestxdet10dataset'
train_image_folder = os.path.join(dataset_path, 'train_data', 'train-old')
test_image_folder = os.path.join(dataset_path, 'test_data', 'test_data')
train_annotation_file = os.path.join(dataset_path, 'train.json')
test_annotation_file = os.path.join(dataset_path, 'test.json')

# Verify dataset paths
print("Dataset contents:", os.listdir(dataset_path))
print("Train images folder:", train_image_folder)
print("Test images folder:", test_image_folder)
print("Train annotation file:", train_annotation_file)
print("Test annotation file:", test_annotation_file)

# Mapping of abnormality categories to integer indices
category_to_index = {
    'Consolidation': 0, 'Pneumothorax': 1, 'Emphysema': 2, 'Calcification': 3,
    'Nodule': 4, 'Mass': 5, 'Fracture': 6, 'Effusion': 7, 'Atelectasis': 8, 'Fibrosis': 9
}

# Dataset Class Definition
class ChestXRayDataset(Dataset):
    def __init__(self, annotations, image_folder, transform=None):
        # Filter out annotations without bounding boxes to avoid errors
        self.annotations = [ann for ann in annotations if ann['boxes']]
        self.image_folder = image_folder
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image and annotation
        ann = self.annotations[idx]
        image_path = os.path.join(self.image_folder, ann['file_name'])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Format bounding boxes and labels for training
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.tensor([category_to_index[sym] for sym in ann['syms']], dtype=torch.long)
        target = {'boxes': boxes, 'labels': labels}
        return image, target

# Model Setup
def get_model(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Training Function
def train_model(model, train_loader, device, num_epochs=2, lr=0.001):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            # Send images and targets to the device (GPU or CPU)
            images = [img.to(device) for img in images if img is not None]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass and compute loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and parameter update
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        # Step the scheduler
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation Function
# Evaluates the model on the test dataset and computes precision, recall, and F1-score.
def evaluate_model(model, data_loader, device, confidence_threshold=0.3):
    model.eval()  # Set to evaluation mode
    all_true_labels, all_pred_labels = [], []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for i, output in enumerate(outputs):
            # Get true labels and predictions with confidence threshold
            true_labels = targets[i]['labels'].cpu().numpy().tolist()
            pred_labels = output['labels'][output['scores'] >= confidence_threshold].cpu().numpy().tolist()
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

    # Compute metrics
    precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=0)
    print(f"Evaluation Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Save Predictions Function
# Saves predictions in JSON format, with bounding boxes and labels for each test image.
def save_predictions(model, data_loader, device, predictions_dir, confidence_threshold=0.3):
    model.eval()
    os.makedirs(predictions_dir, exist_ok=True)

    for idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for i, output in enumerate(outputs):
            # Filter predictions by confidence threshold
            boxes = output['boxes'][output['scores'] >= confidence_threshold].cpu().tolist()
            scores = output['scores'][output['scores'] >= confidence_threshold].cpu().tolist()
            labels = output['labels'][output['scores'] >= confidence_threshold].cpu().tolist()

            # Prepare predictions for JSON format
            detections = [{"label": int(labels[j]), "box": boxes[j], "score": float(scores[j])} for j in range(len(boxes))]
            image_id = targets[i].get("image_id", f"image_{idx}_{i}")
            json_filename = os.path.join(predictions_dir, f"{image_id}.json")

            # Save predictions
            with open(json_filename, "w") as f:
                json.dump(detections, f)
    print("Predictions saved successfully.")

# Load Dataset
with open(train_annotation_file, 'r') as f:
    train_annotations = json.load(f)
with open(test_annotation_file, 'r') as f:
    test_annotations = json.load(f)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = ChestXRayDataset(train_annotations, train_image_folder, transform)
test_dataset = ChestXRayDataset(test_annotations, test_image_folder, transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(category_to_index) + 1)  # +1 for background class
    model.to(device)

    # Train the Model
    print("Starting model training...")
    train_model(model, train_loader, device, num_epochs=10, lr=0.0005)

    # Save the Trained Model
    torch.save(model.state_dict(), 'fasterrcnn_chestxray.pth')
    print("Model training completed and saved as 'fasterrcnn_chestxray.pth'.")

    # Evaluate the Model
    print("Evaluating model on the test dataset...")
    evaluate_model(model, test_loader, device)

    # Save Predictions
    predictions_dir = "/kaggle/working/predictions"
    print("Saving predictions...")
    save_predictions(model, test_loader, device, predictions_dir)
