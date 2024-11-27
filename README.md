# 🩺 Thoracic Abnormality Detection: YOLOv8 and Faster R-CNN 🖼️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-✔️-green?logo=ultralytics)
![FasterRCNN](https://img.shields.io/badge/FasterRCNN-ResNet--50-orange)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

This project leverages advanced object detection architectures to identify thoracic abnormalities in chest X-rays. We aim to compare **YOLOv8** (single-stage) and **Faster R-CNN** (two-stage) models, optimizing their performance for real-world medical diagnostics. 🩻

## Project Overview

![Chest X-Ray Abnormality Detection Overview](results/task_overview.png)

---

## 🎯 **Purpose of the Project**
Chest X-rays are a crucial diagnostic tool for thoracic diseases. This project seeks to:
- 🔍 **Train and Evaluate**: Develop models for detecting abnormalities like Consolidation, Nodule, and Pneumothorax.
- ⚖️ **Compare Architectures**: Analyze YOLOv8 and Faster R-CNN in terms of speed, accuracy, and precision.
- 🛠️ **Optimize Performance**: Apply techniques like class-weighting, augmentation, and learning rate adjustment.
- 📊 **Provide Insights**: Highlight the strengths and weaknesses of each architecture in real-world applications.

---

## 📂 **Dataset Overview**
### Dataset: [ChestX-Det10](https://www.kaggle.com/datasets) 

The dataset includes 3,543 chest X-ray images annotated with bounding boxes for 10 thoracic abnormalities, including Nodule, Mass, and Pneumothorax. This project uses the ChestX-Det10 dataset, a subset of NIH ChestX-14.
The dataset includes chest X-rays annotated with bounding boxes for 10 thoracic abnormalities:
- **Classes**: 
  - Consolidation, Pneumothorax, Emphysema, Calcification, Nodule, Mass, Fracture, Effusion, Atelectasis, Fibrosis.
- **Statistics**:
  - Training Images: **3,001**
  - Testing Images: **1,000+**
  - Missing Data: **22.69% treated as background.**

---

## 🛠️ **Project Workflow**
1. **Data Preparation** 📦
   - Organize the dataset into YOLO-compatible format (`images/`, `labels/`).
   - Convert annotations from `train.json` and `test.json` to YOLO format.
   - Handle missing data by treating unannotated images as background.
   
2. **Model Architectures** 🤖
   - **YOLOv8 (Single-Stage Detection)**:
     - Fast and lightweight, optimized for real-time applications.
   - **Faster R-CNN (Two-Stage Detection)**:
     - Accurate and reliable, with ResNet-50 backbone for feature extraction.
     
3. **Training Configuration** 🏋️‍♀️
   - Data augmentation: Mosaic, horizontal flipping, CutMix.
   - Optimizer: AdamW for YOLOv8, SGD for Faster R-CNN.
   - Custom class weighting for imbalance correction.

4. **Evaluation** 📊
   - Compare metrics like Precision, Recall, mAP@0.5, and mAP@0.5:0.95.

---

## 🚀 **How to Run the Project**
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yasirusama61/YOLOv8-Object-Detection.git
cd thoracic-abnormality-detection
```