from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from pathlib import Path

# Paths to images and annotations
train_images_dir = "/kaggle/working/yolo_dataset/images/train"
train_labels_dir = "/kaggle/working/yolo_dataset/labels/train"
augmented_images_dir = "/kaggle/working/yolo_dataset/images/augmented"
augmented_labels_dir = "/kaggle/working/yolo_dataset/labels/augmented"

# Ensure augmented directories exist
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)

# Augmentation pipeline with bbox_params to handle bounding boxes
augmentation_pipeline = A.Compose(
    [
        A.RandomRotate90(p=0.5),  # Random 90-degree rotations
        A.HorizontalFlip(p=0.5),  # Horizontal flip
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),  # Shifts and rotations
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.5),  # Random dropouts
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

def apply_augmentations(image_path, label_path, save_image_path, save_label_path):
    """Apply augmentations to an image and save the augmented image and labels."""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load YOLO format labels
    boxes = []
    class_ids = []
    with open(label_path, "r") as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.split())
            boxes.append([x_center, y_center, width, height])
            class_ids.append(int(class_id))

    # Apply augmentations
    augmented = augmentation_pipeline(image=image, bboxes=boxes, class_labels=class_ids)
    augmented_image = augmented["image"]

    # Save augmented image
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_image_path, augmented_image)

    # Save augmented labels
    augmented_boxes = augmented["bboxes"]
    augmented_class_ids = augmented["class_labels"]
    with open(save_label_path, "w") as f:
        for (x_center, y_center, width, height), class_id in zip(augmented_boxes, augmented_class_ids):
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def visualize_augmentations(original_image_path, augmented_image_path):
    """Visualize the original and augmented images side by side."""
    original_image = cv2.imread(original_image_path)
    augmented_image = cv2.imread(augmented_image_path)

    # Convert images to RGB for visualization
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)

    # Display images
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(augmented_image)
    axes[1].set_title("Augmented Image")
    axes[1].axis("off")

    plt.show()

# Apply augmentations to all training images
for filename in os.listdir(train_images_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(train_images_dir, filename)
        label_path = os.path.join(train_labels_dir, filename.replace(".png", ".txt"))
        save_image_path = os.path.join(augmented_images_dir, filename)
        save_label_path = os.path.join(augmented_labels_dir, filename.replace(".png", ".txt"))

        # Skip if the label file does not exist
        if not os.path.exists(label_path):
            print(f"Label file not found for {filename}, skipping...")
            continue

        # Apply augmentations and save results
        apply_augmentations(image_path, label_path, save_image_path, save_label_path)
        print(f"Augmented {filename}")

        # Visualize one example (optional)
        visualize_augmentations(image_path, save_image_path)

print("Augmentation complete!")
