import os
import yaml
import shutil
import cv2  # Import OpenCV instead of PIL

# Define paths
dataset_path = "traffic_sign_classification_dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Get list of class folders and create class-to-ID mapping
class_folders = sorted(os.listdir(train_path))  # Sort for consistency
class_to_id = {class_name: idx for idx, class_name in enumerate(class_folders)}

# Define paths for YOLO format dataset structure
yolo_train_images = os.path.join(train_path, "images")
yolo_train_labels = os.path.join(train_path, "labels")
yolo_test_images = os.path.join(test_path, "images")
yolo_test_labels = os.path.join(test_path, "labels")

# Create directories for YOLO format dataset
os.makedirs(yolo_train_images, exist_ok=True)
os.makedirs(yolo_train_labels, exist_ok=True)
os.makedirs(yolo_test_images, exist_ok=True)
os.makedirs(yolo_test_labels, exist_ok=True)

def convert_annotation(image_path, label_path, class_id):
    """Create YOLO format annotation file with normalized coordinates."""
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    # YOLO format for full image: x_center, y_center, width, height = 1
    x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0
    with open(label_path, "w") as label_file:
        label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def process_folder(source_folder, target_image_folder, target_label_folder, dataset_type):
    """Organize images and labels into YOLO format."""
    for class_name, class_id in class_to_id.items():
        class_folder = os.path.join(source_folder, class_name)

        # Make class-specific directories in target folders
        os.makedirs(os.path.join(target_image_folder, str(class_id)), exist_ok=True)
        os.makedirs(os.path.join(target_label_folder, str(class_id)), exist_ok=True)

        # Process each image in the class folder
        for image_name in os.listdir(class_folder):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_folder, image_name)

                # Define paths for the YOLO image and label
                target_image_path = os.path.join(target_image_folder, str(class_id), image_name)
                target_label_path = os.path.join(target_label_folder, str(class_id), os.path.splitext(image_name)[0] + ".txt")

                # Copy image and create label file
                shutil.move(image_path, target_image_path)  # Use shutil.move for cross-drive compatibility
                convert_annotation(target_image_path, target_label_path, class_id)

# Process train and test folders
process_folder(train_path, yolo_train_images, yolo_train_labels, "train")
process_folder(test_path, yolo_test_images, yolo_test_labels, "test")

# Create dataset.yaml
data_yaml = {
    "path": dataset_path,
    "train": "train/images",
    "val": "test/images",
    "nc": len(class_to_id),
    "names": [class_name.replace('_', ' ') for class_name in class_to_id.keys()]
}

# Write dataset.yaml
with open(os.path.join(dataset_path, "dataset.yaml"), "w") as yaml_file:
    yaml.dump(data_yaml, yaml_file)

print("Dataset structure and YAML file created successfully.")
