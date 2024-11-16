import cv2
import os
import glob
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('C:/Users/Dhairya Parikh/Desktop/PROJECT-1-Real Time Traffic Sign Detection Using YOLOv8/runs/detect/train11/weights/best.pt')

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Camera not opened.")
    exit()

# Folder containing speed limit sign images
speed_limit_folder = 'C:/Users/Dhairya Parikh/Desktop/PROJECT-1-Real Time Traffic Sign Detection Using YOLOv8/speed_limit_images'  # Update this path

# Load all images in the folder and create a dictionary for quick access by class name
speed_limit_images = {}
for img_path in glob.glob(os.path.join(speed_limit_folder, "*.jpg")):
    # Assumes filename is the class name (e.g., "speed_30.jpg" for a 30 km/h sign)
    class_name = os.path.basename(img_path).split('.')[0]
    speed_limit_images[class_name] = cv2.imread(img_path)

# Initialize variables for displaying the current speed limit
current_speed_limit_img = None
displayed_class_name = None

# Create a persistent "Driver Display" window
cv2.namedWindow("Driver Display", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Driver Display", 300, 300)  # Adjust size as desired

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run inference on the current frame
    results = model(frame)
    result = results[0]  # Get first result from list
    predictions = result.boxes
    names = result.names  # Class names from model

    new_speed_limit_class = None

    # Loop over detected objects to find speed limit signs
    for box in predictions:
        conf = box.conf[0]
        class_id = int(box.cls[0])
        class_name = names[class_id]

        # Check if the detected object is a speed limit sign
        if class_name in speed_limit_images:
            new_speed_limit_class = class_name

            # Only update display if a new speed limit sign is detected
            if new_speed_limit_class != displayed_class_name:
                displayed_class_name = new_speed_limit_class
                current_speed_limit_img = speed_limit_images[class_name]
                # Update the Driver Display window
                cv2.imshow("Driver Display", current_speed_limit_img)

            # Draw bounding box and label on the live frame
            x1, y1, x2, y2 = box.xyxy[0]
            label = f"{class_name} {conf:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the live feed with detections
    cv2.imshow('Webcam Feed with YOLOv8 Detections', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()