import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('C:/Users/Dhairya Parikh/Desktop/PROJECT-1-Real Time Traffic Sign Detection Using YOLOv8/runs/detect/train11/weights/best.pt')  # Path to your trained model

def test_image(image_path):
    # Read the image from file
    image = cv2.imread(image_path)

    # Run inference on the uploaded image
    results = model(image)

    # Accessing the first result from the list returned by YOLOv8
    result = results[0]  # The result is a list, so we access the first element

    # Extracting predictions: class, confidence, and bounding box coordinates
    predictions = result.boxes
    names = result.names  # Get class names from model

    # Loop over the detected objects
    for box in predictions:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the bounding box
        conf = box.conf[0]  # Confidence score of the prediction
        class_id = int(box.cls[0])  # Class ID of the detected object
        class_name = names[class_id]  # Get class name using the class ID

        # Draw bounding box and label on the image
        label = f"{class_name} {conf:.2f}"
        color = (0, 255, 0)  # Green color for the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image with bounding boxes and labels
    cv2.imshow('YOLOv8 Detections', image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ask for image path to test the model
image_path = input("Enter the path of the image to test: ")

# Test the model with the provided image
test_image(image_path)