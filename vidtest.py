import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('C:/Users/Dhairya Parikh/Desktop/PROJECT-1-Real Time Traffic Sign Detection Using YOLOv8/runs/detect/train11/weights/best.pt')  # Path to your trained model

def test_video(video_path):
    # Capture the video from file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame was not successfully read, break the loop (end of video)
        if not ret:
            break

        # Run inference on the frame
        results = model(frame)

        # Access the first result from the list returned by YOLOv8
        result = results[0]

        # Extracting predictions: class, confidence, and bounding box coordinates
        predictions = result.boxes
        names = result.names  # Get class names from model

        # Loop over the detected objects
        for box in predictions:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the bounding box
            conf = box.conf[0]  # Confidence score of the prediction
            class_id = int(box.cls[0])  # Class ID of the detected object
            class_name = names[class_id]  # Get class name using the class ID

            # Draw bounding box and label on the frame
            label = f"{class_name} {conf:.2f}"
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame with bounding boxes and labels
        cv2.imshow('YOLOv8 Video Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Ask for video path to test the model
video_path = input("Enter the path of the video to test: ")

# Test the model with the provided video
test_video(video_path)