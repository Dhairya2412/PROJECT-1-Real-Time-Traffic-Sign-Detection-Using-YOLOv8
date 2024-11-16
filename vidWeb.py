import streamlit as st
import cv2
import os
import glob
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time

# Initialize Streamlit
st.set_page_config(page_title="Traffic Sign Detection", layout="wide")

# Load the YOLOv8 model
model = YOLO('C:/Users/Dhairya Parikh/Desktop/PROJECT-1-Real Time Traffic Sign Detection Using YOLOv8/runs/detect/train17/weights/best.pt')

# Folder containing speed limit sign images
speed_limit_folder = 'C:/Users/Dhairya Parikh/Desktop/PROJECT-1-Real Time Traffic Sign Detection Using YOLOv8/speed_limit_images'

# Load all speed limit sign images into a dictionary
speed_limit_images = {}
for img_path in glob.glob(os.path.join(speed_limit_folder, "*.jpg")):
    class_name = os.path.basename(img_path).split('.')[0]
    speed_limit_images[class_name] = cv2.imread(img_path)

# Variables for tracking the current display
current_speed_limit_img = None
displayed_class_name = None

# Streamlit layout
st.title("Live Traffic Sign Detection")

# Divide the page into two columns
col1, col2 = st.columns(2)

# Left column (Driver's display)
with col1:
    st.header("Driver's Display")
    sign_placeholder = st.empty()
    precautions_placeholder = st.empty()
    sound_placeholder = st.empty()  # Placeholder for sound alerts

# Right column (Webcam feed)
with col2:
    st.header("Webcam Feed")
    live_feed_placeholder = st.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    st.error("Error: Camera not opened.")
    st.stop()

# Initialize the TTS engine
engine = pyttsx3.init()

# Function to play sound alert for new sign
def play_alert_sound():
    sound_placeholder.audio("https://www.soundjay.com/button/beep-07.wav")

# Function to play voice alert
def play_voice_alert(sign_name):
    alert_message = f"Caution {sign_name}."
    engine.say(alert_message)
    engine.runAndWait()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture image.")
        break

    # Run inference on the frame
    results = model(frame)
    result = results[0]
    predictions = result.boxes
    names = result.names

    new_speed_limit_class = None

    for box in predictions:
        conf = box.conf[0]
        class_id = int(box.cls[0])
        class_name = names[class_id]

        if class_name in speed_limit_images:
            new_speed_limit_class = class_name

            if new_speed_limit_class != displayed_class_name:
                displayed_class_name = new_speed_limit_class
                current_speed_limit_img = speed_limit_images[class_name]

                # Play alert sound and voice alert for new sign
                play_alert_sound()
                play_voice_alert(displayed_class_name)

                # Update Driver's display with detected sign and precaution
                with sign_placeholder.container():
                    st.image(cv2.cvtColor(current_speed_limit_img, cv2.COLOR_BGR2RGB), caption=f"Detected Sign: {displayed_class_name}", width=150)
                    precautions_placeholder.write(
                        f"Precautions: Follow speed limit for {displayed_class_name.split(' ')[-1]} km/h.")

            # Draw bounding box and label on frame
            x1, y1, x2, y2 = box.xyxy[0]
            label = f"{class_name} ({conf:.2f})"
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display live feed on the right column
    live_feed_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Stop the app when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()