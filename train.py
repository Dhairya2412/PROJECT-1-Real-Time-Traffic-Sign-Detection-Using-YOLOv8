from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()  # Ensures proper multiprocessing setup in Windows
    model = YOLO('C:/Users/Dhairya Parikh/Desktop/PROJECT-1-Real Time Traffic Sign Detection Using YOLOv8/runs/detect/train17/weights/best.pt')  # Update this path if necessary
    model.train(data="C:/Users/Dhairya Parikh/Desktop/PROJECT-1-Real Time Traffic Sign Detection Using YOLOv8/datasets/roboflow/data.yaml", epochs=5, imgsz=640, batch=32, device='cuda:0')

