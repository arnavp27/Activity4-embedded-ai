# main_yolo.py
# Orchestrates full live YOLOv5 pipeline on Jetson Nano

from camera.webcam import Webcam
from pipeline.sampler import FrameSampler
from pipeline.preprocess import preprocess
from inference.yolo import YoloInference
from app_utils.metrics import Monitor
import torch

# 🧠 Detect Jetson GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

#  Initialize all modules
cam = Webcam(0,640,480)                                # Live video source
sampler = FrameSampler(target_fps=5)          # FPS controller
monitor = Monitor()                           # Performance monitor
model = YoloInference(device=device)          # YOLOv5 Classifier/Detector

try:
    while True:
        #  Step 1: Read from webcam
        frame = cam.read()
        if frame is None:
            continue

        # Step 2: Throttle FPS
        if not sampler.allow():
            continue

        # Step 3: Preprocess for YOLO
        # Note: preprocess() resizes to 224x224 which is small for YOLO,
        # but we maintain pipeline consistency.
        img = preprocess(frame)

        #  Step 4: Predict (Object Detection)
        results = model.predict(img)

        # 🏷 Step 5: Decode results
        # results.xyxy[0] contains tensor with [x1, y1, x2, y2, conf, cls]
        # We can also use pandas() if pandas is installed for nicer formatting
        # Here we extract top detection for printing
        
        # Get detections as dataframe
        try:
            detections = results.pandas().xyxy[0]
            if not detections.empty:
                # Pick the most confident detection
                top = detections.iloc[0]
                label_str = f"{top['name']} {top['confidence']:.2f}"
                count = len(detections)
                display_text = f"{label_str} (+{count-1} others)" if count > 1 else label_str
            else:
                display_text = "No objects"
        except Exception:
            # Fallback if pandas is not installed or other error
            det_tensor = results.xyxy[0] # tensor
            if len(det_tensor) > 0:
                display_text = f"{len(det_tensor)} objects detected"
            else:
                display_text = "No objects"

        # 📊 Step 6: Monitor performance
        fps, mem = monitor.update()
        print(f"Prediction: {display_text} | FPS: {fps:.2f} | Mem: {mem:.2f} MB")

except KeyboardInterrupt:
    cam.release()
    print("[INFO] Camera stopped. Exiting.")
