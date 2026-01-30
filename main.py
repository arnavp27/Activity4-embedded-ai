# main.py
# Orchestrates full live MobileNet pipeline on Jetson Nano

from camera.webcam import Webcam
from pipeline.sampler import FrameSampler
from pipeline.preprocess import preprocess
from inference.mobilenet import MobileNetInference
from app_utils.metrics import Monitor
from app_utils.labels import load_labels
import torch

# 🧠 Detect Jetson GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

#  Initialize all modules
cam = Webcam(0, 640,480)                                # Live video source
sampler = FrameSampler(target_fps=5)          # FPS controller
monitor = Monitor()                           # Performance monitor
model = MobileNetInference(device=device)     # Classifier
labels = load_labels()                        # Class names (0–999)

try:
    while True:
        #  Step 1: Read from webcam
        frame = cam.read()
        if frame is None:
            continue

        # Step 2: Throttle FPS
        if not sampler.allow():
            continue

        # Step 3: Preprocess for MobileNet
        img = preprocess(frame)

        #  Step 4: Predict class probabilities
        probs = model.predict(img)

        # 🏷 Step 5: Decode top-1 class
        top = probs.argmax().item()
        label = labels[top]

        # 📊 Step 6: Monitor performance
        fps, mem = monitor.update()
        print(f"Prediction: {label} | FPS: {fps:.2f} | Mem: {mem:.2f} MB")

except KeyboardInterrupt:
    cam.release()
    print("[INFO] Camera stopped. Exiting.")
