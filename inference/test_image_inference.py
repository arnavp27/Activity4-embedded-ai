# test_image_inference.py
# Run MobileNet on a test image using the same pipeline modules

import cv2
import torch
from inference.mobilenet import MobileNetInference
from utils.labels import load_labels

# Path to test image
IMAGE_PATH = "test.jpg"  # change as needed

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and labels
model = MobileNetInference(device=device)
labels = load_labels()

# Load and preprocess image
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

#  Run inference
probs = model.predict(image)
top = probs.argmax().item()
label = labels[top]

# Print result
print(f"Top-1 Prediction: {label} (class {top})")