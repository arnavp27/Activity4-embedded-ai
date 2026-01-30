# pipeline/preprocess.py
# Webcam-Compatible Preprocessing
# Input: OpenCV BGR frame (HWC, uint8)
# Output: RGB image (224x224, uint8) ready for MobileNet + torchvision

import cv2

def preprocess(frame):
    """
    Takes a BGR frame from OpenCV, converts to RGB,
    resizes to 224x224 (standard for MobileNet input),
    and returns the processed image.
    """
    # Convert BGR to RGB (MobileNet expects RGB ordering)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to 224x224 (what MobileNet expects)
    frame = cv2.resize(frame, (224, 224))

    return frame
