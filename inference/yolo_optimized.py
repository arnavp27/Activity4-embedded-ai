# inference/yolo_optimized.py
# Optimized YOLO using TorchScript for faster inference

import torch
import os

class YoloOptimizedInference:
    def __init__(self, model_path="models/yolov5n_scripted.pt", device="cpu"):
        """
        Load optimized TorchScript YOLO model.
        """
        self.device = device
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Optimized model not found: {model_path}\n"
                f"Run quantization/quantize_yolo.py first!"
            )
        
        print(f"[INFO] Loading optimized YOLO from {model_path} on {device}...")
        
        # Load TorchScript model
        self.model = torch.jit.load(model_path)
        self.model.to(device)
        self.model.eval()
        
        print(f"[INFO] ✓ Optimized YOLO loaded successfully on {device}")
    
    def predict(self, image):
        """
        Run inference on input image.
        
        Args:
            image: RGB numpy array (H, W, 3)
            
        Returns:
            Detections (simplified format for comparison)
        """
        import cv2
        import numpy as np
        
        # Preprocess: resize to 640x640 (YOLO optimal size)
        img_resized = cv2.resize(image, (640, 640))
        
        # Convert to tensor (C, H, W) and normalize
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device) / 255.0
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        # Simple result wrapper
        class SimpleResult:
            def __init__(self, pred):
                self.pred = pred
                
            def __len__(self):
                # Count detections (rough approximation)
                if isinstance(self.pred, (list, tuple)):
                    return len(self.pred[0]) if len(self.pred) > 0 else 0
                return 0
        
        return SimpleResult(predictions)
