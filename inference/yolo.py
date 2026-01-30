import torch
import os

class YoloInference:
    def __init__(self, model_path="inference/yolov5n.pt", device="cpu"):
        """
        Initializes the YOLOv5 inference engine using torch.hub.
        """
        self.device = device
        
        # Ensure absolute path for the model
        if not os.path.exists(model_path):
            # Try resolving relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(current_dir, "yolov5n.pt")
            if os.path.exists(potential_path):
                model_path = potential_path
            else:
                 print(f"[WARN] Model file not found at {model_path} or {potential_path}")
        
        print(f"[INFO] Loading YOLOv5 model from {model_path} on {device}...")
        
        # Load model using torch.hub
        # We use 'ultralytics/yolov5' from GitHub, loading our custom weights (yolov5n.pt)
        # trust_repo=True is often required for cleaner automation
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device, force_reload=False, _verbose=False)
        except Exception as e:
            print(f"[ERROR] Failed to load YOLOv5 via torch.hub: {e}")
            raise e
            
        # Optimize execution
        self.model.eval()

    def predict(self, image):
        """
        Runs inference on the input image.
        
        Args:
            image: RGB numpy array (H, W, 3) - output of preprocess.py
            
        Returns:
            results: Detections object (YOLOv5 format)
        """
        # YOLOv5 auto-handles normalization and resizing if needed, 
        # but since we digest from preprocess(), we pass the image directly.
        # Note: preprocess() returns specific size (224x224). 
        # YOLOv5n works best at 640, but will function at 224 (imprecise).
        
        results = self.model(image)
        return results