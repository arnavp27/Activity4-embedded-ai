# inference/mobilenet.py

import torch
import torchvision.transforms as T
from torchvision import models


class MobileNetInference:
    def __init__(self, device="cpu"):
        # Store the device (CPU or CUDA)
        self.device = device

        # Load pretrained MobileNetV2 model from torchvision
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval().to(self.device)

        # Define image transform: tensor + normalize (ImageNet standard)
        self.transform = T.Compose([
            T.ToTensor(),  # Converts HWC uint8 image → CHW float32 tensor
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet RGB mean
                std=[0.229, 0.224, 0.225]    # ImageNet RGB std
            )
        ])

    def predict(self, image):
        """
        image: RGB image, numpy array (HWC, uint8)
        returns: softmax probabilities as torch.Tensor
        """
        # Apply transforms and add batch dimension
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run model inference
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.nn.functional.softmax(output, dim=1)

        return prob
