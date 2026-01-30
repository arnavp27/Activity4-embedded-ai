student@ubuntu:~/Activity4-embedded-ai$ # Model sizes
ls -lh inference/yolov5n.pt
du -h inference/yolov5n.pt

# MobileNet info (download size)
python3 -c "from torchvision import models; m = models.mobilenet_v2(weights='DEFAULT'); import torch; torch.save(m.state_dict(), 'temp.pth'); import os; print(f'MobileNet size: {os.path.getsize(\"temp.pth\")/1024/1024:.2f} MB'); os.remove('temp.pth')"

# Both models detailed info
python3 << EOF
from torchvision import models
import torch
import os

# MobileNet
m = models.mobilenet_v2(weights='DEFAULT')
torch.save(m.state_dict(), 'temp.pth')
m_size = os.path.getsize('temp.pth')/1024/1024
m_params = sum(p.numel() for p in m.parameters())
os.remove('temp.pth')

# YOLO
y_size = os.path.getsize('inference/yolov5n.pt')/1024/1024

print(f"MobileNetV2: {m_size:.2f} MB, {m_params/1e6:.2f}M params")
print(f"YOLOv5n: {y_size:.2f} MB")
EOF
-rw-rw-r-- 1 student student 3.9M Jan 30 01:25 inference/yolov5n.pt
3.9M	inference/yolov5n.pt
Downloading: "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth" to /home/student/.cache/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13.6M/13.6M [00:12<00:00, 1.17MB/s]
MobileNet size: 13.58 MB
MobileNetV2: 13.58 MB, 3.50M params
YOLOv5n: 3.87 MB

