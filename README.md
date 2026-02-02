# Embedded AI - MobileNet on Edge Updated

**Course Assignment**: Embedded AI  
**Instructor**: Dr. Dheeraj D, Atria University  
**Target Platform**: NVIDIA Jetson Nano

---

## 📁 Project Structure

```
act4/
│   imagenet_classes.txt          # ImageNet class labels
│   main.py                       # MobileNet live inference
│   main_yolo.py                  # YOLO live inference
│   README.md                     # This file
│
├───app_utils/
│   ├───labels.py                 # Label loader
│   └───metrics.py                # FPS & memory monitoring
│
├───camera/
│   └───webcam.py                 # Webcam capture
│
├───inference/
│   ├───mobilenet.py              # MobileNet inference engine
│   ├───yolo.py                   # YOLO inference engine
│   ├───yolov5n.pt               # YOLO weights
│   └───test_image_inference.py   # Test on single image
│
├───pipeline/
│   ├───preprocess.py             # Image preprocessing
│   └───sampler.py                # FPS control
│
├───experiments/                   # 🆕 Assignment experiments
│   ├───experiment_1_fps.py
│   ├───experiment_2_resolution.py
│   ├───experiment_3_cpu_vs_gpu.py
│   └───experiment_4_mobilenet_vs_yolo.py
│
├───quantization/                  # 🆕 Model optimization
│   ├───quantize_and_export.py
│   └───test_quantized_inference.py
│
├───models/                        # Generated models (after quantization)
│   ├───mobilenet_v2_quantized_int8.pth
│   ├───mobilenet_v2_fp32.onnx
│   └───mobilenet_v2_fp16.onnx
│
└───results/                       # Experiment results (generated)
    ├───experiment_1_results.txt
    ├───experiment_2_results.txt
    ├───experiment_3_results.txt
    ├───experiment_4_results.txt
    └───quantization_results.txt
```

---

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip3 install torch torchvision opencv-python psutil onnx
```

For YOLO (if not already installed):
```bash
pip3 install pandas  # Optional, for YOLO result formatting
```

### 2. Verify Installation

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 📊 Running Experiments

### Experiment 1: FPS Control
Tests target FPS at 2, 5, and 10 to find optimal operating point.

```bash
cd experiments
python3 experiment_1_fps.py
```

**Expected Output**: 
- FPS performance at each setting
- Optimal FPS recommendation
- Results saved to `results/experiment_1_results.txt`

---

### Experiment 2: Resolution Change
Tests camera capture at 640×480 vs 1280×720.

```bash
python3 experiment_2_resolution.py
```

**Expected Output**: 
- FPS at each resolution
- Percentage FPS drop
- Results saved to `results/experiment_2_results.txt`

---

### Experiment 3: CPU vs GPU
Compares CPU and CUDA performance.

```bash
python3 experiment_3_cpu_vs_gpu.py
```

**Expected Output**: 
- FPS on CPU vs GPU
- Speedup factor
- Results saved to `results/experiment_3_results.txt`

---

### Experiment 4: MobileNet vs YOLO
Compares MobileNet and YOLO performance.

```bash
python3 experiment_4_mobilenet_vs_yolo.py
```

**Expected Output**: 
- FPS comparison
- Memory usage comparison
- Optimization recommendations
- Results saved to `results/experiment_4_results.txt`

---

## 🔧 Quantization Pipeline

### Step 1: Run Quantization
This performs all quantization steps in one script:
1. Load MobileNetV2
2. Apply Dynamic Quantization (INT8)
3. Compare sizes before/after
4. Test inference speed
5. Save quantized model
6. Export to ONNX (FP32 and FP16)
7. Validate ONNX models

```bash
cd quantization
python3 quantize_and_export.py
```

**Expected Output**: 
- Model size reduction (~75%)
- Inference speedup measurements
- 3 model files created in `models/`
- Results saved to `results/quantization_results.txt`

---

### Step 2: Test Quantized Model
Test the quantized model on a single image.

```bash
# Place a test image as test.jpg in project root
python3 test_quantized_inference.py
```

**Expected Output**: 
- Top-5 predictions
- Inference time
- Comparison with FP32 model

---

## 🎯 Quick Test (Live Inference)

### MobileNet Live Demo
```bash
python3 main.py
```

### YOLO Live Demo
```bash
python3 main_yolo.py
```

Press `Ctrl+C` to stop.

---

## 📈 Expected Results on Jetson Nano

| Experiment | Metric | Expected Value |
|------------|--------|----------------|
| Exp 1 | Optimal FPS | 5-10 FPS |
| Exp 2 | FPS Drop (1280×720) | 30-50% |
| Exp 3 | GPU Speedup | 3-5x |
| Exp 4 | YOLO FPS Drop | 60-70% |
| Quantization | Size Reduction | ~75% |
| Quantization | Inference Speedup | 1.5-2.5x |

---

## 🐛 Troubleshooting

### CUDA Not Available
```bash
# Check PyTorch CUDA support
python3 -c "import torch; print(torch.version.cuda)"

# If None, reinstall PyTorch with CUDA
# Visit: https://pytorch.org/get-started/locally/
```

### Webcam Not Working
```bash
# Test webcam
ls /dev/video*

# If no camera found, check connections
# Try different camera ID in Webcam(cam_id=0)
```

### YOLO Model Not Found
```bash
# Download YOLOv5n weights
cd inference
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt
```

### Low FPS on Jetson
- Use CUDA (device="cuda")
- Lower target FPS (2-5 FPS)
- Reduce resolution (640×480)
- Use quantized models

---

## 📝 Assignment Deliverables

1. ✅ **Experiment 1**: FPS Control results
2. ✅ **Experiment 2**: Resolution comparison
3. ✅ **Experiment 3**: CPU vs GPU analysis
4. ✅ **Experiment 4**: MobileNet vs YOLO comparison
5. ✅ **Quantization**: Complete pipeline with ONNX export

All results automatically saved to `results/` folder.

---

## 📚 Key Concepts

### Why MobileNet on Edge?
- **Lightweight**: ~3.5M parameters vs ResNet's ~23M
- **Fast**: Uses depthwise separable convolutions
- **Efficient**: 5-7x speedup vs ResNet on edge devices

### Depthwise Separable Convolutions
1. **Depthwise**: Apply one filter per input channel (spatial features)
2. **Pointwise**: 1×1 conv across channels (combine information)
3. **Result**: Same output, 10-15% of computation cost

### Quantization Benefits
- **Size**: 4× smaller (FP32 → INT8)
- **Speed**: 2-3× faster on CPU, even faster with specialized hardware
- **Power**: Lower energy consumption
- **Accuracy**: Minimal loss (<1% top-1 accuracy)

---

## 🔗 Resources

- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Documentation](https://onnx.ai/)
- [Jetson Nano Setup](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

---

## 👨‍🏫 Course Information

**Embedded AI**  
Atria University  
Dr. Dheeraj D

**Assignment**: MobileNet Edge Deployment & Optimization  
**Target**: NVIDIA Jetson Nano  
**Focus**: Real-time inference, quantization, ONNX deployment