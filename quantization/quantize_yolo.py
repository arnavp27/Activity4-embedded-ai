# quantization/quantize_yolo.py
# YOLO Quantization Pipeline: Export to ONNX with FP16 optimization

import torch
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.yolo import YoloInference

def get_model_size_mb(filepath):
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)

def test_inference_speed(model, num_runs=50):
    """Test YOLO inference speed."""
    import cv2
    import numpy as np
    
    # Create dummy image (640x640 is optimal for YOLO)
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(5):
        _ = model.predict(dummy_image)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model.predict(dummy_image)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    return avg_time

def main():
    print("\n" + "="*70)
    print("YOLO QUANTIZATION PIPELINE")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # =====================================================================
    # STEP 1: Load Original YOLO Model
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading Original YOLOv5n Model")
    print("-"*70)
    
    original_model_path = "inference/yolov5n.pt"
    
    if not os.path.exists(original_model_path):
        print(f"❌ Model not found: {original_model_path}")
        print("   Download it with:")
        print("   cd inference && wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt")
        return
    
    original_size = get_model_size_mb(original_model_path)
    print(f"Original YOLOv5n size: {original_size:.2f} MB")
    
    # Load YOLO
    print("Loading YOLO model...")
    yolo_model = YoloInference(model_path=original_model_path, device=device)
    print("✓ YOLO loaded successfully")
    
    # =====================================================================
    # STEP 2: Test Original Model Inference Speed
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 2: Testing Original YOLO Inference Speed")
    print("-"*70)
    
    print("Running 50 inference tests...")
    original_inference_time = test_inference_speed(yolo_model)
    print(f"Original inference time: {original_inference_time:.2f} ms")
    
    # =====================================================================
    # STEP 3: Export to ONNX (FP32)
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 3: Exporting YOLO to ONNX (FP32)")
    print("-"*70)
    
    os.makedirs("models", exist_ok=True)
    
    # YOLOv5 has built-in export function
    print("Exporting via YOLOv5 export.py...")
    
    # Access the underlying model
    try:
        import subprocess
        
        # Use YOLOv5's export script
        export_cmd = [
            sys.executable,
            "-m", "torch.hub",
        ]
        
        # Alternative: Direct export using the loaded model
        onnx_fp32_path = "models/yolov5n_fp32.onnx"
        
        # Get the actual PyTorch model from YOLOv5 wrapper
        torch_model = yolo_model.model.model  # YOLOv5 wraps the actual model
        torch_model.eval()
        
        # Create dummy input (640x640 is YOLO default)
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            torch_model,
            dummy_input,
            onnx_fp32_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch'}
            }
        )
        
        print(f"✓ FP32 ONNX exported to: {onnx_fp32_path}")
        onnx_fp32_size = get_model_size_mb(onnx_fp32_path)
        print(f"  Size: {onnx_fp32_size:.2f} MB")
        
    except Exception as e:
        print(f"⚠ ONNX export failed: {e}")
        print("  Continuing with PyTorch optimization...")
        onnx_fp32_path = None
        onnx_fp32_size = None
    
    # =====================================================================
    # STEP 4: Export to ONNX (FP16 - Optimized for Edge)
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 4: Creating FP16 Optimized Version")
    print("-"*70)
    
    try:
        # Convert model to FP16
        torch_model_fp16 = torch_model.half()
        dummy_input_fp16 = torch.randn(1, 3, 640, 640).half().to(device)
        
        onnx_fp16_path = "models/yolov5n_fp16.onnx"
        
        torch.onnx.export(
            torch_model_fp16,
            dummy_input_fp16,
            onnx_fp16_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch'}
            }
        )
        
        print(f"✓ FP16 ONNX exported to: {onnx_fp16_path}")
        onnx_fp16_size = get_model_size_mb(onnx_fp16_path)
        print(f"  Size: {onnx_fp16_size:.2f} MB")
        
    except Exception as e:
        print(f"⚠ FP16 export failed: {e}")
        onnx_fp16_path = None
        onnx_fp16_size = None
    
    # =====================================================================
    # STEP 5: Validate ONNX Models
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 5: Validating ONNX Models")
    print("-"*70)
    
    try:
        import onnx
        
        if onnx_fp32_path and os.path.exists(onnx_fp32_path):
            onnx_model = onnx.load(onnx_fp32_path)
            onnx.checker.check_model(onnx_model)
            print(f"✓ FP32 ONNX model is valid")
        
        if onnx_fp16_path and os.path.exists(onnx_fp16_path):
            onnx_model = onnx.load(onnx_fp16_path)
            onnx.checker.check_model(onnx_model)
            print(f"✓ FP16 ONNX model is valid")
            
    except ImportError:
        print("⚠ ONNX package not available for validation")
    except Exception as e:
        print(f"⚠ Validation warning: {e}")
    
    # =====================================================================
    # STEP 6: Create Optimized PyTorch Version
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 6: Creating Optimized PyTorch Model (TorchScript)")
    print("-"*70)
    
    try:
        # TorchScript optimization
        scripted_model = torch.jit.trace(torch_model, dummy_input)
        torchscript_path = "models/yolov5n_scripted.pt"
        torch.jit.save(scripted_model, torchscript_path)
        
        torchscript_size = get_model_size_mb(torchscript_path)
        print(f"✓ TorchScript model saved to: {torchscript_path}")
        print(f"  Size: {torchscript_size:.2f} MB")
        
        # Test TorchScript inference
        print("Testing TorchScript inference speed...")
        scripted_model.eval()
        
        # Warmup
        for _ in range(5):
            _ = scripted_model(dummy_input)
        
        # Timing
        times = []
        for _ in range(50):
            start = time.time()
            with torch.no_grad():
                _ = scripted_model(dummy_input)
            times.append((time.time() - start) * 1000)
        
        scripted_inference_time = sum(times) / len(times)
        print(f"TorchScript inference time: {scripted_inference_time:.2f} ms")
        
    except Exception as e:
        print(f"⚠ TorchScript optimization failed: {e}")
        torchscript_path = None
        torchscript_size = None
        scripted_inference_time = None
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Model Type':<30} {'Size (MB)':<15} {'Inference (ms)':<15}")
    print("-"*70)
    print(f"{'Original YOLOv5n (PyTorch)':<30} {original_size:>10.2f} MB   {original_inference_time:>10.2f} ms")
    
    if onnx_fp32_size:
        print(f"{'ONNX FP32':<30} {onnx_fp32_size:>10.2f} MB   {'N/A':>14}")
    
    if onnx_fp16_size:
        size_reduction = ((original_size - onnx_fp16_size) / original_size) * 100
        print(f"{'ONNX FP16 (Optimized)':<30} {onnx_fp16_size:>10.2f} MB   {'N/A':>14}")
        print(f"{'  → Size reduction':<30} {size_reduction:>10.1f}%")
    
    if torchscript_size and scripted_inference_time:
        speedup = original_inference_time / scripted_inference_time
        print(f"{'TorchScript (Optimized)':<30} {torchscript_size:>10.2f} MB   {scripted_inference_time:>10.2f} ms")
        print(f"{'  → Speedup':<30} {speedup:>10.2f}x")
    
    print("\n" + "-"*70)
    print("FILES CREATED:")
    print("-"*70)
    
    files_created = []
    if onnx_fp32_path and os.path.exists(onnx_fp32_path):
        files_created.append(f"  1. {onnx_fp32_path}")
    if onnx_fp16_path and os.path.exists(onnx_fp16_path):
        files_created.append(f"  2. {onnx_fp16_path}")
    if torchscript_path and os.path.exists(torchscript_path):
        files_created.append(f"  3. {torchscript_path}")
    
    for f in files_created:
        print(f)
    
    print("\n" + "-"*70)
    print("DEPLOYMENT RECOMMENDATIONS:")
    print("-"*70)
    print("  1. For Jetson with TensorRT: Use ONNX FP16")
    print("  2. For PyTorch deployment: Use TorchScript")
    print("  3. Expected speedup on Jetson: 1.5-2x")
    print("  4. FP16 provides best size/speed trade-off")
    
    print("\n" + "="*70)
    print("✓ YOLO QUANTIZATION COMPLETE")
    print("="*70 + "\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/yolo_quantization_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("YOLO QUANTIZATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Original YOLOv5n (PyTorch):\n")
        f.write(f"  Size: {original_size:.2f} MB\n")
        f.write(f"  Inference time: {original_inference_time:.2f} ms\n\n")
        
        if onnx_fp16_size:
            f.write(f"ONNX FP16 (Optimized):\n")
            f.write(f"  Size: {onnx_fp16_size:.2f} MB\n")
            f.write(f"  Size reduction: {size_reduction:.1f}%\n\n")
        
        if torchscript_size and scripted_inference_time:
            f.write(f"TorchScript (Optimized):\n")
            f.write(f"  Size: {torchscript_size:.2f} MB\n")
            f.write(f"  Inference time: {scripted_inference_time:.2f} ms\n")
            f.write(f"  Speedup: {speedup:.2f}x\n\n")
        
        f.write(f"Files Created:\n")
        for i, f_path in enumerate(files_created, 1):
            f.write(f"  {i}. {f_path.strip()}\n")
    
    print("✓ Results saved to results/yolo_quantization_results.txt")

if __name__ == "__main__":
    main()
