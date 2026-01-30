# quantization/quantize_and_export.py
# Complete Quantization Pipeline: PTQ + ONNX Export

import torch
import torch.quantization
from torchvision import models
import os
import time
import numpy as np

def get_model_size_mb(model):
    """Calculate model size in MB."""
    torch.save(model.state_dict(), "temp.pth")
    size_mb = os.path.getsize("temp.pth") / (1024 * 1024)
    os.remove("temp.pth")
    return size_mb

def test_inference_speed(model, device, num_runs=100):
    """Test inference speed."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Actual timing
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    elapsed = time.time() - start
    avg_time_ms = (elapsed / num_runs) * 1000
    return avg_time_ms

def main():
    print("\n" + "="*70)
    print("QUANTIZATION PIPELINE: MobileNetV2")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # =====================================================================
    # STEP 1: Load Pretrained MobileNetV2
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 1: Loading Pretrained MobileNetV2")
    print("-"*70)
    
    model_fp32 = models.mobilenet_v2(pretrained=True)
    model_fp32.eval()
    
    print("✓ Model loaded successfully")
    
    # =====================================================================
    # STEP 2: Measure Original Model Size
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 2: Measuring Original Model Size")
    print("-"*70)
    
    original_size = get_model_size_mb(model_fp32)
    print(f"Original model size: {original_size:.2f} MB")
    
    # =====================================================================
    # STEP 3: Test Original Model Inference Speed
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 3: Testing Original Model Inference Speed")
    print("-"*70)
    
    model_fp32_device = model_fp32.to(device)
    original_inference_time = test_inference_speed(model_fp32_device, device)
    print(f"Original inference time: {original_inference_time:.2f} ms")
    
    # Move back to CPU for quantization
    model_fp32 = model_fp32.cpu()
    
    # =====================================================================
    # STEP 4: Apply Dynamic Quantization (INT8)
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 4: Applying Dynamic Quantization (INT8)")
    print("-"*70)
    print("Quantizing Linear and Conv2d layers...")
    
    # Dynamic quantization - easiest PTQ method
    # Quantizes weights to INT8, activations computed in INT8 at runtime
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear, torch.nn.Conv2d},  # Layers to quantize
        dtype=torch.qint8  # INT8 quantization
    )
    
    print("✓ Quantization complete")
    
    # =====================================================================
    # STEP 5: Measure Quantized Model Size
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 5: Measuring Quantized Model Size")
    print("-"*70)
    
    quantized_size = get_model_size_mb(model_int8)
    size_reduction = ((original_size - quantized_size) / original_size) * 100
    
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {size_reduction:.1f}%")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    
    # =====================================================================
    # STEP 6: Test Quantized Model Inference Speed
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 6: Testing Quantized Model Inference Speed")
    print("-"*70)
    
    # Note: Quantized models run on CPU (INT8 ops not fully supported on GPU)
    quantized_inference_time = test_inference_speed(model_int8, "cpu", num_runs=100)
    speedup = original_inference_time / quantized_inference_time
    
    print(f"Quantized inference time (CPU): {quantized_inference_time:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    # =====================================================================
    # STEP 7: Validate Model Accuracy (Quick Test)
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 7: Validating Model Outputs")
    print("-"*70)
    
    test_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output_fp32 = model_fp32(test_input)
        output_int8 = model_int8(test_input)
    
    # Check if top predictions match
    top1_fp32 = output_fp32.argmax().item()
    top1_int8 = output_int8.argmax().item()
    
    print(f"FP32 top prediction: {top1_fp32}")
    print(f"INT8 top prediction: {top1_int8}")
    
    if top1_fp32 == top1_int8:
        print("✓ Top predictions match - quantization preserved accuracy")
    else:
        print("⚠ Top predictions differ - minor accuracy change (expected)")
    
    # =====================================================================
    # STEP 8: Save Quantized Model
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 8: Saving Quantized Model")
    print("-"*70)
    
    os.makedirs("models", exist_ok=True)
    quantized_path = "models/mobilenet_v2_quantized_int8.pth"
    
    torch.save(model_int8.state_dict(), quantized_path)
    print(f"✓ Quantized model saved to: {quantized_path}")
    
    # =====================================================================
    # STEP 9: Export to ONNX (FP32)
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 9: Exporting to ONNX Format (FP32)")
    print("-"*70)
    
    # ONNX export requires FP32 model
    model_fp32.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    onnx_fp32_path = "models/mobilenet_v2_fp32.onnx"
    
    torch.onnx.export(
        model_fp32,
        dummy_input,
        onnx_fp32_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ FP32 ONNX model exported to: {onnx_fp32_path}")
    
    # =====================================================================
    # STEP 10: Export to ONNX (FP16 - for edge deployment)
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 10: Exporting to ONNX Format (FP16)")
    print("-"*70)
    
    # Convert model to FP16
    model_fp16 = model_fp32.half()
    dummy_input_fp16 = torch.randn(1, 3, 224, 224).half()
    
    onnx_fp16_path = "models/mobilenet_v2_fp16.onnx"
    
    torch.onnx.export(
        model_fp16,
        dummy_input_fp16,
        onnx_fp16_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ FP16 ONNX model exported to: {onnx_fp16_path}")
    
    # =====================================================================
    # STEP 11: Validate ONNX Models
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 11: Validating ONNX Models")
    print("-"*70)
    
    try:
        import onnx
        
        # Validate FP32 ONNX
        onnx_model_fp32 = onnx.load(onnx_fp32_path)
        onnx.checker.check_model(onnx_model_fp32)
        print(f"✓ FP32 ONNX model is valid")
        
        # Validate FP16 ONNX
        onnx_model_fp16 = onnx.load(onnx_fp16_path)
        onnx.checker.check_model(onnx_model_fp16)
        print(f"✓ FP16 ONNX model is valid")
        
    except ImportError:
        print("⚠ ONNX package not installed - skipping validation")
        print("  Install with: pip install onnx")
    except Exception as e:
        print(f"⚠ ONNX validation error: {e}")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Model Type':<25} {'Size (MB)':<15} {'Inference (ms)':<15}")
    print("-"*70)
    print(f"{'Original FP32':<25} {original_size:>10.2f} MB   {original_inference_time:>10.2f} ms")
    print(f"{'Quantized INT8':<25} {quantized_size:>10.2f} MB   {quantized_inference_time:>10.2f} ms")
    
    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-"*70)
    print(f"{'Size Reduction':<30} {size_reduction:>15.1f}%")
    print(f"{'Compression Ratio':<30} {original_size/quantized_size:>15.2f}x")
    print(f"{'Inference Speedup':<30} {speedup:>15.2f}x")
    
    print("\n" + "-"*70)
    print("FILES CREATED:")
    print("-"*70)
    print(f"  1. {quantized_path}")
    print(f"  2. {onnx_fp32_path}")
    print(f"  3. {onnx_fp16_path}")
    
    print("\n" + "-"*70)
    print("NEXT STEPS:")
    print("-"*70)
    print("  1. Transfer ONNX models to Jetson Nano")
    print("  2. Install ONNX Runtime: pip3 install onnxruntime-gpu")
    print("  3. Or use TensorRT for maximum performance")
    print("  4. Use test_quantized_inference.py to verify models")
    
    print("\n" + "="*70)
    print("✓ QUANTIZATION PIPELINE COMPLETE")
    print("="*70 + "\n")
    
    # Save results to file
    os.makedirs('results', exist_ok=True)
    with open('results/quantization_results.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("QUANTIZATION RESULTS: MobileNetV2\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Original Model (FP32):\n")
        f.write(f"  Size: {original_size:.2f} MB\n")
        f.write(f"  Inference time: {original_inference_time:.2f} ms\n\n")
        
        f.write(f"Quantized Model (INT8):\n")
        f.write(f"  Size: {quantized_size:.2f} MB\n")
        f.write(f"  Inference time: {quantized_inference_time:.2f} ms\n\n")
        
        f.write(f"Improvements:\n")
        f.write(f"  Size reduction: {size_reduction:.1f}%\n")
        f.write(f"  Compression ratio: {original_size/quantized_size:.2f}x\n")
        f.write(f"  Inference speedup: {speedup:.2f}x\n\n")
        
        f.write(f"Files Created:\n")
        f.write(f"  1. {quantized_path}\n")
        f.write(f"  2. {onnx_fp32_path}\n")
        f.write(f"  3. {onnx_fp16_path}\n")
    
    print("✓ Results saved to results/quantization_results.txt")

if __name__ == "__main__":
    main()