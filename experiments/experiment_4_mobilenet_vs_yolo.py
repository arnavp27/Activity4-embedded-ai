# experiments/experiment_4_mobilenet_vs_yolo.py
# Experiment 4: MobileNet vs YOLO - Model comparison

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.webcam import Webcam
from pipeline.sampler import FrameSampler
from pipeline.preprocess import preprocess
from inference.mobilenet import MobileNetInference
from inference.yolo import YoloInference
from app_utils.metrics import Monitor
from app_utils.labels import load_labels
import torch
import time

def run_model_test(model_type, duration=30):
    """
    Run inference with specified model.
    
    Args:
        model_type: "mobilenet" or "yolo"
        duration: Test duration in seconds
    
    Returns:
        dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize components
    cam = Webcam(0, 640, 480)
    sampler = FrameSampler(target_fps=10)  # High target to test max throughput
    monitor = Monitor()
    
    if model_type == "mobilenet":
        model = MobileNetInference(device=device)
        labels = load_labels()
    else:  # yolo
        model = YoloInference(device=device)
        labels = None
    
    start_time = time.time()
    frame_count = 0
    total_inference_time = 0
    
    try:
        while time.time() - start_time < duration:
            # Read frame
            frame = cam.read()
            if frame is None:
                continue
            
            # Check if frame should be processed
            if not sampler.allow():
                continue
            
            # Preprocess
            img = preprocess(frame)
            
            # Inference with timing
            inf_start = time.time()
            
            if model_type == "mobilenet":
                probs = model.predict(img)
                top = probs.argmax().item()
                prediction = labels[top]
            else:  # yolo
                results = model.predict(img)
                try:
                    detections = results.pandas().xyxy[0]
                    if not detections.empty:
                        prediction = f"{len(detections)} objects"
                    else:
                        prediction = "No objects"
                except:
                    det_tensor = results.xyxy[0]
                    prediction = f"{len(det_tensor)} objects"
            
            inf_time = time.time() - inf_start
            
            total_inference_time += inf_time
            frame_count += 1
            
            # Update metrics
            fps, mem = monitor.update()
            
            if frame_count % 10 == 0:
                print(f"[{frame_count:3d}] {prediction[:30]:<30} | "
                      f"FPS: {fps:.2f} | Inf: {inf_time*1000:.1f}ms | Mem: {mem:.0f} MB")
    
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted")
    
    finally:
        cam.release()
    
    # Calculate final metrics
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    avg_inference_time = (total_inference_time / frame_count * 1000) if frame_count > 0 else 0
    
    results = {
        'model': model_type,
        'actual_fps': actual_fps,
        'frames_processed': frame_count,
        'avg_inference_ms': avg_inference_time,
        'total_inference_s': total_inference_time,
        'total_time': elapsed,
        'final_memory_mb': mem
    }
    
    return results

def main():
    """Run MobileNet vs YOLO comparison and save results."""
    
    print("\n" + "="*60)
    print("EXPERIMENT 4: MOBILENET vs YOLO")
    print("="*60)
    print("Duration: 30 seconds each")
    print()
    
    # Test both models
    models = ["mobilenet", "yolo"]
    all_results = []
    
    for model_type in models:
        try:
            results = run_model_test(model_type=model_type, duration=30)
            all_results.append(results)
            
            print(f"\n{'─'*60}")
            print(f"Results for {model_type.upper()}:")
            print(f"  Actual FPS: {results['actual_fps']:.2f}")
            print(f"  Frames processed: {results['frames_processed']}")
            print(f"  Avg inference time: {results['avg_inference_ms']:.2f} ms")
            print(f"  Total inference time: {results['total_inference_s']:.2f} s")
            print(f"  Memory usage: {results['final_memory_mb']:.0f} MB")
            print(f"{'─'*60}")
            
            time.sleep(2)  # Brief pause between tests
            
        except Exception as e:
            print(f"\n❌ Error testing {model_type}: {e}")
            print(f"   Skipping {model_type}...")
    
    # Comparison
    if len(all_results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        
        mobilenet_result = all_results[0]
        yolo_result = all_results[1]
        
        fps_drop = ((mobilenet_result['actual_fps'] - yolo_result['actual_fps']) / mobilenet_result['actual_fps'] * 100) if mobilenet_result['actual_fps'] > 0 else 0
        inference_increase = ((yolo_result['avg_inference_ms'] - mobilenet_result['avg_inference_ms']) / mobilenet_result['avg_inference_ms'] * 100) if mobilenet_result['avg_inference_ms'] > 0 else 0
        mem_increase = yolo_result['final_memory_mb'] - mobilenet_result['final_memory_mb']
        
        print(f"\nMobileNet Performance:")
        print(f"  FPS: {mobilenet_result['actual_fps']:.2f}")
        print(f"  Inference time: {mobilenet_result['avg_inference_ms']:.2f} ms")
        print(f"  Memory: {mobilenet_result['final_memory_mb']:.0f} MB")
        
        print(f"\nYOLO Performance:")
        print(f"  FPS: {yolo_result['actual_fps']:.2f}")
        print(f"  Inference time: {yolo_result['avg_inference_ms']:.2f} ms")
        print(f"  Memory: {yolo_result['final_memory_mb']:.0f} MB")
        
        print(f"\n{'─'*60}")
        print(f"FPS DROP with YOLO: {fps_drop:.1f}%")
        print(f"Inference time increase: {inference_increase:.1f}%")
        print(f"Memory increase: {mem_increase:.0f} MB ({mem_increase/mobilenet_result['final_memory_mb']*100:.1f}%)")
        print(f"{'─'*60}")
        
        print(f"\nWHY YOLO IS SLOWER:")
        print(f"  1. Architecture complexity:")
        print(f"     - MobileNet: Lightweight depthwise separable convolutions")
        print(f"     - YOLO: Full convolutions + multi-scale feature pyramid")
        print(f"  2. Task complexity:")
        print(f"     - MobileNet: Classification (1000 classes, single prediction)")
        print(f"     - YOLO: Detection (bounding boxes + classes + confidence)")
        print(f"  3. Post-processing:")
        print(f"     - MobileNet: Simple softmax")
        print(f"     - YOLO: NMS (Non-Max Suppression) + anchor processing")
        
        print(f"\nOPTIMIZATIONS REQUIRED FOR YOLO ON EDGE:")
        print(f"  1. Use smaller YOLO variant:")
        print(f"     - YOLOv5n (nano) already in use")
        print(f"     - Consider YOLOv8n or YOLO-FastestV2 for even better speed")
        print(f"  2. Reduce input resolution:")
        print(f"     - Current: 224×224 (too small for YOLO, hurts accuracy)")
        print(f"     - Try: 416×416 or 320×320 (better accuracy-speed trade-off)")
        print(f"  3. Quantization:")
        print(f"     - Apply INT8 quantization (2-3x speedup)")
        print(f"     - Use TensorRT optimization on Jetson")
        print(f"  4. Lower FPS target:")
        print(f"     - Detection doesn't need 30 FPS")
        print(f"     - 5-10 FPS often sufficient for many applications")
        print(f"  5. Frame skipping:")
        print(f"     - Run detection every N frames")
        print(f"     - Use tracking (e.g., SORT) between detections")
        
        print(f"\nRECOMMENDATION:")
        if fps_drop > 50:
            print(f"  ⚠ YOLO causes severe FPS drop ({fps_drop:.0f}%)")
            print(f"  → Use MobileNet for classification tasks")
            print(f"  → Only use YOLO when object detection is required")
            print(f"  → Apply all optimizations listed above for YOLO")
        else:
            print(f"  ⚡ YOLO performance acceptable ({fps_drop:.0f}% drop)")
            print(f"  → Can use YOLO if detection capabilities needed")
        
        print(f"{'='*60}\n")
        
        # Save results to file
        os.makedirs('results', exist_ok=True)
        with open('results/experiment_4_results.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("EXPERIMENT 4: MOBILENET vs YOLO\n")
            f.write("="*60 + "\n\n")
            
            for result in all_results:
                f.write(f"Model: {result['model'].upper()}\n")
                f.write(f"  Actual FPS: {result['actual_fps']:.2f}\n")
                f.write(f"  Frames processed: {result['frames_processed']}\n")
                f.write(f"  Avg inference time: {result['avg_inference_ms']:.2f} ms\n")
                f.write(f"  Total inference time: {result['total_inference_s']:.2f} s\n")
                f.write(f"  Memory usage: {result['final_memory_mb']:.0f} MB\n")
                f.write("-"*60 + "\n\n")
            
            f.write("COMPARISON:\n")
            f.write(f"  MobileNet FPS: {mobilenet_result['actual_fps']:.2f}\n")
            f.write(f"  YOLO FPS: {yolo_result['actual_fps']:.2f}\n")
            f.write(f"  FPS DROP: {fps_drop:.1f}%\n")
            f.write(f"  Memory increase: {mem_increase:.0f} MB\n\n")
            
            f.write("OPTIMIZATIONS REQUIRED:\n")
            f.write("  1. Use smaller YOLO variant (YOLOv8n)\n")
            f.write("  2. Reduce input resolution (416×416)\n")
            f.write("  3. Apply INT8 quantization\n")
            f.write("  4. Use TensorRT optimization\n")
            f.write("  5. Lower FPS target (5-10 FPS)\n")
            f.write("  6. Implement frame skipping + tracking\n")
        
        print("✓ Results saved to results/experiment_4_results.txt")
    
    else:
        print("\n❌ Could not complete comparison (one or both models failed)")
        print("   Check error messages above")

if __name__ == "__main__":
    main()