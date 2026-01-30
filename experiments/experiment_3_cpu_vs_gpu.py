# experiments/experiment_3_cpu_vs_gpu.py
# Experiment 3: CPU vs GPU - Compare device performance

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.webcam import Webcam
from pipeline.sampler import FrameSampler
from pipeline.preprocess import preprocess
from inference.mobilenet import MobileNetInference
from app_utils.metrics import Monitor
from app_utils.labels import load_labels
import torch
import time

def run_device_test(device, duration=30):
    """
    Run MobileNet inference on specified device.
    
    Args:
        device: "cpu" or "cuda"
        duration: Test duration in seconds
    
    Returns:
        dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing on device = {device.upper()}")
    print(f"{'='*60}")
    
    # Initialize components
    cam = Webcam(0, 640, 480)
    sampler = FrameSampler(target_fps=10)  # High target to test max throughput
    monitor = Monitor()
    model = MobileNetInference(device=device)
    labels = load_labels()
    
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
            probs = model.predict(img)
            inf_time = time.time() - inf_start
            
            total_inference_time += inf_time
            frame_count += 1
            
            # Get prediction
            top = probs.argmax().item()
            label = labels[top]
            
            # Update metrics
            fps, mem = monitor.update()
            
            if frame_count % 10 == 0:
                print(f"[{frame_count:3d}] {label[:30]:<30} | "
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
        'device': device,
        'actual_fps': actual_fps,
        'frames_processed': frame_count,
        'avg_inference_ms': avg_inference_time,
        'total_inference_s': total_inference_time,
        'total_time': elapsed,
        'final_memory_mb': mem
    }
    
    return results

def main():
    """Run CPU vs GPU tests and save results."""
    
    print("\n" + "="*60)
    print("EXPERIMENT 3: CPU vs GPU COMPARISON")
    print("="*60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    print("Duration: 30 seconds each")
    print()
    
    # Test devices
    devices = ["cpu"]
    if cuda_available:
        devices.append("cuda")
    else:
        print("⚠ CUDA not available. Testing CPU only.")
        print("  → On Jetson Nano, make sure PyTorch is installed with CUDA support")
    
    all_results = []
    
    for device in devices:
        results = run_device_test(device=device, duration=30)
        all_results.append(results)
        
        print(f"\n{'─'*60}")
        print(f"Results for {device.upper()}:")
        print(f"  Actual FPS: {results['actual_fps']:.2f}")
        print(f"  Frames processed: {results['frames_processed']}")
        print(f"  Avg inference time: {results['avg_inference_ms']:.2f} ms")
        print(f"  Total inference time: {results['total_inference_s']:.2f} s")
        print(f"  Memory usage: {results['final_memory_mb']:.0f} MB")
        print(f"{'─'*60}")
        
        time.sleep(2)  # Brief pause between tests
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    
    if len(all_results) == 2:
        cpu_result = all_results[0]
        gpu_result = all_results[1]
        
        fps_speedup = gpu_result['actual_fps'] / cpu_result['actual_fps'] if cpu_result['actual_fps'] > 0 else 0
        inf_speedup = cpu_result['avg_inference_ms'] / gpu_result['avg_inference_ms'] if gpu_result['avg_inference_ms'] > 0 else 0
        mem_diff = gpu_result['final_memory_mb'] - cpu_result['final_memory_mb']
        
        print(f"\nCPU Performance:")
        print(f"  FPS: {cpu_result['actual_fps']:.2f}")
        print(f"  Inference time: {cpu_result['avg_inference_ms']:.2f} ms")
        print(f"  Memory: {cpu_result['final_memory_mb']:.0f} MB")
        
        print(f"\nGPU Performance:")
        print(f"  FPS: {gpu_result['actual_fps']:.2f}")
        print(f"  Inference time: {gpu_result['avg_inference_ms']:.2f} ms")
        print(f"  Memory: {gpu_result['final_memory_mb']:.0f} MB")
        
        print(f"\n{'─'*60}")
        print(f"GPU SPEEDUP: {fps_speedup:.2f}x faster")
        print(f"Inference speedup: {inf_speedup:.2f}x")
        print(f"Memory overhead: {mem_diff:.0f} MB")
        print(f"{'─'*60}")
        
        print(f"\nCONCLUSION:")
        if fps_speedup > 3:
            print(f"  ✓ GPU provides significant speedup ({fps_speedup:.1f}x)")
            print(f"  → Always use CUDA on Jetson Nano for real-time applications")
        elif fps_speedup > 1.5:
            print(f"  ✓ GPU provides moderate speedup ({fps_speedup:.1f}x)")
            print(f"  → GPU recommended for better performance")
        else:
            print(f"  ⚠ Minimal GPU benefit ({fps_speedup:.1f}x)")
            print(f"  → Check CUDA installation or model optimization")
    else:
        cpu_result = all_results[0]
        print(f"\nCPU-only results:")
        print(f"  FPS: {cpu_result['actual_fps']:.2f}")
        print(f"  Inference time: {cpu_result['avg_inference_ms']:.2f} ms")
        print(f"\n⚠ Run on Jetson Nano with CUDA to see GPU comparison")
    
    print(f"{'='*60}\n")
    
    # Save results to file
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_3_results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("EXPERIMENT 3: CPU vs GPU COMPARISON\n")
        f.write("="*60 + "\n\n")
        
        for result in all_results:
            f.write(f"Device: {result['device'].upper()}\n")
            f.write(f"  Actual FPS: {result['actual_fps']:.2f}\n")
            f.write(f"  Frames processed: {result['frames_processed']}\n")
            f.write(f"  Avg inference time: {result['avg_inference_ms']:.2f} ms\n")
            f.write(f"  Total inference time: {result['total_inference_s']:.2f} s\n")
            f.write(f"  Memory usage: {result['final_memory_mb']:.0f} MB\n")
            f.write("-"*60 + "\n\n")
        
        if len(all_results) == 2:
            cpu_result = all_results[0]
            gpu_result = all_results[1]
            fps_speedup = gpu_result['actual_fps'] / cpu_result['actual_fps']
            
            f.write("COMPARISON:\n")
            f.write(f"  CPU FPS: {cpu_result['actual_fps']:.2f}\n")
            f.write(f"  GPU FPS: {gpu_result['actual_fps']:.2f}\n")
            f.write(f"  GPU SPEEDUP: {fps_speedup:.2f}x\n")
            f.write(f"  Memory overhead: {gpu_result['final_memory_mb'] - cpu_result['final_memory_mb']:.0f} MB\n")
    
    print("✓ Results saved to results/experiment_3_results.txt")

if __name__ == "__main__":
    main()