# experiments/experiment_2_resolution.py
# Experiment 2: Resolution Change - Test 640×480 vs 1280×720

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

def run_resolution_test(width, height, duration=30):
    """
    Run MobileNet inference at specified resolution.
    
    Args:
        width: Camera capture width
        height: Camera capture height
        duration: Test duration in seconds
    
    Returns:
        dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing with resolution = {width}×{height}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize components
    cam = Webcam(0, width, height)
    sampler = FrameSampler(target_fps=5)  # Fixed FPS for fair comparison
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
            
            # Preprocess (this includes resize to 224x224)
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
                      f"FPS: {fps:.2f} | Mem: {mem:.0f} MB")
    
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted")
    
    finally:
        cam.release()
    
    # Calculate final metrics
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    avg_inference_time = (total_inference_time / frame_count * 1000) if frame_count > 0 else 0
    
    results = {
        'resolution': f"{width}×{height}",
        'width': width,
        'height': height,
        'actual_fps': actual_fps,
        'frames_processed': frame_count,
        'avg_inference_ms': avg_inference_time,
        'total_time': elapsed,
        'final_memory_mb': mem
    }
    
    return results

def main():
    """Run resolution tests and save results."""
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: RESOLUTION CHANGE")
    print("="*60)
    print("Testing resolutions: 640×480 and 1280×720")
    print("Duration: 30 seconds each")
    print()
    
    # Test different resolutions
    resolutions = [
        (640, 480),
        (1280, 720)
    ]
    
    all_results = []
    
    for width, height in resolutions:
        results = run_resolution_test(width, height, duration=30)
        all_results.append(results)
        
        print(f"\n{'─'*60}")
        print(f"Results for {width}×{height}:")
        print(f"  Actual FPS: {results['actual_fps']:.2f}")
        print(f"  Frames processed: {results['frames_processed']}")
        print(f"  Avg inference time: {results['avg_inference_ms']:.2f} ms")
        print(f"  Memory usage: {results['final_memory_mb']:.0f} MB")
        print(f"{'─'*60}")
        
        time.sleep(2)  # Brief pause between tests
    
    # Calculate FPS drop
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    baseline = all_results[0]  # 640×480
    high_res = all_results[1]  # 1280×720
    
    fps_drop = ((baseline['actual_fps'] - high_res['actual_fps']) / baseline['actual_fps'] * 100) if baseline['actual_fps'] > 0 else 0
    mem_increase = high_res['final_memory_mb'] - baseline['final_memory_mb']
    
    print(f"\nBaseline (640×480):")
    print(f"  FPS: {baseline['actual_fps']:.2f}")
    print(f"  Memory: {baseline['final_memory_mb']:.0f} MB")
    
    print(f"\nHigh Resolution (1280×720):")
    print(f"  FPS: {high_res['actual_fps']:.2f}")
    print(f"  Memory: {high_res['final_memory_mb']:.0f} MB")
    
    print(f"\n{'─'*60}")
    print(f"FPS DROP: {fps_drop:.1f}%")
    print(f"Memory increase: {mem_increase:.0f} MB ({mem_increase/baseline['final_memory_mb']*100:.1f}%)")
    print(f"{'─'*60}")
    
    print(f"\nOBSERVATION:")
    if fps_drop > 20:
        print(f"  ⚠ Significant FPS drop at higher resolution")
        print(f"  → Camera capture overhead increases with resolution")
        print(f"  → Recommend 640×480 for real-time applications")
    elif fps_drop > 10:
        print(f"  ⚡ Moderate FPS drop at higher resolution")
        print(f"  → Acceptable for some use cases")
    else:
        print(f"  ✓ Minimal FPS impact at higher resolution")
        print(f"  → Hardware can handle 1280×720 efficiently")
    
    print(f"{'='*60}\n")
    
    # Save results to file
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_2_results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("EXPERIMENT 2: RESOLUTION CHANGE\n")
        f.write("="*60 + "\n\n")
        
        for result in all_results:
            f.write(f"Resolution: {result['resolution']}\n")
            f.write(f"  Actual FPS: {result['actual_fps']:.2f}\n")
            f.write(f"  Frames processed: {result['frames_processed']}\n")
            f.write(f"  Avg inference time: {result['avg_inference_ms']:.2f} ms\n")
            f.write(f"  Memory usage: {result['final_memory_mb']:.0f} MB\n")
            f.write("-"*60 + "\n\n")
        
        f.write("ANALYSIS:\n")
        f.write(f"  FPS at 640×480: {baseline['actual_fps']:.2f}\n")
        f.write(f"  FPS at 1280×720: {high_res['actual_fps']:.2f}\n")
        f.write(f"  FPS DROP: {fps_drop:.1f}%\n")
        f.write(f"  Memory increase: {mem_increase:.0f} MB\n")
    
    print("✓ Results saved to results/experiment_2_results.txt")

if __name__ == "__main__":
    main()