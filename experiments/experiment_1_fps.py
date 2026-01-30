# experiments/experiment_1_fps.py
# Experiment 1: FPS Control - Test target_fps at 2, 5, 10

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

def run_fps_test(target_fps, duration=30):
    """
    Run MobileNet inference at specified FPS for given duration.
    
    Args:
        target_fps: Target frames per second
        duration: Test duration in seconds
    
    Returns:
        dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing with target_fps = {target_fps}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize components
    cam = Webcam(0, 640, 480)
    sampler = FrameSampler(target_fps=target_fps)
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
                print(f"[{frame_count:3d}] Prediction: {label[:30]:<30} | "
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
        'target_fps': target_fps,
        'actual_fps': actual_fps,
        'frames_processed': frame_count,
        'avg_inference_ms': avg_inference_time,
        'total_time': elapsed,
        'final_memory_mb': mem
    }
    
    return results

def main():
    """Run FPS tests and save results."""
    
    print("\n" + "="*60)
    print("EXPERIMENT 1: FPS CONTROL")
    print("="*60)
    print("Testing target_fps = 2, 5, 10")
    print("Duration: 30 seconds each")
    print()
    
    # Test different FPS values
    fps_values = [2, 5, 10]
    all_results = []
    
    for fps in fps_values:
        results = run_fps_test(target_fps=fps, duration=30)
        all_results.append(results)
        
        print(f"\n{'─'*60}")
        print(f"Results for target_fps = {fps}:")
        print(f"  Actual FPS achieved: {results['actual_fps']:.2f}")
        print(f"  Frames processed: {results['frames_processed']}")
        print(f"  Avg inference time: {results['avg_inference_ms']:.2f} ms")
        print(f"  Memory usage: {results['final_memory_mb']:.0f} MB")
        print(f"{'─'*60}")
        
        time.sleep(2)  # Brief pause between tests
    
    # Find optimal FPS
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    # Recommendation logic
    best_fps = None
    best_score = -1
    
    for result in all_results:
        # Score based on: actual FPS close to target, low memory, good throughput
        target = result['target_fps']
        actual = result['actual_fps']
        accuracy = min(actual / target, 1.0) if target > 0 else 0
        score = accuracy * actual  # Prefer higher FPS if achievable
        
        print(f"Target FPS: {target:2d} | Actual: {actual:5.2f} | "
              f"Efficiency: {accuracy*100:5.1f}% | Score: {score:.2f}")
        
        if score > best_score:
            best_score = score
            best_fps = result
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: Optimal FPS = {best_fps['target_fps']}")
    print(f"  Achieves {best_fps['actual_fps']:.2f} FPS consistently")
    print(f"  Memory usage: {best_fps['final_memory_mb']:.0f} MB")
    print(f"  Inference time: {best_fps['avg_inference_ms']:.2f} ms")
    print(f"{'='*60}\n")
    
    # Save results to file
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_1_results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("EXPERIMENT 1: FPS CONTROL\n")
        f.write("="*60 + "\n\n")
        
        for result in all_results:
            f.write(f"Target FPS: {result['target_fps']}\n")
            f.write(f"  Actual FPS: {result['actual_fps']:.2f}\n")
            f.write(f"  Frames processed: {result['frames_processed']}\n")
            f.write(f"  Avg inference time: {result['avg_inference_ms']:.2f} ms\n")
            f.write(f"  Memory usage: {result['final_memory_mb']:.0f} MB\n")
            f.write("-"*60 + "\n\n")
        
        f.write("RECOMMENDATION:\n")
        f.write(f"  Optimal FPS = {best_fps['target_fps']}\n")
        f.write(f"  Achieves {best_fps['actual_fps']:.2f} FPS\n")
        f.write(f"  Memory: {best_fps['final_memory_mb']:.0f} MB\n")
    
    print("✓ Results saved to results/experiment_1_results.txt")

if __name__ == "__main__":
    main()