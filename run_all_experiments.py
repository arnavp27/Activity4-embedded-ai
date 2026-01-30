# run_all_experiments_updated.py
# Master script with YOLO quantization and final comparison

import os
import sys
import subprocess
import time

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")
        return False

def main():
    print("\n" + "="*80)
    print("EMBEDDED AI - COMPLETE EXPERIMENT SUITE")
    print("="*80)
    print("\nThis script will run:")
    print("  1-4: Basic experiments (FPS, Resolution, CPU/GPU, Models)")
    print("    5: YOLO Quantization & Optimization")
    print("    6: Final Comparison (MobileNet vs Optimized YOLO)")
    print("\nTotal estimated time: 8-10 minutes")
    print("\nPress Ctrl+C at any time to stop")
    print("\n" + "="*80)
    
    input("\nPress ENTER to start...")
    
    # Check if we're in the correct directory
    if not os.path.exists("experiments") or not os.path.exists("quantization"):
        print("\n❌ Error: Please run this script from the project root directory")
        print("   Expected structure: Activity4-embedded-ai/ with experiments/ and quantization/ folders")
        return
    
    # List of experiments to run
    experiments = [
        ("experiments/experiment_1_fps.py", "Experiment 1: FPS Control"),
        ("experiments/experiment_2_resolution.py", "Experiment 2: Resolution Change"),
        ("experiments/experiment_3_cpu_vs_gpu.py", "Experiment 3: CPU vs GPU"),
        ("experiments/experiment_4_mobilenet_vs_yolo.py", "Experiment 4: MobileNet vs YOLO"),
        ("quantization/quantize_yolo.py", "YOLO Quantization Pipeline"),
        ("experiments/experiment_5_final_comparison.py", "Experiment 5: Final Comparison"),
    ]
    
    results = []
    start_time = time.time()
    
    for script_path, description in experiments:
        success = run_script(script_path, description)
        results.append((description, success))
        
        if not success:
            response = input("\nExperiment failed. Continue with remaining experiments? (y/n): ")
            if response.lower() != 'y':
                print("\n⚠ Stopping experiment suite")
                break
        
        time.sleep(2)  # Brief pause between experiments
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("EXPERIMENT SUITE COMPLETE")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"\nResults Summary:")
    print("-"*80)
    
    for description, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}  {description}")
    
    # Count successes
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print("-"*80)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All experiments completed successfully!")
    else:
        print(f"\n⚠ {total - passed} experiment(s) failed - check output above")
    
    print("\n" + "="*80)
    print("RESULTS LOCATION")
    print("="*80)
    print("\nAll results saved to:")
    print("  - results/experiment_1_results.txt")
    print("  - results/experiment_2_results.txt")
    print("  - results/experiment_3_results.txt")
    print("  - results/experiment_4_results.txt")
    print("  - results/yolo_quantization_results.txt")
    print("  - results/experiment_5_results.txt")
    
    print("\nOptimized models created:")
    print("  - models/yolov5n_fp32.onnx")
    print("  - models/yolov5n_fp16.onnx")
    print("  - models/yolov5n_scripted.pt")
    
    print("\n" + "="*80)
    print("ASSIGNMENT DELIVERABLES:")
    print("="*80)
    print("✓ Experiment 1: FPS Control (optimal = 5-10 FPS)")
    print("✓ Experiment 2: Resolution comparison (FPS drop analysis)")
    print("✓ Experiment 3: CPU vs GPU (speedup factor)")
    print("✓ Experiment 4: Model comparison (MobileNet vs YOLO)")
    print("✓ Quantization: YOLO optimization with ONNX export")
    print("✓ Experiment 5: Final comparison showing optimization impact")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Experiment suite interrupted by user")
        print("Partial results may be available in results/ folder\n")
