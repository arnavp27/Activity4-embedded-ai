# quantization/test_quantized_inference.py
# Test the quantized model with a sample image

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.quantization
from torchvision import models
import torchvision.transforms as T
import cv2
from app_utils.labels import load_labels
import time

def load_quantized_model(model_path):
    """Load the quantized INT8 model."""
    print(f"Loading quantized model from: {model_path}")
    
    # Create base model structure
    model = models.mobilenet_v2(pretrained=False)
    
    # Apply quantization (same as during export)
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    # Load quantized weights
    model_quantized.load_state_dict(torch.load(model_path))
    model_quantized.eval()
    
    return model_quantized

def preprocess_image(image_path):
    """Load and preprocess an image."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    
    # Convert to tensor and normalize
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return tensor

def test_inference(model, image_tensor, labels, num_runs=10):
    """Run inference and measure performance."""
    
    # Warmup
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Timed inference
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            output = model(image_tensor)
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms
    
    # Get prediction
    probs = torch.nn.functional.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probs, 5)
    
    avg_time = sum(times) / len(times)
    
    return top5_idx[0], top5_prob[0], avg_time

def main():
    print("\n" + "="*70)
    print("TESTING QUANTIZED MODEL")
    print("="*70)
    
    # Paths
    quantized_model_path = "models/mobilenet_v2_quantized_int8.pth"
    test_image_path = "test.jpg"  # Place a test image here
    
    # Check if files exist
    if not os.path.exists(quantized_model_path):
        print(f"\n❌ Quantized model not found: {quantized_model_path}")
        print("   Run quantize_and_export.py first!")
        return
    
    if not os.path.exists(test_image_path):
        print(f"\n❌ Test image not found: {test_image_path}")
        print("   Place a test image (test.jpg) in the project root")
        print("   Or modify test_image_path in this script")
        return
    
    # Load labels
    print("\nLoading ImageNet labels...")
    labels = load_labels()
    
    # Load quantized model
    print("\nLoading quantized model...")
    model_quantized = load_quantized_model(quantized_model_path)
    print("✓ Model loaded")
    
    # Load and preprocess image
    print(f"\nLoading test image: {test_image_path}")
    image_tensor = preprocess_image(test_image_path)
    print("✓ Image preprocessed")
    
    # Run inference
    print("\nRunning inference...")
    top5_idx, top5_prob, avg_time = test_inference(model_quantized, image_tensor, labels)
    
    # Display results
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    print(f"\nAverage inference time: {avg_time:.2f} ms")
    print(f"\nTop-5 Predictions:")
    print(f"{'Rank':<6} {'Confidence':<12} {'Class'}")
    print("-"*70)
    
    for i, (idx, prob) in enumerate(zip(top5_idx, top5_prob), 1):
        class_name = labels[idx.item()]
        confidence = prob.item() * 100
        print(f"{i:<6} {confidence:>6.2f}%      {class_name}")
    
    print("\n" + "="*70)
    print("✓ INFERENCE TEST COMPLETE")
    print("="*70 + "\n")
    
    # Compare with original model (optional)
    print("\n" + "-"*70)
    print("COMPARISON WITH ORIGINAL MODEL (Optional)")
    print("-"*70)
    
    try:
        print("\nLoading original FP32 model...")
        model_fp32 = models.mobilenet_v2(pretrained=True)
        model_fp32.eval()
        
        print("Running inference with FP32 model...")
        top5_idx_fp32, top5_prob_fp32, avg_time_fp32 = test_inference(
            model_fp32, image_tensor, labels
        )
        
        print(f"\nInference time comparison:")
        print(f"  FP32 model:  {avg_time_fp32:.2f} ms")
        print(f"  INT8 model:  {avg_time:.2f} ms")
        print(f"  Speedup:     {avg_time_fp32/avg_time:.2f}x")
        
        # Check if predictions match
        if top5_idx[0] == top5_idx_fp32[0]:
            print(f"\n✓ Top prediction matches between FP32 and INT8")
        else:
            print(f"\n⚠ Top predictions differ:")
            print(f"  FP32: {labels[top5_idx_fp32[0].item()]}")
            print(f"  INT8: {labels[top5_idx[0].item()]}")
    
    except Exception as e:
        print(f"\n⚠ Could not compare with FP32 model: {e}")

if __name__ == "__main__":
    main()