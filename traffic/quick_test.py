"""
Quick validation script for iSENSOR repo setup.
Tests imports, model building, and basic functionality WITHOUT requiring pre-trained weights.
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("iSENSOR Quick Test & Validation")
print("=" * 60)

# Test 1: Core imports
print("\n[TEST 1] Checking core imports...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"    Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
except Exception as e:
    print(f"  ✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"  ✗ OpenCV import failed: {e}")
    sys.exit(1)

try:
    import yaml
    import numpy as np
    from PIL import Image
    print(f"  ✓ Supporting libraries (yaml, numpy, PIL)")
except Exception as e:
    print(f"  ✗ Supporting libraries import failed: {e}")
    sys.exit(1)

# Test 2: Model architecture loading
print("\n[TEST 2] Loading model architecture from YAML...")
try:
    from models.yolo import Model
    
    # Load standard YOLOv7 config
    model_yaml = Path('cfg/training/yolov7.yaml')
    with open(model_yaml) as f:
        model_dict = yaml.safe_load(f)
    
    print(f"  ✓ Loaded {model_yaml.name}")
    print(f"    Classes: {model_dict['nc']}")
    print(f"    Depth mult: {model_dict['depth_multiple']}")
    print(f"    Width mult: {model_dict['width_multiple']}")
except Exception as e:
    print(f"  ✗ Model config loading failed: {e}")
    sys.exit(1)

# Test 3: Build model (no pretrained weights)
print("\n[TEST 3] Building YOLOv7 model from scratch (FP32, CPU)...")
try:
    device = torch.device('cpu')
    model = Model(model_yaml, ch=3, nc=80)
    model.to(device)
    print(f"  ✓ Model built successfully")
    print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"    Model device: {next(model.parameters()).device}")
except Exception as e:
    print(f"  ✗ Model building failed: {e}")
    sys.exit(1)

# Test 4: Forward pass with dummy input
print("\n[TEST 4] Running forward pass with dummy input...")
try:
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    print(f"  Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  ✓ Forward pass successful")
    if isinstance(output, (list, tuple)):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                print(f"    Output {i}: {out.shape}")
    else:
        print(f"    Output shape: {output.shape}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 5: Data loading (inference)
print("\n[TEST 5] Testing data loaders...")
try:
    from utils.datasets import LoadImages, LoadStreams
    
    # Check if sample videos exist
    sample_video = Path('sample.mp4')
    if sample_video.exists():
        print(f"  ✓ Found {sample_video}")
        dataset = LoadImages(str(sample_video), img_size=640, stride=32)
        print(f"    Loaded as LoadImages stream")
    else:
        print(f"  ! Sample video not found, skipping video load test")
except Exception as e:
    print(f"  ✗ Data loader test failed: {e}")
    sys.exit(1)

# Test 6: Utils (post-processing)
print("\n[TEST 6] Testing utility functions...")
try:
    from utils.general import non_max_suppression, scale_coords
    
    # Create dummy predictions
    dummy_pred = torch.randn(1, 25200, 85)  # batch_size, num_anchors, (4 bbox + 1 obj + 80 classes)
    dummy_pred[..., :4] = torch.rand(1, 25200, 4) * 640  # normalize to image size
    dummy_pred[..., 4] = 0.5  # objectness confidence
    
    # Test NMS
    result = non_max_suppression(dummy_pred, conf_thres=0.25, iou_thres=0.45)
    print(f"  ✓ NMS post-processing works")
    print(f"    Input predictions: {dummy_pred.shape}")
    print(f"    Output detections (after NMS): {len(result)} batches")
    
except Exception as e:
    print(f"  ✗ Utils test failed: {e}")
    sys.exit(1)

# Test 7: Model export compatibility check
print("\n[TEST 7] Checking export capabilities...")
try:
    import torch.onnx
    print(f"  ✓ ONNX export available")
except:
    print(f"  ! ONNX export not available (optional)")

try:
    # Check for optional export deps
    import tensorboard
    print(f"  ✓ TensorBoard available")
except:
    print(f"  ! TensorBoard not available (optional)")

# Summary
print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nRepository is ready to use. Next steps:")
print("  1. Download model weights:")
print("     - YOLOv7: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
print("     - YOLOv7-tiny: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt")
print("\n  2. Run inference on a video:")
print("     python detect.py --weights yolov7-tiny.pt --source sample.mp4 --conf-thres 0.25")
print("\n  3. Run object tracking:")
print("     python detect_or_track.py --weights yolov7.pt --source sample.mp4")
print("\n  4. Test with pre-built model:")
print("     python hubconf.py")
print("\n" + "=" * 60)
