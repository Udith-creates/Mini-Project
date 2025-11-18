"""
Download YOLOv7-tiny model and test inference on sample.mp4
This is a minimal, Windows-friendly inference test.
"""

import os
import sys
from pathlib import Path
import urllib.request
import torch

print("=" * 70)
print("iSENSOR: Download Model & Test Inference")
print("=" * 70)

# Step 1: Download model
weights_path = Path('yolov7-tiny.pt')
if not weights_path.exists():
    print(f"\n[STEP 1] Downloading YOLOv7-tiny model (~24 MB)...")
    url = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt'
    try:
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, weights_path, reporthook=lambda a, b, c: print(f"  Downloaded: {min(100, a*b*100//c)}%", end='\r'))
        print(f"  ✓ Downloaded: {weights_path.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print(f"  Manual download: {url}")
        sys.exit(1)
else:
    print(f"\n[STEP 1] Model weights already exist: {weights_path}")
    print(f"  Size: {weights_path.stat().st_size / 1e6:.1f} MB")

# Step 2: Load model
print(f"\n[STEP 2] Loading model...")
try:
    from models.experimental import attempt_load
    from utils.general import check_img_size
    
    device = torch.device('cpu')  # Use CPU for now
    
    # Handle PyTorch 2.6+ weights_only requirement
    try:
        model = attempt_load(str(weights_path), map_location=device)
    except Exception as load_err:
        if "weights_only" in str(load_err) or "Unsupported global" in str(load_err):
            print(f"  ℹ️  PyTorch 2.6+ weight loading - retrying with safe globals...")
            from models.yolo import Model
            import torch as torch_module
            if hasattr(torch_module.serialization, 'safe_globals'):
                with torch_module.serialization.safe_globals([Model]):
                    model = attempt_load(str(weights_path), map_location=device)
            else:
                # Fallback for PyTorch 2.6+
                model = attempt_load(str(weights_path), map_location=device)
        else:
            raise
    
    stride = int(model.stride.max())
    img_size = check_img_size(640, s=stride)
    print(f"  OK Model loaded")
    print(f"    Stride: {stride}")
    print(f"    Image size: {img_size}")
    print(f"    Classes: {model.nc}")
    print(f"    Device: {device}")
except Exception as e:
    print(f"  ERROR Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test inference on sample video
print(f"\n[STEP 3] Testing inference on sample.mp4...")
try:
    from utils.datasets import LoadImages
    from utils.general import non_max_suppression, scale_coords, xyxy2xywh
    from utils.plots import plot_one_box
    import cv2
    import numpy as np
    
    source = Path('sample.mp4')
    if not source.exists():
        print(f"  ! sample.mp4 not found. Skipping inference test.")
    else:
        dataset = LoadImages(str(source), img_size=img_size, stride=stride)
        print(f"  ✓ Loaded video: {source}")
        
        # Process first 5 frames
        frame_count = 0
        detections_total = 0
        
        for path, img, im0s, vid_cap in dataset:
            if frame_count >= 5:
                break
            
            # Prepare image
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                pred = model(img, augment=False)[0]
            
            # NMS
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
            
            # Count detections
            num_dets = 0
            for det in pred:
                if len(det):
                    num_dets += len(det)
            
            detections_total += num_dets
            frame_count += 1
            print(f"    Frame {frame_count}: {num_dets} detections")
        
        print(f"  ✓ Processed {frame_count} frames")
        print(f"    Total detections: {detections_total}")
        
except Exception as e:
    print(f"  ✗ Inference test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✓ MODEL TEST COMPLETE!")
print("=" * 70)
print("\nYou can now run full inference with:")
print(f"  python detect.py --weights {weights_path} --source sample.mp4 --view-img")
print("\nOther commands:")
print(f"  python detect.py --weights {weights_path} --source street.mp4")
print(f"  python detect_or_track.py --weights {weights_path} --source sample.mp4")
print("\n" + "=" * 70)
