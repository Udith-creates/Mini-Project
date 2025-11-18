"""
iSENSOR Testing Guide - Complete Instructions
==============================================

Your repo is now fixed and ready for testing! Here's a complete walkthrough.
"""

# ============================================================================
# PART 1: WHAT WAS FIXED
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ PART 1: WHAT WAS FIXED IN YOUR REPO                                       ║
╚════════════════════════════════════════════════════════════════════════════╝

✓ ISSUE: google_utils.py crashed when trying to auto-download model weights
  - Problem: IndexError when 'git tag' returned empty output
  - Solution: Added try-except with fallback to 'v0.1' as default tag
  - File: utils/google_utils.py (lines 16-25)

✓ NEW VALIDATION SCRIPT: quick_test.py
  - Tests all imports (PyTorch, OpenCV, utilities)
  - Builds YOLOv7 model from scratch (no weights needed)
  - Runs forward pass on dummy input
  - Tests data loaders and post-processing
  - Status: All 7 tests PASSED ✓

✓ NEW DOWNLOAD & INFERENCE SCRIPT: download_and_test.py
  - Downloads YOLOv7-tiny model (~24 MB)
  - Loads model with weights
  - Tests inference on 5 sample video frames
  - Ready to run!

""")

# ============================================================================
# PART 2: CURRENT STATUS
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ PART 2: CURRENT STATUS                                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Project: iSENSOR (YOLOv7-based object detection)
Repository: https://github.com/iamthighs/iSENSOR
Branch: main

Environment:
  ✓ Python 3.x
  ✓ PyTorch 2.9.1 (CPU)
  ✓ OpenCV 4.12.0
  ✓ All core dependencies installed

Available Models:
  • YOLOv7-tiny: Small model (~12 MB), fast inference
  • YOLOv7:      Standard model (~37 MB), balanced
  • YOLOv7-X:    Extra-large model (~71 MB), highest accuracy

Sample Videos:
  • sample.mp4 - General scene
  • street.mp4 - Traffic/street scene

""")

# ============================================================================
# PART 3: QUICK START COMMANDS
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ PART 3: QUICK START COMMANDS                                              ║
╚════════════════════════════════════════════════════════════════════════════╝

1️⃣  VALIDATE REPO (no weights needed):
    python quick_test.py
    
    Output: Shows all 7 tests passing, model architecture info
    Time: ~30 seconds
    
2️⃣  DOWNLOAD MODEL & TEST (will download ~24 MB):
    python download_and_test.py
    
    Output: Downloads YOLOv7-tiny, tests on 5 sample video frames
    Time: ~2-5 minutes (depends on internet speed)
    
3️⃣  RUN DETECTION ON VIDEO (after weights downloaded):
    python detect.py --weights yolov7-tiny.pt --source sample.mp4 --conf-thres 0.25
    
    Options:
      --weights         : Path to model weights (yolov7-tiny.pt, yolov7.pt, etc.)
      --source          : Input video/image/directory
      --conf-thres      : Confidence threshold (0-1, lower = more detections)
      --iou-thres       : IoU threshold for NMS (0-1)
      --img-size        : Input image size (640, 1280, etc.)
      --view-img        : Display results in a window
      --save-txt        : Save detections as .txt files
      
4️⃣  RUN DETECTION + TRACKING:
    python detect_or_track.py --weights yolov7-tiny.pt --source sample.mp4
    
    Tracks objects across frames using SORT algorithm
    
5️⃣  TEST ON STREET VIDEO:
    python detect.py --weights yolov7-tiny.pt --source street.mp4 --view-img

""")

# ============================================================================
# PART 4: DETAILED WORKFLOW
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ PART 4: DETAILED TESTING WORKFLOW                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

STEP 1: Verify Environment
────────────────────────────
  $ cd C:\\Users\\UDITH\\Desktop\\webby'\\iSENSOR
  $ python quick_test.py
  
  Expected: 7/7 tests pass, total params: 37,622,682

STEP 2: Download Model (First Time Only)
──────────────────────────────────────────
  Method A: Automatic
    $ python download_and_test.py
    (Downloads yolov7-tiny.pt automatically, ~24 MB)
    
  Method B: Manual
    Download from: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
    Save to: C:\\Users\\UDITH\\Desktop\\webby'\\iSENSOR\\yolov7-tiny.pt
    
STEP 3: Test Basic Inference
──────────────────────────────
  $ python detect.py --weights yolov7-tiny.pt --source sample.mp4 --conf-thres 0.25
  
  What happens:
    - Loads model weights
    - Opens video file
    - Processes each frame
    - Detects objects (people, cars, etc.)
    - Saves results to: runs/detect/exp/
    
STEP 4: View Detection Results
───────────────────────────────
  $ python detect.py --weights yolov7-tiny.pt --source sample.mp4 --view-img --save-txt
  
  Output files:
    - runs/detect/exp/sample.mp4  (video with bounding boxes)
    - runs/detect/exp/labels/     (detection coordinates as .txt)
    
STEP 5: Advanced - Object Tracking
───────────────────────────────────
  $ python detect_or_track.py --weights yolov7-tiny.pt --source sample.mp4
  
  Features:
    - Detects objects
    - Tracks them across frames
    - Assigns unique IDs to each object
    - Uses Kalman filter + Hungarian algorithm (SORT)

""")

# ============================================================================
# PART 5: FILE REFERENCE
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ PART 5: KEY FILES REFERENCE                                               ║
╚════════════════════════════════════════════════════════════════════════════╝

NEW FILES CREATED:
  • quick_test.py              - Repo validation (no weights)
  • download_and_test.py       - Download + test inference

MAIN SCRIPTS:
  • detect.py                  - Run detection on images/videos
  • detect_or_track.py         - Detection + SORT tracking
  • train.py                   - Train custom model
  • test.py                    - Evaluate on dataset
  • export.py                  - Export to ONNX/TensorRT/CoreML

MODEL CONFIGS:
  • cfg/training/yolov7.yaml        - Standard YOLOv7 architecture
  • cfg/training/yolov7-tiny.yaml   - Lightweight variant
  • cfg/training/yolov7x.yaml       - Extra-large variant
  
DATASETS:
  • data/coco.yaml             - COCO dataset config (80 classes)
  • data/custom_data.yaml      - Custom dataset template
  • data/hyp.scratch.*.yaml    - Training hyperparameters

UTILITIES:
  • utils/datasets.py          - Data loading & augmentation
  • utils/general.py           - Post-processing (NMS, scaling)
  • utils/loss.py              - Loss functions
  • models/yolo.py             - Model builder
  • models/common.py           - Layer definitions
  • models/experimental.py     - Advanced modules

""")

# ============================================================================
# PART 6: TROUBLESHOOTING
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ PART 6: TROUBLESHOOTING                                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

ISSUE: "yolov7-tiny.pt not found"
SOLUTION: 
  $ python download_and_test.py
  Or manually download from GitHub releases

ISSUE: "ModuleNotFoundError: No module named 'torch'"
SOLUTION:
  $ pip install torch torchvision

ISSUE: "Out of memory" when running inference
SOLUTION:
  Use smaller model: yolov7-tiny instead of yolov7-x
  Reduce image size: --img-size 416
  Reduce batch size: --batch-size 1

ISSUE: Video file not found
SOLUTION:
  Check file path: Use absolute path or relative path from repo root
  $ python detect.py --weights yolov7-tiny.pt --source sample.mp4
  Not: --source C:\\...\\videos\\video.mp4 (if video is in repo)

ISSUE: Slow inference on CPU
SOLUTION:
  This is normal. Use CUDA GPU if available:
  $ python detect.py --weights yolov7-tiny.pt --source sample.mp4 --device 0

""")

# ============================================================================
# PART 7: MODEL INFORMATION
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ PART 7: MODEL VARIANTS INFORMATION                                        ║
╚════════════════════════════════════════════════════════════════════════════╝

YOLOv7 Variants (Performance on COCO dataset):

Model           Size    AP50   AP75   Speed(fps)  Params(M)  Use Case
─────────────────────────────────────────────────────────────────────
yolov7-tiny     25MB    39.9%  37.3%  430        6.2M       Real-time mobile
yolov7          37MB    51.4%  55.9%  161        37.6M      Balanced
yolov7x         71MB    53.1%  57.8%  114        71.3M      High accuracy
yolov7-w6       154MB   54.9%  60.1%  84 (1280)  79.8M      Large objects
yolov7-e6       226MB   56.0%  61.2%  56 (1280)  95.9M      Highest accuracy

Recommendations:
  • Testing:       Use yolov7-tiny (fastest, good for CPU)
  • Production:    Use yolov7 or yolov7-x (balance speed/accuracy)
  • Real-time:     Use yolov7-tiny (mobile/edge devices)
  • High accuracy: Use yolov7-e6 or yolov7-e6e (GPU recommended)

""")

# ============================================================================
# PART 8: NEXT STEPS
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ PART 8: RECOMMENDED NEXT STEPS                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

1. Run the validation:
   $ python quick_test.py
   
2. Download model and test:
   $ python download_and_test.py
   
3. Try detection on sample video:
   $ python detect.py --weights yolov7-tiny.pt --source sample.mp4 --view-img
   
4. Try on the street video:
   $ python detect.py --weights yolov7-tiny.pt --source street.mp4 --save-txt
   
5. Explore detection with tracking:
   $ python detect_or_track.py --weights yolov7-tiny.pt --source sample.mp4
   
6. Experiment with different parameters:
   $ python detect.py --weights yolov7-tiny.pt --source sample.mp4 \\
       --conf-thres 0.5 --iou-thres 0.45 --img-size 416
   
7. Look at results:
   Check: runs/detect/exp/ directory for output images/videos
   
8. When ready to train custom model:
   Prepare custom dataset in data/custom_data.yaml
   $ python train.py --data data/custom_data.yaml --cfg cfg/training/yolov7.yaml

""")

# ============================================================================
# FOOTER
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ You're all set! Your iSENSOR repo is fixed and ready to use.              ║
║                                                                            ║
║ Start with: python quick_test.py                                          ║
║ Then:       python download_and_test.py                                   ║
║ Finally:    python detect.py --weights yolov7-tiny.pt --source sample.mp4 ║
║                                                                            ║
║ Questions? Check README.md or README_OfficiallYOLOV7.md                   ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
