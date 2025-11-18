# iSENSOR Streamlit Interface - Setup & Usage Guide

## 📋 Installation

### Step 1: Install Streamlit Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### Step 2: Download Model Weights (First Time Only)
If you haven't already downloaded the model weights, run:
```bash
python download_and_test.py
```

This will download YOLOv7-Tiny (~24MB) and test it. You can then manually download other models or they'll be auto-downloaded on first use.

### Step 3: Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at: `http://localhost:8501`

---

## 🎯 Features Overview

### 🖼️ Image Detection Tab
- **Upload** image files (JPG, PNG, BMP)
- **Configure** detection parameters (confidence, IoU)
- **Run** real-time detection
- **View** results with bounding boxes and class labels
- **Save** detection results to `runs/detect/streamlit_results/`

**Steps:**
1. Click "Choose an image" and upload a file
2. Adjust detection parameters in sidebar
3. Click "🚀 Run Detection"
4. View detected objects with confidence scores

**Tips:**
- Confidence threshold: Lower (0.1-0.3) = more detections, Higher (0.5+) = fewer false positives
- Check "Show Confidence" to see confidence scores on boxes
- Check "Save Results" to keep detection images

### 🎥 Video Detection Tab
- **Upload** video files or use sample videos
- **Process** entire videos frame-by-frame
- **Track** detections throughout video
- **Download** processed video with detections
- **View** statistics (total frames, detections)

**Steps:**
1. Choose upload or sample video
2. Select video file
3. Adjust detection parameters
4. Click "🚀 Process Video"
5. Wait for processing (shows progress)
6. Download or view result

**Supported formats:** MP4, AVI, MOV, MKV

**Tips:**
- Lower image size for faster processing
- YOLOv7-Tiny recommended for long videos
- GPU significantly speeds up processing

### 📊 Analytics Tab
- **View** system information (device, library versions)
- **Compare** YOLOv7 model variants
- **Understand** performance metrics
- **Read** usage tips and troubleshooting

### ℹ️ About Tab
- Project overview and features
- Technology stack
- Documentation links
- Quick start guide
- Model information

---

## ⚙️ Configuration Guide

### Sidebar Settings

**Model Selection:**
- `YOLOv7-Tiny`: Fastest (430 fps), best for real-time
- `YOLOv7`: Balanced (161 fps), recommended default
- `YOLOv7-X`: Most accurate (114 fps), best precision

**Detection Parameters:**
- `Confidence Threshold` (0.0-1.0, default 0.25): Minimum confidence to show detection
- `IoU Threshold` (0.0-1.0, default 0.45): Non-Maximum Suppression threshold
- `Input Image Size` (320-1280, default 640): Detection resolution

**Advanced Options:**
- `Save Results`: Keep detection outputs to disk
- `Show Labels`: Display class labels on boxes
- `Show Confidence`: Display confidence scores

**System Info:**
- Device (CPU/CUDA)
- PyTorch version
- CUDA availability

---

## 📁 Output Structure

### Image Detection Results
```
runs/
└── detect/
    └── streamlit_results/
        └── result_1234567890.jpg
```

### Video Detection Results
```
runs/
└── detect/
    └── streamlit_video/
        └── result_1234567890.mp4
```

---

## 🔧 Troubleshooting

### "Model weights not found" Error
**Solution:** Run `python download_and_test.py` to download weights

### "No objects detected"
**Solutions:**
- Lower the confidence threshold (try 0.15-0.2)
- Try a different model (YOLOv7-X is more accurate)
- Ensure input image has clear objects

### Slow Processing
**Solutions:**
- Lower image size (try 416 instead of 640)
- Use YOLOv7-Tiny model
- Enable GPU if available (see System Info)

### "Out of Memory" Error
**Solutions:**
- Lower image size
- Use YOLOv7-Tiny
- Process shorter videos
- Restart Streamlit app

### Unicode Encoding Errors (Windows)
- Automatically handled in this version
- If issues persist, check Windows console encoding

---

## 💻 System Requirements

### Minimum (CPU Mode)
- RAM: 8GB
- Storage: 2GB for models + temp files
- Processor: Intel i5 or equivalent

### Recommended (GPU Mode)
- GPU: NVIDIA GTX 1060 or better
- CUDA: 11.8+
- RAM: 16GB
- VRAM: 8GB+

---

## 🚀 Performance Tips

### For Real-time Processing
```
- Model: YOLOv7-Tiny
- Image Size: 320-416
- Confidence: 0.3+
- Device: GPU (if available)
```

### For High Accuracy
```
- Model: YOLOv7-X
- Image Size: 1280
- Confidence: 0.25-0.35
- Device: GPU (recommended)
```

### For Balanced Performance
```
- Model: YOLOv7 (default)
- Image Size: 640
- Confidence: 0.25-0.4
- Device: GPU (recommended) or CPU
```

---

## 📊 Expected Performance

### YOLOv7-Tiny (12 MB)
- **Speed:** ~430 fps (GPU), ~50 fps (CPU)
- **Accuracy:** Good for most use cases
- **Memory:** ~500MB (inference)
- **Best for:** Real-time video, embedded systems

### YOLOv7 (37 MB)
- **Speed:** ~161 fps (GPU), ~20 fps (CPU)
- **Accuracy:** High precision and recall
- **Memory:** ~1.5GB (inference)
- **Best for:** Balanced production use

### YOLOv7-X (71 MB)
- **Speed:** ~114 fps (GPU), ~10 fps (CPU)
- **Accuracy:** Highest accuracy
- **Memory:** ~2.5GB (inference)
- **Best for:** High-precision applications

---

## 📝 Example Workflows

### Workflow 1: Quick Detection
1. Upload image
2. Use default settings
3. Run detection
4. Download result

### Workflow 2: Fine-tuned Detection
1. Upload image
2. Lower confidence to 0.15
3. Use YOLOv7-X for accuracy
4. Save results
5. Analyze detections

### Workflow 3: Video Analysis
1. Upload video
2. Use YOLOv7-Tiny for speed
3. Set image size to 416
4. Process video
5. Download result
6. Post-process if needed

---

## 🔗 Resources

- **GitHub:** https://github.com/iamthighs/iSENSOR
- **YOLOv7 Paper:** https://arxiv.org/abs/2207.02696
- **Streamlit Docs:** https://docs.streamlit.io/
- **PyTorch Hub:** https://pytorch.org/hub/

---

## 📞 Support

For issues:
1. Check the "Analytics & Information" tab for system details
2. Review this guide's troubleshooting section
3. Ensure model weights are downloaded
4. Check available disk space and RAM
5. Try with smaller images/videos first

---

## 🎯 Next Steps

1. ✓ Install Streamlit: `pip install -r requirements_streamlit.txt`
2. ✓ Download models: `python download_and_test.py`
3. ✓ Run app: `streamlit run streamlit_app.py`
4. ✓ Test with sample images/videos
5. ✓ Customize as needed

**Ready to detect!** 🚀
