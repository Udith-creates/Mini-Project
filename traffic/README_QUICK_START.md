# iSENSOR - Quick Start Guide

## 🚀 Getting Started

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_streamlit.txt
```

### Step 2: Download Model Weights
```bash
python download_and_test.py
```

This will:
- Download `yolov7-tiny.pt` (~12 MB)
- Test model loading
- Verify inference works
- Show sample detections

### Step 3: Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

Open browser at: **http://localhost:8501**

---

## 📊 What's New - PyTorch 2.6+ Support

✓ **Fixed:** Model loading with PyTorch 2.6+
✓ **Added:** Safe globals allowlisting for weights
✓ **Improved:** Error messages and fallback loading

See `PYTORCH_2.6_FIX.md` for technical details.

---

## 🎯 Available Commands

### Detection (Image/Video)
```bash
# Image detection
python detect.py --weights yolov7-tiny.pt --source image.jpg

# Video detection
python detect.py --weights yolov7-tiny.pt --source sample.mp4

# With tracking
python detect_or_track.py --weights yolov7-tiny.pt --source sample.mp4
```

### Testing
```bash
# Quick validation (no weights needed)
python quick_test.py

# Download and test inference
python download_and_test.py

# PyTorch Hub integration
python hubconf.py
```

### Web Interface
```bash
# Streamlit app (recommended for ease of use)
streamlit run streamlit_app.py

# Setup assistant
python setup_streamlit.py
```

---

## 📁 Output Locations

- **Image detections:** `runs/detect/streamlit_results/`
- **Video detections:** `runs/detect/streamlit_video/`
- **Model weights:** `./*.pt` (current directory)

---

## ⚙️ Configuration Options

### Model Selection
- `yolov7-tiny.pt` - Fast, 12 MB
- `yolov7.pt` - Balanced, 37 MB (auto-downloaded)
- `yolov7x.pt` - Accurate, 71 MB (auto-downloaded)

### Detection Parameters
- **Confidence Threshold:** 0.0-1.0 (default: 0.25)
- **IoU Threshold (NMS):** 0.0-1.0 (default: 0.45)
- **Image Size:** 320-1280 (default: 640)

---

## 🔍 Troubleshooting

### "Model weights not found"
```bash
python download_and_test.py  # Download weights
```

### "No objects detected"
- Lower confidence threshold (try 0.15-0.2)
- Use larger model (yolov7.pt or yolov7x.pt)
- Check image/video quality

### Slow Processing
- Lower image size (416 instead of 640)
- Use yolov7-tiny.pt
- Enable GPU if available

### PyTorch 2.6+ Loading Error
- Already fixed! See `PYTORCH_2.6_FIX.md`
- No action needed - just works

---

## 📚 Documentation

- **Setup Guide:** `STREAMLIT_GUIDE.md`
- **PyTorch 2.6+ Fix:** `PYTORCH_2.6_FIX.md`
- **Testing Guide:** `TESTING_GUIDE.py`
- **Quick Start:** `QUICK_START.md`

---

## 🎓 Model Performance

| Model | Speed (GPU) | Accuracy | Size |
|-------|-----------|----------|------|
| YOLOv7-Tiny | 430 fps | Good | 12 MB |
| YOLOv7 | 161 fps | High | 37 MB |
| YOLOv7-X | 114 fps | Very High | 71 MB |

---

## 🔗 Links

- **GitHub:** https://github.com/iamthighs/iSENSOR
- **YOLOv7 Paper:** https://arxiv.org/abs/2207.02696
- **PyTorch Docs:** https://pytorch.org/
- **Streamlit Docs:** https://docs.streamlit.io/

---

## ✨ Features

✓ Real-time object detection
✓ Video processing with frame-by-frame analysis
✓ Object tracking (SORT algorithm)
✓ Web-based Streamlit interface
✓ Multiple model sizes (speed/accuracy tradeoff)
✓ GPU/CPU support
✓ ONNX/TensorRT export
✓ COCO 80-class detection

---

## 🚦 Status Check

Run this to verify everything is working:

```bash
python quick_test.py
```

Expected output: **7/7 TESTS PASSED**

---

**Ready to detect objects!** 🎯
