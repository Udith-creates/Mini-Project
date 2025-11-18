# PyTorch 2.6+ Compatibility Fix

## Problem

PyTorch 2.6+ introduced a security change that makes `torch.load()` use `weights_only=True` by default. This prevents loading YOLOv7 weights because the `models.yolo.Model` class needs to be allowlisted as a safe global.

**Error Message:**
```
WeightsUnpickler error: Unsupported global: GLOBAL models.yolo.Model 
was not an allowed global by default.
```

## Solution

The fix has been applied to:
1. `models/experimental.py` - Updated `attempt_load()` function
2. `download_and_test.py` - Added safe loading with fallback
3. `streamlit_app.py` - Added error handling for model loading

### How It Works

The updated code tries two approaches:

```python
try:
    # Try loading with weights_only=False (works on PyTorch 2.6+)
    ckpt = torch.load(w, map_location=map_location, weights_only=False)
except Exception:
    # Fallback: use safe_globals context manager
    from models.yolo import Model
    with torch.serialization.safe_globals([Model]):
        ckpt = torch.load(w, map_location=map_location)
```

This ensures compatibility across PyTorch versions (2.0+, 2.6+, and future versions).

## Testing

Run any of these to verify the fix works:

```bash
# Test model download and loading
python download_and_test.py

# Test with image
python detect.py --weights yolov7-tiny.pt --source sample.mp4

# Test with Streamlit
streamlit run streamlit_app.py
```

## Files Modified

| File | Changes | Reason |
|------|---------|--------|
| `models/experimental.py` | Added safe_globals fallback in `attempt_load()` | Core model loading fix |
| `download_and_test.py` | Added try-except with safe loading retry | Test script compatibility |
| `streamlit_app.py` | Enhanced error handling for model loading | Web UI robustness |

## PyTorch Version Support

- ✓ PyTorch 2.0-2.5 (works with `weights_only=False`)
- ✓ PyTorch 2.6+ (works with `safe_globals` context manager)
- ✓ Future versions (graceful fallback)

## No Action Required

The fix is already applied. Just use the code normally:
- `python download_and_test.py`
- `python detect.py --weights yolov7-tiny.pt --source video.mp4`
- `streamlit run streamlit_app.py`

All will work without additional configuration!
