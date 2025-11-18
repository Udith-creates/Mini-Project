"""
iSENSOR - Streamlit Web Interface
Real-time object detection and tracking using YOLOv7
"""

import streamlit as st
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import time
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="iSENSOR - Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #1f77b4;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 16px;
        margin-bottom: 20px;
    }
    .stat-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="title">🎯 iSENSOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time Object Detection & Tracking with YOLOv7</div>', unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'detections' not in st.session_state:
    st.session_state.detections = None

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
col1, col2 = st.sidebar.columns(2)
with col1:
    st.sidebar.markdown("### 📦 Model Selection")
    model_choice = st.sidebar.radio(
        "Select Model",
        ["YOLOv7-Tiny", "YOLOv7", "YOLOv7-X"],
        help="Choose model based on speed/accuracy trade-off"
    )

# Detection parameters
st.sidebar.markdown("### 🎚️ Detection Parameters")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.25,
    help="Only show detections above this confidence"
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)",
    0.0, 1.0, 0.45,
    help="Remove overlapping boxes using Non-Maximum Suppression"
)

# Image size option
img_size = st.sidebar.selectbox(
    "Input Image Size",
    [320, 416, 512, 640, 1280],
    index=3,
    help="Larger = better accuracy, slower inference"
)

# Advanced options
st.sidebar.markdown("### 🔧 Advanced Options")
save_results = st.sidebar.checkbox("Save Results", value=True)
show_labels = st.sidebar.checkbox("Show Labels", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence", value=True)

# Device info
st.sidebar.markdown("---")
st.sidebar.markdown("### 💻 System Info")
st.sidebar.info(f"""
**Device:** {str(st.session_state.device).upper()}
**CUDA Available:** {torch.cuda.is_available()}
**PyTorch Version:** {torch.__version__}
""")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Image Detection",
    "🎥 Video Detection", 
    "📊 Analytics",
    "ℹ️ About"
])

# ==================== TAB 1: IMAGE DETECTION ====================
with tab1:
    st.header("Image Detection")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key='image_upload'
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save to temp file for processing
            temp_image_path = Path(tempfile.gettempdir()) / uploaded_file.name
            image.save(temp_image_path)
            
            if st.button("🚀 Run Detection", key='detect_image'):
                with st.spinner("Loading model..."):
                    try:
                        from models.experimental import attempt_load
                        from utils.general import non_max_suppression, scale_coords
                        from utils.plots import plot_one_box
                        
                        # Load model
                        model_map = {
                            "YOLOv7-Tiny": "yolov7-tiny.pt",
                            "YOLOv7": "yolov7.pt",
                            "YOLOv7-X": "yolov7x.pt"
                        }
                        weights = model_map[model_choice]
                        
                        if not Path(weights).exists():
                            st.error(f"❌ Model weights not found: {weights}")
                            st.info(f"Run: `python download_and_test.py` to download weights")
                        else:
                            try:
                                model = attempt_load(weights, map_location=st.session_state.device)
                                model.eval()
                            except Exception as load_err:
                                st.error(f"❌ Failed to load model: {str(load_err)}")
                                st.info("Try: `python download_and_test.py` to re-download weights")
                                raise
                            
                            # Prepare image
                            img_cv = cv2.imread(str(temp_image_path))
                            img_h, img_w = img_cv.shape[:2]
                            
                            # Resize and normalize
                            img_resized = cv2.resize(img_cv, (img_size, img_size))
                            img_tensor = torch.from_numpy(img_resized).float() / 255.0
                            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                            img_tensor = img_tensor.to(st.session_state.device)
                            
                            # Inference
                            with torch.no_grad():
                                pred = model(img_tensor)[0]
                            
                            # NMS
                            pred = non_max_suppression(pred, conf_threshold, iou_threshold)
                            
                            # Draw results
                            det_count = 0
                            if len(pred) > 0 and len(pred[0]) > 0:
                                dets = pred[0]
                                det_count = len(dets)
                                
                                # Scale coordinates back to original size
                                scale_x = img_w / img_size
                                scale_y = img_h / img_size
                                
                                img_display = img_cv.copy()
                                for det in dets:
                                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                                    x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                                    
                                    # Draw box
                                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Draw label
                                    if show_labels:
                                        label = f"Class {int(cls)}: {conf:.2f}"
                                        cv2.putText(img_display, label, (x1, y1-5),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # Display result
                                st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), 
                                       caption=f"Detection Results ({det_count} objects)", 
                                       use_column_width=True)
                                
                                # Save if requested
                                if save_results:
                                    output_path = Path("runs/detect") / "streamlit_results"
                                    output_path.mkdir(parents=True, exist_ok=True)
                                    result_file = output_path / f"result_{int(time.time())}.jpg"
                                    cv2.imwrite(str(result_file), img_display)
                                    st.success(f"✓ Saved to {result_file}")
                            else:
                                st.info("No objects detected")
                                st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error during detection: {str(e)}")
                        st.info("Make sure model weights are downloaded. Run: `python download_and_test.py`")
    
    with col2:
        st.subheader("Statistics")
        if st.session_state.detections:
            st.metric("Detections", len(st.session_state.detections))
        
        st.markdown("---")
        st.subheader("Tips")
        st.markdown("""
        - Adjust confidence threshold to filter detections
        - Higher image size = better accuracy but slower
        - YOLOv7-Tiny is fastest, best for real-time
        - YOLOv7-X is most accurate, best for precision
        """)

# ==================== TAB 2: VIDEO DETECTION ====================
with tab2:
    st.header("Video Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Video or Select Sample")
        
        # Sample videos or upload
        video_source = st.radio("Choose source:", ["Upload Video", "Use Sample Video"])
        
        if video_source == "Upload Video":
            uploaded_video = st.file_uploader(
                "Choose a video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key='video_upload'
            )
            video_path = uploaded_video
        else:
            sample_videos = list(Path(".").glob("*.mp4"))
            if sample_videos:
                selected_video = st.selectbox(
                    "Select sample video",
                    sample_videos,
                    format_func=lambda x: x.name
                )
                video_path = selected_video
            else:
                st.warning("No sample videos found in current directory")
                video_path = None
        
        if video_path:
            st.info(f"Selected: {video_path.name if hasattr(video_path, 'name') else 'uploaded'}")
            
            if st.button("🚀 Process Video", key='detect_video'):
                st.warning("⏳ Video processing may take time. You can view results after completion.")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    from models.experimental import attempt_load
                    from utils.general import non_max_suppression
                    
                    # Save uploaded video if needed
                    if video_source == "Upload Video":
                        temp_video_path = Path(tempfile.gettempdir()) / uploaded_video.name
                        with open(temp_video_path, 'wb') as f:
                            f.write(uploaded_video.getbuffer())
                        video_path = temp_video_path
                    
                    # Load model
                    model_map = {
                        "YOLOv7-Tiny": "yolov7-tiny.pt",
                        "YOLOv7": "yolov7.pt",
                        "YOLOv7-X": "yolov7x.pt"
                    }
                    weights = model_map[model_choice]
                    
                    if not Path(weights).exists():
                        st.error(f"❌ Model not found: {weights}")
                    else:
                        try:
                            model = attempt_load(weights, map_location=st.session_state.device)
                            model.eval()
                        except Exception as load_err:
                            st.error(f"❌ Failed to load model: {str(load_err)}")
                            st.info("Try: `python download_and_test.py` to re-download weights")
                            raise
                        
                        # Open video
                        cap = cv2.VideoCapture(str(video_path))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # Output video setup
                        output_dir = Path("runs/detect/streamlit_video")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_file = output_dir / f"result_{int(time.time())}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
                        
                        frame_idx = 0
                        total_detections = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Inference
                            img_resized = cv2.resize(frame, (img_size, img_size))
                            img_tensor = torch.from_numpy(img_resized).float() / 255.0
                            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                            img_tensor = img_tensor.to(st.session_state.device)
                            
                            with torch.no_grad():
                                pred = model(img_tensor)[0]
                            
                            pred = non_max_suppression(pred, conf_threshold, iou_threshold)
                            
                            # Draw
                            frame_display = frame.copy()
                            if len(pred) > 0 and len(pred[0]) > 0:
                                dets = pred[0]
                                total_detections += len(dets)
                                
                                scale_x = width / img_size
                                scale_y = height / img_size
                                
                                for det in dets:
                                    x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                                    x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    if show_labels:
                                        label = f"{int(cls)}: {conf:.2f}"
                                        cv2.putText(frame_display, label, (x1, y1-5),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            out.write(frame_display)
                            
                            # Update progress
                            frame_idx += 1
                            progress = min(frame_idx / total_frames, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: Frame {frame_idx}/{total_frames} | Detections: {total_detections}")
                        
                        cap.release()
                        out.release()
                        
                        st.success(f"✓ Video saved to {output_file}")
                        st.info(f"📊 Total frames: {total_frames} | Total detections: {total_detections}")
                        
                        # Display video
                        with open(output_file, 'rb') as f:
                            st.video(f.read())
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("Video Settings")
        st.markdown("""
        **Processing Tips:**
        - Longer videos take more time
        - Lower image size for faster processing
        - GPU recommended for long videos
        
        **Output:**
        - Saved to `runs/detect/streamlit_video/`
        - Can download processed video
        """)

# ==================== TAB 3: ANALYTICS ====================
with tab3:
    st.header("Analytics & Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Device", str(st.session_state.device).upper())
        st.metric("CUDA Available", "Yes" if torch.cuda.is_available() else "No")
    
    with col2:
        st.metric("PyTorch Version", torch.__version__)
        st.metric("OpenCV Version", cv2.__version__)
    
    with col3:
        st.metric("NumPy Version", np.__version__)
        try:
            from PIL import __version__ as pil_version
            st.metric("PIL Version", pil_version)
        except:
            st.metric("PIL Version", "N/A")
    
    st.markdown("---")
    
    st.subheader("📊 Model Comparison")
    comparison_data = {
        "Model": ["YOLOv7-Tiny", "YOLOv7", "YOLOv7-X"],
        "Size (MB)": [12, 37, 71],
        "Parameters (M)": [6.2, 37.6, 71.3],
        "Speed (fps)": [430, 161, 114],
        "AP50 (%)": [39.9, 51.4, 53.1],
        "Best For": ["Real-time/Mobile", "Balanced", "High Accuracy"]
    }
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🎯 Usage Instructions")
    
    st.markdown("""
    ### Image Detection
    1. Upload an image (JPG, PNG, BMP)
    2. Adjust confidence and IoU thresholds
    3. Click "Run Detection"
    4. Results are displayed and optionally saved
    
    ### Video Detection
    1. Upload a video or select sample
    2. Configure detection parameters
    3. Click "Process Video"
    4. Watch progress and download results
    
    ### Tips for Best Results
    - **Confidence Threshold**: Lower (0.1-0.3) for more detections, higher (0.5+) for fewer false positives
    - **Image Size**: 640 is balanced, use 416 for speed, 1280 for accuracy
    - **IoU Threshold**: 0.45 is default, increase for stricter NMS
    """)

# ==================== TAB 4: ABOUT ====================
with tab4:
    st.header("About iSENSOR")
    
    st.markdown("""
    ### 🚀 Project Overview
    iSENSOR is a real-time object detection and tracking system built on YOLOv7.
    
    **Features:**
    - 🎯 Real-time object detection
    - 📊 Multi-model support (YOLOv7-Tiny, YOLOv7, YOLOv7-X)
    - 🎥 Video processing with frame-by-frame detection
    - 💾 Result saving and export
    - 🌐 Web-based Streamlit interface
    
    ### 📦 Technologies
    - **Framework:** PyTorch, YOLOv7
    - **Interface:** Streamlit
    - **Vision:** OpenCV, Pillow
    - **Tracking:** SORT algorithm
    
    ### 📚 Documentation
    - Main Repository: [iamthighs/iSENSOR](https://github.com/iamthighs/iSENSOR)
    - YOLOv7 Paper: [arXiv:2207.02696](https://arxiv.org/abs/2207.02696)
    
    ### 🔧 Getting Started
    1. Install dependencies: `pip install -r requirements.txt`
    2. Download model weights: `python download_and_test.py`
    3. Run Streamlit app: `streamlit run streamlit_app.py`
    
    ### 📞 Support
    - For issues with detection: Check model weights are downloaded
    - For slow inference: Use YOLOv7-Tiny or GPU device
    - For video processing: Ensure sufficient disk space
    
    ### 📋 Model Information
    """)
    
    st.info("""
    **YOLOv7 Models Available:**
    - **yolov7-tiny.pt** (12 MB) - Fastest, mobile-friendly
    - **yolov7.pt** (37 MB) - Balanced speed/accuracy
    - **yolov7x.pt** (71 MB) - Most accurate
    
    Download with: `python download_and_test.py`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px;">
    iSENSOR v1.0 | Built with Streamlit & YOLOv7
</div>
""", unsafe_allow_html=True)
