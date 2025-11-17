import streamlit as st
import cv2
import numpy as np
from detection import AccidentDetectionModel
from pathlib import Path
import tempfile
import os
import time
import requests
import time

st.set_page_config(page_title="Accident Detection Dashboard", layout="wide")
st.title("Accident Detection Dashboard")


# Sidebar: select video source or batch mode
video_source = st.sidebar.text_input("Video file path (or 0 for webcam)", value="0")
run_button = st.sidebar.button("Start Detection")
batch_button = st.sidebar.button("Run on all test videos")
image_batch_button = st.sidebar.button("Run on all test images")
def process_image(image_path):
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(rgb_img, (250, 250))
    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    if pred == "Accident":
        prob_val = round(prob[0][0] * 100, 2)
        return {
            "Image": str(image_path.name),
            "Probability": prob_val
        }
    return None

if image_batch_button:
    test_dir = Path("data/test")
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
# Sidebar: SMTP credentials and location
st.sidebar.header("Email Notification Settings")
smtp_server = st.sidebar.text_input("SMTP server", value="smtp.gmail.com")
smtp_port = st.sidebar.number_input("SMTP port", value=587)
smtp_user = st.sidebar.text_input("SMTP username (your Gmail)", value="threadstogether23@gmail.com")
smtp_pass = st.sidebar.text_input("SMTP password (app password recommended)", value="ccfsbvbatnxasvgs", type="password")

st.sidebar.header("Location Info")
if "latitude" not in st.session_state:
    st.session_state["latitude"] = ""
if "longitude" not in st.session_state:
    st.session_state["longitude"] = ""

def fetch_location():
    try:
        resp = requests.get("https://ipinfo.io/json", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if "loc" in data:
                lat, lon = data["loc"].split(",")
                st.session_state["latitude"] = lat
                st.session_state["longitude"] = lon
                # Reverse geocode to get place name
                place = get_place_name(lat, lon)
                st.session_state["place_name"] = place
                st.success(f"Location fetched: {lat}, {lon} ({place})")
            else:
                st.warning("Location not found in response.")
        else:
            st.warning(f"Failed to fetch location: {resp.status_code}")
    except Exception as e:
        st.warning(f"Error fetching location: {e}")

# Reverse geocode lat/lon to place name
def get_place_name(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
        resp = requests.get(url, headers={"User-Agent": "accident-detection-app"}, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if "display_name" in data:
                return data["display_name"]
        return "Unknown location"
    except Exception:
        return "Unknown location"

if st.sidebar.button("Auto-fetch my location"):
    fetch_location()

latitude = st.sidebar.text_input("Latitude", value=st.session_state["latitude"])
longitude = st.sidebar.text_input("Longitude", value=st.session_state["longitude"])
place_name = st.session_state.get("place_name", "")
if latitude and longitude and place_name:
    st.sidebar.info(f"Place: {place_name}")

def send_accident_email(log_data, total_frames, video_name, highest_prob_img_path=None):
    # Only send if any accident with prob >= 99
    high_prob_events = [e for e in log_data if e["Probability"] >= 99]
    if not high_prob_events:
        return
    # Use default sender/recipient if not set
    sender = smtp_user or "threadstogether23@gmail.com"
    password = smtp_pass or "ccfsbvbatnxasvgs"
    recipient = "udithsnair@gmail.com"
    if not sender or not password:
        return
    # Get place name if possible
    place_name = ""
    if latitude and longitude:
        place_name = get_place_name(latitude, longitude)
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = "Accident Detected! (High Probability)"
    body = f"Accident detected in video: {video_name}\n"
    if latitude and longitude:
        body += f"Location: {latitude}, {longitude}"
        if place_name:
            body += f" ({place_name})"
        body += "\n"
    body += f"Frames processed: {total_frames}\n"
    body += f"Number of high-probability accidents (>=99%): {len(high_prob_events)}\n"
    body += f"Total accident events: {len(log_data)}\n"
    body += "\nDetails (prob >= 99%):\n"
    for entry in high_prob_events:
        body += f"Frame: {entry['Frame']}, Time: {entry['Time (s)']}s, Probability: {entry['Probability']}%\n"
    body += "\nTech stack: Python, TensorFlow/Keras, OpenCV, Streamlit\n"
    msg.attach(MIMEText(body, "plain"))
    # Attach highest prob frame image if available
    if highest_prob_img_path and os.path.exists(highest_prob_img_path):
        with open(highest_prob_img_path, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(highest_prob_img_path)}"')
        msg.attach(part)
    try:
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        st.success(f"Accident email sent to {recipient}!")
    except Exception as e:
        pass

st.set_page_config(page_title="Accident Detection", layout="wide")
st.title("Accident Detection: Video Upload & Test")

# Load model
@st.cache_resource
def load_model():
    return AccidentDetectionModel("model.json", "model_weights.h5")
model = load_model()

# Upload video
uploaded_file = st.file_uploader("Upload a video file (.mp4)", type=["mp4"])

# Or select test video
test_videos = list(Path("data/test").glob("*.mp4"))
test_video = st.selectbox("Or select a test video", [None] + [str(v) for v in test_videos])

run_button = st.button("Run Accident Detection")

if run_button:
    # Determine video source
    if uploaded_file is not None:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name
    elif test_video:
        video_path = test_video
    else:
        st.error("Please upload a video or select a test video.")
        st.stop()

    st.success(f"Processing: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        st.stop()
    frame_count = 0
    accident_count = 0
    log_data = []
    stframe = st.empty()
    log_table = st.empty()
    highest_prob = -1
    highest_prob_frame = None
    highest_prob_frame_num = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            accident_count += 1
            prob_val = round(prob[0][0] * 100, 2)
            log_data.append({
                "Frame": frame_count,
                "Time (s)": round(frame_count / cap.get(cv2.CAP_PROP_FPS), 2),
                "Probability": prob_val
            })
            if prob_val > highest_prob:
                highest_prob = prob_val
                highest_prob_frame = frame.copy()
                highest_prob_frame_num = frame_count
            # Draw overlay
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Accident {prob_val}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        # Show video frame
        stframe.image(frame, channels="BGR", caption=f"Frame {frame_count}", use_column_width=True)
        # Show log table
        if log_data:
            log_table.dataframe(log_data)
        time.sleep(0.01)
    cap.release()
    st.success(f"Done. Total frames: {frame_count}, Accidents detected: {accident_count}")
    # Save highest prob frame if >=99
    highest_prob_img_path = None
    if highest_prob >= 99 and highest_prob_frame is not None:
        import tempfile
        highest_prob_img_path = os.path.join(tempfile.gettempdir(), f"accident_frame_{highest_prob_frame_num}.jpg")
        cv2.imwrite(highest_prob_img_path, highest_prob_frame)
    if log_data:
        st.dataframe(log_data)
        send_accident_email(log_data, frame_count, os.path.basename(video_path), highest_prob_img_path)
    else:
        st.info("No accidents detected.")
    # Clean up temp file
    if uploaded_file is not None:
        os.remove(video_path)
