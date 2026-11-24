import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Smart Queue Monitor", layout="wide")

st.title("Smart Queue Monitor")
st.caption("Live snapshot analysis for Streamlit Cloud")

img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    # Decode the captured image
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Run detection on the captured image
    results = YOLO("yolov8n.pt")(frame, conf=0.45)
    annotated_frame = results[0].plot()

    # Display results
    st.image(annotated_frame, channels="BGR", use_container_width=True)
    st.success(f"Detected {len(results[0].boxes)} people")