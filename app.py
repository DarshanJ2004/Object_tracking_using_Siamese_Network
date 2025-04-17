import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from model import SiameseNet

# Your model file
yolo_model = YOLO('yolov8s.pt')
# ------------------- Load Models -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Siamese tracker
siamese_model = SiameseNet().to(device)
siamese_model.load_state_dict(torch.load("siamese_tracker.pt", map_location=device))
siamese_model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ------------------- Helper Functions -------------------
def extract_patch(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    patch = frame[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

def get_best_match(template_tensor, search_frame):
    h, w, _ = search_frame.shape
    best_score = float('inf')
    best_box = None
    step = 32
    box_size = 128

    for y in range(0, h - box_size, step):
        for x in range(0, w - box_size, step):
            patch = search_frame[y:y+box_size, x:x+box_size]
            if patch.shape[0] != 128 or patch.shape[1] != 128:
                continue
            patch_tensor = transform(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = siamese_model(template_tensor, patch_tensor)
            loss = torch.nn.functional.mse_loss(pred, torch.tensor([[0.0, 0.0, 1.0, 1.0]]).to(device))
            if loss.item() < best_score:
                best_score = loss.item()
                best_box = (x, y, x+box_size, y+box_size)

    return best_box

def detect_objects_with_yolo(frame):
    results = yolo_model(frame)
    return results[0].boxes.xyxy.cpu().numpy()

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Live Tracker", layout="wide")
st.title("🎯 Live Webcam Object Tracking SiameseNet")

start_cam = st.checkbox("Start Webcam")

if start_cam:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Failed to access webcam. Check permissions or close other apps.")
    else:
        stframe = st.empty()
        template_tensor = None
        tracking_enabled = False
        init_bbox = None

        # Initialize object index
        obj_index = 0

        while True:
            success, frame = cap.read()
            if not success:
                st.warning("📷 Could not read frame from webcam.")
                break

            display_frame = frame.copy()

            if not tracking_enabled:
                detections = detect_objects_with_yolo(frame)

                for i, box in enumerate(detections):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"{i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                stframe.image(display_frame, channels="BGR", caption="Select object to track")

                if len(detections) > 0:
                    # Move the selectbox and button outside the loop
                    obj_index = st.selectbox(
                        "Select Object Index to Track", 
                        list(range(len(detections))),
                        key="object_select_1"  # Unique key
                    )
                    if st.button("🎯 Start Tracking"):
                        init_bbox = tuple(map(int, detections[obj_index]))
                        template_patch = extract_patch(frame, init_bbox)
                        template_tensor = transform(template_patch).unsqueeze(0).to(device)
                        tracking_enabled = True
            else:
                best_box = get_best_match(template_tensor, frame)
                if best_box:
                    x1, y1, x2, y2 = best_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                stframe.image(frame, channels="BGR", caption="Tracking...")

        cap.release()
else:
    st.info("👈 Enable 'Start Webcam' checkbox to begin tracking.")
