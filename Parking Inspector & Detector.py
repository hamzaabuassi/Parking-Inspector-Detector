# parking_inspector_debug.py

import cv2
import pickle
import tempfile
import time
import os

import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("🔍 Parking Inspector & Detector (Debug Mode)")

# ---------------------------------------
# Sidebar settings
MODEL_NAME     = st.sidebar.text_input("YOLO Model Path", "yolov8m.pt")
CONF_THRESHOLD = st.sidebar.slider("Detection Confidence ≥", 0.0, 1.0, 0.3, 0.01)
IOU_THRESHOLD  = st.sidebar.slider("IoU Threshold ≥",       0.0, 1.0, 0.1, 0.01)
SLOT_W         = st.sidebar.number_input("Slot Width",    min_value=10, max_value=500, value=107)
SLOT_H         = st.sidebar.number_input("Slot Height",   min_value=10, max_value=500, value=48)
# ---------------------------------------

# helper IoU function
def calc_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(areaA + areaB - inter)

tabs = st.tabs(["🔎 Inspect IoU", "🎥 Detect Video"])

with tabs[0]:
    st.header("1️⃣ Inspect IoU on Static Frame")
    img_file = st.file_uploader("Upload sample frame (jpg/png/jpeg)", type=["jpg","png","jpeg"])
    pkl_file = st.file_uploader("Upload CarParkPos.pkl", type=["pkl"])
    if img_file and pkl_file:
        # load image
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # load slots
        posList = pickle.load(pkl_file)
        # load and run model
        model = YOLO(MODEL_NAME)
        results = model(frame, conf=CONF_THRESHOLD)[0]
        dets    = results.boxes.xyxy.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        names   = results.names

        disp = frame.copy()
        # draw YOLO detections in blue
        for (x1, y1, x2, y2), cid in zip(dets, cls_ids):
            label = names[int(cid)]
            if label in ['car','truck','bus','motorcycle']:
                cv2.rectangle(disp, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
                cv2.putText(disp, label, (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        # draw slots in green
        for idx, (sx, sy) in enumerate(posList, start=1):
            cv2.rectangle(disp, (sx, sy), (sx+SLOT_W, sy+SLOT_H), (0,255,0), 2)
        st.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), use_column_width=True)

        # compute IoU per slot
        rows = []
        for idx, (sx, sy) in enumerate(posList, start=1):
            slot_box = (sx, sy, sx+SLOT_W, sy+SLOT_H)
            best_iou = 0.0
            for d in dets:
                best_iou = max(best_iou, calc_iou(slot_box, d))
            status = "Occupied" if best_iou >= IOU_THRESHOLD else "Free"
            rows.append({"Slot": idx, "IoU": round(best_iou,2), "Status": status})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

with tabs[1]:
    st.header("2️⃣ Detect on Video (with IoU Overlay)")
    pkl_file2 = st.file_uploader("Upload CarParkPos.pkl", type=["pkl"], key="pkl2")
    vid_file   = st.file_uploader("Upload MP4 video", type=["mp4"])
    start_btn  = st.button("▶️ Start")
    stop_btn   = st.button("⏹ Stop")

    if pkl_file2 and vid_file:
        posList = pickle.load(pkl_file2)
        # save temp video file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vid_file.read())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)

        model = YOLO(MODEL_NAME)
        placeholder = st.empty()
        running = False

        while True:
            # handle start/stop
            if start_btn:
                running = True
            if stop_btn:
                running = False
            if not running:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # detect
            results = model(frame, conf=CONF_THRESHOLD)[0]
            dets    = results.boxes.xyxy.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            names   = results.names

            # draw detections
            for (x1,y1,x2,y2), cid in zip(dets, cls_ids):
                label = names[int(cid)]
                if label in ['car','truck','bus','motorcycle']:
                    cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)

            # draw slots and overlay IoU & status
            for idx, (sx, sy) in enumerate(posList, start=1):
                slot_box = (sx, sy, sx+SLOT_W, sy+SLOT_H)
                best_iou = 0.0
                for d in dets:
                    best_iou = max(best_iou, calc_iou(slot_box, d))
                occupied = best_iou >= IOU_THRESHOLD
                col = (0,0,255) if occupied else (0,255,0)
                status = "Occ" if occupied else "Free"
                # draw slot rectangle
                cv2.rectangle(frame, (sx, sy), (sx+SLOT_W, sy+SLOT_H), col, 2)
                # overlay text: idx, status, IoU
                text = f"{idx}:{status} IoU={best_iou:.2f}"
                cv2.putText(frame, text, (sx, sy-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            time.sleep(0.03)

        cap.release()
        try:
            os.remove(tmp.name)
        except PermissionError:
            pass
        st.success("Detection finished.")
