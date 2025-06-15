# 🚗🔍 Parking Inspector & Detector (YOLOv8 + Streamlit)

An intelligent system for detecting parking slot occupancy in images and videos using YOLOv8 and an interactive Streamlit dashboard.

## 📸 Overview

This project allows you to:
- Analyze images or video to determine which parking slots are **occupied** or **free**.
- Manually calibrate parking slot positions with a visual tool.
- Calculate **Intersection over Union (IoU)** between each slot and detected objects.
- Display real-time results in an interactive dashboard.
- Support both static image and live video modes.

---

## 🧠 Key Features

✅ YOLOv8-based vehicle detection (cars, trucks, buses, motorcycles)  
✅ Manual slot calibration with `matplotlib`  
✅ Real-time IoU calculation for each slot  
✅ Visual overlays for detection and status  
✅ Interactive interface built with Streamlit  
✅ Adjustable confidence, IoU thresholds, and slot dimensions

---

## 📁 Project Structure

| File                          | Description |
|-------------------------------|-------------|
| `parking_inspector_debug.py`  | Main Streamlit application for image and video analysis |
| `calibrate_slots_matplotlib.py` | Slot calibration tool to manually mark top-left corners |
| `CarParkPos.pkl`              | Pickled file storing parking slot positions |
| `requirements.txt`            | Python dependencies for the project |

---

## 🖥️ Application Demo

### 🔎 Static Image Analysis
- Upload a sample image
- Upload `CarParkPos.pkl`
- The app will draw detected vehicles and show IoU & status for each slot

### 🎥 Video Analysis
- Upload an `.mp4` video
- Upload the same `CarParkPos.pkl`
- The app will process the video frame-by-frame, showing live detections

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run parking_inspector_debug.py
