# 🕷️ Keypoints Spidy Vision

### 🧠 Real-Time Human Keypoint Detection on NVIDIA Jetson

---

## 🚀 Project Overview

Welcome to **Keypoints Spidy Vision** — a real-time Edge AI system that detects human body keypoints and streams them to a live dashboard.

Inspired by a **Spiderman-style vision interface**, this project combines **computer vision, real-time inference, and interactive UI** to create an intelligent perception system.

---

## ⚡ What This Project Does

🔹 Detects human keypoints in real-time
🔹 Streams processed frames to a live dashboard
🔹 Runs efficiently on NVIDIA Jetson (Edge AI)
🔹 Forms the base for gesture recognition & anomaly detection

---

## 🧠 System Pipeline

```text
Camera Feed → Keypoint Detection → Frame Processing → Flask Server → Live Dashboard
```

---

## 🕷️ Spidy Vision Dashboard

* Real-time visual output
* Keypoint overlays
* Live streaming interface
* Designed for future alert systems

👉 Open in browser:

```
http://<JETSON_IP>:5000
```

---

## 🖼️ Image-Based Detection (Extension)

This system can also be extended to process static images:

* Upload image → Detect keypoints → Display output
* Useful for testing without camera
* Supports debugging and analysis

⚠️ *Planned enhancement (can be implemented easily)*

---

## 🛠️ Tech Stack

* 🐍 Python
* 🎥 OpenCV
* ⚡ NVIDIA Jetson Inference
* 🌐 Flask
* 🧠 Computer Vision Models

---

## ▶️ How to Run

### 1️⃣ Clone Repo

```bash
git clone https://github.com/Venub1/Keypoints_spidy_vision.git
cd Keypoints_spidy_vision
```

### 2️⃣ Install Dependencies

```bash
pip install flask opencv-python numpy
```

### 3️⃣ Run the System

```bash
python keypoints_spidy_vision.py
```

---

## 📸 Demo

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/85404155-d5a8-4637-890e-571be13c1b82" />


Example:

```
/screenshots/dashboard.png
```

---

## 🎯 Use Cases

🚨 Anomaly Detection Systems
🧠 Gesture Recognition
🧑‍🚒 Silent Communication (Security / Emergency)
🏭 Smart Surveillance
🤖 Human Activity Recognition

---

## 🔮 Future Enhancements

* 🧠 LSTM-based gesture classification
* 🚨 Forbidden object detection (guns, phones, etc.)
* ⚠️ Real-time alert system
* 🎨 Advanced Spiderman-style UI
* ⚡ Performance optimization

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Occlusion can affect accuracy
* Requires Jetson setup for optimal performance

---

## 🧠 Project Vision

This project is part of a broader goal to build **real-time intelligent systems on edge devices**, enabling:

* Faster decision-making
* Reduced latency
* Real-world deployment readiness

---

## 👨‍💻 Author

**Venu Bandi**
MPS Data Science – UMBC

---

## ⭐ Acknowledgements

* NVIDIA Jetson Inference
* OpenCV Community
* Real-time pose estimation research

---
