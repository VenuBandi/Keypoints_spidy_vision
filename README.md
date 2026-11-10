<div align="center">

# Keypoints Spidy Vision

Real-time human keypoint detection on NVIDIA Jetson, streamed to a live web dashboard.

[![GitHub stars](https://img.shields.io/github/stars/VenuBandi/Keypoints_spidy_vision?style=for-the-badge&color=yellow)](https://github.com/VenuBandi/Keypoints_spidy_vision/stargazers)

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![Jetson](https://img.shields.io/badge/NVIDIA-Jetson-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-nano)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

<br>

![Keypoint Detection Demo](https://github.com/user-attachments/assets/b2212283-8730-4a58-8f17-3ee31c5f38e2)

---

## What It Does

Detects human body keypoints from a camera feed in real time and streams the annotated output to a Flask web dashboard. Runs on-device using NVIDIA Jetson Inference — no cloud round-trip.

```
Camera Feed → Keypoint Detection (Jetson) → Frame Processing → Flask Server → Live Dashboard
```

Access the dashboard at `http://<JETSON_IP>:5000`

---

## Use Cases

- Foundation for gesture recognition systems
- Anomaly detection (unauthorized zones, posture alerts)
- Silent communication research
- Human activity recognition on edge devices

---

## Quick Start

```bash
git clone https://github.com/VenuBandi/Keypoints_spidy_vision
cd Keypoints_spidy_vision
pip install flask opencv-python numpy
python keypoints_spidy_vision.py
```

---

## Stack

`Python` `OpenCV` `NVIDIA Jetson Inference` `Flask`

---

<div align="center">

If this project was useful, a ⭐ helps others find it.

</div>
