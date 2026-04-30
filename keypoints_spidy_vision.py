#!/usr/bin/env python3
"""
Keypoints Spidy Vision - STRICT VERSION (FIXED)

Distance logic fixes:
1. Height estimation now uses a smarter approach:
   - Tries head-to-ankle/hip keypoint span first (most accurate)
   - Falls back to bbox height only if full-body span is available
   - Applies a scale correction factor because partial poses shrink the bbox
2. Anchor point is the body CENTER (not bottom) for measuring person-to-person
   distance — bottom anchors cause near-zero Y distance for people standing
   side by side, which breaks the ratio math.
3. NORMALIZED_DISTANCE_THRESHOLD lowered to a realistic value for a camera
   that sees people at normal indoor distances.
4. Fallback pixel threshold also corrected for typical camera FOV.
5. /status endpoint now returns all fields the W.E.B dashboard expects.
6. /stream route alias added (dashboard polls /stream, not /video).

Run inside Jetson container:
    cd /jetson-inference/python/examples/keypoints_spidy_vision_strict
    pip3 install flask
    python3 keypoints_spidy_vision.py
"""

import os
import cv2
import csv
import math
import time
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
from flask import Flask, Response, render_template, jsonify, request, send_from_directory

from jetson_inference import poseNet
from jetson_utils import videoSource, loadImage, cudaToNumpy


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME       = "Keypoints Spidy Vision"
POSE_NETWORK   = "resnet18-body"
POSE_THRESHOLD = 0.15
CAMERA_URI     = "/dev/video0"

# ---------------------------------------------------------------
# SOCIAL DISTANCE THRESHOLDS
#
# How the check works:
#   estimated_height  = best guess at real body height in pixels
#   center_distance   = Euclidean distance between body CENTERS
#   ratio             = center_distance / estimated_height
#
# Typical real-world ratios (camera ~3-5 m from subjects):
#   literally touching   -> ~0.3 – 0.5
#   arms-length apart    -> ~0.6 – 0.9
#   ~1 metre apart       -> ~1.0 – 1.4
#   safe distance (2 m)  -> ~1.8 – 2.5
#
# BREACH fires when ratio < NORMALIZED_DISTANCE_THRESHOLD.
# 0.8 = "arms-length or closer" — good general-purpose value.
# ---------------------------------------------------------------
NORMALIZED_DISTANCE_THRESHOLD = 0.8

# Fallback raw-pixel threshold used when height cannot be estimated
# reliably (too few keypoints on both people).
# 150 px is "clearly too close" for a 720p feed where a person
# typically spans 200-400 px of frame height.
FALLBACK_PIXEL_THRESHOLD = 150

# Minimum keypoints per person to trust the height estimate.
MIN_KEYPOINTS_FOR_SCALE = 4

# When height must be estimated from a partial bbox (not head-to-toe
# keypoints), the raw bbox is much smaller than true body height.
# This factor scales it up.  Tune if your camera angle is unusual.
PARTIAL_HEIGHT_SCALE = 2.2

# Live stream violation save cooldown (seconds).
LIVE_SAVE_COOLDOWN_SECONDS = 2.0

HOST = "0.0.0.0"
PORT = 5050


# resnet18-body keypoint indices (COCO order used by jetson-inference)
KP_NOSE       = 0
KP_L_EYE      = 1
KP_R_EYE      = 2
KP_L_EAR      = 3
KP_R_EAR      = 4
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_ELBOW    = 7
KP_R_ELBOW    = 8
KP_L_WRIST    = 9
KP_R_WRIST    = 10
KP_L_HIP      = 11
KP_R_HIP      = 12
KP_L_KNEE     = 13
KP_R_KNEE     = 14
KP_L_ANKLE    = 15
KP_R_ANKLE    = 16

HEAD_KPS  = [KP_NOSE, KP_L_EYE, KP_R_EYE, KP_L_EAR, KP_R_EAR]
FOOT_KPS  = [KP_L_ANKLE, KP_R_ANKLE, KP_L_KNEE, KP_R_KNEE]
UPPER_KPS = [KP_L_SHOULDER, KP_R_SHOULDER, KP_NOSE]
LOWER_KPS = [KP_L_HIP, KP_R_HIP, KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE]


# ============================================================
# FOLDERS
# ============================================================

BASE_DIR          = Path(__file__).resolve().parent
UPLOAD_DIR        = BASE_DIR / "uploads"
LIVE_CAPTURE_DIR  = BASE_DIR / "violation_captures" / "live_stream"
IMAGE_CAPTURE_DIR = BASE_DIR / "violation_captures" / "uploaded_images"
LOG_DIR           = BASE_DIR / "logs"
LOG_FILE          = LOG_DIR  / "spidy_detection_log.csv"

for folder in [UPLOAD_DIR, LIVE_CAPTURE_DIR, IMAGE_CAPTURE_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# APP STATE  (all fields that spiderman_dashboard_final.html polls)
# ============================================================

app        = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
state_lock = threading.Lock()

dashboard_state = {
    # W.E.B dashboard fields
    "fps":                  0.0,
    "frame":                0,
    "persons":              0,
    "objects":              0,       # always 0 — pose-only backend
    "avg_conf":             0,       # 0-100 proxy based on keypoint count
    "anomalies":            0,       # cumulative breach-pair count
    "status":               "NOMINAL",
    "last_anomaly_time":    0,       # epoch float, 0 = never
    "last_anomaly_reason":  "",
    "last_anomaly_type":    "",
    "time_since_last":      "N/A",
    "current_log":          [],
    # extra internal fields
    "mode":                     "IDLE",
    "violations":               0,
    "last_update":              "-",
    "last_saved":               "-",
    "normalized_threshold":     NORMALIZED_DISTANCE_THRESHOLD,
    "fallback_pixel_threshold": FALLBACK_PIXEL_THRESHOLD,
    "last_pairs":               []
}

last_live_save_time = 0.0


# ============================================================
# MODEL INIT
# ============================================================

print("=" * 80)
print(f"Spider  Starting {APP_NAME} - STRICT VERSION (FIXED)")
print("Loading NVIDIA Jetson poseNet ...")
net = poseNet(POSE_NETWORK, threshold=POSE_THRESHOLD)
print("poseNet loaded.")
print("=" * 80)


# ============================================================
# HELPERS
# ============================================================

def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def file_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def safe_filename(prefix, ext="jpg"):
    return f"{prefix}_{file_stamp()}.{ext}"

def allowed_file(filename):
    allowed = {"jpg", "jpeg", "png", "bmp"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def ensure_log_header():
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "mode", "people_detected", "violations",
                "distance_mode", "normalized_threshold",
                "fallback_pixel_threshold", "pair_details", "saved_image"
            ])


def write_log(mode, people, violations, pair_details, saved_image=""):
    ensure_log_header()
    compact = [
        f"P{p['i']+1}-P{p['j']+1}: {int(p['pixel_distance'])}px "
        f"ratio={p['normalized_distance']:.2f} "
        f"h~{int(p['avg_height'])}px "
        f"{'BREACH' if p['violation'] else 'SAFE'} ({p['rule_used']})"
        for p in pair_details
    ]
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            now_string(), mode, people, violations,
            "normalized center-dist / est-height",
            NORMALIZED_DISTANCE_THRESHOLD, FALLBACK_PIXEL_THRESHOLD,
            " | ".join(compact), saved_image
        ])


def cuda_img_to_bgr(img):
    frame = np.array(cudaToNumpy(img), copy=True)
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def keypoint_xy(kp):
    """Return (int_x, int_y) from a jetson-inference keypoint, or None."""
    x = getattr(kp, "x", None)
    y = getattr(kp, "y", None)
    if x is None:
        x = getattr(kp, "X", None)
    if y is None:
        y = getattr(kp, "Y", None)
    if x is None or y is None:
        return None
    return int(x), int(y)


def get_kp_map(pose, frame_w, frame_h):
    """
    Returns {kp_index: (x, y)} for all valid in-frame keypoints.
    Handles jetson-inference builds that store index in .ID or .id.
    """
    kp_map = {}
    try:
        keypoints = pose.Keypoints
    except Exception:
        return kp_map
    for kp in keypoints:
        xy = keypoint_xy(kp)
        if xy is None:
            continue
        x, y = xy
        if not (0 <= x < frame_w and 0 <= y < frame_h):
            continue
        idx = getattr(kp, "ID", getattr(kp, "id", None))
        if idx is not None:
            kp_map[int(idx)] = (x, y)
    return kp_map


def estimate_height(kp_map, bbox_height, frame_h):
    """
    Best-effort body height in pixels.

    Priority:
      1. head keypoints Y  ->  lowest foot/knee Y   (most reliable)
      2. shoulder Y        ->  hip/knee/ankle Y     (upper-body partial)
      3. bbox_height * PARTIAL_HEIGHT_SCALE          (last resort)
      4. 35% of frame height                         (absolute fallback)

    Returns (height_px, method_name).
    """
    head_ys = [kp_map[k][1] for k in HEAD_KPS if k in kp_map]
    foot_ys = [kp_map[k][1] for k in FOOT_KPS if k in kp_map]
    if head_ys and foot_ys:
        span = max(foot_ys) - min(head_ys)
        if span > 20:
            return max(span, 1), "head_to_foot"

    upper_ys = [kp_map[k][1] for k in UPPER_KPS if k in kp_map]
    lower_ys = [kp_map[k][1] for k in LOWER_KPS if k in kp_map]
    if upper_ys and lower_ys:
        span = max(lower_ys) - min(upper_ys)
        if span > 15:
            # shoulder-to-ankle ≈ 80% of full height
            return max(int(span * 1.25), 1), "shoulder_to_lower"

    if bbox_height > 10:
        return max(int(bbox_height * PARTIAL_HEIGHT_SCALE), 1), "scaled_bbox"

    return max(int(frame_h * 0.35), 1), "frame_fraction"


def extract_person_info(pose, frame_w, frame_h):
    """
    Returns a dict for one detected person, or None if unusable.
    """
    kp_map = get_kp_map(pose, frame_w, frame_h)
    points = list(kp_map.values())

    if len(points) < 2:
        try:
            cx = int(pose.Center[0])
            cy = int(pose.Center[1])
        except Exception:
            return None
        fb_h = max(int(frame_h * 0.35), 1)
        return {
            "center":         (cx, cy),
            "anchor":         (cx, cy + fb_h // 2),
            "bbox":           (cx - 30, cy - fb_h // 2, cx + 30, cy + fb_h // 2),
            "height":         fb_h,
            "height_method":  "frame_fraction",
            "keypoint_count": 0,
            "scale_reliable": False,
            "kp_map":         kp_map
        }

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(min(xs), 0)
    y1 = max(min(ys), 0)
    x2 = min(max(xs), frame_w - 1)
    y2 = min(max(ys), frame_h - 1)

    bbox_h           = max(y2 - y1, 1)
    height, method   = estimate_height(kp_map, bbox_h, frame_h)

    # CENTER of bbox — this is what we measure distance from
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # Bottom anchor — only used for drawing the line endpoint
    anchor_x = center_x
    anchor_y = int(y2)

    reliable = (
        len(points) >= MIN_KEYPOINTS_FOR_SCALE
        and method in ("head_to_foot", "shoulder_to_lower")
    )

    return {
        "center":         (center_x, center_y),
        "anchor":         (anchor_x, anchor_y),
        "bbox":           (x1, y1, x2, y2),
        "height":         height,
        "height_method":  method,
        "keypoint_count": len(points),
        "scale_reliable": reliable,
        "kp_map":         kp_map
    }


def analyze_people(people):
    """
    Check every pair.  Distance = center-to-center / avg estimated height.
    """
    pair_details = []
    violations   = 0

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1 = people[i]
            p2 = people[j]

            cx1, cy1 = p1["center"]
            cx2, cy2 = p2["center"]

            pixel_distance      = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
            avg_height          = max((p1["height"] + p2["height"]) / 2.0, 1.0)
            normalized_distance = pixel_distance / avg_height

            reliable = p1["scale_reliable"] and p2["scale_reliable"]

            if reliable:
                violation = normalized_distance < NORMALIZED_DISTANCE_THRESHOLD
                rule_used = "normalized"
            else:
                violation = pixel_distance < FALLBACK_PIXEL_THRESHOLD
                rule_used = "fallback_pixel"

            if violation:
                violations += 1

            pair_details.append({
                "i":                   i,
                "j":                   j,
                "p1":                  p1["anchor"],
                "p2":                  p2["anchor"],
                "pixel_distance":      pixel_distance,
                "avg_height":          avg_height,
                "normalized_distance": normalized_distance,
                "violation":           violation,
                "rule_used":           rule_used
            })

    return violations, pair_details


def draw_overlay(frame, people, pair_details, mode):
    h, w = frame.shape[:2]

    for pair in pair_details:
        x1, y1 = pair["p1"]
        x2, y2 = pair["p2"]
        color  = (0, 0, 255) if pair["violation"] else (0, 255, 0)
        status = "BREACH"    if pair["violation"] else "SAFE"

        cv2.line(frame, (x1, y1), (x2, y2), color, 3)

        mx = int((x1 + x2) / 2)
        my = int((y1 + y2) / 2)

        label = (
            f"{status}  dist={int(pair['pixel_distance'])}px "
            f"ratio={pair['normalized_distance']:.2f}"
            f"(thr {NORMALIZED_DISTANCE_THRESHOLD:.2f}) "
            f"h~{int(pair['avg_height'])}px [{pair['rule_used']}]"
        )
        cv2.rectangle(frame,
                      (max(mx - 165, 0),  max(my - 28, 0)),
                      (min(mx + 265, w-1), min(my + 8,  h-1)),
                      (0, 0, 0), -1)
        cv2.putText(frame, label,
                    (max(mx - 158, 0), max(my - 7, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)

    for idx, person in enumerate(people):
        x1, y1, x2, y2 = person["bbox"]
        ax, ay          = person["anchor"]
        cx, cy          = person["center"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 255), 2)
        cv2.circle(frame, (ax, ay),  8, (0, 255, 255), -1)
        cv2.circle(frame, (ax, ay), 14, (0, 0, 255),   2)
        cv2.circle(frame, (cx, cy),  5, (255, 255, 0), -1)  # yellow = center used for measure

        cv2.putText(frame, f"P{idx+1}",
                    (x1, max(y1 - 8, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame,
                    f"h~{person['height']}px ({person.get('height_method','?')}) kp={person['keypoint_count']}",
                    (x1, min(y2 + 22, h - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    violations = sum(1 for p in pair_details if p["violation"])
    banner       = f"SPIDY ALERT: {violations} DISTANCE BREACH" if violations else "SPIDY VISION: SAFE DISTANCE"
    banner_color = (0, 0, 255) if violations else (0, 255, 0)

    cv2.rectangle(frame, (10, 10), (min(w - 10, 980), 108), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (min(w - 10, 980), 108), (0, 0, 255), 2)
    cv2.putText(frame, banner, (24, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_color, 2)
    cv2.putText(frame,
                f"Mode: {mode} | People: {len(people)} | "
                f"Norm-thr: {NORMALIZED_DISTANCE_THRESHOLD:.2f} | Fallback: {FALLBACK_PIXEL_THRESHOLD}px",
                (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(frame,
                "Rule: BREACH when (center-dist / est-height) < threshold  [yellow dot = center]",
                (24, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 220, 90), 1)

    return frame


def process_image(img, mode):
    poses = net.Process(img, overlay="links,keypoints")
    frame = cuda_img_to_bgr(img)
    h, w  = frame.shape[:2]

    people = []
    for pose in poses:
        info = extract_person_info(pose, w, h)
        if info is not None:
            people.append(info)

    violations, pair_details = analyze_people(people)
    frame = draw_overlay(frame, people, pair_details, mode)
    return frame, people, violations, pair_details


def update_state(mode, people, violations, pair_details, fps=0.0, saved_image=""):
    with state_lock:
        dashboard_state["mode"]       = mode
        dashboard_state["fps"]        = round(float(fps), 1)
        dashboard_state["persons"]    = len(people)
        dashboard_state["objects"]    = 0
        dashboard_state["violations"] = violations
        dashboard_state["last_update"]= now_string()
        dashboard_state["normalized_threshold"]     = NORMALIZED_DISTANCE_THRESHOLD
        dashboard_state["fallback_pixel_threshold"] = FALLBACK_PIXEL_THRESHOLD

        # Confidence proxy: avg keypoints / 17 total COCO keypoints * 100
        if people:
            avg_kp   = sum(p["keypoint_count"] for p in people) / len(people)
            avg_conf = min(int(avg_kp / 17 * 100), 100)
        else:
            avg_conf = 0
        dashboard_state["avg_conf"] = avg_conf

        # Increment frame counter
        dashboard_state["frame"] = dashboard_state.get("frame", 0) + 1

        # Status & anomaly tracking
        dashboard_state["status"] = "THREAT DETECTED" if violations > 0 else "NOMINAL"

        if violations > 0:
            dashboard_state["anomalies"]           = dashboard_state.get("anomalies", 0) + violations
            dashboard_state["last_anomaly_time"]   = time.time()
            dashboard_state["last_anomaly_type"]   = "DISTANCE BREACH"
            dashboard_state["last_anomaly_reason"] = (
                f"{violations} pair(s) within safe distance "
                f"(ratio < {NORMALIZED_DISTANCE_THRESHOLD:.2f})"
            )

        # Time-since-last
        lat = dashboard_state.get("last_anomaly_time", 0)
        if lat:
            secs = int(time.time() - lat)
            dashboard_state["time_since_last"] = (
                f"{secs}s ago" if secs < 60 else f"{secs // 60}m {secs % 60}s ago"
            )
        else:
            dashboard_state["time_since_last"] = "N/A"

        # Pair details for any consumer reading /status
        dashboard_state["last_pairs"] = [
            {
                "pair":                f"P{p['i']+1}-P{p['j']+1}",
                "pixel_distance":      int(p["pixel_distance"]),
                "normalized_distance": round(p["normalized_distance"], 2),
                "avg_height":          int(p["avg_height"]),
                "rule_used":           p["rule_used"],
                "status":              "BREACH" if p["violation"] else "SAFE"
            }
            for p in pair_details
        ]

        if saved_image:
            dashboard_state["last_saved"] = saved_image


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/status")
def status():
    with state_lock:
        return jsonify(dashboard_state)


@app.route("/logs")
def logs():
    ensure_log_header()
    rows = []
    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify(rows[-25:][::-1])


@app.route("/violation_captures/<path:filename>")
def captures(filename):
    return send_from_directory(BASE_DIR / "violation_captures", filename)


@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "Empty file name"}), 400
    if not allowed_file(file.filename):
        return jsonify({"ok": False, "error": "Use jpg, jpeg, png, or bmp"}), 400

    ext        = file.filename.rsplit(".", 1)[1].lower()
    input_name = safe_filename("uploaded_input", ext)
    input_path = UPLOAD_DIR / input_name
    file.save(input_path)

    try:
        img = loadImage(str(input_path))
        frame, people, violations, pair_details = process_image(img, "UPLOADED IMAGE")

        result_name = safe_filename("uploaded_image_result", "jpg")
        result_path = IMAGE_CAPTURE_DIR / result_name
        cv2.imwrite(str(result_path), frame)
        rel = f"uploaded_images/{result_name}"

        write_log("UPLOADED IMAGE", len(people), violations, pair_details, rel)
        update_state("UPLOADED IMAGE", people, violations, pair_details, 0.0, rel)

        return jsonify({
            "ok":         True,
            "people":     len(people),
            "violations": violations,
            "result_url": f"/violation_captures/{rel}",
            "saved_image": rel,
            "pairs":      dashboard_state["last_pairs"]
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def generate_frames():
    global last_live_save_time

    camera      = videoSource(CAMERA_URI)
    frame_count = 0
    start_time  = time.time()

    while True:
        img = camera.Capture()
        if img is None:
            continue

        try:
            frame, people, violations, pair_details = process_image(img, "LIVE STREAM")
        except Exception as e:
            print("Live frame processing error:", e)
            continue

        frame_count += 1
        elapsed      = time.time() - start_time
        fps          = frame_count / elapsed if elapsed > 0 else 0.0

        saved_image = ""
        if violations > 0:
            now_t = time.time()
            if now_t - last_live_save_time >= LIVE_SAVE_COOLDOWN_SECONDS:
                live_name           = safe_filename("live_violation", "jpg")
                live_path           = LIVE_CAPTURE_DIR / live_name
                cv2.imwrite(str(live_path), frame)
                saved_image         = f"live_stream/{live_name}"
                last_live_save_time = now_t

        if violations > 0:
            write_log("LIVE STREAM", len(people), violations, pair_details, saved_image)
        elif frame_count % 60 == 0:
            write_log("LIVE STREAM", len(people), 0, pair_details, "")

        update_state("LIVE STREAM", people, violations, pair_details, fps, saved_image)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        jpg = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")


# /video and /stream both serve the MJPEG feed.
# The W.E.B dashboard <img> points to /stream.
@app.route("/video")
@app.route("/stream")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    ensure_log_header()
    print("=" * 80)
    print("Spider  KEYPOINTS SPIDY VISION - STRICT VERSION (FIXED)")
    print("Open browser: http://<JETSON_IP>:5000")
    print("Find IP:      hostname -I")
    print("=" * 80)
    app.run(host=HOST, port=PORT, threaded=True)
