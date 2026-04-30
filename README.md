# 🕷 Keypoints Spidy Vision - Strict Version

This is the corrected strict version.

## What changed

The old version used a fixed pixel threshold. That was not reliable.

This version uses:

```text
normalized_distance = pixel_distance / average_detected_person_height
```

A pair is unsafe when:

```text
normalized_distance < NORMALIZED_DISTANCE_THRESHOLD
```

Default:

```python
NORMALIZED_DISTANCE_THRESHOLD = 1.05
FALLBACK_PIXEL_THRESHOLD = 500
```

## Why this is better

A 500px distance means different things depending on:
- image size
- camera distance
- whether people are close to the camera
- whether people are far away

Using body height gives a stricter and more visual estimate.

## Run

Copy the whole folder to:

```bash
~/Desktop/spring_26/jetson_apps/jetson-inference/python/examples/keypoints_spidy_vision
```

Inside container:

```bash
cd /jetson-inference/python/examples/keypoints_spidy_vision
pip3 install flask
python3 keypoints_spidy_vision.py
```

Open:

```text
http://JETSON_IP:5000
```

## Tune strictness

Edit:

```python
NORMALIZED_DISTANCE_THRESHOLD = 1.05
```

Try:

```text
0.85 = moderate
1.05 = strict
1.25 = very strict
1.50 = extremely strict
```

## Saved outputs

Live violation images:

```text
violation_captures/live_stream/
```

Uploaded image results:

```text
violation_captures/uploaded_images/
```

Logs:

```text
logs/spidy_detection_log.csv
```
