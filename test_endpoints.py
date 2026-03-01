import base64
import json

import cv2
import httpx

cap = cv2.VideoCapture("data/stream-26feb09arizku.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
ret, frame = cap.read()
cap.release()
print(f"Frame: {frame.shape}")

_, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
b64 = base64.b64encode(buf).decode()

BASE = "https://adityapdev13--cv-inference-rfdetrsam2inference-serve.modal.run"

# Debug: see raw class IDs
print("\n--- DEBUG (raw RF-DETR detections) ---")
r = httpx.post(f"{BASE}/v1/debug", json={"base64": b64}, timeout=60.0)
d = r.json()
print(f"Total dets: {d['total']}")
print(f"Unique classes: {d['unique_classes']}")
for det in d.get("detections", [])[:10]:
    print(f"  class={det['class_id']}  conf={det['confidence']}  bbox={det['bbox']}")

# Segment
print("\n--- SEGMENT ---")
r2 = httpx.post(
    f"{BASE}/v1/segment", json={"base64": b64, "timestamp": 30.0}, timeout=60.0
)
s = r2.json()
print(f"Count: {s['count']}, Masks: {len(s.get('mask_contours', {}))}")
for det in s.get("detections", [])[:5]:
    print(f"  {det}")
