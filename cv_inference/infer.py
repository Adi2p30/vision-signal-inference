from ultralytics import YOLO
import cv2
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stream-26feb09arizku.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO26s model
model = YOLO("yolo26s.pt")

# Run inference on the video
results = model(VIDEO_PATH, stream=True, save=True, project=OUTPUT_DIR, name="stream_results")

# Process results frame by frame
for r in results:
    print(f"Frame: {r.path} | Detections: {len(r.boxes)}")

print(f"\nDone! Output saved to {OUTPUT_DIR}/stream_results/")
