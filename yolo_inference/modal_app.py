import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libxcb1")
    .pip_install("ultralytics", "opencv-python-headless", "fastapi[standard]", "lapx")
)

app = modal.App("yolo26s-inference")


@app.cls(
    image=image,
    gpu="L4",
    timeout=600,
    scaledown_window=300,
    min_containers=8,
    max_containers=10,
)
class YOLOInference:
    @modal.enter()
    def load(self):
        from ultralytics import YOLO

        self.model = YOLO("yolo26m.pt")
        # Warm up
        import numpy as np

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.track(dummy, persist=True, verbose=False)
        print("YOLO model loaded and warmed up")

    @modal.asgi_app()
    def serve(self):
        import base64
        import io
        from collections import Counter

        import cv2
        import numpy as np
        from fastapi import FastAPI
        from fastapi.requests import Request
        from fastapi.responses import JSONResponse

        web = FastAPI()

        @web.post("/v1/detect")
        async def detect(request: Request):
            body = await request.json()
            img_b64 = body["base64"]
            timestamp = body.get("timestamp", 0)

            # Decode image
            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return JSONResponse(
                    {"error": "Failed to decode image"}, status_code=400
                )

            # Run YOLO with ByteTrack tracker
            results = self.model.track(frame, persist=True, verbose=False)
            result = results[0]

            # Draw annotations using ultralytics built-in plot()
            annotated = result.plot()

            # Encode annotated frame to base64 JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = base64.b64encode(buf.tobytes()).decode()

            # Extract detections
            detections = []
            summary = Counter()
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    cls_name = result.names[cls_id]
                    conf = float(boxes.conf[i])
                    bbox = boxes.xyxy[i].tolist()
                    track_id = int(boxes.id[i]) if boxes.id is not None else None
                    detections.append(
                        {
                            "class": cls_name,
                            "confidence": round(conf, 3),
                            "track_id": track_id,
                            "bbox": [round(v, 1) for v in bbox],
                        }
                    )
                    summary[cls_name] += 1

            return {
                "annotated_frame": annotated_b64,
                "detections": detections,
                "summary": dict(summary),
                "timestamp": timestamp,
            }

        @web.get("/health")
        async def health():
            return {"status": "ok"}

        return web


@app.local_entrypoint()
def main():
    print("Use `modal serve yolo_inference/modal_app.py` for dev")
    print("or `modal deploy yolo_inference/modal_app.py` for prod.")
