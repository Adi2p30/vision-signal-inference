import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("libgl1", "libglib2.0-0", "libxcb1")
    .pip_install(
        "ultralytics",
        "rfdetr",
        "opencv-python-headless",
        "fastapi[standard]",
        "lapx",
        "supervision",
    )
)

app = modal.App("cv-inference")


@app.cls(
    image=image,
    gpu="L4",
    timeout=600,
    scaledown_window=300,
    min_containers=5,
    max_containers=7,
)
class YOLOInference:
    @modal.enter()
    def load(self):
        from collections import defaultdict, deque

        import numpy as np
        import supervision as sv
        from ultralytics import YOLO

        self.model = YOLO("yolo26m.pt")

        # Supervision tracker (ByteTrack)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        # Trail history: track_id -> deque of (x, y) bottom-center positions
        self.trails = defaultdict(lambda: deque(maxlen=30))

        # Ball trail history
        self.ball_trail = deque(maxlen=30)

        # Warm up
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("YOLO + ByteTrack loaded and warmed up")

    @modal.asgi_app()
    def serve(self):
        import base64

        import cv2
        import numpy as np
        import supervision as sv
        from fastapi import FastAPI
        from fastapi.requests import Request
        from fastapi.responses import JSONResponse

        web = FastAPI()

        PERSON_CLASS = 0  # COCO class 0 = person
        BALL_CLASS = 32  # COCO class 32 = sports ball

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

            h, w = frame.shape[:2]

            # 1. YOLO detection (person + sports ball)
            results = self.model(
                frame, verbose=False, classes=[PERSON_CLASS, BALL_CLASS]
            )

            # 2. Convert to Supervision Detections
            all_dets = sv.Detections.from_ultralytics(results[0])

            # Split person vs ball detections
            person_mask = all_dets.class_id == PERSON_CLASS
            ball_mask = all_dets.class_id == BALL_CLASS
            person_dets = all_dets[person_mask]
            yolo_ball_dets = all_dets[ball_mask]

            # 3. Track persons with ByteTrack
            person_dets = self.tracker.update_with_detections(person_dets)

            # 4. Build person JSON + update trails
            det_list = []
            for i in range(len(person_dets)):
                tid = (
                    int(person_dets.tracker_id[i])
                    if person_dets.tracker_id is not None
                    else None
                )
                conf = (
                    float(person_dets.confidence[i])
                    if person_dets.confidence is not None
                    else 0
                )
                bbox = person_dets.xyxy[i].tolist()
                x1, y1, x2, y2 = bbox
                cx = round((x1 + x2) / 2, 1)
                cy = round(y2, 1)
                if tid is not None:
                    self.trails[tid].append([cx, cy])
                det_list.append(
                    {
                        "label": "P",
                        "track_id": tid,
                        "bbox": [round(v, 1) for v in bbox],
                        "confidence": round(conf, 2),
                        "center": [cx, cy],
                    }
                )

            # 5. Build trail snapshot for active person tracks
            trail_data = {}
            for det in det_list:
                tid = det["track_id"]
                if tid is not None and tid in self.trails:
                    trail_data[str(tid)] = list(self.trails[tid])

            # 6. Ball detection: YOLO sports-ball
            ball = _detect_ball(yolo_ball_dets)
            if ball:
                self.ball_trail.append(ball["center"])
            ball_trail_list = list(self.ball_trail)

            return {
                "detections": det_list,
                "trails": trail_data,
                "ball": ball,
                "ball_trail": ball_trail_list,
                "frame_size": [w, h],
                "count": len(person_dets),
                "timestamp": timestamp,
            }

        def _detect_ball(yolo_ball_dets):
            """Detect basketball via YOLO sports-ball class."""
            if len(yolo_ball_dets) == 0:
                return None

            i = 0
            if yolo_ball_dets.confidence is not None:
                i = int(np.argmax(yolo_ball_dets.confidence))
            bbox = yolo_ball_dets.xyxy[i].tolist()
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            r = max(x2 - x1, y2 - y1) / 2
            conf = (
                float(yolo_ball_dets.confidence[i])
                if yolo_ball_dets.confidence is not None
                else 0.5
            )
            return {
                "center": [round(cx, 1), round(cy, 1)],
                "radius": round(r, 1),
                "confidence": round(conf, 2),
                "source": "yolo",
            }

        @web.post("/v1/court-detect")
        async def court_detect(request: Request):
            """Auto-detect court corners from a video frame."""
            body = await request.json()
            img_b64 = body["base64"]

            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return JSONResponse(
                    {"error": "Failed to decode image"}, status_code=400
                )

            fh, fw = frame.shape[:2]
            result = _auto_detect_court(frame, fw, fh)

            return {
                "corners": result["corners"],
                "partial": result.get("partial", False),
                "edges_touching": result.get("edges_touching", []),
                "frame_size": [fw, fh],
            }

        def _auto_detect_court(frame, w, h):
            """Detect court area using edge detection.

            Strategy:
            1. Convert to grayscale, apply adaptive threshold to
               isolate the court surface (typically lighter/uniform).
            2. Use Canny + Hough lines to find dominant straight
               edges and derive court corners.
            3. Fall back to largest contour from threshold mask.
            4. Returns 4 ordered points + partial flag + edge info.
            """
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Approach 1: Hough lines on Canny edges ---
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            # Dilate to connect nearby edge fragments
            kernel_d = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel_d, iterations=1)

            corners_from_lines = _line_refined_corners(edges, None, w, h)
            if corners_from_lines is not None:
                corners_arr = np.array(corners_from_lines)
                margin = 6
                edges_touching = []
                if np.any(corners_arr[:, 1] <= margin):
                    edges_touching.append("top")
                if np.any(corners_arr[:, 1] >= h - margin):
                    edges_touching.append("bottom")
                if np.any(corners_arr[:, 0] <= margin):
                    edges_touching.append("left")
                if np.any(corners_arr[:, 0] >= w - margin):
                    edges_touching.append("right")
                corners = [[round(c[0], 1), round(c[1], 1)] for c in corners_from_lines]
                return {
                    "corners": corners,
                    "partial": len(edges_touching) > 0,
                    "edges_touching": edges_touching,
                }

            # --- Approach 2: Adaptive threshold + largest contour ---
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                51,
                -5,
            )
            kernel = np.ones((7, 7), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area >= h * w * 0.10:
                    margin = 6
                    pts_arr = largest.reshape(-1, 2)
                    edges_touching = []
                    if bool(np.any(pts_arr[:, 1] <= margin)):
                        edges_touching.append("top")
                    if bool(np.any(pts_arr[:, 1] >= h - margin)):
                        edges_touching.append("bottom")
                    if bool(np.any(pts_arr[:, 0] <= margin)):
                        edges_touching.append("left")
                    if bool(np.any(pts_arr[:, 0] >= w - margin)):
                        edges_touching.append("right")

                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect).tolist()
                    corners = _order_corners(box)
                    corners = [[round(c[0], 1), round(c[1], 1)] for c in corners]
                    return {
                        "corners": corners,
                        "partial": len(edges_touching) > 0,
                        "edges_touching": edges_touching,
                    }

            # --- Fallback: frame with margin ---
            mx, my = int(w * 0.05), int(h * 0.05)
            return {
                "corners": [
                    [mx, my],
                    [w - mx, my],
                    [w - mx, h - my],
                    [mx, h - my],
                ],
                "partial": True,
                "edges_touching": ["top", "bottom", "left", "right"],
            }

        def _line_refined_corners(edge_img, contour, w, h):
            """Use Hough lines to find court corners."""
            lines = cv2.HoughLinesP(
                edge_img,
                1,
                np.pi / 180,
                threshold=60,
                minLineLength=min(w, h) // 6,
                maxLineGap=20,
            )
            if lines is None or len(lines) < 2:
                return None

            # Separate ~horizontal and ~vertical lines
            horiz, vert = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                angle = np.degrees(np.arctan2(dy, dx))
                if angle < 30:
                    horiz.append((x1, y1, x2, y2))
                elif angle > 60:
                    vert.append((x1, y1, x2, y2))

            if len(horiz) < 2 or len(vert) < 2:
                # Not enough lines — use rotated rect
                return None

            # Cluster horizontal lines into top/bottom by y-midpoint
            horiz.sort(key=lambda ln: (ln[1] + ln[3]) / 2)
            top_y = (horiz[0][1] + horiz[0][3]) / 2
            bot_y = (horiz[-1][1] + horiz[-1][3]) / 2

            # Cluster vertical lines into left/right by x-midpoint
            vert.sort(key=lambda ln: (ln[0] + ln[2]) / 2)
            left_x = (vert[0][0] + vert[0][2]) / 2
            right_x = (vert[-1][0] + vert[-1][2]) / 2

            # Form corners from intersections
            tl = [left_x, top_y]
            tr = [right_x, top_y]
            br = [right_x, bot_y]
            bl = [left_x, bot_y]

            # Validate — corners should span meaningful area
            area = abs(right_x - left_x) * abs(bot_y - top_y)
            if area < w * h * 0.08:
                return None

            return [tl, tr, br, bl]

        def _order_corners(corners):
            """Order as TL, TR, BR, BL."""
            corners = sorted(corners, key=lambda p: p[0] + p[1])
            tl, br = corners[0], corners[3]
            mid = sorted(corners[1:3], key=lambda p: p[0] - p[1])
            bl, tr = mid[0], mid[1]
            return [tl, tr, br, bl]

        @web.get("/health")
        async def health():
            return {"status": "ok"}

        return web


# ─── RF-DETR + SAM2: Detection + Precise Segmentation ──────────────────────


@app.cls(
    image=image,
    gpu="L4",
    timeout=600,
    scaledown_window=300,
    min_containers=2,
    max_containers=5,
)
class RFDETRSam2Inference:
    @modal.enter()
    def load(self):
        from collections import defaultdict, deque

        import numpy as np
        import supervision as sv
        from rfdetr import RFDETRBase
        from ultralytics import SAM

        self.detector = RFDETRBase()
        self.sam = SAM("sam2.1_s.pt")

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        self.trails = defaultdict(lambda: deque(maxlen=60))
        self.ball_trail = deque(maxlen=30)
        self.movement_history = defaultdict(lambda: deque(maxlen=150))

        # Warm up RF-DETR
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.predict(dummy, threshold=0.5)
        print("RF-DETR + SAM2 + ByteTrack loaded and warmed up")

    @modal.asgi_app()
    def serve(self):
        import base64

        import cv2
        import numpy as np
        import supervision as sv
        from fastapi import FastAPI
        from fastapi.requests import Request
        from fastapi.responses import JSONResponse

        web = FastAPI()

        PERSON_CLASS = 0
        BALL_CLASS = 32

        @web.post("/v1/segment")
        async def segment(request: Request):
            body = await request.json()
            img_b64 = body["base64"]
            timestamp = body.get("timestamp", 0)

            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return JSONResponse(
                    {"error": "Failed to decode image"}, status_code=400
                )

            h, w = frame.shape[:2]

            # 1. RF-DETR detection
            all_dets = self.detector.predict(frame, threshold=0.4)

            person_mask_idx = all_dets.class_id == PERSON_CLASS
            ball_mask_idx = all_dets.class_id == BALL_CLASS
            person_dets = all_dets[person_mask_idx]
            ball_dets = all_dets[ball_mask_idx]

            # 2. ByteTrack
            person_dets = self.tracker.update_with_detections(person_dets)

            # 3. SAM2 segmentation masks
            mask_contours = {}
            if len(person_dets) > 0:
                bboxes = person_dets.xyxy.tolist()
                track_ids = (
                    person_dets.tracker_id.tolist()
                    if person_dets.tracker_id is not None
                    else list(range(len(bboxes)))
                )

                try:
                    sam_results = self.sam(frame, bboxes=bboxes)
                    if sam_results and sam_results[0].masks is not None:
                        masks_np = sam_results[0].masks.data.cpu().numpy()
                        for idx, tid in enumerate(track_ids):
                            if idx >= len(masks_np):
                                break
                            mask_u8 = (masks_np[idx] > 0.5).astype(np.uint8) * 255
                            contours_cv, _ = cv2.findContours(
                                mask_u8,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE,
                            )
                            polys = []
                            for c in contours_cv:
                                if len(c) >= 4:
                                    eps = 0.003 * cv2.arcLength(c, True)
                                    approx = cv2.approxPolyDP(c, eps, True)
                                    pts = [
                                        [round(p[0], 1), round(p[1], 1)]
                                        for p in approx.reshape(-1, 2).tolist()
                                    ]
                                    polys.append(pts)
                            if polys:
                                mask_contours[str(int(tid))] = polys
                except Exception as e:
                    print(f"SAM2 segmentation error: {e}")

            # 4. Detections + trails + movement
            det_list = []
            for i in range(len(person_dets)):
                tid = (
                    int(person_dets.tracker_id[i])
                    if person_dets.tracker_id is not None
                    else None
                )
                conf = (
                    float(person_dets.confidence[i])
                    if person_dets.confidence is not None
                    else 0
                )
                bbox = person_dets.xyxy[i].tolist()
                x1, y1, x2, y2 = bbox
                cx = round((x1 + x2) / 2, 1)
                cy = round(y2, 1)
                if tid is not None:
                    self.trails[tid].append([cx, cy])
                    self.movement_history[tid].append((cx, cy, timestamp))
                det_list.append(
                    {
                        "label": "P",
                        "track_id": tid,
                        "bbox": [round(v, 1) for v in bbox],
                        "confidence": round(conf, 2),
                        "center": [cx, cy],
                    }
                )

            trail_data = {}
            for det in det_list:
                tid = det["track_id"]
                if tid is not None and tid in self.trails:
                    trail_data[str(tid)] = list(self.trails[tid])

            # 5. Movement analytics
            movement = {}
            for det in det_list:
                tid = det["track_id"]
                if tid is None or tid not in self.movement_history:
                    continue
                hist = list(self.movement_history[tid])
                if len(hist) < 2:
                    continue
                window = hist[-min(15, len(hist)) :]
                dist_w = 0.0
                for j in range(1, len(window)):
                    ddx = window[j][0] - window[j - 1][0]
                    ddy = window[j][1] - window[j - 1][1]
                    dist_w += (ddx**2 + ddy**2) ** 0.5
                t_span = window[-1][2] - window[0][2]
                speed = dist_w / max(t_span, 0.033)
                dx = hist[-1][0] - hist[-2][0]
                dy = hist[-1][1] - hist[-2][1]
                mag = max((dx**2 + dy**2) ** 0.5, 0.01)
                total_d = 0.0
                for j in range(1, len(hist)):
                    ddx = hist[j][0] - hist[j - 1][0]
                    ddy = hist[j][1] - hist[j - 1][1]
                    total_d += (ddx**2 + ddy**2) ** 0.5
                movement[str(tid)] = {
                    "speed": round(speed, 1),
                    "direction": [round(dx / mag, 2), round(dy / mag, 2)],
                    "distance": round(total_d, 1),
                    "samples": len(hist),
                }

            # 6. Ball
            ball = _detect_ball(ball_dets)
            if ball:
                self.ball_trail.append(ball["center"])

            return {
                "detections": det_list,
                "trails": trail_data,
                "mask_contours": mask_contours,
                "movement": movement,
                "ball": ball,
                "ball_trail": list(self.ball_trail),
                "frame_size": [w, h],
                "count": len(person_dets),
                "timestamp": timestamp,
                "model": "rfdetr+sam2",
            }

        def _detect_ball(ball_dets):
            if len(ball_dets) == 0:
                return None
            i = 0
            if ball_dets.confidence is not None:
                i = int(np.argmax(ball_dets.confidence))
            bbox = ball_dets.xyxy[i].tolist()
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            r = max(x2 - x1, y2 - y1) / 2
            conf = (
                float(ball_dets.confidence[i])
                if ball_dets.confidence is not None
                else 0.5
            )
            return {
                "center": [round(cx, 1), round(cy, 1)],
                "radius": round(r, 1),
                "confidence": round(conf, 2),
                "source": "rfdetr",
            }

        @web.post("/v1/detect")
        async def detect(request: Request):
            """RF-DETR detect only (no masks), backward compatible."""
            body = await request.json()
            img_b64 = body["base64"]
            timestamp = body.get("timestamp", 0)

            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return JSONResponse(
                    {"error": "Failed to decode image"}, status_code=400
                )

            h, w = frame.shape[:2]
            all_dets = self.detector.predict(frame, threshold=0.4)

            person_mask_idx = all_dets.class_id == PERSON_CLASS
            ball_mask_idx = all_dets.class_id == BALL_CLASS
            person_dets = all_dets[person_mask_idx]
            ball_dets = all_dets[ball_mask_idx]

            person_dets = self.tracker.update_with_detections(person_dets)

            det_list = []
            for i in range(len(person_dets)):
                tid = (
                    int(person_dets.tracker_id[i])
                    if person_dets.tracker_id is not None
                    else None
                )
                conf = (
                    float(person_dets.confidence[i])
                    if person_dets.confidence is not None
                    else 0
                )
                bbox = person_dets.xyxy[i].tolist()
                x1, y1, x2, y2 = bbox
                cx = round((x1 + x2) / 2, 1)
                cy = round(y2, 1)
                if tid is not None:
                    self.trails[tid].append([cx, cy])
                det_list.append(
                    {
                        "label": "P",
                        "track_id": tid,
                        "bbox": [round(v, 1) for v in bbox],
                        "confidence": round(conf, 2),
                        "center": [cx, cy],
                    }
                )

            trail_data = {}
            for det in det_list:
                tid = det["track_id"]
                if tid is not None and tid in self.trails:
                    trail_data[str(tid)] = list(self.trails[tid])

            ball = _detect_ball(ball_dets)
            if ball:
                self.ball_trail.append(ball["center"])

            return {
                "detections": det_list,
                "trails": trail_data,
                "ball": ball,
                "ball_trail": list(self.ball_trail),
                "frame_size": [w, h],
                "count": len(person_dets),
                "timestamp": timestamp,
                "model": "rfdetr",
            }

        @web.post("/v1/court-detect")
        async def court_detect(request: Request):
            """Auto-detect court corners from a video frame."""
            body = await request.json()
            img_b64 = body["base64"]

            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return JSONResponse(
                    {"error": "Failed to decode image"}, status_code=400
                )

            fh, fw = frame.shape[:2]
            result = _auto_detect_court(frame, fw, fh)

            return {
                "corners": result["corners"],
                "partial": result.get("partial", False),
                "edges_touching": result.get("edges_touching", []),
                "frame_size": [fw, fh],
            }

        def _auto_detect_court(frame, w, h):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            kernel_d = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel_d, iterations=1)
            corners_from_lines = _line_refined_corners(edges, w, h)
            if corners_from_lines is not None:
                corners_arr = np.array(corners_from_lines)
                margin = 6
                edges_touching = []
                if np.any(corners_arr[:, 1] <= margin):
                    edges_touching.append("top")
                if np.any(corners_arr[:, 1] >= h - margin):
                    edges_touching.append("bottom")
                if np.any(corners_arr[:, 0] <= margin):
                    edges_touching.append("left")
                if np.any(corners_arr[:, 0] >= w - margin):
                    edges_touching.append("right")
                corners = [[round(c[0], 1), round(c[1], 1)] for c in corners_from_lines]
                return {
                    "corners": corners,
                    "partial": len(edges_touching) > 0,
                    "edges_touching": edges_touching,
                }
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                51,
                -5,
            )
            kernel = np.ones((7, 7), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            contours_list, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours_list:
                largest = max(contours_list, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area >= h * w * 0.10:
                    margin = 6
                    pts_arr = largest.reshape(-1, 2)
                    edges_touching = []
                    if bool(np.any(pts_arr[:, 1] <= margin)):
                        edges_touching.append("top")
                    if bool(np.any(pts_arr[:, 1] >= h - margin)):
                        edges_touching.append("bottom")
                    if bool(np.any(pts_arr[:, 0] <= margin)):
                        edges_touching.append("left")
                    if bool(np.any(pts_arr[:, 0] >= w - margin)):
                        edges_touching.append("right")
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect).tolist()
                    corners = _order_corners(box)
                    corners = [[round(c[0], 1), round(c[1], 1)] for c in corners]
                    return {
                        "corners": corners,
                        "partial": len(edges_touching) > 0,
                        "edges_touching": edges_touching,
                    }
            mx, my = int(w * 0.05), int(h * 0.05)
            return {
                "corners": [[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]],
                "partial": True,
                "edges_touching": ["top", "bottom", "left", "right"],
            }

        def _line_refined_corners(edge_img, w, h):
            lines = cv2.HoughLinesP(
                edge_img,
                1,
                np.pi / 180,
                threshold=60,
                minLineLength=min(w, h) // 6,
                maxLineGap=20,
            )
            if lines is None or len(lines) < 2:
                return None
            horiz, vert = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                ddx, ddy = abs(x2 - x1), abs(y2 - y1)
                angle = np.degrees(np.arctan2(ddy, ddx))
                if angle < 30:
                    horiz.append((x1, y1, x2, y2))
                elif angle > 60:
                    vert.append((x1, y1, x2, y2))
            if len(horiz) < 2 or len(vert) < 2:
                return None
            horiz.sort(key=lambda ln: (ln[1] + ln[3]) / 2)
            top_y = (horiz[0][1] + horiz[0][3]) / 2
            bot_y = (horiz[-1][1] + horiz[-1][3]) / 2
            vert.sort(key=lambda ln: (ln[0] + ln[2]) / 2)
            left_x = (vert[0][0] + vert[0][2]) / 2
            right_x = (vert[-1][0] + vert[-1][2]) / 2
            area = abs(right_x - left_x) * abs(bot_y - top_y)
            if area < w * h * 0.08:
                return None
            return [
                [left_x, top_y],
                [right_x, top_y],
                [right_x, bot_y],
                [left_x, bot_y],
            ]

        def _order_corners(corners):
            corners = sorted(corners, key=lambda p: p[0] + p[1])
            tl, br = corners[0], corners[3]
            mid = sorted(corners[1:3], key=lambda p: p[0] - p[1])
            bl, tr = mid[0], mid[1]
            return [tl, tr, br, bl]

        @web.get("/health")
        async def health():
            return {"status": "ok", "model": "rfdetr+sam2"}

        return web


@app.local_entrypoint()
def main():
    print("Use `modal serve cv_inference/modal_app.py` for dev")
    print("or `modal deploy cv_inference/modal_app.py` for prod.")
