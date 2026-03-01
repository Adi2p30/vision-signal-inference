"""
Render RF-DETR + SAM2 detections onto a video file.

Usage:
    python -m cv_inference.render_video \
        --id 26feb09arizku \
        --endpoint https://adityapdev13--cv-inference-rfdetrsam2inference-serve.modal.run/v1/segment \
        --fps 5 \
        --start 0 --end 60
"""

import argparse
import base64
import math
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import httpx
import numpy as np

# ── Color palette (same as frontend) ──
TRACK_COLORS = [
    (230, 25, 75),   (60, 180, 75),   (255, 225, 25),  (67, 99, 216),
    (245, 130, 49),  (145, 30, 180),  (66, 212, 244),  (240, 50, 230),
    (191, 239, 69),  (250, 190, 212), (70, 153, 144),  (220, 190, 255),
    (154, 99, 36),   (255, 250, 200), (128, 0, 0),     (170, 255, 195),
    (128, 128, 0),   (255, 216, 177), (0, 0, 117),     (169, 169, 169),
]

BALL_COLOR = (0, 102, 255)  # BGR for orange


def track_color(tid):
    c = TRACK_COLORS[tid % len(TRACK_COLORS)]
    return (c[2], c[1], c[0])  # RGB -> BGR


def hex_alpha(color_bgr, alpha):
    """Return (B, G, R) for blending."""
    return color_bgr, alpha


def draw_frame(frame, data):
    """Draw masks, boxes, trails, ball, movement onto a frame (in-place)."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    dets = data.get("detections", [])
    mask_contours = data.get("mask_contours", {})
    trails = data.get("trails", {})
    movement = data.get("movement", {})
    ball = data.get("ball")
    ball_trail = data.get("ball_trail", [])

    # 1. SAM2 mask fills (smoothed contours)
    for tid_str, polys in mask_contours.items():
        tid = int(tid_str)
        color = track_color(tid)
        for poly in polys:
            if len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32)
            # Smooth the contour to reduce jaggedness
            epsilon = 0.008 * cv2.arcLength(pts, True)
            pts = cv2.approxPolyDP(pts, epsilon, True)
            if len(pts) < 3:
                continue
            cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

    # 2. Mask contour outlines (smoothed)
    for tid_str, polys in mask_contours.items():
        tid = int(tid_str)
        color = track_color(tid)
        for poly in polys:
            if len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32)
            epsilon = 0.008 * cv2.arcLength(pts, True)
            pts = cv2.approxPolyDP(pts, epsilon, True)
            if len(pts) < 3:
                continue
            cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)

    # 3. Trails (smoothed with fading)
    for tid_str, points in trails.items():
        if len(points) < 2:
            continue
        tid = int(tid_str)
        color = track_color(tid)
        # Build polyline from trail points
        pts_arr = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)
        cv2.polylines(frame, [pts_arr], False, color, 1, cv2.LINE_AA)

    # 4. Bounding boxes + labels
    for d in dets:
        tid = d.get("track_id")
        color = track_color(tid) if tid is not None else (255, 255, 255)
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"P#{tid}" if tid is not None else "P"
        conf = d.get("confidence", 0)
        label_full = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label_full, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # 5. Movement arrows + speed
    for d in dets:
        tid = d.get("track_id")
        if tid is None or str(tid) not in movement:
            continue
        m = movement[str(tid)]
        speed = m["speed"]
        direction = m["direction"]
        x1, y1, x2, y2 = d["bbox"]
        # Speed label
        sx = int(x2) + 4
        sy = int(y1 + (y2 - y1) * 0.3)
        speed_text = f"{speed:.0f} px/s"
        (tw, th), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(frame, (sx - 1, sy - th - 2), (sx + tw + 2, sy + 2), (0, 0, 0), -1)
        cv2.putText(frame, speed_text, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        # Direction arrow
        if speed > 5:
            cx = int((x1 + x2) / 2)
            cy = int(y2) + 8
            arrow_len = min(speed * 0.3, 25)
            dx = direction[0] * arrow_len
            dy = direction[1] * arrow_len
            cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)),
                            (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.35)

    return frame


def send_frame(client, endpoint, frame, timestamp, timeout=30.0):
    """Encode and send a frame, return the JSON response."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    try:
        resp = client.post(endpoint, json={"base64": b64, "timestamp": timestamp}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] Frame {timestamp:.1f}s failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Render RF-DETR+SAM2 detections onto video")
    parser.add_argument("--id", required=True, help="Game ID")
    parser.add_argument("--endpoint", required=True,
                        help="Segment endpoint URL")
    parser.add_argument("--fps", type=float, default=10,
                        help="Frames per second to process (default: 10)")
    parser.add_argument("--start", type=float, default=0,
                        help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0,
                        help="End time in seconds (0 = full video)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel request workers (default: 6)")
    parser.add_argument("--output", default="",
                        help="Output MP4 path (default: data/rendered-{id}.mp4)")
    args = parser.parse_args()
    args.endpoint = args.endpoint.replace("\u00a0", " ").strip()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    video_path = data_dir / f"stream-{args.id}.mp4"
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        sys.exit(1)

    output_path = args.output or str(data_dir / f"rendered-{args.id}.mp4")

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_s = args.start
    end_s = args.end if args.end > 0 else duration
    process_fps = args.fps
    frame_interval = 1.0 / process_fps

    # Collect timestamps to process
    timestamps = []
    t = start_s
    while t < end_s:
        timestamps.append(t)
        t += frame_interval

    print(f"Video: {video_path.name} ({width}x{height}, {src_fps:.1f}fps, {duration:.1f}s)")
    print(f"Processing: {start_s:.1f}s - {end_s:.1f}s @ {process_fps}fps = {len(timestamps)} frames")
    print(f"Endpoint: {args.endpoint}")
    print(f"Output: {output_path}")
    print(f"Workers: {args.workers}")
    print()

    # Phase 1: Read all frames
    print("Phase 1: Reading frames...")
    frames = {}
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if ret:
            frames[ts] = frame
    cap.release()
    print(f"  Read {len(frames)} frames")

    # Phase 2: Send to endpoint in parallel
    print("Phase 2: Running inference...")
    results = {}
    client = httpx.Client(timeout=httpx.Timeout(60.0))
    completed = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for ts in timestamps:
            if ts not in frames:
                continue
            f = pool.submit(send_frame, client, args.endpoint, frames[ts], ts)
            futures[f] = ts

        for future in as_completed(futures):
            ts = futures[future]
            data = future.result()
            if data:
                results[ts] = data
                completed += 1
            else:
                failed += 1
            total = completed + failed
            if total % 10 == 0 or total == len(futures):
                elapsed = time.time() - t0
                fps_rate = total / max(elapsed, 0.1)
                remaining = (len(futures) - total) / max(fps_rate, 0.01)
                print(f"  {total}/{len(futures)} "
                      f"({completed} ok, {failed} fail) "
                      f"[{fps_rate:.1f} req/s, ~{remaining:.0f}s left]")

    client.close()
    print(f"  Inference done: {completed} frames in {time.time() - t0:.1f}s")

    # Phase 3: Render and write video
    print("Phase 3: Rendering video...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, process_fps, (width, height))

    rendered = 0
    for ts in sorted(frames.keys()):
        frame = frames[ts].copy()
        if ts in results:
            frame = draw_frame(frame, results[ts])

        # HUD overlay
        count = results[ts]["count"] if ts in results else 0
        model = results[ts].get("model", "?") if ts in results else "?"
        masks = len(results[ts].get("mask_contours", {})) if ts in results else 0
        hud = f"{ts:.1f}s | {model} | {count}P | {masks}M"
        cv2.rectangle(frame, (8, 8), (len(hud) * 10 + 16, 32), (0, 0, 0), -1)
        cv2.putText(frame, hud, (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        rendered += 1

    writer.release()
    print(f"\nDone! Rendered {rendered} frames to {output_path}")
    print(f"  Duration: {rendered / process_fps:.1f}s @ {process_fps}fps")


if __name__ == "__main__":
    main()
