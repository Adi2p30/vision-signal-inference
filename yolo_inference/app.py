import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Inference Web UI")
    parser.add_argument(
        "--id", required=True, help="Game ID (loads data/stream-{id}.mp4)"
    )
    parser.add_argument(
        "--endpoint", required=True, help="Modal YOLO endpoint URL (e.g. https://...modal.run/v1/detect)"
    )
    parser.add_argument(
        "--port", type=int, default=8081, help="Local web UI port (default: 8081)"
    )
    args = parser.parse_args()
    args.endpoint = args.endpoint.replace("\u00a0", " ").strip()
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"
    video_path = data_dir / f"stream-{args.id}.mp4"
    if not video_path.exists():
        raise SystemExit(f"Error: video file '{video_path}' not found")

    app.state.game_id = args.id
    app.state.data_dir = data_dir
    app.state.endpoint = args.endpoint
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    print(f"Game ID: {args.id}")
    print(f"Endpoint: {args.endpoint}")

    yield

    await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/video")
async def serve_video(request: Request):
    video_path = request.app.state.data_dir / f"stream-{request.app.state.game_id}.mp4"
    return FileResponse(video_path, media_type="video/mp4")


@app.post("/detect")
async def detect(request: Request):
    body = await request.json()
    try:
        resp = await request.app.state.http.post(
            request.app.state.endpoint,
            json=body,
        )
        resp.raise_for_status()
        return JSONResponse(resp.json())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


def main():
    args = parse_args()
    print(f"Starting YOLO web UI at http://localhost:{args.port}")
    uvicorn.run("yolo_inference.app:app", host="0.0.0.0", port=args.port, reload=True)


if __name__ == "__main__":
    main()
