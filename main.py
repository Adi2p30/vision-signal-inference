import argparse
import csv
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from openai import AsyncOpenAI

app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Globals set at startup
client: AsyncOpenAI = None
model_name: str = ""
video_path: str = ""

DEFAULT_PROMPT = (
    "ONLY output one CSV row. No headers, no explanation, no markdown, no extra text.\n"
    "Columns: timestamp,action,team_a_score,team_b_score,momentum_team,momentum_score,momentum_reason\n"
    "- action: <=10 tokens\n"
    "- momentum_team: team_A/team_B/neutral\n"
    "- momentum_score: -5 to +5\n"
    "- momentum_reason: 1-5 tokens\n"
    "Example: 12.0,fast break layup scored,45,42,team_A,3,quick transition"
)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/video")
async def serve_video():
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/play-by-play")
async def play_by_play():
    """Return scoring-relevant plays with (awayScore, homeScore, wallclock)."""
    summary_path = Path(__file__).parent / "data" / "summary.json"
    import json
    with open(summary_path) as f:
        data = json.load(f)
    # Build a list of every play's score state + wallclock
    plays = []
    for p in data["plays"]:
        plays.append({
            "awayScore": p["awayScore"],
            "homeScore": p["homeScore"],
            "wallclock": p["wallclock"],
        })
    return JSONResponse(plays)


@app.get("/kalshi-data")
async def kalshi_data():
    csv_path = Path(__file__).parent / "data" / "kalshi-price-history-kxncaambgame-26feb09arizku-minute.csv"
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kansas = row.get("Kansas", "")
            arizona = row.get("Arizona", "")
            rows.append({
                "timestamp": row["timestamp"],
                "kansas": float(kansas) if kansas else None,
                "arizona": float(arizona) if arizona else None,
            })
    return JSONResponse(rows)


@app.post("/infer")
async def infer(request: Request):
    body = await request.json()
    timestamp = body.get("timestamp", 0)
    base64_img = body.get("base64", "")
    prompt = body.get("prompt", "") or DEFAULT_PROMPT

    content = [
        {"type": "text", "text": f"[{timestamp:.1f}s]"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
        {"type": "text", "text": prompt},
    ]

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=60,
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}


def main():
    global client, model_name, video_path

    parser = argparse.ArgumentParser(description="Video Frame Inference Web UI")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--endpoint", required=True, help="Modal vLLM endpoint URL (e.g. https://your-app--serve.modal.run/v1)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="Model name served by vLLM")
    parser.add_argument("--port", type=int, default=8080, help="Local web UI port (default: 8080)")
    args = parser.parse_args()

    video_path = str(Path(args.video).resolve())
    if not Path(video_path).exists():
        raise SystemExit(f"Error: video file '{video_path}' not found")

    model_name = args.model
    client = AsyncOpenAI(base_url=args.endpoint, api_key="not-needed")

    print(f"Starting web UI at http://localhost:{args.port}")
    print(f"Video: {video_path}")
    print(f"Endpoint: {args.endpoint}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
