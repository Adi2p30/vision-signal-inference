import argparse
import asyncio
import csv
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import uvicorn
from openai import AsyncOpenAI


def strip_think(text):
    """Strip Qwen3 <think>...</think> reasoning tokens from response."""
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Video Frame Inference Web UI")
    parser.add_argument("--id", required=True, help="Game ID (loads data/stream-{id}.mp4, data/pbp-{id}.json, data/kalshi-price-history-{id}.csv)")
    parser.add_argument("--endpoint", required=True, help="Modal vLLM endpoint URL for momentum VLM")
    parser.add_argument("--score-endpoint", default="", help="Modal score VLM endpoint URL (enables dual-VLM mode)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="Model name served by vLLM")
    parser.add_argument("--port", type=int, default=8080, help="Local web UI port (default: 8080)")
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = parse_args()

    data_dir = Path(__file__).parent / "data"
    video_path = data_dir / f"stream-{args.id}.mp4"
    if not video_path.exists():
        raise SystemExit(f"Error: video file '{video_path}' not found")

    app.state.game_id = args.id
    app.state.model_name = args.model
    app.state.client = AsyncOpenAI(base_url=args.endpoint, api_key="not-needed")
    app.state.score_endpoint = args.score_endpoint
    app.state.score_http = httpx.AsyncClient(timeout=httpx.Timeout(60.0)) if args.score_endpoint else None

    if args.score_endpoint:
        print(f"Dual VLM mode enabled")
        print(f"  Momentum endpoint: {args.endpoint}")
        print(f"  Score endpoint:    {args.score_endpoint}")
    else:
        print(f"Legacy single VLM mode")
        print(f"  Endpoint: {args.endpoint}")

    print(f"Game ID: {args.id}")

    yield

    if app.state.score_http:
        await app.state.score_http.aclose()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

MOMENTUM_PROMPT = (
    "/no_think\n"
    "ONLY output one CSV row. No headers, no explanation, no markdown, no extra text.\n"
    "Columns: timestamp,action,momentum_team,momentum_score,momentum_reason\n"
    "- action: <=10 tokens, describe the current play action\n"
    "- momentum_team: team_A/team_B/neutral\n"
    "- momentum_score: -5 to +5\n"
    "- momentum_reason: 1-5 tokens\n"
    "Example: 12.0,fast break layup scored,team_A,3,quick transition"
)

LEGACY_PROMPT = (
    "/no_think\n"
    "ONLY output one CSV row. No headers, no explanation, no markdown, no extra text.\n"
    "If you cannot see the scoreboard or game clock clearly, output exactly: NONE\n"
    "Columns: timestamp,action,team_a_score,team_b_score,game_clock,momentum_team,momentum_score,momentum_reason\n"
    "- action: <=10 tokens\n"
    "- game_clock: countdown timer shown on screen, format MM:SS (e.g. 14:32, 03:07)\n"
    "- momentum_team: team_A/team_B/neutral\n"
    "- momentum_score: -5 to +5\n"
    "- momentum_reason: 1-5 tokens\n"
    "Example: 12.0,fast break layup scored,45,42,14:32,team_A,3,quick transition"
)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/video")
async def serve_video(request: Request):
    return FileResponse(Path(__file__).parent / "data" / f"stream-{request.app.state.game_id}.mp4", media_type="video/mp4")


@app.get("/play-by-play")
async def play_by_play(request: Request):
    """Return scoring-relevant plays with (awayScore, homeScore, wallclock)."""
    import json
    summary_path = Path(__file__).parent / "data" / f"pbp-{request.app.state.game_id}.json"
    with open(summary_path) as f:
        data = json.load(f)
    plays = []
    for p in data["plays"]:
        plays.append({
            "awayScore": p["awayScore"],
            "homeScore": p["homeScore"],
            "clock": p["clock"]["displayValue"],
            "period": p["period"]["number"],
            "wallclock": p["wallclock"],
        })
    return JSONResponse(plays)


@app.get("/price-history")
async def price_history(request: Request):
    csv_path = Path(__file__).parent / "data" / f"kalshi-price-history-{request.app.state.game_id}.csv"
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {"timestamp": row["timestamp"]}
            for col in row:
                if col == "timestamp":
                    continue
                val = row[col]
                entry[col.lower()] = float(val) if val else None
            rows.append(entry)
    return JSONResponse(rows)


@app.get("/config")
async def config(request: Request):
    """Tell the frontend which mode we're running in."""
    return {"dual_vlm": bool(request.app.state.score_endpoint)}


@app.get("/candlesticks")
async def candlesticks(request: Request):
    import json
    cs_path = Path(__file__).parent / "data" / f"candlesticks-{request.app.state.game_id}.json"
    with open(cs_path) as f:
        data = json.load(f)
    return JSONResponse(data["candlesticks"])


@app.post("/infer")
async def infer(request: Request):
    body = await request.json()
    timestamp = body.get("timestamp", 0)
    base64_img = body.get("base64", "")
    prompt = body.get("prompt", "")

    state = request.app.state

    if state.score_endpoint:
        # Dual VLM: momentum + score in parallel
        momentum_prompt = prompt or MOMENTUM_PROMPT

        async def get_momentum():
            resp = await state.client.chat.completions.create(
                model=state.model_name,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": f"[{timestamp:.1f}s]"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    {"type": "text", "text": momentum_prompt},
                ]}],
                max_tokens=60,
            )
            raw = resp.choices[0].message.content or ""
            return strip_think(raw)

        async def get_score():
            resp = await state.score_http.post(
                state.score_endpoint,
                json={"base64": base64_img, "timestamp": timestamp},
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "NONE") or "NONE"
            return strip_think(raw)

        try:
            momentum, score_val = await asyncio.gather(get_momentum(), get_score())
            return {"momentum": momentum or "NONE", "score": score_val or "NONE"}
        except Exception as e:
            return {"error": str(e)}
    else:
        # Legacy single VLM mode
        prompt = prompt or LEGACY_PROMPT
        content = [
            {"type": "text", "text": f"[{timestamp:.1f}s]"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
            {"type": "text", "text": prompt},
        ]
        try:
            response = await state.client.chat.completions.create(
                model=state.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=60,
            )
            raw = response.choices[0].message.content or ""
            return {"response": strip_think(raw)}
        except Exception as e:
            return {"error": str(e)}


def main():
    args = parse_args()
    print(f"Starting web UI at http://localhost:{args.port}")
    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=True)


if __name__ == "__main__":
    main()
