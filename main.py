import argparse
import csv
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI()

# Globals set at startup
vllm_http: httpx.AsyncClient = None
vllm_endpoint: str = ""
model_name: str = ""
video_path: str = ""
summary_data: dict = {}
kalshi_data: list = []
html_content: str = ""
scoreboard_http: httpx.AsyncClient = None
scoreboard_url: str = ""

DEFAULT_PROMPT = (
    "ONLY output one CSV row. No headers, no explanation, no markdown, no extra text.\n"
    "Columns: action,momentum_team,momentum_score,momentum_reason\n"
    "- action: describe the play in <=10 tokens\n"
    "- momentum_team: team_A/team_B/neutral\n"
    "- momentum_score: integer -5 to +5\n"
    "- momentum_reason: 1-5 tokens\n"
    "Example: fast break layup scored,team_A,3,quick transition"
)


@app.get("/", response_class=HTMLResponse)
async def index():
    return html_content


@app.get("/video")
async def serve_video():
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/kalshi")
async def get_kalshi():
    return {"data": kalshi_data}


@app.get("/api/plays")
async def get_plays():
    away = ""
    home = ""
    if summary_data:
        teams = summary_data.get("boxscore", {}).get("teams", [])
        for t in teams:
            ha = ""
            for c in (
                summary_data.get("header", {})
                .get("competitions", [{}])[0]
                .get("competitors", [])
            ):
                if c["team"]["abbreviation"] == t["team"]["abbreviation"]:
                    ha = c.get("homeAway", "")
            if ha == "away":
                away = t["team"]["abbreviation"]
            else:
                home = t["team"]["abbreviation"]

        plays_out = []
        for p in summary_data.get("plays", []):
            plays_out.append(
                {
                    "clock": p.get("clock", {}).get("displayValue", ""),
                    "period": p.get("period", {}).get("displayValue", ""),
                    "text": p.get("text", ""),
                    "awayScore": p.get("awayScore", 0),
                    "homeScore": p.get("homeScore", 0),
                    "scoringPlay": p.get("scoringPlay", False),
                }
            )
        return {"plays": plays_out, "away": away, "home": home}
    return {"plays": [], "away": "TEAM A", "home": "TEAM B"}


@app.post("/scoreboard")
async def scoreboard_proxy(request: Request):
    body = await request.body()
    resp = await scoreboard_http.post(scoreboard_url, content=body, headers={"Content-Type": "application/json"})
    return resp.json()


@app.post("/infer")
async def infer(request: Request):
    body = await request.json()
    frames = body.get("frames", [])
    prompt = body.get("prompt", "") or DEFAULT_PROMPT

    content = []
    for f in frames:
        content.append({"type": "text", "text": f"[{f['timestamp']:.1f}s]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f['base64']}"},
            }
        )
    content.append({"type": "text", "text": prompt})

    try:
        resp = await vllm_http.post(
            f"{vllm_endpoint}/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 60,
            },
        )
        data = resp.json()
        return {"response": data["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"error": str(e)}


def load_kalshi_csv(path: str, start_offset: int = 180, end_offset: int = 0) -> list:
    from datetime import datetime, timedelta

    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        team_a_col = cols[1] if len(cols) > 1 else None
        team_b_col = cols[2] if len(cols) > 2 else None
        for row in reader:
            a_val = row.get(team_a_col, "")
            b_val = row.get(team_b_col, "")
            rows.append(
                {
                    "ts": row.get("timestamp", ""),
                    "a": float(a_val) if a_val else None,
                    "b": float(b_val) if b_val else None,
                }
            )

    # Filter to window: n-start_offset to n-end_offset minutes from end
    if rows:
        last_ts = datetime.fromisoformat(rows[-1]["ts"].replace("Z", "+00:00"))
        window_start = last_ts - timedelta(minutes=start_offset)
        window_end = last_ts - timedelta(minutes=end_offset)
        rows = [
            r
            for r in rows
            if window_start <= datetime.fromisoformat(r["ts"].replace("Z", "+00:00")) <= window_end
        ]

    return rows


def main():
    import json as json_mod

    global vllm_http, vllm_endpoint, model_name, video_path
    global summary_data, kalshi_data, html_content
    global scoreboard_http, scoreboard_url

    parser = argparse.ArgumentParser(description="Video Frame Inference Web UI")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--endpoint", required=True, help="vLLM Modal endpoint URL (e.g. https://...modal.run/v1)")
    parser.add_argument("--scoreboard-endpoint", required=True, help="Scoreboard Modal endpoint URL (e.g. https://...modal.run/scoreboard)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--summary", default="data/summary.json")
    parser.add_argument("--kalshi", default="data/arizonavkansas_kalshi.csv")
    args = parser.parse_args()

    video_path = str(Path(args.video).resolve())
    if not Path(video_path).exists():
        raise SystemExit(f"Error: video file '{video_path}' not found")

    # Load HTML
    html_path = Path(__file__).parent / "static" / "index.html"
    html_content = html_path.read_text()

    # Load data
    summary_path = Path(args.summary)
    if summary_path.exists():
        summary_data = json_mod.loads(summary_path.read_text())
        print(f"Loaded summary: {len(summary_data.get('plays', []))} plays")

    kalshi_path = Path(args.kalshi)
    if kalshi_path.exists():
        kalshi_data = load_kalshi_csv(str(kalshi_path))
        print(f"Loaded Kalshi: {len(kalshi_data)} price points")

    model_name = args.model
    vllm_endpoint = args.endpoint.rstrip("/")
    vllm_http = httpx.AsyncClient(timeout=120.0)

    scoreboard_url = args.scoreboard_endpoint
    scoreboard_http = httpx.AsyncClient(timeout=30.0)

    print(f"Starting web UI at http://localhost:{args.port}")
    print(f"Video: {video_path}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Scoreboard: {scoreboard_url}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
