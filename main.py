import argparse
import csv
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from openai import AsyncOpenAI

app = FastAPI()

# Globals set at startup
client: AsyncOpenAI = None
ocr_http: httpx.AsyncClient = None
ocr_url: str = ""
model_name: str = ""
video_path: str = ""
summary_data: dict = {}
kalshi_data: list = []
html_content: str = ""

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
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=60,
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}


@app.post("/ocr")
async def ocr(request: Request):
    body = await request.json()
    image_b64 = body.get("image", "")
    try:
        resp = await ocr_http.post(ocr_url, json={"image": image_b64})
        return resp.json()
    except Exception as e:
        return {"error": str(e)}



def load_kalshi_csv(path: str, last_minutes: int = 30) -> list:
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

    # Filter to last N minutes of data
    if rows and last_minutes > 0:
        last_ts = datetime.fromisoformat(rows[-1]["ts"].replace("Z", "+00:00"))
        cutoff = last_ts - timedelta(minutes=last_minutes)
        rows = [
            r
            for r in rows
            if datetime.fromisoformat(r["ts"].replace("Z", "+00:00")) >= cutoff
        ]

    return rows


def main():
    import json as json_mod

    global client, ocr_http, ocr_url, model_name, video_path
    global summary_data, kalshi_data, html_content

    parser = argparse.ArgumentParser(description="Video Frame Inference Web UI")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--endpoint", required=True, help="Modal endpoint URL (e.g. https://...modal.run/v1)")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--summary", default="data/summary.json")
    parser.add_argument("--kalshi", default="data/kalshi_price_history.csv")
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
    client = AsyncOpenAI(base_url=args.endpoint, api_key="not-needed")

    # OCR endpoint — same Modal instance, /ocr path
    endpoint_base = args.endpoint.rstrip("/")
    if endpoint_base.endswith("/v1"):
        endpoint_base = endpoint_base[:-3]
    ocr_url = f"{endpoint_base}/ocr"
    ocr_http = httpx.AsyncClient(timeout=10.0)
    print(f"OCR endpoint: {ocr_url}")

    print(f"Starting web UI at http://localhost:{args.port}")
    print(f"Video: {video_path}")
    print(f"Endpoint: {args.endpoint}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
