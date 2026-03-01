import asyncio
import csv
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
import uvicorn
from openai import AsyncOpenAI

# Environment variables (with defaults)
GAME_ID = os.environ.get("GAME_ID", "26feb09arizku")
VLM_ENDPOINT = os.environ.get("VLM_ENDPOINT", "https://adityapdev13--qwen3-vl-inference-inference-serve.modal.run")
SCORE_ENDPOINT = os.environ.get("SCORE_ENDPOINT", "https://adityapdev13--qwen3-vl-inference-scoreinference-serve.modal.run/v1/score")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
PORT = int(os.environ.get("PORT", "8080"))

# Normalize VLM endpoint to end with /v1 so the OpenAI client hits POST /v1/chat/completions
_endpoint = VLM_ENDPOINT.strip().rstrip('/')
if not _endpoint.endswith('/v1'):
    _endpoint = _endpoint + '/v1'
VLM_ENDPOINT = _endpoint


def strip_think(text):
    """Strip Qwen3 <think>...</think> reasoning tokens from response."""
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path(__file__).parent / "data"
    video_path = data_dir / f"stream-{GAME_ID}.mp4"
    if not video_path.exists():
        raise SystemExit(f"Error: video file '{video_path}' not found")

    app.state.game_id = GAME_ID
    app.state.model_name = MODEL_NAME
    app.state.client = AsyncOpenAI(base_url=VLM_ENDPOINT, api_key="not-needed")
    app.state.score_endpoint = SCORE_ENDPOINT
    app.state.score_http = httpx.AsyncClient(timeout=httpx.Timeout(60.0)) if SCORE_ENDPOINT else None

    # Extract team names from PBP data
    pbp_path = data_dir / f"pbp-{GAME_ID}.json"
    away_name, home_name = "Away", "Home"
    away_abbr, home_abbr = "AWAY", "HOME"
    if pbp_path.exists():
        with open(pbp_path) as f:
            pbp_data = json.load(f)
        header = pbp_data.get("header", {})
        comps = header.get("competitions", [])
        if comps and "competitors" in comps[0]:
            for c in comps[0]["competitors"]:
                team = c.get("team", {})
                if c.get("homeAway") == "away":
                    away_name = team.get("displayName", "Away")
                    away_abbr = team.get("abbreviation", "AWAY")
                elif c.get("homeAway") == "home":
                    home_name = team.get("displayName", "Home")
                    home_abbr = team.get("abbreviation", "HOME")

    # team_a = away (left on scoreboard), team_b = home (right on scoreboard)
    app.state.team_a_name = away_name
    app.state.team_a_abbr = away_abbr
    app.state.team_b_name = home_name
    app.state.team_b_abbr = home_abbr

    # Game state for flag evaluation + clock self-correction
    app.state.game = {
        "score_a": 0,
        "score_b": 0,
        "clock_seconds": None,
        "period": 0,
        # Clock self-correction tracking
        "accepted_clock": None,      # last accepted clock reading (seconds)
        "accepted_video_ts": None,   # video timestamp when last accepted
        "accepted_period": 0,        # period when last accepted
        "outlier_count": 0,          # consecutive rejected frames
    }

    mode = "dual VLM" if SCORE_ENDPOINT else "single VLM"
    print(f"Game ID: {GAME_ID} — {away_abbr} @ {home_abbr} [{mode}]")

    yield

    if app.state.score_http:
        await app.state.score_http.aclose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

CLOCK_TOLERANCE = 0.10   # 10% deviation to flag as outlier
OUTLIER_ACCEPT = 30      # accept VLM value after this many consecutive outliers


def parse_clock(clock_str):
    """Parse M:SS or MM:SS game clock to total seconds."""
    if not clock_str:
        return None
    m = re.match(r'^(\d{1,2}):(\d{2})$', clock_str.strip())
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def format_clock(seconds):
    """Convert total seconds to M:SS display string."""
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _accept_clock(gs, clock_seconds, video_ts, period):
    """Accept a clock reading and reset outlier tracking."""
    gs["accepted_clock"] = clock_seconds
    gs["accepted_video_ts"] = video_ts
    gs["accepted_period"] = period
    gs["clock_seconds"] = clock_seconds
    gs["outlier_count"] = 0


def validate_clock(gs, vlm_clock, video_ts, vlm_period=None):
    """Validate VLM clock against expected countdown.

    Tracks an internal countdown: for every second of video time elapsed,
    the expected game clock decreases by one second.  If the VLM reading
    deviates more than CLOCK_TOLERANCE (10%), it is rejected and the
    expected value is used instead.  After OUTLIER_ACCEPT (30) consecutive
    rejections, the VLM value is force-accepted.

    Returns (corrected_clock_seconds, accepted_bool).
    """
    if vlm_clock is None:
        return gs["clock_seconds"], True

    period = vlm_period or gs["accepted_period"] or 0

    # Period change (halftime) -> clock resets, accept immediately
    if vlm_period and vlm_period != gs["accepted_period"]:
        _accept_clock(gs, vlm_clock, video_ts, vlm_period)
        return vlm_clock, True

    # First reading ever -> accept
    if gs["accepted_clock"] is None:
        _accept_clock(gs, vlm_clock, video_ts, period)
        return vlm_clock, True

    # Expected clock = last accepted minus elapsed video time
    elapsed = video_ts - gs["accepted_video_ts"]
    expected = gs["accepted_clock"] - elapsed
    expected = max(0, expected)

    # 10% tolerance (floor reference at 10s to avoid issues near 0:00)
    reference = max(expected, 10)
    deviation = abs(vlm_clock - expected) / reference

    if deviation <= CLOCK_TOLERANCE:
        _accept_clock(gs, vlm_clock, video_ts, period)
        return vlm_clock, True
    else:
        gs["outlier_count"] += 1
        if gs["outlier_count"] >= OUTLIER_ACCEPT:
            # 30 consecutive outliers -> trust the VLM
            _accept_clock(gs, vlm_clock, video_ts, period)
            return vlm_clock, True
        # Reject: use expected countdown instead
        corrected = max(0, int(expected))
        gs["clock_seconds"] = corrected
        return corrected, False


def evaluate_flags(gs, team_a_abbr="AWAY", team_b_abbr="HOME"):
    """Check game conditions and return active flags."""
    flags = []
    sa, sb = gs["score_a"], gs["score_b"]
    clock = gs["clock_seconds"]
    period = gs["period"]
    lead = abs(sa - sb)
    leader = team_a_abbr if sa > sb else (team_b_abbr if sb > sa else None)

    # Flag: 6+ point lead with <=90s left in 2nd half
    if period == 2 and clock is not None and clock <= 90 and lead >= 6 and leader:
        flags.append({
            "id": "late_lead_6",
            "label": "LATE LEAD",
            "message": f"{leader} leads by {lead} with {clock}s left in 2H",
            "severity": "high",
        })

    return flags


def process_score(gs, score_csv, video_ts, team_a_abbr="AWAY", team_b_abbr="HOME"):
    """Parse dual-VLM score CSV, validate clock, evaluate flags.

    Returns (flags, clock_correction).
    """
    no_correction = {"accepted": True, "outlier_streak": 0}
    if not score_csv or score_csv.strip() == "NONE":
        return [], no_correction

    cols = score_csv.split(',')
    if len(cols) < 3:
        return [], no_correction

    try:
        gs["score_a"] = int(cols[0].strip())
        gs["score_b"] = int(cols[1].strip())
    except ValueError:
        pass

    raw_clock = parse_clock(cols[2].strip())
    vlm_period = None
    if len(cols) >= 4:
        try:
            vlm_period = int(cols[3].strip())
        except ValueError:
            pass

    corrected, accepted = validate_clock(gs, raw_clock, video_ts, vlm_period)
    gs["clock_seconds"] = corrected
    if vlm_period:
        gs["period"] = vlm_period

    clock_correction = {
        "raw": raw_clock,
        "corrected": corrected,
        "corrected_display": format_clock(corrected),
        "accepted": accepted,
        "outlier_streak": gs["outlier_count"],
    }
    return evaluate_flags(gs, team_a_abbr, team_b_abbr), clock_correction


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
    "Columns: timestamp,action,team_a_score,team_b_score,game_clock,period,momentum_team,momentum_score,momentum_reason\n"
    "- action: <=10 tokens\n"
    "- game_clock: countdown timer shown on screen, format M:SS or MM:SS exactly as displayed (e.g. 14:32, 3:07, 0:48)\n"
    "- period: current half/period number shown on scoreboard (1 or 2)\n"
    "- momentum_team: team_A/team_B/neutral\n"
    "- momentum_score: -5 to +5\n"
    "- momentum_reason: 1-5 tokens\n"
    "Example: 12.0,fast break layup scored,45,42,14:32,2,team_A,3,quick transition\n"
    "Example under 1 min: 12.0,free throw made,52,50,0:48,2,team_A,2,clutch free throws"
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
                entry[col] = float(val) if val else None
            rows.append(entry)
    return JSONResponse(rows)


@app.get("/flags")
async def flags(request: Request):
    """Return current game state and active flags."""
    gs = request.app.state.game
    return {"game_state": gs, "flags": evaluate_flags(gs, request.app.state.team_a_abbr, request.app.state.team_b_abbr)}


@app.get("/config")
async def config(request: Request):
    """Tell the frontend which mode we're running in + team info."""
    s = request.app.state
    return {
        "dual_vlm": bool(s.score_endpoint),
        "team_a": {"name": s.team_a_name, "abbr": s.team_a_abbr},
        "team_b": {"name": s.team_b_name, "abbr": s.team_b_abbr},
    }


@app.get("/candlesticks")
async def candlesticks(request: Request):
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
            flag_list, clock_correction = process_score(state.game, score_val, timestamp, state.team_a_abbr, state.team_b_abbr)
            return {
                "momentum": momentum or "NONE",
                "score": score_val or "NONE",
                "flags": flag_list,
                "clock_correction": clock_correction,
            }
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
    print(f"Starting web UI at http://localhost:{PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
