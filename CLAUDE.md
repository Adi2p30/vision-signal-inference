# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Real-time sports video inference system that uses a vision-language model (VLM) to analyze basketball game footage. It extracts momentum signals and overlays Kalshi betting odds — all in a browser-based UI.

## Commands

```bash
# First-time setup: download model weights into Modal Volume (only needed once)
modal run modal_app.py

# Deploy Modal backend (dev mode with hot reload)
modal serve modal_app.py

# Deploy Modal backend (production)
modal deploy modal_app.py

# Run local web UI (requires Modal backend running)
python main.py --video data/ArizonavKansas.mp4 --endpoint <MODAL_VLLM_URL>/v1 --scoreboard-endpoint <MODAL_SCOREBOARD_URL>/scoreboard

# Install dependencies
pip install -r requirements.txt
```

## Architecture

Two deployment layers:

**Modal cloud (`modal_app.py`)** — deploys two independent functions under one Modal app:
- `serve_vllm()` — H100 GPU, runs vLLM with Qwen3-VL-8B-Instruct, exposes `/v1/*` (OpenAI-compatible API)
- `serve_scoreboard()` — L4 GPU, runs SmolVLM2-2.2B-Instruct via transformers, exposes `/scoreboard`

Model weights are stored in a shared Modal Volume (`model-weights-vol`), downloaded once via `modal run modal_app.py`. Each function has its own Docker image (pip deps only — no weights baked in), so image rebuilds are fast and don't re-download models.

**Local FastAPI server (`main.py`)** — browser-facing app that:
- Serves the single-page UI from `static/index.html`
- Proxies `/infer` → Modal vLLM (multi-frame vision inference for momentum analysis)
- Serves `/api/plays` (ESPN play-by-play from `data/summary.json`) and `/api/kalshi` (betting odds from `data/kalshi_price_history.csv`)

**Frontend (`static/index.html`)** — single HTML file with embedded JS:
- Video player with manual/auto capture (2-frame batches every 2s in auto mode)
- VLM returns CSV row: `action,momentum_team,momentum_score,momentum_reason`
- Kalshi odds chart synced to video position
- ESPN play-by-play matched by game clock

## Data Files

Expected in `data/` directory (not committed):
- `*.mp4` — game video
- `summary.json` — ESPN game summary (boxscore, plays, header)
- `kalshi_price_history.csv` — columns: `timestamp`, team_a_price, team_b_price

## Key Details

- Python 3.12+ (Modal images use 3.12, local venv uses 3.14)
- VLM prompt is hardcoded in `DEFAULT_PROMPT` in `main.py`; can be overridden per-request via the UI text input
- Video playback is clamped to the last 30 minutes; Kalshi CSV is also filtered to last 30 minutes
- `main.py` uses `httpx` directly to call the vLLM OpenAI-compatible API (no `openai` dependency)
- Modal containers have a 5-minute scaledown window for both functions
