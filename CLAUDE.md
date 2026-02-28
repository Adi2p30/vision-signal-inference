# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Real-time sports video inference system that combines a vision-language model (VLM) with OCR to analyze basketball game footage. It extracts momentum signals, reads scoreboards, and overlays Kalshi betting odds — all in a browser-based UI.

## Commands

```bash
# Deploy Modal backend (dev mode with hot reload)
modal serve modal_app.py

# Deploy Modal backend (production)
modal deploy modal_app.py

# Run local web UI (requires Modal backend running)
python main.py --video data/ArizonavKansas.mp4 --endpoint <MODAL_VLLM_URL>/v1

# Install dependencies
pip install -r requirements.txt
```

## Architecture

Two deployment layers:

**Modal cloud (`modal_app.py`)** — deploys two services to Modal:
- `serve`: vLLM server running Qwen3-VL-8B-Instruct on H100 GPU. Exposes OpenAI-compatible API.
- `OCR` class: CPU-based RapidOCR service with FastAPI endpoint (`POST /extract`). Reads scoreboard crops (base64 image → clock, period, scores).

**Local FastAPI server (`main.py`)** — browser-facing app that:
- Serves the single-page UI from `static/index.html`
- Proxies `/infer` → Modal vLLM (multi-frame vision inference for momentum analysis)
- Proxies `/ocr` → Modal OCR service (scoreboard reading)
- Serves `/api/plays` (ESPN play-by-play from `data/summary.json`) and `/api/kalshi` (betting odds from `data/kalshi_price_history.csv`)
- The OCR endpoint URL is auto-derived from the vLLM endpoint URL via `derive_ocr_url()`, or can be overridden with `--ocr-endpoint`

**Frontend (`static/index.html`)** — single HTML file with embedded JS:
- Video player with manual/auto capture (2-frame batches every 2s in auto mode)
- Fires OCR + VLM requests in parallel per capture
- OCR updates scoreboard (score, clock, period, scoring droughts)
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
- The `openai` client is used with `api_key="not-needed"` since vLLM doesn't require auth
- Modal containers have a 5-minute scaledown window for vLLM, 2-minute for OCR
