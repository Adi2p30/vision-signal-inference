import argparse
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
from openai import AsyncOpenAI

app = FastAPI()

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

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Frame Inference</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: #000; color: #d4d4d4; height: 100vh; overflow: hidden; }
  .layout { display: flex; height: 100vh; }

  .left { flex: 1; display: flex; flex-direction: column; padding: 24px 28px; gap: 16px; min-width: 0; }
  .title { font-size: 13px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #fff; }
  video { width: 100%; flex: 1; min-height: 0; background: #000; border: 1px solid #1a1a1a; }
  .bar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .bar button {
    padding: 10px 22px; border: 1px solid #fff; background: transparent; color: #fff;
    font-family: inherit; font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; cursor: pointer; transition: background 0.15s, color 0.15s;
  }
  .bar button:hover { background: #fff; color: #000; }
  .bar button:disabled { border-color: #333; color: #333; cursor: not-allowed; background: transparent; }
  .ts { font-size: 13px; color: #555; font-variant-numeric: tabular-nums; }
  .auto-label {
    display: flex; align-items: center; gap: 6px; font-size: 11px; color: #555;
    text-transform: uppercase; letter-spacing: 0.08em; cursor: pointer; user-select: none;
  }
  .auto-label input { accent-color: #fff; cursor: pointer; }
  .auto-label.active { color: #fff; }
  .stats {
    font-size: 10px; color: #333; letter-spacing: 0.06em; text-transform: uppercase;
  }
  .stats .s-val { color: #666; }
  .prompt-row input {
    width: 100%; padding: 10px 14px; border: 1px solid #1a1a1a; background: transparent;
    color: #888; font-family: inherit; font-size: 12px; outline: none;
  }
  .prompt-row input:focus { border-color: #333; color: #d4d4d4; }
  .prompt-row input::placeholder { color: #333; }

  .right { width: 480px; border-left: 1px solid #1a1a1a; display: flex; flex-direction: column; }
  .log-head {
    padding: 24px 20px 16px; display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid #1a1a1a;
  }
  .log-head span { font-size: 13px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #fff; }
  .log-head-btns { display: flex; gap: 6px; }
  .log-head button {
    padding: 4px 10px; border: 1px solid #222; background: transparent; color: #444;
    font-family: inherit; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; cursor: pointer;
  }
  .log-head button:hover { color: #aaa; border-color: #444; }

  .scoreboard {
    padding: 16px 20px; border-bottom: 1px solid #1a1a1a;
    display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 8px;
  }
  .sb-team { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #555; }
  .sb-team.left-t { text-align: right; }
  .sb-team.right-t { text-align: left; }
  .sb-score { font-size: 28px; font-weight: 700; color: #fff; text-align: center; letter-spacing: 0.05em; }
  .sb-momentum { grid-column: 1 / -1; text-align: center; font-size: 10px; color: #444; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; }
  .sb-momentum .val { color: #888; }

  .drought {
    padding: 10px 20px; border-bottom: 1px solid #1a1a1a; font-size: 10px;
    color: #444; letter-spacing: 0.06em; text-transform: uppercase;
    display: flex; justify-content: space-between;
  }
  .drought .d-val { color: #888; }

  .entries { flex: 1; overflow-y: auto; padding: 12px 20px; }
  .entries::-webkit-scrollbar { width: 4px; }
  .entries::-webkit-scrollbar-thumb { background: #222; }

  .entry { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #111; }
  .entry:last-child { border-bottom: none; }
  .entry .at { font-size: 10px; color: #444; margin-bottom: 4px; letter-spacing: 0.06em; }
  .entry .out { font-size: 11px; line-height: 1.5; color: #999; white-space: pre-wrap; font-variant-numeric: tabular-nums; }
  .entry.err .out { color: #c44; }
  .entry.pending .out { color: #333; }

  .spin {
    display: inline-block; width: 8px; height: 8px;
    border: 1.5px solid #333; border-top-color: #888;
    border-radius: 50%; animation: sp 0.7s linear infinite; margin-right: 4px; vertical-align: middle;
  }
  @keyframes sp { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="layout">
  <div class="left">
    <div class="title">Frame Inference</div>
    <video id="video" controls>
      <source src="/video" type="video/mp4">
    </video>
    <div class="bar">
      <button id="captureBtn" onclick="manualCapture()">Capture</button>
      <label class="auto-label" id="autoLabel">
        <input type="checkbox" id="autoCheck" onchange="toggleAuto()"> Auto
      </label>
      <span class="ts" id="tsDisplay">0:00.0</span>
    </div>
    <div class="stats">
      inflight: <span class="s-val" id="inflightCount">0</span>
      &nbsp; done: <span class="s-val" id="doneCount">0</span>
      &nbsp; fps: <span class="s-val" id="inferFps">0</span>
    </div>
    <div class="prompt-row">
      <input id="promptInput" type="text" placeholder="custom prompt...">
    </div>
  </div>
  <div class="right">
    <div class="log-head">
      <span>Log</span>
      <div class="log-head-btns">
        <button onclick="exportCSV()">CSV</button>
        <button onclick="clearLog()">Clear</button>
      </div>
    </div>
    <div class="scoreboard">
      <div class="sb-team left-t">Team A</div>
      <div class="sb-score" id="sbScore">0 - 0</div>
      <div class="sb-team right-t">Team B</div>
      <div class="sb-momentum">momentum <span class="val" id="sbMomentum">neutral 0</span></div>
    </div>
    <div class="drought">
      <span>A drought: <span class="d-val" id="droughtA">0s</span></span>
      <span>B drought: <span class="d-val" id="droughtB">0s</span></span>
    </div>
    <div class="entries" id="logEntries"></div>
  </div>
</div>
<canvas id="canvas" style="display:none;"></canvas>
<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const logEntries = document.getElementById('logEntries');
const tsDisplay = document.getElementById('tsDisplay');
const SCALE_W = 480; // downscale width for speed

// State
let autoRunning = false;
let autoRafId = null;
let lastCaptureTime = -1;
let inflight = 0;
let doneTotal = 0;
let scoreA = 0, scoreB = 0;
let lastScoreA_ts = 0, lastScoreB_ts = 0;
const allRows = [];
let fpsTimestamps = [];

video.addEventListener('timeupdate', () => {
  const t = video.currentTime;
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1);
  tsDisplay.textContent = m + ':' + s.padStart(4, '0');
});

function captureFrame() {
  const scale = Math.min(1, SCALE_W / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return {
    timestamp: video.currentTime,
    base64: canvas.toDataURL('image/jpeg', 0.5).split(',')[1]
  };
}

function updateStats() {
  document.getElementById('inflightCount').textContent = inflight;
  document.getElementById('doneCount').textContent = doneTotal;
  // fps over last 10s
  const now = performance.now();
  fpsTimestamps = fpsTimestamps.filter(t => now - t < 10000);
  const fps = fpsTimestamps.length > 0 ? (fpsTimestamps.length / ((now - fpsTimestamps[0]) / 1000)).toFixed(1) : '0';
  document.getElementById('inferFps').textContent = fps;
}

// --- Auto: capture every frame, fire concurrently ---
function toggleAuto() {
  const on = document.getElementById('autoCheck').checked;
  document.getElementById('autoLabel').classList.toggle('active', on);
  if (on) {
    autoRunning = true;
    lastCaptureTime = -1;
    autoLoop();
  } else {
    autoRunning = false;
    if (autoRafId) cancelAnimationFrame(autoRafId);
  }
}

function autoLoop() {
  if (!autoRunning) return;
  autoRafId = requestAnimationFrame(autoLoop);
  if (video.paused || video.ended) return;
  const t = video.currentTime;
  // Capture every unique second of video
  const tRounded = Math.floor(t);
  if (tRounded === lastCaptureTime) return;
  lastCaptureTime = tRounded;
  fireInference(captureFrame());
}

function manualCapture() {
  fireInference(captureFrame());
}

// --- Fire & forget inference ---
function fireInference(frame) {
  inflight++;
  updateStats();

  const ts = frame.timestamp.toFixed(1) + 's';
  const entryId = 'e' + Date.now() + Math.random().toString(36).slice(2, 6);

  logEntries.insertAdjacentHTML('afterbegin',
    '<div class="entry pending" id="' + entryId + '">' +
    '<div class="at"><span class="spin"></span>' + ts + '</div>' +
    '<div class="out">...</div></div>'
  );

  const prompt = document.getElementById('promptInput').value || '';

  fetch('/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ timestamp: frame.timestamp, base64: frame.base64, prompt: prompt })
  })
  .then(r => r.json())
  .then(data => {
    const el = document.getElementById(entryId);
    if (!el) return;
    el.classList.remove('pending');
    if (data.error) {
      el.classList.add('err');
      el.querySelector('.out').textContent = data.error;
    } else {
      el.querySelector('.at').textContent = ts;
      el.querySelector('.out').textContent = data.response;
      updateScore(data.response, frame.timestamp);
    }
  })
  .catch(err => {
    const el = document.getElementById(entryId);
    if (!el) return;
    el.classList.remove('pending');
    el.classList.add('err');
    el.querySelector('.out').textContent = err.message;
  })
  .finally(() => {
    inflight--;
    doneTotal++;
    fpsTimestamps.push(performance.now());
    updateStats();
  });
}

// --- Score & drought ---
function updateScore(csvText, currentTs) {
  const lines = csvText.trim().split('\\n');
  for (const line of lines) {
    const cols = line.split(',');
    if (cols.length >= 7) {
      const a = parseInt(cols[2]);
      const b = parseInt(cols[3]);
      if (!isNaN(a) && !isNaN(b)) {
        if (a > scoreA) lastScoreA_ts = currentTs;
        if (b > scoreB) lastScoreB_ts = currentTs;
        scoreA = a;
        scoreB = b;
      }
      const mTeam = cols[4]?.trim() || 'neutral';
      const mScore = cols[5]?.trim() || '0';
      document.getElementById('sbMomentum').textContent = mTeam + ' ' + mScore;
      allRows.push(line.trim());
    }
  }
  document.getElementById('sbScore').textContent = scoreA + ' - ' + scoreB;
  const dA = currentTs - lastScoreA_ts;
  const dB = currentTs - lastScoreB_ts;
  document.getElementById('droughtA').textContent = formatDrought(dA);
  document.getElementById('droughtB').textContent = formatDrought(dB);
}

function formatDrought(sec) {
  if (sec < 60) return sec.toFixed(0) + 's';
  return Math.floor(sec / 60) + 'm ' + (sec % 60).toFixed(0) + 's';
}

function exportCSV() {
  const header = 'timestamp,action,team_a_score,team_b_score,momentum_team,momentum_score,momentum_reason';
  const csv = header + '\\n' + allRows.join('\\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'inference_log.csv';
  a.click();
}

function clearLog() {
  logEntries.innerHTML = '';
  allRows.length = 0;
  scoreA = 0; scoreB = 0;
  lastScoreA_ts = 0; lastScoreB_ts = 0;
  doneTotal = 0;
  document.getElementById('sbScore').textContent = '0 - 0';
  document.getElementById('sbMomentum').textContent = 'neutral 0';
  document.getElementById('droughtA').textContent = '0s';
  document.getElementById('droughtB').textContent = '0s';
  updateStats();
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/video")
async def serve_video():
    return FileResponse(video_path, media_type="video/mp4")


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
