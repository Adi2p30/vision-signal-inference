import argparse
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn
from openai import OpenAI

app = FastAPI()

# Globals set at startup
client: OpenAI = None
model_name: str = ""
video_path: str = ""

DEFAULT_PROMPT = (
    "Output CSV only. Columns: timestamp,action,momentum_team,momentum_score,momentum_reason\n"
    "action: describe play in <=10 tokens.\n"
    "momentum_team: which team has momentum (team_A/team_B/neutral).\n"
    "momentum_score: -5 to +5 (negative=team_B, positive=team_A).\n"
    "momentum_reason: 1-5 tokens why momentum shifted.\n"
    "One row per frame. No headers. No explanation."
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
  .bar { display: flex; gap: 10px; align-items: center; }
  .bar button {
    padding: 10px 22px; border: 1px solid #fff; background: transparent; color: #fff;
    font-family: inherit; font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; cursor: pointer; transition: background 0.15s, color 0.15s;
  }
  .bar button:hover { background: #fff; color: #000; }
  .bar button:disabled { border-color: #333; color: #333; cursor: not-allowed; background: transparent; }
  .ts { font-size: 13px; color: #555; font-variant-numeric: tabular-nums; }
  .prompt-row input {
    width: 100%; padding: 10px 14px; border: 1px solid #1a1a1a; background: transparent;
    color: #888; font-family: inherit; font-size: 12px; outline: none;
  }
  .prompt-row input:focus { border-color: #333; color: #d4d4d4; }
  .prompt-row input::placeholder { color: #333; }

  .right { width: 440px; border-left: 1px solid #1a1a1a; display: flex; flex-direction: column; }
  .log-head {
    padding: 24px 20px 16px; display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid #1a1a1a;
  }
  .log-head span { font-size: 13px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #fff; }
  .log-head button {
    padding: 4px 10px; border: 1px solid #222; background: transparent; color: #444;
    font-family: inherit; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; cursor: pointer;
  }
  .log-head button:hover { color: #aaa; border-color: #444; }
  .entries { flex: 1; overflow-y: auto; padding: 16px 20px; }
  .entries::-webkit-scrollbar { width: 4px; }
  .entries::-webkit-scrollbar-thumb { background: #222; }

  .entry { margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #111; }
  .entry:last-child { border-bottom: none; }
  .entry .at { font-size: 11px; color: #444; margin-bottom: 8px; letter-spacing: 0.06em; }
  .entry .thumbs { display: flex; gap: 4px; margin-bottom: 10px; }
  .entry .thumbs img { height: 40px; opacity: 0.8; }
  .entry .out { font-size: 12px; line-height: 1.7; color: #999; white-space: pre-wrap; font-variant-numeric: tabular-nums; }
  .entry.err .out { color: #c44; }

  .spin {
    display: inline-block; width: 10px; height: 10px;
    border: 1.5px solid #333; border-top-color: #888;
    border-radius: 50%; animation: sp 0.7s linear infinite; margin-right: 6px; vertical-align: middle;
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
      <button id="captureBtn" onclick="captureAndInfer(1)">Capture 1</button>
      <button id="capture4Btn" onclick="captureAndInfer(4)">Capture 4</button>
      <span class="ts" id="tsDisplay">0:00.0</span>
    </div>
    <div class="prompt-row">
      <input id="promptInput" type="text" placeholder="custom prompt...">
    </div>
  </div>
  <div class="right">
    <div class="log-head">
      <span>Log</span>
      <button onclick="clearLog()">Clear</button>
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

video.addEventListener('timeupdate', () => {
  const t = video.currentTime;
  const m = Math.floor(t / 60);
  const s = (t % 60).toFixed(1);
  tsDisplay.textContent = m + ':' + s.padStart(4, '0');
});

function captureFrame() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);
  return { timestamp: video.currentTime, dataUrl: canvas.toDataURL('image/jpeg', 0.85) };
}

async function captureAndInfer(count) {
  const captureBtn = document.getElementById('captureBtn');
  const capture4Btn = document.getElementById('capture4Btn');
  captureBtn.disabled = true;
  capture4Btn.disabled = true;

  const frames = [];
  const wasPlaying = !video.paused;
  if (wasPlaying) video.pause();

  frames.push(captureFrame());

  if (count > 1) {
    for (let i = 1; i < count; i++) {
      video.currentTime = Math.min(video.currentTime + 1, video.duration);
      await new Promise(r => { video.onseeked = r; });
      frames.push(captureFrame());
    }
  }

  const timestamps = frames.map(f => f.timestamp.toFixed(1) + 's').join(', ');
  const entryId = 'entry-' + Date.now();
  const previews = frames.map(f => '<img src="' + f.dataUrl + '">').join('');

  logEntries.insertAdjacentHTML('afterbegin',
    '<div class="entry" id="' + entryId + '">' +
    '  <div class="at"><span class="spin"></span>' + timestamps + '</div>' +
    '  <div class="thumbs">' + previews + '</div>' +
    '  <div class="out"></div>' +
    '</div>'
  );

  const prompt = document.getElementById('promptInput').value || '';
  const b64Frames = frames.map(f => ({ timestamp: f.timestamp, base64: f.dataUrl.split(',')[1] }));

  try {
    const resp = await fetch('/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: b64Frames, prompt: prompt })
    });

    const entry = document.getElementById(entryId);
    const outEl = entry.querySelector('.out');

    if (!resp.ok) {
      entry.classList.add('err');
      entry.querySelector('.at').textContent = timestamps;
      outEl.textContent = 'HTTP ' + resp.status;
    } else {
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let text = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const payload = line.slice(6);
            if (payload === '[DONE]') continue;
            try {
              const parsed = JSON.parse(payload);
              const delta = parsed.choices?.[0]?.delta?.content;
              if (delta) {
                text += delta;
                outEl.textContent = text;
              }
              if (parsed.error) {
                entry.classList.add('err');
                outEl.textContent = parsed.error;
              }
            } catch {}
          }
        }
      }
      entry.querySelector('.at').textContent = timestamps;
    }
  } catch (err) {
    const entry = document.getElementById(entryId);
    entry.classList.add('err');
    entry.querySelector('.at').textContent = timestamps;
    entry.querySelector('.out').textContent = err.message;
  }

  captureBtn.disabled = false;
  capture4Btn.disabled = false;
  if (wasPlaying) video.play();
}

function clearLog() { logEntries.innerHTML = ''; }
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
    frames = body.get("frames", [])
    prompt = body.get("prompt", "") or DEFAULT_PROMPT

    content = []
    for f in frames:
        content.append({"type": "text", "text": f"[{f['timestamp']:.1f}s]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{f['base64']}"},
        })
    content.append({"type": "text", "text": prompt})

    def generate():
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=150,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    payload = json.dumps({"choices": [{"delta": {"content": delta}}]})
                    yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
    client = OpenAI(base_url=args.endpoint, api_key="not-needed")

    print(f"Starting web UI at http://localhost:{args.port}")
    print(f"Video: {video_path}")
    print(f"Endpoint: {args.endpoint}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
