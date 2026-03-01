import modal

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_DIR = "/model"
VLLM_PORT = 8000

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("vllm>=0.13.0", "huggingface-hub[hf_xet]")
)
vllm_image = base_image
score_vllm_image = base_image.pip_install("Pillow")

app = modal.App("qwen3-vl-inference")

model_vol = modal.Volume.from_name("llm-weights", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

SCORE_PROMPT = (
    "/no_think\n"
    "ONLY output one CSV row. No headers, no explanation, no markdown, no extra text.\n"
    "If you cannot see the scoreboard or game clock clearly, output exactly: NONE\n"
    "Columns: team_a_score,team_b_score,game_clock\n"
    "- team_a_score: left team's score as integer\n"
    "- team_b_score: right team's score as integer\n"
    "- game_clock: countdown timer shown on screen, format MM:SS (e.g. 14:32, 03:07)\n"
    "Example: 45,42,14:32"
)


@app.cls(
    image=vllm_image,
    gpu="H100",
    volumes={MODEL_DIR: model_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    scaledown_window=300,
    min_containers=1,
    max_containers=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class Inference:
    @modal.enter(snap=True)
    def load(self):
        import os
        from huggingface_hub import snapshot_download

        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            snapshot_download(MODEL_NAME, local_dir=MODEL_DIR)
            model_vol.commit()

        import json
        import subprocess

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_DIR,
            "--served-model-name", MODEL_NAME,
            "--port", str(VLLM_PORT),
            "--limit-mm-per-prompt", json.dumps({"image": 1}),
            "--max-model-len", "2048",
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.95",
            "--max-num-seqs", "32",
            "--trust-remote-code",
        ]
        self.proc = subprocess.Popen(cmd)

        # Wait for vLLM to be ready before snapshot
        import time
        import urllib.request
        health_url = f"http://localhost:{VLLM_PORT}/health"
        for _ in range(120):
            try:
                urllib.request.urlopen(health_url, timeout=2)
                print("vLLM is ready, snapshotting...")
                break
            except Exception:
                time.sleep(5)

    @modal.web_server(port=VLLM_PORT)
    def serve(self):
        pass  # vLLM subprocess already running from load()


@app.cls(
    image=score_vllm_image,
    gpu="H100",
    volumes={MODEL_DIR: model_vol, "/root/.cache/vllm": vllm_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
    scaledown_window=300,
    min_containers=1,
    max_containers=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class ScoreInference:
    @modal.enter(snap=True)
    def load(self):
        import os
        from huggingface_hub import snapshot_download

        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            snapshot_download(MODEL_NAME, local_dir=MODEL_DIR)
            model_vol.commit()

        import json
        import subprocess

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_DIR,
            "--served-model-name", MODEL_NAME,
            "--port", str(VLLM_PORT),
            "--limit-mm-per-prompt", json.dumps({"image": 1}),
            "--max-model-len", "2048",
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.95",
            "--max-num-seqs", "32",
            "--trust-remote-code",
        ]
        self.proc = subprocess.Popen(cmd)

        import time
        import urllib.request
        health_url = f"http://localhost:{VLLM_PORT}/health"
        for _ in range(120):
            try:
                urllib.request.urlopen(health_url, timeout=2)
                print("Score vLLM is ready, snapshotting...")
                break
            except Exception:
                time.sleep(5)

    @modal.asgi_app()
    def serve(self):
        import base64
        import io
        import re

        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from PIL import Image

        def strip_think(text):
            if not text:
                return ""
            return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

        score_app = FastAPI()

        @score_app.post("/v1/score")
        async def score_infer(request: Request):
            body = await request.json()
            base64_img = body["base64"]
            timestamp = body.get("timestamp", 0)

            # Decode image
            img_bytes = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_bytes))

            # Crop: bottom half with 15% left/right margins
            w, h = img.size
            left = int(w * 0.15)
            right = int(w * 0.85)
            top = h // 2
            bottom = h
            cropped = img.crop((left, top, right, bottom))

            # Re-encode to base64 JPEG
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=80)
            cropped_b64 = base64.b64encode(buf.getvalue()).decode()

            # Call local vLLM for score extraction
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                    json={
                        "model": MODEL_NAME,
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": f"[{timestamp:.1f}s]"},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{cropped_b64}"
                            }},
                            {"type": "text", "text": SCORE_PROMPT},
                        ]}],
                        "max_tokens": 30,
                    },
                    timeout=60,
                )
            result = resp.json()
            raw = result["choices"][0]["message"]["content"] or ""
            content = strip_think(raw)
            return JSONResponse({"response": content or "NONE"})

        @score_app.get("/health")
        async def health():
            return {"status": "ok"}

        return score_app


@app.local_entrypoint()
def health_check():
    print(f"Deploying {MODEL_NAME} with GPU memory snapshots...")
    print("Use `modal serve modal_app.py` for dev or `modal deploy modal_app.py` for prod.")
