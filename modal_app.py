import modal
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
SCOREBOARD_MODEL_NAME = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODELS_DIR = Path("/models")
VLLM_MODEL_DIR = MODELS_DIR / "vllm"
SCOREBOARD_MODEL_DIR = MODELS_DIR / "scoreboard"
VLLM_PORT = 8001  # internal port for vLLM subprocess

# ---------------------------------------------------------------------------
# Persistent volume for model weights — downloaded once, reused across deploys
# ---------------------------------------------------------------------------
model_weights_vol = modal.Volume.from_name("model-weights-vol", create_if_missing=True)

# Lightweight image for downloading weights (no GPU needed)
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# ---------------------------------------------------------------------------
# Serving images — pip layers cached independently, no model weights baked in
# ---------------------------------------------------------------------------
_cuda_base = modal.Image.from_registry(
    "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
)

vllm_image = _cuda_base.pip_install(
    "vllm>=0.13.0",
    "huggingface-hub",
    "httpx",
    "fastapi",
)

scoreboard_image = _cuda_base.pip_install(
    "transformers",
    "torch",
    "accelerate",
    "num2words",
    "Pillow",
    "numpy",
    "huggingface-hub",
    "httpx",
    "fastapi",
)

app = modal.App("qwen3-vl-inference")


# ---------------------------------------------------------------------------
# Download functions — run once to populate the volume
#   modal run modal_app.py
# ---------------------------------------------------------------------------
@app.function(
    volumes={MODELS_DIR.as_posix(): model_weights_vol},
    image=download_image,
)
def download_vllm_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME, local_dir=str(VLLM_MODEL_DIR))
    print(f"Downloaded {MODEL_NAME} to {VLLM_MODEL_DIR}")


@app.function(
    volumes={MODELS_DIR.as_posix(): model_weights_vol},
    image=download_image,
)
def download_scoreboard_model():
    from huggingface_hub import snapshot_download

    snapshot_download(SCOREBOARD_MODEL_NAME, local_dir=str(SCOREBOARD_MODEL_DIR))
    print(f"Downloaded {SCOREBOARD_MODEL_NAME} to {SCOREBOARD_MODEL_DIR}")


@app.local_entrypoint()
def download_models():
    """Download both models into the volume. Run: modal run modal_app.py"""
    download_vllm_model.remote()
    download_scoreboard_model.remote()


# ---------------------------------------------------------------------------
# vLLM function — H100, proxies /v1/* to the vLLM subprocess
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={MODELS_DIR.as_posix(): model_weights_vol},
    timeout=1800,
    scaledown_window=300,
    min_containers=1,
    max_containers=2,
)
@modal.asgi_app()
def serve_vllm():
    import json
    import os
    import socket
    import subprocess
    import time

    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    # Ensure model weights are in the volume (downloads on first start)
    if not os.path.exists(str(VLLM_MODEL_DIR / "config.json")):
        print(f"Model not found in volume, downloading {MODEL_NAME}...")
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL_NAME, local_dir=str(VLLM_MODEL_DIR))
        print(f"Downloaded {MODEL_NAME} to {VLLM_MODEL_DIR}")

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(VLLM_MODEL_DIR),
        "--served-model-name",
        MODEL_NAME,
        "--port",
        str(VLLM_PORT),
        "--limit-mm-per-prompt",
        json.dumps({"image": 2}),
        "--max-model-len",
        "2048",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--max-num-seqs",
        "32",
        "--trust-remote-code",
    ]
    subprocess.Popen(cmd)

    # Wait for vLLM to be ready (up to 10 minutes)
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", VLLM_PORT), timeout=1):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)

    vllm_client = httpx.AsyncClient(
        base_url=f"http://localhost:{VLLM_PORT}", timeout=120.0
    )
    web_app = FastAPI()

    @web_app.api_route(
        "/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    async def proxy_vllm(request: Request, path: str):
        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        resp = await vllm_client.request(
            method=request.method,
            url=f"/v1/{path}",
            content=body,
            headers=headers,
        )
        resp_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding")
        }
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
        )

    return web_app


# ---------------------------------------------------------------------------
# Scoreboard function — L4, loads SmolVLM2 into VRAM, exposes /scoreboard
# ---------------------------------------------------------------------------
@app.function(
    image=scoreboard_image,
    gpu="L4",
    volumes={MODELS_DIR.as_posix(): model_weights_vol},
    timeout=300,
    scaledown_window=300,
    min_containers=1,
    max_containers=2,
)
@modal.asgi_app()
def serve_scoreboard():
    import asyncio
    import base64
    import io
    import os
    import re

    import torch
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Ensure model weights are in the volume (downloads on first start)
    if not os.path.exists(str(SCOREBOARD_MODEL_DIR / "config.json")):
        print(f"Model not found in volume, downloading {SCOREBOARD_MODEL_NAME}...")
        from huggingface_hub import snapshot_download

        snapshot_download(SCOREBOARD_MODEL_NAME, local_dir=str(SCOREBOARD_MODEL_DIR))
        print(f"Downloaded {SCOREBOARD_MODEL_NAME} to {SCOREBOARD_MODEL_DIR}")

    print("Loading SmolVLM2 scoreboard model...")
    sb_processor = AutoProcessor.from_pretrained(str(SCOREBOARD_MODEL_DIR), local_files_only=True)
    sb_model = AutoModelForImageTextToText.from_pretrained(
        str(SCOREBOARD_MODEL_DIR),
        dtype=torch.float16,
        device_map="cuda",
        local_files_only=True,
    )
    sb_model.eval()
    print("SmolVLM2 scoreboard model loaded.")

    web_app = FastAPI()

    def _run_scoreboard(image_b64: str) -> dict:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        prompt_text = (
            "Read the scoreboard. Output ONLY one CSV row: score_a,score_b,clock,half\n"
            "Example: 64,67,8:13,2nd"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        inputs = sb_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(sb_model.device)

        with torch.no_grad():
            output_ids = sb_model.generate(**inputs, max_new_tokens=30)

        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        raw = sb_processor.decode(generated_ids, skip_special_tokens=True).strip()

        result = {"score_a": "", "score_b": "", "clock": "", "half": "", "raw": raw}
        match = re.match(r"(\d+)\s*,\s*(\d+)\s*,\s*([\d:]+)\s*,\s*(.+)", raw)
        if match:
            result["score_a"] = match.group(1)
            result["score_b"] = match.group(2)
            result["clock"] = match.group(3)
            result["half"] = match.group(4).strip()

        return result

    @web_app.post("/scoreboard")
    async def scoreboard(request: Request):
        body = await request.json()
        image_b64 = body.get("image", "")
        if not image_b64:
            return JSONResponse({"error": "No image provided"}, status_code=400)
        try:
            result = await asyncio.to_thread(_run_scoreboard, image_b64)
            return result
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return web_app
