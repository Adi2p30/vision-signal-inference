import modal

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_DIR = "/model"
VLLM_PORT = 8001  # internal port for vLLM subprocess


def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_NAME, local_dir=MODEL_DIR)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "vllm>=0.13.0",
        "huggingface-hub",
        "opencv-python-headless",
        "Pillow",
        "numpy",
        "httpx",
        "fastapi",
    )
    .pip_install("easyocr", extra_options="--no-deps")
    .pip_install("python-bidi", "scikit-image")
    .run_function(download_model)
)

app = modal.App("qwen3-vl-inference")

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/root/.cache/vllm": vllm_cache_vol},
    timeout=1800,
    scaledown_window=300,
    max_containers=2,
)
@modal.asgi_app()
def serve():
    import base64
    import io
    import json
    import re
    import socket
    import subprocess
    import time

    import httpx
    import numpy as np
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
    from PIL import Image
    import easyocr

    # --- Start vLLM on internal port ---
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_DIR,
        "--served-model-name", MODEL_NAME,
        "--port", str(VLLM_PORT),
        "--limit-mm-per-prompt", json.dumps({"image": 2}),
        "--max-model-len", "2048",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.95",
        "--max-num-seqs", "32",
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

    # --- Init ---
    vllm_client = httpx.AsyncClient(
        base_url=f"http://localhost:{VLLM_PORT}", timeout=120.0
    )
    ocr_engine = easyocr.Reader(["en"], gpu=True)

    web_app = FastAPI()

    # --- OCR endpoint ---
    @web_app.post("/ocr")
    async def ocr_endpoint(data: dict):
        t0 = time.perf_counter()

        img_b64 = data.get("image", "")
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        result = ocr_engine.readtext(np.array(img))

        clock = None
        period = None
        score_candidates = []  # (value, center_x, box_area)
        all_text = []

        for box, text, _conf in result:
            all_text.append(text)

            # Clock: M:SS or MM:SS
            cm = re.search(r"(\d{1,2}:\d{2})", text)
            if cm and not clock:
                clock = cm.group(1)
                continue

            # Period (may include shot clock like "25 2nd")
            if re.search(r"(1st|2nd|1ST|2ND|OT|HALF|Half)", text):
                period = text.strip()
                continue

            # Standalone 2-3 digit numbers as score candidates
            text_clean = text.strip()
            if re.fullmatch(r"\d{2,3}", text_clean):
                val = int(text_clean)
                if val <= 200:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    cx = sum(xs) / len(xs)
                    score_candidates.append((val, cx, area))

        # No clock means no scoreboard visible
        score_a = None
        score_b = None
        if clock and len(score_candidates) >= 2:
            # Pick the two largest-area standalone numbers (game scores
            # are displayed in the biggest font on the scoreboard)
            score_candidates.sort(key=lambda x: -x[2])
            top2 = sorted(score_candidates[:2], key=lambda x: x[1])
            score_a = top2[0][0]  # left
            score_b = top2[1][0]  # right
        elif clock and len(score_candidates) == 1:
            score_a = score_candidates[0][0]

        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        return {
            "clock": clock,
            "period": period,
            "score_a": score_a,
            "score_b": score_b,
            "raw": all_text,
            "elapsed_ms": elapsed_ms,
        }

    # --- Proxy /v1/* to vLLM subprocess ---
    @web_app.api_route(
        "/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    async def proxy_vllm(request: Request, path: str):
        body = await request.body()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        resp = await vllm_client.request(
            method=request.method,
            url=f"/v1/{path}",
            content=body,
            headers=headers,
        )
        resp_headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding")
        }
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
        )

    return web_app


@app.local_entrypoint()
def health_check():
    print(f"Deploying {MODEL_NAME}...")
    print("Use `modal serve modal_app.py` for dev or `modal deploy modal_app.py` for prod.")
