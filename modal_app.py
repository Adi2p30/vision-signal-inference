import modal

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_DIR = "/model"
VLLM_PORT = 8000

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("vllm>=0.13.0", "huggingface-hub[hf_xet]")
)

app = modal.App("qwen3-vl-inference")

model_vol = modal.Volume.from_name("llm-weights", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="H100",
    volumes={MODEL_DIR: model_vol, "/root/.cache/vllm": vllm_cache_vol},
    timeout=1800,
    scaledown_window=300,
    min_containers=1
)
@modal.web_server(port=VLLM_PORT, startup_timeout=600)
def serve():
    import json
    import os
    import subprocess

    from huggingface_hub import snapshot_download

    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        snapshot_download(MODEL_NAME, local_dir=MODEL_DIR)
        model_vol.commit()

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
    subprocess.Popen(cmd)


@app.local_entrypoint()
def health_check():
    print(f"Deploying {MODEL_NAME}...")
    print("Use `modal serve modal_app.py` for dev or `modal deploy modal_app.py` for prod.")
