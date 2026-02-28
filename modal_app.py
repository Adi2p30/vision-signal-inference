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


@app.cls(
    image=vllm_image,
    gpu="H100",
    volumes={MODEL_DIR: model_vol, "/root/.cache/vllm": vllm_cache_vol},
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


@app.local_entrypoint()
def health_check():
    print(f"Deploying {MODEL_NAME} with GPU memory snapshots...")
    print("Use `modal serve modal_app.py` for dev or `modal deploy modal_app.py` for prod.")
