"""
deploy/modal_maya.py  -  Maya 2 Native as a SERVERLESS GPU service on Modal (https://modal.com).

Any story, any time: the laptop sends each line to this endpoint and gets a wav back. The GPU is only running (and only billed, per second)
while lines are being rendered; it goes to sleep a few minutes after the last request. Nothing runs on your laptop's CPU.

One-time setup (your Modal account - this script never sees credentials):
    pip install modal
    modal setup                                   # opens the browser, logs you in
    modal deploy deploy/modal_maya.py             # builds the image once (~10 min: installs maya3, downloads ~5 GB of weights), prints the URL
Protect it (otherwise anyone who learns the URL can use your GPU): Modal dashboard -> Settings -> Proxy Auth Tokens -> create one, then
    set MAYA_URL=https://<workspace>--hinglish-maya-serve.modal.run
    set MAYA_PROXY_KEY=wk-...        set MAYA_PROXY_SECRET=ws-...
Check it:  python -m tools.check_maya_endpoint     then pick "Maya" in the UI.

Tuning:  MAYA_GPU (default "T4"; "L4" is a little faster), MAYA_WARM=1 keeps one container always up (no cold start, ~$0.6/h while up).
The endpoint speaks the same HTTP API as `python -m maya3.server` (GET /health, POST /v1/audio/speech), so audio.tts.MayaBackend needs no changes.
"""
import os
import subprocess

import modal

GPU = os.environ.get("MAYA_GPU", "T4")
KEEP_WARM = os.environ.get("MAYA_WARM", "0") in ("1", "true", "True")
CKPT = "maya-research/maya2-native"
SEATS = 8                       # concurrent lines batched on the GPU (~0.3 GB each on top of ~7.7 GB)

app = modal.App("hinglish-maya")


def _warm_caches():
    """Runs once at image-build time on a GPU: loads the model, which downloads the backbone, the Mimi codec and the speaker encoder into the image."""
    from maya3 import Maya3TTS
    tts = Maya3TTS.from_pretrained(CKPT, max_seats=1)
    tts.generate("नमस्ते", lang="hi", voice="Ahana")
    tts.close()


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .pip_install("maya3[server] @ git+https://github.com/MayaResearch/maya2-native", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("huggingface_hub")
    .run_function(_warm_caches, gpu=GPU)
)


@app.function(image=image, gpu=GPU, timeout=60 * 60, scaledown_window=300, min_containers=1 if KEEP_WARM else 0, max_containers=2)
@modal.concurrent(max_inputs=SEATS)
@modal.web_server(8080, startup_timeout=600, requires_proxy_auth=True)
def serve():
    subprocess.Popen(["python", "-m", "maya3.server", "--ckpt", CKPT, "--host", "0.0.0.0", "--port", "8080", "--max-seats", str(SEATS)])
