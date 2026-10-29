"""
tools/check_maya_endpoint.py  -  is the Maya server (Modal / Kaggle tunnel / local GPU) reachable and does it speak? Times a cold start and one line.

    set MAYA_URL=https://...        (and MAYA_PROXY_KEY / MAYA_PROXY_SECRET for a protected Modal endpoint)
    python -m tools.check_maya_endpoint            -> writes outputs/maya_endpoint_check.wav
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()                                       # MAYA_URL / MAYA_PROXY_KEY / MAYA_PROXY_SECRET from the project's .env
from audio.tts import MayaBackend, maya_key


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    be = MayaBackend(warm=False)
    if not be.url:
        sys.exit("MAYA_URL is not set.")
    print("server:", be.url, "| proxy auth token:", "yes" if all(be.proxy) else "no")
    t0 = time.time()
    be.warm()
    up = be.server_up()
    print(f"health: {'UP (model loaded)' if up else 'NOT READY'} after {time.time() - t0:.1f}s (a first call to a sleeping serverless GPU takes up to ~2 minutes)")
    if not up:
        print("reason:", be.last_error() or "(no error recorded)")
        sys.exit("Not reachable. 401/403 = missing or wrong MAYA_PROXY_KEY / MAYA_PROXY_SECRET; 404/5xx = the app is not running or still starting "
                 "(see the reason above); a timeout = the URL is wrong.")
    job = {"text": "नमस्ते, यह एक परीक्षण है। मुझे बहुत खुशी हो रही है!", "maya_voice": "Charvi", "maya_lang": "hi"}
    be.cache_dir = "outputs/_maya_check_cache"
    os.makedirs(be.cache_dir, exist_ok=True)
    t1 = time.time()
    out = be.synth_many([job])[0]
    if isinstance(out, Exception):
        sys.exit(f"render failed: {out}")
    import shutil
    import soundfile as sf
    os.makedirs("outputs", exist_ok=True)
    shutil.copy(out, "outputs/maya_endpoint_check.wav")
    d = sf.info(out).duration
    print(f"render OK: {d:.1f}s of speech in {time.time() - t1:.1f}s -> outputs/maya_endpoint_check.wav  (key {maya_key(job)})")


if __name__ == "__main__":
    main()
