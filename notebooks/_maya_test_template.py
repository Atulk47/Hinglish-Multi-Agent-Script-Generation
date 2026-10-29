# Maya 2 Native listening test  -  upload as a Kaggle notebook (File -> Import), session options:
#   Accelerator: GPU T4 x1   |   Internet: ON   |   run in a FRESH session (the install replaces torch)
# Output: /kaggle/working/maya_test_out.zip  (download it and put it in  outputs/maya_test/ )
#
# %% 1. environment check
import json, os, subprocess, sys, time

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", shell=True, capture_output=True, text=True).stdout)
print("python", sys.version)

# %% 2. install  (the card pins torch 2.8.0 / cu128, transformers 5.6.2, speechbrain 1.1.2; this takes a few minutes)
cmd = ('pip install -q "maya3[server] @ git+https://github.com/MayaResearch/maya2-native" '
       "--extra-index-url https://download.pytorch.org/whl/cu128")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-2500:])
assert r.returncode == 0, "install failed - see the log above and send it back"
# Kaggle ships torchvision/torchaudio built for ITS torch; the install above swaps torch to 2.8.0, so they no longer match and transformers
# crashes on import ("operator torchvision::nms does not exist"). Maya needs neither of them.
r = subprocess.run("pip uninstall -y -q torchvision torchaudio", shell=True, capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
if "torch" in sys.modules:
    raise SystemExit("torch was already imported in this session: Run -> Restart session (or 'Factory reset'), then run all cells again.")

# %% 3. load the model (first run downloads ~5 GB of weights + the Mimi codec)
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU: turn on Accelerator -> GPU T4 x1 in the session options"
from maya3 import Maya3TTS

t0 = time.time()
tts = Maya3TTS.from_pretrained("maya-research/maya2-native", max_seats=1)
print(f"loaded in {time.time() - t0:.0f}s ; GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")
try:
    print("voices:", tts.voices)
except Exception as e:
    print("voices not listable:", e)

# %% 4. the test lines (made by tools/make_maya_test.py from the cricket/birthday demo)
LINES = json.loads(r'''__LINES_JSON__''')

# %% 5. render: for every line  A) plain Devanagari   B) with the emotion tag in front   C) the same line in a second voice (first 4 lines)
import numpy as np, soundfile as sf

OUT = "/kaggle/working/maya_test_out"
os.makedirs(OUT, exist_ok=True)
SR = 24000


def gen(text, voice):
    last = None
    for kw in ({"normalize": True}, {}):
        try:
            return np.asarray(tts.generate(text, lang="hi", voice=voice, **kw), dtype=np.float32)
        except TypeError as e:                                   # this release may not take `normalize` on generate()
            last = e
    raise last


ALT = {"Vrinda": "Charvi", "Shaunak": "Satvik", "Malini": "Vaidehi", "Nivedita": "Tanushree", "Akshat": "Siddhant"}
log, t0 = [], time.time()
for i, l in enumerate(LINES):
    jobs = [("plain", l["voice"], l["text_deva"])]
    if l["tag"]:
        jobs.append(("tagged", l["voice"], f'{l["tag"]} {l["text_deva"]}'))
    if i < 4 and l["voice"] in ALT:
        jobs.append(("altvoice", ALT[l["voice"]], l["text_deva"]))
    jobs.append(("roman", l["voice"], l["text_roman"])) if i in (4, 8) else None      # does it cope with raw romanized Hinglish?
    for kind, voice, text in jobs:
        s = time.time()
        try:
            wav = gen(text, voice)
        except Exception as e:
            print("FAILED", l["id"], kind, repr(e)[:200])
            continue
        sf.write(f"{OUT}/{l['id']}__{kind}.wav", wav, SR)
        log.append({"id": l["id"], "kind": kind, "voice": voice, "secs_audio": round(len(wav) / SR, 2), "secs_gen": round(time.time() - s, 2)})
        print(f"{l['id']:34s} {kind:9s} {voice:9s} audio {len(wav) / SR:5.1f}s  took {time.time() - s:4.1f}s")
json.dump(log, open(f"{OUT}/render_log.json", "w"), indent=1)
json.dump(LINES, open(f"{OUT}/lines.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
tot_a = sum(x["secs_audio"] for x in log)
print(f"\n{len(log)} clips, {tot_a:.0f}s of audio in {time.time() - t0:.0f}s  (real-time factor {(time.time() - t0) / max(tot_a, 1):.2f})")

# %% 6. zip for download
tts.close()
import shutil
shutil.make_archive("/kaggle/working/maya_test_out", "zip", OUT)
print("ready: /kaggle/working/maya_test_out.zip  ->  Output tab -> download")
