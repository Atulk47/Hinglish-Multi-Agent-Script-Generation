# Maya 2 Native BATCH RENDER  -  renders every line of one or more audio dramas on a free Kaggle GPU.
# Upload as a Kaggle notebook (File -> Import).  Session options:  Accelerator = GPU T4 x1  |  Internet = ON  |  FRESH session.
# Output: /kaggle/working/maya_batch.zip  ->  download, then on your laptop:   python -m tools.import_maya_zip maya_batch.zip
# After that the stories play with Maya voices with NO GPU and NO internet (pick "Maya" in the UI).
#
# %% 1. environment check
import json, os, subprocess, sys, time

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", shell=True, capture_output=True, text=True).stdout)
print("python", sys.version)

# %% 2. jobs (made by tools/export_maya_jobs.py: key = cache file name, voice, language, text)
_RAW = r'''__JOBS_JSON__'''
assert _RAW.lstrip().startswith("["), "this is the TEMPLATE - upload the generated  notebooks/maya_batch_<name>.py  instead"
JOBS = json.loads(_RAW)
print(f"{len(JOBS)} clips to render, voices: {sorted({j['voice'] for j in JOBS})}")

# %% 3. install  (the model card pins torch 2.8.0 / cu128, transformers 5.6.2, speechbrain 1.1.2)
cmd = ('pip install -q "maya3[server] @ git+https://github.com/MayaResearch/maya2-native" '
       "--extra-index-url https://download.pytorch.org/whl/cu128")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-2500:])
assert r.returncode == 0, "install failed - see the log above and send it back"
# Kaggle's torchvision / torchaudio were built for ITS torch; after the swap they crash transformers on import. Maya needs neither.
r = subprocess.run("pip uninstall -y -q torchvision torchaudio", shell=True, capture_output=True, text=True)
if "torch" in sys.modules:
    raise SystemExit("torch was already imported in this session: Run -> Restart session, then run all cells again.")

# %% 4. load the model
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU: turn on Accelerator -> GPU T4 x1 in the session options"
from maya3 import Maya3TTS

WORKERS = 4                                             # concurrent requests are batched on the GPU (~0.3 GB per extra seat)
t0 = time.time()
tts = Maya3TTS.from_pretrained("maya-research/maya2-native", max_seats=WORKERS)
print(f"loaded in {time.time() - t0:.0f}s ; GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# %% 5. render everything (a failed clip is logged and skipped, never stops the batch)
import numpy as np, soundfile as sf
from concurrent.futures import ThreadPoolExecutor

OUT = "/kaggle/working/maya_batch"
os.makedirs(OUT, exist_ok=True)
SR = 24000
log = {}


def gen(job):
    last = None
    for kw in ({"normalize": True}, {}):
        try:
            return np.asarray(tts.generate(job["text"], lang=job["lang"], voice=job["voice"], **kw), dtype=np.float32)
        except TypeError as e:                          # this release may not take `normalize` on generate()
            last = e
    raise last


def one(job):
    path = f"{OUT}/{job['key']}.wav"
    if os.path.exists(path):
        return
    s = time.time()
    try:
        wav = gen(job)
        if len(wav) < 0.2 * SR:
            raise RuntimeError("audio too short")
        sf.write(path + ".part.wav", wav, SR)
        os.replace(path + ".part.wav", path)
        log[job["key"]] = {"ok": True, "secs_audio": round(len(wav) / SR, 2), "secs_gen": round(time.time() - s, 2)}
    except Exception as e:
        log[job["key"]] = {"ok": False, "error": repr(e)[:200]}
        print("FAILED", job["key"], job.get("speaker"), repr(e)[:160])


t0 = time.time()
try:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, _ in enumerate(ex.map(one, JOBS), 1):
            if i % 10 == 0 or i == len(JOBS):
                print(f"  {i}/{len(JOBS)} done, {time.time() - t0:.0f}s")
except Exception as e:                                  # parallel mode is the less-tested path: finish whatever is left one at a time
    print("parallel rendering failed (", repr(e)[:120], ") - continuing sequentially")
    for j in JOBS:
        one(j)
for j in JOBS:                                          # second pass: anything missing is retried once, one at a time
    if not os.path.exists(f"{OUT}/{j['key']}.wav"):
        one(j)
ok = [k for k, v in log.items() if v.get("ok")]
missing = [j["key"] for j in JOBS if not os.path.exists(f"{OUT}/{j['key']}.wav")]
tot_a = sum(log[k]["secs_audio"] for k in ok)
print(f"\nrendered {len(ok)} new clips, {tot_a:.0f}s of audio in {time.time() - t0:.0f}s; missing: {len(missing)}")
json.dump({"log": log, "missing": missing, "n_jobs": len(JOBS)}, open(f"{OUT}/batch_log.json", "w"), indent=1)

# %% 6. zip for download
tts.close()
import shutil
shutil.make_archive("/kaggle/working/maya_batch", "zip", OUT)
print("ready: /kaggle/working/maya_batch.zip  ->  right panel 'Output' -> download   (missing clips:", len(missing), ")")
