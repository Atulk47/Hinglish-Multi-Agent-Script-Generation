# Maya 2 Native LIVE SERVER on a free Kaggle GPU  (EXPERIMENTAL - not tested end to end: the batch notebook is the safe route for a demo).
# Session options:  Accelerator = GPU T4 x1  |  Internet = ON  |  FRESH session.   Keep the tab open: free sessions stop when idle / after hours.
# It prints a public URL.  On your laptop:   set MAYA_URL=<that url>   then pick "Maya" in the UI (or  --tts maya ).
# Lines the server cannot make, or a URL that has expired, fall back to the standard voices automatically.
# SECURITY: the server has no authentication. The tunnel URL is random and public to anyone who has it - stop the session after the demo.
#
# %% 1. install  (same as the batch notebook)
import os, re, subprocess, sys, time, urllib.request

cmd = ('pip install -q "maya3[server] @ git+https://github.com/MayaResearch/maya2-native" '
       "--extra-index-url https://download.pytorch.org/whl/cu128")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-1500:])
assert r.returncode == 0, "install failed"
subprocess.run("pip uninstall -y -q torchvision torchaudio", shell=True)
if "torch" in sys.modules:
    raise SystemExit("torch was already imported: Run -> Restart session, then run all cells.")

# %% 2. start the model server (loads ~5 GB, then listens on :8080)
srv = subprocess.Popen([sys.executable, "-m", "maya3.server", "--ckpt", "maya-research/maya2-native", "--host", "0.0.0.0", "--port", "8080",
                        "--max-seats", "8"], stdout=open("/kaggle/working/maya_server.log", "w"), stderr=subprocess.STDOUT)
for i in range(180):
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3)
        print(f"server up after ~{i * 5}s")
        break
    except Exception:
        time.sleep(5)
else:
    print(open("/kaggle/working/maya_server.log").read()[-3000:])
    raise SystemExit("the server did not start - send the log above back")

# %% 3. public URL through a Cloudflare quick tunnel (no account needed)
subprocess.run("curl -sL -o /kaggle/working/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
               "&& chmod +x /kaggle/working/cloudflared", shell=True, check=True)
tun = subprocess.Popen(["/kaggle/working/cloudflared", "tunnel", "--url", "http://127.0.0.1:8080", "--no-autoupdate"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
t0 = time.time()
while time.time() - t0 < 90 and not url:
    line = tun.stdout.readline()
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line or "")
    url = m.group(0) if m else None
print("\n" + "=" * 70 + f"\nMAYA_URL = {url}\n" + "=" * 70)
print("On your laptop (PowerShell):   $env:MAYA_URL=\"" + str(url) + "\"   then start the UI and pick the Maya voices.")

# %% 4. keep the session alive while you demo (stop this cell, or the session, when finished)
while True:
    time.sleep(60)
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5)
    except Exception as e:
        print("server health check failed:", repr(e)[:100])
