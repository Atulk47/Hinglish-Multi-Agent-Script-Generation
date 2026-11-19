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
LINES = json.loads(r'''[
 {
  "id": "01_01_Ananya_excited",
  "speaker": "Ananya",
  "emotion": "excited",
  "voice": "Vrinda",
  "text_roman": "Bhaiya, Nani ka janamdin agle hafte hai na? Mujhe bahut khushi ho rahi hai!",
  "text_deva": "भैया, नानी का जन्मदिन अगले हफ्ते है ना? मुझे बहुत खुशी हो रही है!",
  "tag": "[excited]"
 },
 {
  "id": "02_01_Vikram_happy",
  "speaker": "Vikram",
  "emotion": "happy",
  "voice": "Shaunak",
  "text_roman": "Theek hai yaar, main cake laoonga. Tu decoration ki zimmedari sambhal lena.",
  "text_deva": "ठीक है यार, मैं केक लाऊंगा। तू डेकोरेशन की ज़िम्मेदारी संभाल लेना।",
  "tag": "[excited]"
 },
 {
  "id": "03_02_Vikram_frustrated",
  "speaker": "Vikram",
  "emotion": "frustrated",
  "voice": "Shaunak",
  "text_roman": "Aaj shaam bazaar mein ek bahut sundar cake dikha. Lekin uski keemat bahut zyada hai, mujhe bura lag raha hai.",
  "text_deva": "आज शाम बाज़ार में एक बहुत सुंदर केक दिखा। लेकिन उसकी कीमत बहुत ज़्यादा है, मुझे बुरा लग रहा है।",
  "tag": ""
 },
 {
  "id": "04_02_Dukaandaar_neutral",
  "speaker": "Dukaandaar",
  "emotion": "neutral",
  "voice": "Akshat",
  "text_roman": "Beta, isse sasta cake bhi mil jaayega. Woh bhi bahut accha hai.",
  "text_deva": "बेटा, इससे सस्ता केक भी मिल जाएगा। वह भी बहुत अच्छा है।",
  "tag": ""
 },
 {
  "id": "05_02_Vikram_frustrated",
  "speaker": "Vikram",
  "emotion": "frustrated",
  "voice": "Shaunak",
  "text_roman": "Nahi, Nani ke liye kuch achha hona chahiye. Mujhe is cake ki problem nahi, bas paise ki full tension hai.",
  "text_deva": "नहीं, नानी के लिए कुछ अच्छा होना चाहिए। मुझे इस केक की प्रॉब्लम नहीं, बस पैसे की फुल टेंशन है।",
  "tag": ""
 },
 {
  "id": "06_03_Ananya_scared",
  "speaker": "Ananya",
  "emotion": "scared",
  "voice": "Vrinda",
  "text_roman": "Bhaiya, baarish bahut tez hai aur bijli bhi chali gayi! Sab saman bheeg jayega, mujhe bahut darr lag raha hai.",
  "text_deva": "भैया, बारिश बहुत तेज़ है और बिजली भी चली गई! सब सामान भीग जाएगा, मुझे बहुत डर लग रहा है।",
  "tag": "[whispers]"
 },
 {
  "id": "07_03_Vikram_determined",
  "speaker": "Vikram",
  "emotion": "determined",
  "voice": "Shaunak",
  "text_roman": "Tu chinta mat kar, Ananya. Hum dono milkar sab kuch theek kar lenge.",
  "text_deva": "तू चिंता मत कर, अनन्या। हम दोनों मिलकर सब कुछ ठीक कर लेंगे।",
  "tag": ""
 },
 {
  "id": "08_03_Ananya_scared",
  "speaker": "Ananya",
  "emotion": "scared",
  "voice": "Vrinda",
  "text_roman": "Theek hai bhaiya. Mujhe ab bhi darr lag raha hai, par aap saath ho toh himmat aa rahi hai.",
  "text_deva": "ठीक है भैया। मुझे अब भी डर लग रहा है, पर आप साथ हो तो हिम्मत आ रही है।",
  "tag": "[whispers]"
 },
 {
  "id": "09_04_Priya_amused",
  "speaker": "Priya",
  "emotion": "amused",
  "voice": "Nivedita",
  "text_roman": "Main aa gayi hoon, madad karne! Gubbaare darwaze ke paas laga do.",
  "text_deva": "मैं आ गयी हूँ, मदद करने! गुब्बारे दरवाज़े के पास लगा दो।",
  "tag": "[laughs]"
 },
 {
  "id": "10_04_Ananya_happy",
  "speaker": "Ananya",
  "emotion": "happy",
  "voice": "Vrinda",
  "text_roman": "Haan, mera dil bahut zor se dhadak raha hai, itni khushi ho rahi hai mujhe!",
  "text_deva": "हाँ, मेरा दिल बहुत ज़ोर से धड़क रहा है, इतनी खुशी हो रही है मुझे!",
  "tag": "[excited]"
 },
 {
  "id": "11_05_Nani_affectionate",
  "speaker": "Nani",
  "emotion": "affectionate",
  "voice": "Malini",
  "text_roman": "Aaj bas itna ho ki tum sab mere saath ho, beta.",
  "text_deva": "आज बस इतना हो कि तुम सब मेरे साथ हो, बेटा।",
  "tag": ""
 },
 {
  "id": "12_05_Nani_affectionate",
  "speaker": "Nani",
  "emotion": "affectionate",
  "voice": "Malini",
  "text_roman": "Mere aankhon mein aansu aa gaye hain, aur main tum teeno ko gale lagati hoon. Yeh meri zindagi ka sabse achha tohfa hai. Poora ghar hansi aur khushi se bhar gaya hai.",
  "text_deva": "मेरी आँखों में आँसू आ गए हैं, और मैं तुम तीनों को गले लगाती हूँ। यह मेरी ज़िंदगी का सबसे अच्छा तोहफ़ा है। पूरा घर हँसी और ख़ुशी से भर गया है।",
  "tag": ""
 },
 {
  "id": "13_05_Vikram_happy",
  "speaker": "Vikram",
  "emotion": "happy",
  "voice": "Shaunak",
  "text_roman": "Nani, aap darwaza kholte hi ruk gayi, kya hua?",
  "text_deva": "नानी, आप दरवाज़ा खोलते ही रुक गयीं, क्या हुआ?",
  "tag": "[excited]"
 }
]''')

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
