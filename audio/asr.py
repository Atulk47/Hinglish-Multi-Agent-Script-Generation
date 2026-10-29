"""
audio/asr.py  -  VOICE INPUT: a person tells the story aloud in Hindi (Devanagari)  ->  text + pause map for the pipeline.

Two backends behind one function:
  local  faster-whisper (Whisper small / medium / large-v3-turbo through CTranslate2, int8 on CPU). Offline after the first model download.
  cloud  an audio-capable LLM through the same gateway as the other agents (default google/gemini-2.5-flash): usually the most accurate
         Devanagari with proper punctuation; costs a fraction of a cent per minute.
  auto   cloud when a key is configured, else local; if the chosen one fails it falls back to the other.

Besides the words the module measures the SILENCES in the recording (energy based, backend independent) and aligns them to the sentence
gaps of the transcript, so the Scene Segmenter can use "the narrator paused for 1.8 s here" as evidence of a scene change.
"""
from __future__ import annotations
import base64
import io
import os
import re
import time
from typing import List, Optional, Tuple

import numpy as np

SR = 16000
CLOUD_MODEL = os.environ.get("ASR_CLOUD_MODEL", "google/gemini-2.5-flash")
LOCAL_MODEL = os.environ.get("ASR_LOCAL_MODEL", "small")         # small | medium | large-v3-turbo
PROMPT_HINT = "यह एक हिंदी कहानी है। कबीर और रोहन क्रिकेट खेलते थे। फिर उन्होंने कहा, \"हम जीतेंगे!\""   # steers Whisper to Devanagari + punctuation


# ----------------------------------------------------------------------------------------------------------------------- audio io
def load_audio(path: str) -> np.ndarray:
    """Any common format (wav, mp3, ogg, flac, m4a, webm ...) -> mono float32 at 16 kHz. Uses PyAV, so no ffmpeg binary is needed."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        x = data if data.ndim == 1 else data.mean(axis=1)
    except Exception:
        import av
        chunks = []
        with av.open(path) as c:
            st = c.streams.audio[0]
            sr = st.rate
            for fr in c.decode(st):
                a = fr.to_ndarray()
                chunks.append(a.mean(axis=0) if a.ndim > 1 else a)
        x = np.concatenate(chunks).astype(np.float32)
        if x.dtype.kind in "iu":
            x = x / np.iinfo(x.dtype).max
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x.astype(np.float32)


def silences(x: np.ndarray, min_s: float = 0.45) -> List[Tuple[float, float]]:
    """Silent stretches (start, end) in seconds. The threshold adapts to the recording's own noise floor."""
    win = int(0.025 * SR)
    n = len(x) // win
    if n < 4:
        return []
    rms = np.sqrt(np.mean(x[:n * win].reshape(n, win) ** 2, axis=1))
    floor = np.percentile(rms, 12)
    speech = np.percentile(rms, 90)
    thr = max(floor * 2.2, speech * 0.06, 1e-4)
    quiet = rms < thr
    out, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * 0.025 >= min_s:
                out.append((i * 0.025, j * 0.025))
            i = j
        else:
            i += 1
    return out


# ----------------------------------------------------------------------------------------------------------------------- backends
_LOCAL = {}


def _transcribe_local(x: np.ndarray, model_size: str) -> dict:
    from faster_whisper import WhisperModel
    if model_size not in _LOCAL:
        _LOCAL[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=max(4, (os.cpu_count() or 8) - 2))
    segs, info = _LOCAL[model_size].transcribe(x, language="hi", task="transcribe", beam_size=5, vad_filter=True,
                                               initial_prompt=PROMPT_HINT, condition_on_previous_text=False)
    segments = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segs]
    return {"text": " ".join(s["text"] for s in segments).strip(), "segments": segments, "backend": f"local:{model_size}"}


CLOUD_PROMPT = ("Transcribe this Hindi speech EXACTLY as spoken, in Devanagari script. Add normal punctuation (, । ? !) and put quotation marks "
                "around words a character speaks aloud. Keep English words as they are said in Hindi (e.g. मैच, टूर्नामेंट). Do not translate, "
                "summarise or add anything. Return ONLY the transcript text.")


def _wav_bytes(x: np.ndarray) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _transcribe_cloud(x: np.ndarray, model: str = CLOUD_MODEL) -> dict:
    import json
    import urllib.request
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    b64 = base64.b64encode(_wav_bytes(x)).decode()
    body = {"model": model, "temperature": 0.0, "max_tokens": 6000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": CLOUD_PROMPT},
                                                      {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        out = json.loads(r.read().decode())
    txt = (out["choices"][0]["message"]["content"] or "").strip()
    txt = re.sub(r"^```\w*\n?|\n?```$", "", txt).strip()
    if len(re.findall(r"[ऀ-ॿ]", txt)) < 5:
        raise RuntimeError("cloud ASR returned no Devanagari text: " + txt[:80])
    return {"text": txt, "segments": [], "backend": f"cloud:{model}", "cost_usd": float((out.get("usage") or {}).get("cost") or 0)}


# ----------------------------------------------------------------------------------------------------------------------- pauses -> sentence gaps
def align_pauses(sentences: List[str], duration: float, sil: List[Tuple[float, float]],
                 segments: Optional[List[dict]] = None) -> List[float]:
    """Seconds of silence after each sentence (len = sentences-1).

    A sentence's end is estimated from how much of the SPOKEN text has been read by then: characters are proportional to SPEAKING time, so
    the silences (which carry no text) are taken out of the clock first - otherwise ten long scene pauses would push every later estimate
    seconds off. The estimate is then snapped to the nearest silence within 1.4 s."""
    n = len(sentences)
    if n < 2 or duration <= 0:
        return [0.0] * max(0, n - 1)
    sil = sorted(sil)
    lead = sil[0][1] if sil and sil[0][0] < 0.3 else 0.0                       # leading / trailing silence is not between sentences
    tail = sil[-1][0] if sil and sil[-1][1] > duration - 0.3 else duration
    inner = [(a, b) for (a, b) in sil if a >= lead - 1e-6 and b <= tail + 1e-6 and (b - a) > 0]
    speech_total = (tail - lead) - sum(b - a for a, b in inner)
    lens = [max(1, len(s)) for s in sentences]
    total = sum(lens)

    def time_at_speech(u: float) -> float:
        """wall-clock time at which `u` seconds of speech have elapsed (skipping the silences)."""
        t, left = lead, u
        for (a, b) in inner:
            seg = a - t
            if left <= seg:
                return t + left
            left -= max(0.0, seg)
            t = b
        return t + left

    ends, acc = [], 0
    for ln in lens[:-1]:
        acc += ln
        ends.append(time_at_speech(speech_total * acc / total))
    out = []
    for t in ends:
        best, best_d = 0.0, 1.4                       # the NEAREST silence within 1.4 s (a longer one further away is a different gap)
        for (a, b) in inner:
            d = 0.0 if a <= t <= b else min(abs(t - a), abs(t - b))
            if d < best_d - 1e-9 or (abs(d - best_d) < 1e-9 and (b - a) > best):
                best_d, best = d, (b - a)
        out.append(round(best, 2))
    return out


def transcribe(path: str, backend: str = "auto", local_model: str = LOCAL_MODEL, cloud_model: str = CLOUD_MODEL) -> dict:
    """-> {'text','backend','duration','pauses' (per sentence gap), 'sentences','silences','seconds','notes'}"""
    t0 = time.time()
    x = load_audio(path)
    dur = len(x) / SR
    notes = []
    res = None
    order = {"local": ["local"], "cloud": ["cloud"], "auto": ["cloud", "local"]}.get(backend, ["cloud", "local"])
    if backend == "auto" and not os.environ.get("OPENROUTER_API_KEY"):
        order = ["local"]
    for b in order:
        try:
            res = _transcribe_cloud(x, cloud_model) if b == "cloud" else _transcribe_local(x, local_model)
            break
        except Exception as e:
            notes.append(f"{b} ASR failed: {str(e)[:110]}")
    if res is None:
        raise RuntimeError("all speech-recognition backends failed: " + " | ".join(notes))
    from preprocessing.segmenter import split_sentences
    sents = split_sentences(res["text"])
    sil = silences(x)
    pauses = None
    segs = res.get("segments") or []
    if len(segs) >= 3 and len(sents) < 0.6 * len(segs):
        # Whisper-small writes commas where a sentence ends: trust its TIMED segments as the sentences instead, and the real gaps between
        # them as the pauses.
        sents = [re.sub(r"[,;\s]+$", "", sg["text"]).strip() for sg in segs]
        sents = [t + ("" if re.search(r"[\u0964.!?\"\u201d]$", t) else "\u0964") for t in sents if t]
        res["text"] = " ".join(sents)
        pauses = [round(max(0.0, segs[i + 1]["start"] - segs[i]["end"]), 2) for i in range(len(segs) - 1)]
        res["notes"] = notes + ["sentences taken from Whisper segments (its punctuation was sparse)"]
        notes = res["notes"]
    res.update({"duration": round(dur, 1), "sentences": sents, "silences": [(round(a, 2), round(b, 2)) for a, b in sil],
                "pauses": pauses if pauses is not None else align_pauses(sents, dur, sil, segs),
                "seconds": round(time.time() - t0, 1), "notes": notes})
    return res
