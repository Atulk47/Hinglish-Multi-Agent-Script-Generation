"""
audio/mix.py  -  turn per-line clips into one finished recording.

  * trims leading / trailing silence of every clip (TTS adds ~200 ms) but keeps a short natural margin,
  * levels each clip to the same loudness (RMS) so no character is louder just because of the voice,
  * inserts human turn-taking gaps (longer after a question or a pause cue, shorter when the same person keeps talking, a longer
    breather between scenes) with 12 ms fades so there are no clicks,
  * returns the audio plus an exact timeline (start / end of every line) that drives subtitles and the HTML player.
"""
from __future__ import annotations
from typing import List

import numpy as np
import soundfile as sf

from audio.tts import load_mono

SR = 24000
TARGET_RMS = 0.085          # ~ -21 dBFS
FADE = int(0.012 * SR)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def trim_silence(x: np.ndarray, thresh_db: float = -42.0, keep_ms: int = 70) -> np.ndarray:
    if len(x) == 0:
        return x
    env = np.abs(x)
    win = int(0.01 * SR)
    if len(env) > win:
        env = np.convolve(env, np.ones(win) / win, mode="same")
    thr = 10 ** (thresh_db / 20)
    idx = np.where(env > thr)[0]
    if len(idx) == 0:
        return x
    keep = int(keep_ms / 1000 * SR)
    return x[max(0, idx[0] - keep): min(len(x), idx[-1] + keep)]


def level(x: np.ndarray, target_rms: float = TARGET_RMS, max_gain: float = 6.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    if rms < 1e-5:
        return x
    y = x * min(max_gain, target_rms / rms)
    return np.tanh(y * 1.1) / 1.1 * 0.98 if np.abs(y).max() > 0.95 else y        # soft limiter only when needed


def fade(x: np.ndarray) -> np.ndarray:
    n = min(FADE, len(x) // 2)
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x


def gap_ms(prev: dict | None, cur: dict) -> int:
    """Silence BEFORE `cur` (ms)."""
    if prev is None:
        return 250
    if prev["scene_id"] != cur["scene_id"]:
        return 1100
    g = 190 if prev["speaker"] == cur["speaker"] else 300
    t = prev.get("text", "").rstrip()
    if t.endswith("?"):
        g += 80
    if t.endswith(("...", "…")):
        g += 220
    return g + int(prev.get("pause_after_ms", 0)) + int(cur.get("pause_before_ms", 0))


def assemble(items: List[dict], scene_fx: dict | None = None) -> tuple:
    """items: [{'path','scene_id','speaker','text','pause_before_ms','pause_after_ms', ...}] in play order.
    scene_fx: optional {scene_id: {'reverb': wet, 'bed': 'rain'|'stadium'|'market'|'room'|None}} (see audio/ambience.py).
    -> (audio float32 @24k, timeline items with 'start' / 'end' seconds)"""
    from audio import ambience
    chunks, timeline, t, prev = [], [], 0.0, None
    for it in items:
        x, sr = load_mono(it["path"])
        x = fade(level(trim_silence(_resample(x, sr))))
        fx = (scene_fx or {}).get(it["scene_id"]) or {}
        if fx.get("reverb", 0) > 0.005:
            x = ambience.reverb(np.concatenate([x, np.zeros(int(0.22 * SR), np.float32)]), fx["reverb"], seed=it["scene_id"])
            x = x[:len(x) - int(0.22 * SR) // 2]                                           # keep half of the tail, the rest is lost in the gap
        g = gap_ms(prev, it) / 1000.0
        chunks.append(np.zeros(int(g * SR), np.float32))
        t += g
        start = t
        chunks.append(x)
        t += len(x) / SR
        timeline.append({**{k: v for k, v in it.items() if k != "path"}, "start": round(start, 3), "end": round(t, 3)})
        prev = it
    chunks.append(np.zeros(int(0.5 * SR), np.float32))
    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    if scene_fx:
        by_scene: dict = {}
        for it in timeline:
            lo, hi = by_scene.get(it["scene_id"], (it["start"], it["end"]))
            by_scene[it["scene_id"]] = (min(lo, it["start"]), max(hi, it["end"]))
        for sid, (lo, hi) in by_scene.items():
            kind = (scene_fx.get(sid) or {}).get("bed")
            if kind:
                ambience.add_bed(audio, kind, max(0.0, lo - 0.4), hi, seed=sid)
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio * (0.89 / peak)                                                # peak at about -1 dBFS
    return audio.astype(np.float32), timeline


def save_audio(audio: np.ndarray, path_noext: str) -> dict:
    out = {}
    sf.write(path_noext + ".wav", audio, SR)
    out["wav"] = path_noext + ".wav"
    try:
        sf.write(path_noext + ".mp3", audio, SR, format="MP3")
        out["mp3"] = path_noext + ".mp3"
    except Exception as e:                                                           # libsndfile without MP3 encoder
        out["mp3_error"] = str(e)[:80]
    return out


def _ts(sec: float, comma=True) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"


def to_srt(timeline: List[dict]) -> str:
    out = []
    for i, it in enumerate(timeline, 1):
        if it.get("kind") == "narration":
            label = "[Narrator]"
        else:
            label = it["speaker"].title()
        out.append(f"{i}\n{_ts(it['start'])} --> {_ts(it['end'])}\n{label}: {it['text']}\n")
    return "\n".join(out)
