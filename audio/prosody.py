"""
audio/prosody.py  -  emotion (+ intensity + punctuation) -> speaking style.

Each emotion has a (valence, arousal) position (agents/agent2b_persona.EMOTIONS). Arousal drives speaking RATE and VOLUME, valence and
arousal together drive PITCH: excited / scared / angry speech is faster and higher, sad speech slower, softer and lower, proud speech
is steady and a little louder. Intensity (1-3) scales the effect. Punctuation adds small local cues ("!" lifts pitch and volume,
"?" lifts pitch, "..." slows down and adds a pause). All values are CLAMPED so no line ever sounds cartoonish.
"""
from __future__ import annotations
import re

from agents.agent2b_persona import EMOTIONS

LIMITS = {"rate_pct": (-16, 18), "pitch_hz": (-12, 12), "volume_pct": (-16, 12)}
# emotion-specific fine tuning on top of the valence/arousal rule: (rate, pitch, volume) offsets at intensity 2
TWEAK = {"proud": (-3, 1, 5), "angry": (2, -2, 8), "sad": (-4, -3, -6), "scared": (3, 5, -2), "anxious": (3, 2, -2),
         "ashamed": (-5, -4, -8), "affectionate": (-4, 1, -3), "determined": (2, -1, 4), "relieved": (-3, 0, -2),
         "surprised": (3, 6, 3), "amused": (2, 3, 0)}


def _clamp(key, v):
    lo, hi = LIMITS[key]
    return int(max(lo, min(hi, round(v))))


def emotion_prosody(emotion: str, intensity: int = 2, text: str = "") -> dict:
    """-> {'rate_pct','pitch_hz','volume_pct','pause_before_ms','pause_after_ms'} offsets relative to a character's base voice."""
    val, aro = EMOTIONS.get(emotion, EMOTIONS["neutral"])
    k = {1: 0.55, 2: 0.8, 3: 1.1}.get(int(intensity), 0.8)       # softer than a theatre voice: a screenplay read, not a cartoon
    rate = (aro - 0.35) * 20 * k
    pitch = (val * 4 + (aro - 0.35) * 8) * k
    vol = (aro - 0.35) * 12 * k
    tr, tp, tv = TWEAK.get(emotion, (0, 0, 0))
    rate, pitch, vol = rate + tr * k, pitch + tp * k, vol + tv * k
    text = text or ""
    pause_before, pause_after = 0, 0
    if text.rstrip().endswith("!"):
        pitch, vol = pitch + 2, vol + 4
    if "?" in text:
        pitch += 2
    if "..." in text or "…" in text:
        rate, pause_after = rate - 4, 350
    if emotion in ("sad", "ashamed", "relieved"):
        pause_before = 120
    return {"rate_pct": _clamp("rate_pct", rate), "pitch_hz": _clamp("pitch_hz", pitch),
            "volume_pct": _clamp("volume_pct", vol), "pause_before_ms": pause_before, "pause_after_ms": pause_after}


def combine(base: dict, delta: dict) -> dict:
    """Character's base voice settings + the per-line emotion offsets -> final settings (clamped, formatted for the TTS)."""
    out = {}
    for key in ("rate_pct", "pitch_hz", "volume_pct"):
        lo, hi = {"rate_pct": (-28, 24), "pitch_hz": (-24, 24), "volume_pct": (-30, 20)}[key]      # combined with the character's base voice
        out[key] = int(max(lo, min(hi, base.get(key, 0) + delta.get(key, 0))))
    out["pause_before_ms"] = delta.get("pause_before_ms", 0)
    out["pause_after_ms"] = delta.get("pause_after_ms", 0)
    return out


def edge_args(settings: dict) -> dict:
    """Format for edge-tts (strict regexes: '+0%', '-5%', '+4Hz')."""
    return {"rate": f"{settings['rate_pct']:+d}%", "pitch": f"{settings['pitch_hz']:+d}Hz", "volume": f"{settings['volume_pct']:+d}%"}
