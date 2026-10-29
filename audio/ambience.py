"""
audio/ambience.py - sound design that makes synthetic voices feel like they are IN a place.

A dry TTS voice sounds "pasted on silence". Two cheap, procedural fixes (no downloads, no copyright):
  * ROOM    a short synthetic reverb (exponentially decaying noise impulse response) mixed in at a low level: indoors a little, outdoors almost none,
  * BED     a quiet background layer per scene - rain, a market / crowd murmur, a stadium crowd, or room tone - chosen from the scene's location and
            events, faded in and out and mixed ~30 dB under the speech.
Everything is deterministic (seeded noise), so the same story always sounds the same.
"""
from __future__ import annotations
import re
from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter

SR = 24000
INDOOR = ("home", "house", "room", "kitchen", "hospital", "school", "classroom", "office", "hall", "living", "inside", "ghar", "mandir", "temple")
OUTDOOR = ("field", "ground", "stadium", "market", "bazaar", "street", "road", "park", "maidan", "outside", "station", "village")


def _band(x: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    b, a = butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band")
    return lfilter(b, a, x)


def _rng(seed: str) -> np.random.RandomState:
    return np.random.RandomState(sum(ord(c) * (i + 7) for i, c in enumerate(seed)) % (2 ** 31))


def detect(location: str, events: str = "", text: str = "") -> Dict[str, object]:
    """-> {'reverb': wet level, 'bed': 'rain'|'crowd'|'stadium'|'market'|'room'|None}"""
    blob = f"{location} {events} {text}".lower()
    indoor = any(w in blob for w in INDOOR)
    outdoor = any(w in blob for w in OUTDOOR)
    bed = None
    if re.search(r"\b(rain|raining|storm|baarish|barish|monsoon|downpour)\b", blob):
        bed = "rain"
    elif re.search(r"\b(stadium|match|tournament|cricket|crowd|final)\b", blob) and (outdoor or "field" in blob or "stadium" in blob):
        bed = "stadium"
    elif re.search(r"\b(market|bazaar|shop|vendor)\b", blob):
        bed = "market"
    elif indoor:
        bed = "room"
    wet = 0.10 if indoor and not outdoor else (0.04 if outdoor else 0.07)
    if bed == "rain":
        wet = 0.09
    return {"reverb": wet, "bed": bed}


def reverb(x: np.ndarray, wet: float = 0.08, rt60: float = 0.38, seed: str = "ir") -> np.ndarray:
    if wet <= 0.005 or len(x) == 0:
        return x
    rng = _rng(seed)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.randn(n) * np.exp(-6.9 * t / rt60)                       # -60 dB at rt60
    ir = _band(ir, 180, 7000)
    ir[:int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))         # soft onset
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    y = fftconvolve(x, ir)[:len(x)]
    y *= np.sqrt(np.mean(x ** 2) + 1e-12) / (np.sqrt(np.mean(y ** 2)) + 1e-12)
    return ((1 - wet) * x + wet * y).astype(np.float32)


def _pink(n: int, rng) -> np.ndarray:
    w = rng.randn(n)
    b, a = butter(1, 900 / (SR / 2))
    return lfilter(b, a, w) * 3 + w * 0.25


def bed(kind: str, seconds: float, seed: str = "bed") -> np.ndarray:
    """Background layer at roughly speech-relative level 1.0 RMS=0.05 (the caller scales it)."""
    n = int(seconds * SR)
    rng = _rng(seed + kind)
    t = np.arange(n) / SR
    if kind == "rain":
        hiss = _band(rng.randn(n), 1500, 9000)
        env = 0.8 + 0.2 * np.sin(2 * np.pi * 0.13 * t + rng.rand() * 6)
        drops = np.zeros(n)
        for i in rng.randint(0, max(1, n - 400), int(seconds * 14)):             # sparse drips on a surface
            k = np.exp(-np.arange(300) / 40.0) * rng.rand() * 2.5
            drops[i:i + 300] += k[:min(300, n - i)] * rng.randn(min(300, n - i))
        x = hiss * env + _band(drops, 700, 5000) * 0.6 + _band(rng.randn(n), 80, 400) * 0.25
    elif kind in ("crowd", "stadium", "market"):
        x = np.zeros(n)
        for f0, f1 in ((200, 900), (500, 1800), (900, 3000)):                     # babble = several voice-band noises modulated at syllable rates
            m = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * (3 + 4 * rng.rand()) * t + rng.rand() * 6)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * rng.rand() * t))
            x += _band(rng.randn(n), f0, f1) * m
        if kind == "stadium":                                                  # roar swells + low rumble
            swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t + 1.3) ** 2
            x = x * swell + _band(rng.randn(n), 60, 300) * 0.9
        if kind == "market":
            x += _band(rng.randn(n), 2500, 6000) * 0.12                       # distant clatter
    else:                                                                      # room tone
        x = _pink(n, rng) * 0.6 + _band(rng.randn(n), 50, 160) * 0.5
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.05
    fade = int(0.9 * SR)
    if n > 2 * fade:
        x[:fade] *= np.linspace(0, 1, fade)
        x[-fade:] *= np.linspace(1, 0, fade)
    return x.astype(np.float32)


BED_LEVEL_DB = {"rain": -25, "stadium": -27, "crowd": -29, "market": -29, "room": -36}


def add_bed(speech: np.ndarray, kind: Optional[str], start: float, end: float, seed: str = "bed", speech_rms: float = 0.085) -> np.ndarray:
    """Mix `kind` ambience under the speech between start and end (seconds), in place. Level is set relative to the (already levelled) speech."""
    if not kind:
        return speech
    a, b = int(start * SR), min(len(speech), int((end + 0.6) * SR))
    if b - a < SR:
        return speech
    layer = bed(kind, (b - a) / SR, seed)[:b - a]                               # normalised to RMS 0.05
    layer = layer * (speech_rms * 10 ** (BED_LEVEL_DB.get(kind, -32) / 20) / 0.05)
    speech[a:a + len(layer)] += layer
    return speech
