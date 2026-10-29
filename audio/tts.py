"""
audio/tts.py  -  text-to-speech backends behind one small interface.

  EdgeBackend  (default)  Microsoft neural voices through the `edge-tts` package: free, no key, ~1-2 s per line, native Hindi voices
                          (hi-IN-Madhur / Swara) + Indian-English voices, per-line rate / pitch / volume. Needs internet.
  FakeBackend             offline stand-in used by the tests (a short tone per line).
  (Parler / IndicF5 backends can be added later behind the same `synth_many` signature for GPU pre-rendering.)

Every clip is cached on disk by a hash of (voice, text, prosody) so re-running a demo costs nothing and works offline.
"""
from __future__ import annotations
import asyncio
import hashlib
import os
import time
from typing import List

import numpy as np
import soundfile as sf

CACHE_DIR = "outputs/_tts_cache"


def _key(job: dict) -> str:
    raw = "|".join(str(job.get(k)) for k in ("voice", "text", "rate", "pitch", "volume"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


class EdgeBackend:
    name = "edge-tts"

    def __init__(self, concurrency: int = 4, retries: int = 3, cache_dir: str = CACHE_DIR):
        self.concurrency, self.retries, self.cache_dir = concurrency, retries, cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.stats = {"cached": 0, "synthesized": 0, "failed": 0}

    async def _one(self, sem, job):
        import edge_tts
        path = os.path.join(self.cache_dir, _key(job) + ".mp3")
        if os.path.exists(path) and os.path.getsize(path) > 800:
            self.stats["cached"] += 1
            return path
        last = None
        async with sem:
            for attempt in range(self.retries):
                try:
                    comm = edge_tts.Communicate(job["text"], job["voice"], rate=job.get("rate", "+0%"),
                                                pitch=job.get("pitch", "+0Hz"), volume=job.get("volume", "+0%"))
                    await asyncio.wait_for(comm.save(path + ".part"), timeout=45)
                    os.replace(path + ".part", path)
                    if os.path.getsize(path) < 800:
                        raise RuntimeError("empty audio returned")
                    self.stats["synthesized"] += 1
                    return path
                except Exception as e:                       # network blip / throttling
                    last = e
                    await asyncio.sleep(1.5 * (attempt + 1))
        self.stats["failed"] += 1
        raise RuntimeError(f"TTS failed for {job['voice']!r} / {job['text'][:40]!r}: {last}")

    def synth_many(self, jobs: List[dict]) -> List[str]:
        """jobs: [{'text','voice','rate','pitch','volume'}] -> list of mp3 paths (same order)."""
        async def run():
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(*[self._one(sem, j) for j in jobs])
        return list(asyncio.run(run()))


class GPTAudioBackend:
    """Expressive, instruction-following speech from an audio-capable LLM (OpenAI gpt-audio / gpt-audio-mini) through OpenRouter.

    Unlike a plain reader voice, the model is told WHO is speaking and HOW they feel and acts the line; it has 11 distinct voices, so a cast of
    nine does not collapse onto two voices. Roughly $0.0005 per line with gpt-audio-mini. Streams 24 kHz PCM, written as wav."""
    name = "gpt-audio"
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str = None, concurrency: int = 4, retries: int = 2, cache_dir: str = CACHE_DIR):
        import config
        self.model = model or getattr(config, "AUDIO_GPT_MODEL", "openai/gpt-audio-mini")
        self.concurrency, self.retries, self.cache_dir = concurrency, retries, cache_dir
        self.stats = {"cached": 0, "synthesized": 0, "failed": 0, "cost_usd": 0.0}
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, job):
        raw = "|".join(str(job.get(k)) for k in ("gpt_voice", "text", "instructions")) + "|" + self.model
        return "g" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def _call(self, job):
        import base64
        import json
        import urllib.request
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        body = {"model": self.model, "stream": True, "modalities": ["text", "audio"],
                "audio": {"voice": job["gpt_voice"], "format": "pcm16"},
                "messages": [{"role": "system", "content": job["instructions"]},
                             {"role": "user", "content": "Say exactly this line, nothing else, no greeting, no commentary:\n" + job["text"]}]}
        req = urllib.request.Request(self.URL, data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        pcm, cost = bytearray(), 0.0
        with urllib.request.urlopen(req, timeout=90) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                try:
                    ch = json.loads(line[5:])
                except Exception:
                    continue
                if ch.get("error"):
                    raise RuntimeError(str(ch["error"])[:160])
                cost = float((ch.get("usage") or {}).get("cost") or cost)
                for c in ch.get("choices", []):
                    a = (c.get("delta") or {}).get("audio")
                    if a and a.get("data"):
                        pcm += base64.b64decode(a["data"])
        if len(pcm) < 4800:
            raise RuntimeError("no audio returned")
        return np.frombuffer(bytes(pcm), dtype="<i2").astype(np.float32) / 32768, cost

    async def _one(self, sem, job):
        path = os.path.join(self.cache_dir, self._key(job) + ".wav")
        if os.path.exists(path) and os.path.getsize(path) > 2000:
            self.stats["cached"] += 1
            return path
        last = None
        async with sem:
            for attempt in range(self.retries + 1):
                try:
                    x, cost = await asyncio.to_thread(self._call, job)
                    sf.write(path + ".part.wav", x, 24000)
                    os.replace(path + ".part.wav", path)
                    self.stats["synthesized"] += 1
                    self.stats["cost_usd"] += cost
                    return path
                except Exception as e:
                    last = e
                    await asyncio.sleep(1.2 * (attempt + 1))
        self.stats["failed"] += 1
        raise RuntimeError(f"gpt-audio failed for {job['text'][:40]!r}: {last}")

    def synth_many(self, jobs: List[dict]) -> List[str]:
        async def run():
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(*[self._one(sem, j) for j in jobs], return_exceptions=True)
        return list(asyncio.run(run()))          # failed lines come back as Exception objects (the hybrid backend repairs them)


MAYA_CACHE = os.path.join(CACHE_DIR, "maya")


def maya_text(job: dict) -> str:
    """What Maya is asked to say: the (Devanagari) line, with the optional inline emotion tag in front."""
    tag = (job.get("maya_tag") or "").strip()
    return f"{tag} {job['text']}".strip() if tag else job["text"]


def maya_key(job: dict) -> str:
    """Cache key of one Maya clip. The Kaggle batch notebook names its files with the same key, so a rendered batch drops straight into the cache."""
    raw = "|".join([job.get("maya_voice") or "", job.get("maya_lang") or "hi", maya_text(job)])
    return "m" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


class MayaBackend:
    """Maya 2 Native (maya-research/maya2-native): natural Indic speech, 20 voices. It needs an NVIDIA GPU, so there are two ways to have it:

      1. PRE-RENDERED: tools/export_maya_jobs.py writes a Kaggle notebook for a story; its output zip is imported into the cache
         (tools/import_maya_zip.py). Playing a pre-rendered story needs no GPU, no network, no server.
      2. LIVE: the same HTTP API served from a GPU, URL in MAYA_URL / config.AUDIO_MAYA_URL: a serverless GPU (deploy/modal_maya.py - any
         story, pay per second, sleeps when idle) or `python -m maya3.server` on any GPU machine (notebooks/serve_maya_kaggle.py).

    Cached clips are always used first. If something is missing and no server answers, the WHOLE call fails, so the drama is read by one
    consistent set of fallback voices instead of a character changing voice halfway (HybridBackend then uses the free neural voices)."""
    name = "maya"
    COLD_START_S = 170              # a serverless GPU (deploy/modal_maya.py) needs up to ~2 minutes to boot and load the model when it was asleep
    _WARM: dict = {}                # url -> (started, thread, result): one wake-up call per server, shared by every backend instance

    def __init__(self, url: str | None = None, api_key: str | None = None, concurrency: int = 6, timeout: int = 180, cache_dir: str = MAYA_CACHE,
                 warm: bool = True):
        import config
        self.url = (url if url is not None else (os.environ.get("MAYA_URL") or getattr(config, "AUDIO_MAYA_URL", ""))).rstrip("/")
        self.api_key = api_key or os.environ.get("MAYA_API_KEY", "")
        # Modal "proxy auth" token (Settings -> Proxy Auth Tokens): sent as  Modal-Key / Modal-Secret
        self.proxy = (os.environ.get("MAYA_PROXY_KEY", ""), os.environ.get("MAYA_PROXY_SECRET", ""))
        self.concurrency, self.timeout, self.cache_dir = concurrency, timeout, cache_dir
        self.stats = {"cached": 0, "synthesized": 0, "failed": 0, "missing": 0}
        self._up = None
        os.makedirs(cache_dir, exist_ok=True)
        if warm:
            self.warm()

    def _req(self, path, data=None, timeout=None):
        import urllib.request
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        if all(self.proxy):
            h["Modal-Key"], h["Modal-Secret"] = self.proxy
        return urllib.request.urlopen(urllib.request.Request(self.url + path, data=data, headers=h), timeout=timeout or self.timeout)

    @staticmethod
    def _describe(e) -> str:
        """The real reason a call failed (HTTP status + the start of the server's answer), not just 'unreachable'."""
        body = ""
        try:
            body = e.read(300).decode("utf-8", "ignore").strip().replace("\n", " ")
        except Exception:
            pass
        code = getattr(e, "code", None)
        return (f"HTTP {code} " if code else f"{type(e).__name__}: ") + (body or str(e))[:260]

    def last_error(self) -> str:
        w = MayaBackend._WARM.get(self.url)
        return (w[2].get("err") or "") if w else ""

    def warm(self):
        """Wake a sleeping serverless server in the background (the UI calls this before the screenplay is written, so the GPU boots while
        the agents work and the voices are ready when the text is)."""
        import threading
        old = MayaBackend._WARM.get(self.url)
        if not self.url or (old and time.time() - old[0] < 200):
            return
        res = {"up": None}

        def go():
            end = time.time() + self.COLD_START_S
            while True:
                try:
                    res["up"] = self._req("/health", timeout=max(5, end - time.time())).status < 500
                    return
                except Exception as e:
                    res["up"], res["err"] = False, self._describe(e)
                    # the server answers 503 {"ready": false, "loading": true} while it loads the model: that is "starting", keep asking
                    if getattr(e, "code", None) in (502, 503, 504) and time.time() < end:
                        time.sleep(3)
                        continue
                    return
        t = threading.Thread(target=go, daemon=True)
        t.start()
        MayaBackend._WARM[self.url] = (time.time(), t, res)

    def server_up(self) -> bool:
        if self._up is None:
            self._up = False
            if self.url:
                w = MayaBackend._WARM.get(self.url)
                if w:                                           # a wake-up call is (or was) running: wait for its answer
                    w[1].join(self.COLD_START_S + 5)
                    self._up = bool(w[2]["up"])
                else:
                    try:
                        self._up = self._req("/health", timeout=8).status < 500
                    except Exception:
                        self._up = False
        return self._up

    def _render(self, job: dict, path: str):
        import json
        body = {"model": "maya3", "input": maya_text(job), "language": job.get("maya_lang") or "hi", "voice": job["maya_voice"],
                "response_format": "wav"}
        last = None
        for attempt in range(2):
            try:
                with self._req("/v1/audio/speech", json.dumps(body).encode("utf-8")) as r:
                    data = r.read()
                with open(path + ".part", "wb") as f:
                    f.write(data)
                if sf.info(path + ".part").duration < 0.2:
                    raise RuntimeError("empty audio returned")
                os.replace(path + ".part", path)
                return path
            except Exception as e:
                last = e
                time.sleep(1.0)
        raise RuntimeError(f"Maya server failed for {job['maya_voice']}/{job['text'][:30]!r}: {str(last)[:120]}")

    def synth_many(self, jobs: List[dict]):
        paths = [os.path.join(self.cache_dir, maya_key(j) + ".wav") if j.get("maya_voice") else None for j in jobs]
        have = [bool(p) and os.path.exists(p) and os.path.getsize(p) > 2000 for p in paths]
        miss = [i for i, h in enumerate(have) if not h]
        self.stats["cached"] += len(jobs) - len(miss)
        if not miss:
            return paths
        if not self.server_up():
            self.stats["missing"] += len(miss)
            print(f"  [maya] {len(jobs) - len(miss)} of {len(jobs)} lines are pre-rendered and no Maya server is reachable"
                  f"{'' if self.url else ' (MAYA_URL is not set)'}: this drama uses the standard voices.", flush=True)
            return [RuntimeError("Maya: story not pre-rendered and no server reachable") for _ in jobs]
        from concurrent.futures import ThreadPoolExecutor
        out = list(paths)

        def work(i):
            try:
                out[i] = self._render(jobs[i], paths[i])
                self.stats["synthesized"] += 1
            except Exception as e:
                out[i] = e
                self.stats["failed"] += 1
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            list(ex.map(work, miss))
        return out


class HybridBackend:
    """Expressive voice first; any line it cannot make (no credit, rate limit, network) is read by the free neural voice instead, so the
    drama always completes."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}+{fallback.name}"
        self.stats = {}

    def synth_many(self, jobs):
        out = self.primary.synth_many(jobs)
        bad = [i for i, o in enumerate(out) if isinstance(o, Exception) or not o]
        if bad:
            errs = sorted({str(out[i])[:170] for i in bad if isinstance(out[i], Exception)})[:2]
            print(f"  [tts] {len(bad)} of {len(out)} lines could not be made by {self.primary.name}; the free voice reads them. e.g. {errs}")
            alt = self.fallback.synth_many([dict(jobs[i], voice=jobs[i].get("edge_voice", jobs[i].get("voice"))) for i in bad])
            for i, a in zip(bad, alt):
                out[i] = a
        self.stats = {"primary": dict(self.primary.stats), "fallback_lines": len(bad)}
        return out


class FakeBackend:
    """Offline: writes a tone whose length is proportional to the text (tests, CI)."""
    name = "fake"

    def __init__(self, cache_dir: str = "outputs/_tts_cache_fake"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.stats = {"cached": 0, "synthesized": 0, "failed": 0}

    def synth_many(self, jobs):
        out = []
        for j in jobs:
            path = os.path.join(self.cache_dir, _key(j) + ".wav")
            if not os.path.exists(path):
                n = max(0.4, 0.06 * len(j["text"]))
                t = np.linspace(0, n, int(24000 * n), endpoint=False)
                f = 180 + (abs(hash(j["voice"])) % 120)
                sig = (0.3 * np.sin(2 * np.pi * f * t) * np.hanning(len(t))).astype(np.float32)
                sig = np.concatenate([np.zeros(2400, np.float32), sig, np.zeros(2400, np.float32)])
                sf.write(path, sig, 24000)
                self.stats["synthesized"] += 1
            out.append(path)
        return out


def load_mono(path: str):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr
