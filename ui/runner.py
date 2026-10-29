"""
ui/runner.py - runs the pipeline for the demo UI: models are loaded ONCE, each request gets a fresh run, the pipeline's console log is
streamed to the page while it works, and every artefact (trace JSON, script, subtitles, audio) is written under outputs/.
"""
from __future__ import annotations
import contextlib
import io
import json
import os
import sys
import threading
import time
from typing import Iterator, Optional

import config


class _Tee(io.TextIOBase):
    def __init__(self, real):
        self.real, self.buf, self.lock = real, [], threading.Lock()

    def write(self, s):
        with self.lock:
            self.buf.append(s)
        try:
            self.real.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass

    def text(self):
        with self.lock:
            return "".join(self.buf)


class DemoRunner:
    def __init__(self):
        self.pipe = None
        self.error = None
        self.ready = False
        self._busy = threading.Lock()

    def load(self):
        t0 = time.time()
        try:
            from orchestrator.pipeline import HinglishPipeline
            self.pipe = HinglishPipeline("ui_session", verbose=True)
            self.ready = True
        except Exception as e:                                  # surfaced in the page, not a crash
            self.error = f"{type(e).__name__}: {e}"
        return time.time() - t0

    # ------------------------------------------------------------------------------------------------------------------
    def _work(self, text: str, pauses, make_audio: bool, out: dict, studio: bool = False):
        import llm_client
        from evaluation.compare import build_comparison
        pipe = self.pipe
        t0 = time.time()
        c0 = float(llm_client.STATS["cost"])
        kind = studio if isinstance(studio, str) else ("gpt" if studio else "edge")
        be = None
        if make_audio and kind == "maya":
            from audio.synthesize import default_backend
            be = default_backend("maya")                      # created now: a sleeping serverless GPU boots while the agents write the screenplay
        final = pipe.run(text, pauses=pauses)
        sp = pipe.scratchpad
        session = time.strftime("ui_%Y%m%d_%H%M%S")
        neutral, slang = sp.read("neutral_script"), sp.read("slang_script")
        trace = {"session": session, "romanized_text": sp.read("romanized_text"), "deva_scenes": sp.read("deva_scenes"),
                 "segmentation": sp.read("segmentation"), "scene_chunks": sp.read("scene_chunks"), "event_chain": sp.read("event_chain"),
                 "character_profiles": sp.read("character_profiles"), "persona": sp.read("persona"),
                 "neutral_script": neutral, "slang_script": slang, "comparison": build_comparison(neutral, slang), "final_script": final}
        os.makedirs("outputs", exist_ok=True)
        files = []
        for path, content in ((f"outputs/{session}_script.txt", final),):
            open(path, "w", encoding="utf-8").write(content)
            files.append(path)
        tpath = f"outputs/{session}_trace.json"
        json.dump(trace, open(tpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        files.append(tpath)
        trace["seconds_pipeline"] = round(time.time() - t0, 1)
        trace["llm_cost_usd"] = round(float(llm_client.STATS["cost"]) - c0, 4)
        trace["seconds_total"] = round(time.time() - t0, 1)
        trace["files"] = [f for f in files if f]
        if make_audio:
            trace["audio_pending"] = True
        out["trace"] = dict(trace)                            # PHASE 1: the screenplay is ready - the page shows it while the audio renders
        out["phase"] = 1
        if make_audio:
            t1 = time.time()
            try:
                from audio.synthesize import synthesize_trace, default_backend
                be = be or default_backend(kind)
                res = synthesize_trace(trace, f"outputs/{session}_audio", "both", backend=be)
                trace["audio"] = {v: {"duration_s": res[v]["duration_s"], "timeline": res[v]["timeline"],
                                      "mp3": os.path.join(f"outputs/{session}_audio", res[v]["files"].get("mp3", res[v]["files"].get("wav"))),
                                      "srt": os.path.join(f"outputs/{session}_audio", f"{v}.srt")}
                                  for v in ("slang", "neutral")}
                trace["audio"]["player"] = res.get("player")
                files += [trace["audio"]["slang"]["mp3"], trace["audio"]["neutral"]["mp3"], trace["audio"]["slang"]["srt"], res.get("player")]
            except Exception as e:
                trace["audio_error"] = f"{type(e).__name__}: {e}"
            trace["seconds_audio"] = round(time.time() - t1, 1)
        trace.pop("audio_pending", None)
        trace["llm_cost_usd"] = round(float(llm_client.STATS["cost"]) - c0, 4)
        trace["seconds_total"] = round(time.time() - t0, 1)
        trace["files"] = [f for f in files if f]
        out["trace"] = trace                                   # PHASE 2: with audio
        out["phase"] = 2

    def run_stream(self, text: str, pauses=None, make_audio: bool = True, studio: bool = False) -> Iterator[tuple]:
        """Yields (log_text, trace_or_None, error_or_None) about twice a second until the run is finished."""
        if not self.ready:
            yield "", None, self.error or "The models are still loading - give it a minute."
            return
        if not self._busy.acquire(blocking=False):
            yield "", None, "Another story is being generated - please wait for it to finish."
            return
        tee = _Tee(sys.stdout)
        out: dict = {}
        err: dict = {}

        def target():
            try:
                with contextlib.redirect_stdout(tee):
                    self._work(text, pauses, make_audio, out, studio)
            except Exception as e:
                import traceback
                err["e"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-900:]}"

        th = threading.Thread(target=target, daemon=True)
        th.start()
        try:
            sent_phase = 0
            while th.is_alive():
                ph = out.get("phase", 0)
                if ph > sent_phase:                              # the text results are ready: show them now, audio follows
                    sent_phase = ph
                    yield tee.text(), out.get("trace"), None
                else:
                    yield tee.text(), None, None
                time.sleep(0.6)
            yield tee.text(), out.get("trace"), err.get("e")
        finally:
            self._busy.release()
