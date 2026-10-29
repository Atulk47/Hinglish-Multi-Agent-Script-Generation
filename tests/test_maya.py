"""Offline tests for the Maya 2 voice path: casting, cache keys, pre-rendered cache, a (fake) live server, fallback, batch export / import."""
import io
import json
import os
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import soundfile as sf

from audio import synthesize, tts, voices
from audio.tts import FakeBackend, HybridBackend, MayaBackend, maya_key, maya_text

PROFILES = {"Ananya": {"gender": "female", "age_group": "teen", "turn_count": 9}, "Vikram": {"gender": "male", "age_group": "young_adult", "turn_count": 8},
            "Priya": {"gender": "female", "age_group": "adult", "turn_count": 5}, "Nani": {"gender": "female", "age_group": "elder", "social_role": "elder", "turn_count": 4},
            "Dukaandaar": {"gender": "male", "age_group": "adult", "turn_count": 3}}
TRACE = {"character_profiles": PROFILES, "neutral_script": [{"scene_id": "scene_01", "dialogue": "Ananya: Bhaiya, chalo.\nVikram: Haan, chal.", "events": {"location": "Home"}}],
         "slang_script": [{"scene_id": "scene_01", "dialogue": "Ananya: Bhaiya, chalo.\nVikram: Haan, chal yaar.", "events": {"location": "Home"}}]}


def _job(text="नमस्ते", voice="Charvi", lang="hi", tag=""):
    return {"text": text, "voice": "hi-IN-SwaraNeural", "maya_voice": voice, "maya_lang": lang, "maya_tag": tag}


def _wav_bytes(seconds=0.6):
    buf = io.BytesIO()
    sf.write(buf, (0.2 * np.sin(np.linspace(0, 400, int(24000 * seconds)))).astype(np.float32), 24000, format="WAV")
    return buf.getvalue()


def test_maya_casting_matches_age_gender_and_is_distinct():
    cast = voices.assign_maya_voices(voices.cast_voices(PROFILES))
    got = {n: c["maya_voice"] for n, c in cast.items()}
    assert got["Ananya"] == "Charvi" and got["Vikram"] == "Satvik" and got["Dukaandaar"] == "Siddhant"        # the voices liked in the listening test
    assert got["Nani"] == "Malini" and got["Priya"] == "Jahnavi"                                                # mature voices for the elder / the adult woman
    assert len({got[n] for n in ("Ananya", "Priya", "Nani")}) == 3
    many = {f"M{i}": {"gender": "male", "age_group": "young_adult", "turn_count": 10 - i} for i in range(6)}
    assert len({c["maya_voice"] for c in voices.assign_maya_voices(voices.cast_voices(many)).values()}) == 6


def test_maya_key_is_stable_and_sensitive_to_voice_text_lang_tag():
    a = _job()
    assert maya_key(a) == maya_key(dict(a)) and maya_key(a).startswith("m") and len(maya_key(a)) == 21
    assert len({maya_key(a), maya_key(_job(voice="Satvik")), maya_key(_job(text="alvida")), maya_key(_job(lang="en")), maya_key(_job(tag="[sad]"))}) == 5
    assert maya_text(_job(tag="[sad]")) == "[sad] नमस्ते" and maya_text(a) == "नमस्ते"
    assert maya_key({**a, "voice": "other-edge-voice"}) == maya_key(a)           # the fallback voice does not change the Maya clip


def test_plan_carries_maya_fields_narrator_in_english():
    plan, _ = synthesize.prepare_plan(TRACE, "slang", True, "roman")
    jobs = synthesize.plan_jobs(plan)
    assert jobs[0]["maya_voice"] == "Anirban" and jobs[0]["maya_lang"] == "en"
    assert {j["maya_voice"] for j in jobs[1:]} == {"Charvi", "Satvik"} and all(j["maya_lang"] == "hi" for j in jobs[1:])


def test_speaker_without_profile_gets_own_maya_voice():
    trace = json.loads(json.dumps(TRACE))
    trace["slang_script"][0]["dialogue"] += "\nDukaandaar: Beta, yeh lo.\nDoctor: Aaram karo."
    plan, _ = synthesize.prepare_plan(trace, "slang", False, "roman")
    mv = {p["speaker"]: p["maya_voice"] for p in plan}
    main = {c["maya_voice"] for c in voices.assign_maya_voices(voices.cast_voices(PROFILES)).values()}
    assert mv["Dukaandaar"] == "Siddhant" or mv["Dukaandaar"] not in main
    assert mv["Dukaandaar"] != mv["Doctor"] and mv["Doctor"] not in {mv["Ananya"], mv["Vikram"]} and mv["Dukaandaar"] not in {mv["Ananya"], mv["Vikram"]}


def test_prerendered_cache_is_used_and_partial_coverage_falls_back_for_the_whole_call(tmp_path):
    be = MayaBackend(url="", cache_dir=str(tmp_path / "m"))
    jobs = [_job("एक"), _job("दो")]
    sf.write(os.path.join(be.cache_dir, maya_key(jobs[0]) + ".wav"), np.zeros(24000, np.float32) + 0.01, 24000)
    out = be.synth_many(jobs)
    assert all(isinstance(o, RuntimeError) for o in out)                          # one line missing, no server: nothing is mixed halfway
    sf.write(os.path.join(be.cache_dir, maya_key(jobs[1]) + ".wav"), np.zeros(24000, np.float32) + 0.01, 24000)
    out = MayaBackend(url="", cache_dir=be.cache_dir).synth_many(jobs)
    assert all(isinstance(o, str) and os.path.exists(o) for o in out)
    hy = HybridBackend(MayaBackend(url="", cache_dir=str(tmp_path / "empty")), FakeBackend(str(tmp_path / "fb")))
    res = hy.synth_many([_job("तीन")])
    assert isinstance(res[0], str) and os.path.exists(res[0])                     # nothing cached, no server: the standard voice reads it


def test_live_server_renders_and_caches_and_a_dead_server_is_not_waited_for(tmp_path):
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.end_headers(); self.wfile.write(_wav_bytes())
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = MayaBackend(url=f"http://127.0.0.1:{srv.server_port}", cache_dir=str(tmp_path / "m"))
        jobs = [_job("एक"), _job("दो", voice="Satvik")]
        out = be.synth_many(jobs)
        assert all(isinstance(o, str) and sf.info(o).duration > 0.5 for o in out) and be.stats["synthesized"] == 2
        assert {b["voice"] for b in seen} == {"Charvi", "Satvik"} and seen[0]["model"] == "maya3" and seen[0]["language"] == "hi"
        before = len(seen)
        be.synth_many(jobs)
        assert len(seen) == before                                                 # second time: from the cache
    finally:
        srv.shutdown()
    dead = MayaBackend(url="http://127.0.0.1:9", cache_dir=str(tmp_path / "d"))
    assert not dead.server_up() and all(isinstance(o, Exception) for o in dead.synth_many([_job("चार")]))


def test_proxy_auth_headers_and_cold_start_wake_up(tmp_path, monkeypatch):
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("Modal-Key"), self.headers.get("Modal-Secret")))
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(("POST", self.path, self.headers.get("Modal-Key"), self.headers.get("Modal-Secret")))
            self.send_response(200); self.end_headers(); self.wfile.write(_wav_bytes())
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("MAYA_PROXY_KEY", "wk-test")
        monkeypatch.setenv("MAYA_PROXY_SECRET", "ws-test")
        MayaBackend._WARM.clear()
        be = MayaBackend(url=f"http://127.0.0.1:{srv.server_port}", cache_dir=str(tmp_path / "m"))      # the constructor already wakes the server
        assert be.server_up() and seen[0][:2] == ("GET", "/health")
        assert isinstance(be.synth_many([_job("पाँच")])[0], str)
        assert all(s[2] == "wk-test" and s[3] == "ws-test" for s in seen) and len([s for s in seen if s[0] == "GET"]) == 1     # one wake-up call, not one per line
    finally:
        srv.shutdown()
        MayaBackend._WARM.clear()


def test_batch_export_notebook_and_zip_import_roundtrip(tmp_path, monkeypatch):
    from tools import export_maya_jobs, import_maya_zip
    monkeypatch.setattr(tts, "MAYA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(import_maya_zip, "MAYA_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(export_maya_jobs.tts, "MAYA_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    jobs, st = export_maya_jobs.collect_jobs([TRACE], text_mode="roman")
    assert st["to_render"] == len(jobs) and len(jobs) == st["unique"] == 4       # narrator + Ananya's line (same in both versions) + Vikram's neutral and slang lines
    nb = export_maya_jobs.build_notebook(jobs)
    import ast
    ast.parse(nb)
    assert "__JOBS_JSON__" not in nb and all(j["key"] in nb for j in jobs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for j in jobs:
            z.writestr(f"maya_batch/{j['key']}.wav", _wav_bytes())
        z.writestr("maya_batch/batch_log.json", json.dumps({"missing": []}))
        z.writestr("maya_batch/evil.wav", b"x")                                    # names that are not cache keys are ignored
    zp = tmp_path / "b.zip"
    zp.write_bytes(buf.getvalue())
    r = import_maya_zip.import_zip(str(zp), str(tmp_path / "cache"))
    assert r["imported"] == len(jobs) and not r["bad"]
    jobs2, st2 = export_maya_jobs.collect_jobs([TRACE], text_mode="roman")
    assert jobs2 == [] and st2["already_cached"] == st2["unique"]                  # everything is covered now: nothing left to render
