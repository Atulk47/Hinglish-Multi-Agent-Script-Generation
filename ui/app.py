"""
ui/app.py - the demo UI.      python -m ui.app          (then open http://127.0.0.1:7860)

INPUT   a Hindi story as Devanagari TEXT (type / paste / upload / pick a sample)  or  a VOICE recording (microphone / file / sample narration)
        -> speech recognition -> editable transcript (+ the narrator's pauses, used to find scene breaks)
OUTPUT  live pipeline log, scenes, characters (gender / role / relationships / emotion arcs), the screenplay with the slang edits
        highlighted next to the neutral version, and the multi-voice audio drama (with and without slang) plus downloads.
"""
from __future__ import annotations
import os
import sys
import threading
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv

load_dotenv()
import gradio as gr

from ui import render
from ui.runner import DemoRunner

RUNNER = DemoRunner()
SAMPLES = [("Cricket tournament - 11 scenes, 9 characters", "data/raw/demo_cricket_tournament.txt"),
           ("Science fair - 5 scenes", "data/raw/demo_science_fair.txt"),
           ("Hostel project night - 5 scenes, a secret", "data/raw/demo_hostel_project_night.txt"),
           ("Test story 1", "data/raw/test_story_1.txt"), ("Test story 2", "data/raw/test_story_2.txt"),
           ("Test story 3", "data/raw/test_story_3.txt")]
VOICE_SAMPLES = [("Science fair, narrated (2 min)", "outputs/audio_samples/sciencefair_narrated.mp3"),
                 ("Cricket tournament, narrated (6 min)", "outputs/audio_samples/cricket_narrated.mp3")]


def read_text(path):
    return open(path, encoding="utf-8").read() if path and os.path.exists(path) else ""


def sentences_of(text):
    from preprocessing.segmenter import split_sentences
    return split_sentences(text)


# ---------------------------------------------------------------------------------------------------------------- handlers
def on_sample(path):
    return read_text(path) if path else gr.update(), None, ""


def on_file(f):
    p = getattr(f, "name", f)
    return read_text(p) if p else gr.update(), None, ""


def on_voice_sample(path):
    return path if path and os.path.exists(path) else gr.update()


def on_transcribe(audio_path, backend, local_model):
    if not audio_path:
        return gr.update(), None, "", "Record or upload a voice file first."
    from audio.asr import transcribe
    try:
        r = transcribe(audio_path, backend=backend, local_model=local_model)
    except Exception as e:
        return gr.update(), None, "", f"Transcription failed: {e}"
    long_p = sum(1 for p in r["pauses"] if p >= 1.1)
    status = (f"**Transcribed** with `{r['backend']}` in {r['seconds']}s | audio {r['duration']}s | {len(r['sentences'])} sentences | "
              f"{long_p} long pauses (likely scene breaks)" + (f" | notes: {'; '.join(r['notes'])}" if r["notes"] else "")
              + "  \nCheck the text below, fix any mistakes, then press **Generate**.")
    return r["text"], {"text": r["text"], "pauses": r["pauses"]}, status, ""


def _pauses_for(text, state):
    if state and state.get("text") == text and len(state.get("pauses") or []) == len(sentences_of(text)) - 1:
        return state["pauses"]
    return None


def on_preview(text, state):
    if not text.strip():
        return "Paste or record a story first."
    from preprocessing.segmenter import SceneSegmenter
    seg = SceneSegmenter().segment(text, pauses=_pauses_for(text, state))
    out = [f"**{len(seg.scenes)} scenes** from {len(seg.sentences)} sentences &middot; method `{seg.method}`"
           + (f" &middot; {seg.notes}" if seg.notes else "") + "\n"]
    for i, sc in enumerate(seg.scenes, 1):
        out.append(f"- **Scene {i}** ({len(sentences_of(sc))} sentences): {sc[:120]}...")
    return "\n".join(out)


# label shown in the page -> voice engine of audio.synthesize.default_backend
ENGINES = {"Standard (free)": "edge", "Studio (expressive actors, ~$0.1/min)": "gpt",
           "Maya (natural Indic voices; pre-rendered stories or a GPU server, else Standard)": "maya"}


def on_generate(text, state, make_audio, studio):
    studio = ENGINES.get(studio, studio if isinstance(studio, str) else ("gpt" if studio else "edge"))
    nothing = gr.update()
    if not text or len(text.strip()) < 20:
        yield ("Please provide a story first.", "", nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
        return
    pauses = _pauses_for(text, state)
    head = "Generating... (the first scene appears in the log within a few seconds)"
    for log, trace, err in RUNNER.run_stream(text, pauses, make_audio, studio):
        if err:
            yield (f"**Error:** {err}", log, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
            return
        if trace is None:
            yield (head, log, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
            continue
        audio = trace.get("audio") or {}
        yield (render.summary_html(trace), log, render.scenes_html(trace), render.profiles_df(trace), render.arcs_html(trace),
               render.comparison_html(trace),
               (audio.get("slang") or {}).get("mp3"), (audio.get("neutral") or {}).get("mp3"), render.timeline_df(trace),
               trace.get("files"), trace.get("final_script"))


# ---------------------------------------------------------------------------------------------------------------- layout
def build():
    with gr.Blocks(title="Hinglish Multi-Agent Screenplay Studio") as demo:
        gr.HTML(render.CSS + "<h2 style='margin:4px 0'>Hinglish Multi-Agent Screenplay Studio</h2>"
                "<div style='opacity:.7;margin-bottom:8px'>Hindi story (text or voice) &rarr; scenes &rarr; characters &rarr; neutral dialogue &rarr; "
                "context-aware slang &rarr; multi-voice audio drama</div>")
        status_md = gr.Markdown("")
        pauses_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Text (Devanagari)"):
                        sample_dd = gr.Dropdown(choices=SAMPLES, label="Sample story", value=None)
                        file_in = gr.File(label="...or upload a .txt story", file_types=[".txt"])
                    with gr.Tab("Voice"):
                        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Tell the story in Hindi")
                        vs_dd = gr.Dropdown(choices=VOICE_SAMPLES, label="...or use a sample narration", value=None)
                        with gr.Row():
                            backend = gr.Radio(["auto", "cloud", "local"], value="auto", label="Speech recognition")
                            local_model = gr.Dropdown(["small", "medium", "large-v3-turbo"], value="small", label="Local model")
                        tr_btn = gr.Button("Transcribe", variant="secondary")
                text_in = gr.Textbox(label="Story text (editable - the voice transcript lands here)", lines=11,
                                     placeholder="कबीर अपने स्कूल की क्रिकेट टीम का कप्तान था ...")
                with gr.Row():
                    prev_btn = gr.Button("Preview scenes", variant="secondary")
                    make_audio = gr.Checkbox(value=True, label="Render audio drama")
                    studio = gr.Radio(list(ENGINES), value=list(ENGINES)[0], label="Voices")
                preview_md = gr.Markdown("")
                gen_btn = gr.Button("Generate screenplay", variant="primary", size="lg")
            with gr.Column(scale=7):
                summary = gr.HTML("<div style='opacity:.6;padding:12px'>Results appear here.</div>")
                with gr.Tabs():
                    with gr.Tab("Screenplay"):
                        comparison = gr.HTML()
                    with gr.Tab("Scenes"):
                        scenes = gr.HTML()
                    with gr.Tab("Characters"):
                        profiles = gr.Dataframe(interactive=False, wrap=True)
                        arcs = gr.HTML()
                    with gr.Tab("Audio drama"):
                        gr.Markdown("Two renderings of the same screenplay: with the slang edits, and the neutral first pass.")
                        a_slang = gr.Audio(label="With slang (Pass 2)", type="filepath", interactive=False)
                        a_neutral = gr.Audio(label="Neutral (Pass 1)", type="filepath", interactive=False)
                        timeline = gr.Dataframe(label="Timeline", interactive=False, wrap=True)
                    with gr.Tab("Pipeline log"):
                        log = gr.Textbox(lines=22, max_lines=22, autoscroll=True, interactive=False, show_label=False)
                    with gr.Tab("Downloads"):
                        files = gr.File(file_count="multiple", interactive=False)
                        final_txt = gr.Textbox(label="Final screenplay (plain text)", lines=14, interactive=False)

        sample_dd.change(on_sample, sample_dd, [text_in, pauses_state, preview_md])
        file_in.upload(on_file, file_in, [text_in, pauses_state, preview_md])
        vs_dd.change(on_voice_sample, vs_dd, audio_in)
        tr_btn.click(on_transcribe, [audio_in, backend, local_model], [text_in, pauses_state, status_md, preview_md])
        prev_btn.click(on_preview, [text_in, pauses_state], preview_md)
        gen_btn.click(on_generate, [text_in, pauses_state, make_audio, studio],
                      [summary, log, scenes, profiles, arcs, comparison, a_slang, a_neutral, timeline, files, final_txt])
    return demo


def main():
    port = int(os.environ.get("PORT", "7860"))
    demo = build()
    print("Loading models (NER, MuRIL, slang editor) ... the page is usable immediately; Generate waits until they are ready.", flush=True)
    threading.Thread(target=lambda: print(f"Models ready in {RUNNER.load():.0f}s" + (f" (ERROR: {RUNNER.error})" if RUNNER.error else ""),
                                          flush=True), daemon=True).start()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=port, inbrowser=False,
                allowed_paths=[os.path.abspath("outputs"), os.path.abspath("data")])


if __name__ == "__main__":
    main()
