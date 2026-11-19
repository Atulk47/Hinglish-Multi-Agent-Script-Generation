# Hinglish Multi-Agent Screenplay Generator

Turn a **Hindi story** — typed in Devanagari **or told aloud** — into a **Hinglish screenplay** (the casual Hindi-English mix, in Roman letters, that young
Indians speak) with **context-appropriate slang**, and optionally into a **multi-voice audio drama**.

The slang is not random filler: a small local editor (Qwen2.5-1.5B + LoRA, 5-bit GGUF on llama.cpp) swaps one evaluative word for a slang that fits the
sentence's *mood* ("bahut accha" → "ekdum kamaal", "bahut chinta" → "full tension"), and a chain of deterministic gates and a validator keep it off
elders, health scenes and facts.

```
Voice recording ─► ASR + pause detection ─┐
Devanagari text ──────────────────────────┴► Preprocess (transliterate, Scene Segmenter)
   ► Agent 1  narrative: events, characters, locations              ► Agent 2  character profiles (NER)
   ► Persona Agent  gender, age, relationships, emotion arc, story bible (who is present, secrets)
   ► Agent 3  neutral Hinglish dialogue (Pass 1)  ◄── Fidelity Critic (absent speakers, leaked secrets, invented events → regenerate)
   ► Agent 4  slang editor (Pass 2): eligibility gates → local LoRA editor → rule gate
   ► Validator  centred MuRIL + edit audit; unsafe edits are reverted to the neutral line
   ► Final screenplay  (+ audio drama: Edge / Studio / Maya voices, room reverb, ambience, subtitles)
```

Everything in the slang path runs **locally and deterministically** (no API call per edit). LLM calls (OpenRouter, cents per story) are used for story
analysis and for writing the neutral dialogue.

## Quick start

Python 3.10, git, ~6 GB disk for models. The project's tested environment is a conda env:

```bash
conda create -n hinglish python=3.10 -y && conda activate hinglish
pip install -r requirements.txt
pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git     # transliteration data (not in the repo)
```

Create `.env` in the project root (never commit it):

```
OPENROUTER_API_KEY=...        # story analysis + dialogue (or GROQ_API_KEY as a fallback)
```

Model files are **not in git** (size): put the slang editor in `models/slang-editor-lora-v6/` and `models/gguf/slang-editor-v6-Q5_K_M.gguf`
(or rebuild the GGUF with `python -m tools.export_gguf`). Without them the pipeline falls back to a weaker cloud editor and says so loudly.

```bash
python -m ui.app                                              # the demo UI, http://127.0.0.1:7860 (text or voice in, screenplay + audio out)
python run_pipeline.py --story data/raw/demo_hostel_project_night.txt --session demo --audio
python run_pipeline.py --voice outputs/audio_samples/sciencefair_narrated.mp3 --session v
python -m evaluation.evaluate outputs/demo_trace.json --lexicon data/felix_dataset/slang_bank_v3.csv
python -m pytest tests -q                                     # 130 offline tests (no network, no model weights needed for most)
```

Results land in `outputs/` (`<session>_script.txt`, `<session>_trace.json` with every stage, `<session>_audio/` with the drama).

## Voices

`Standard` (free Edge neural voices, default) · `Studio` (expressive actors through OpenRouter, ~$0.1/min) · `Maya` (Maya 2 Native, the most natural Indic
voices; needs a GPU). Maya works from **pre-rendered clips** (`tools/export_maya_jobs.py` → a Kaggle notebook → `tools/import_maya_zip.py`) or from a
**serverless GPU** (`deploy/modal_maya.py`, any story, pay per second) and falls back to the standard voices per drama if neither is available.

## Validation and evaluation

How the outputs are validated — and how the validation itself is validated — is documented in **[docs/EVALUATION.md](docs/EVALUATION.md)**:

* **Validator v2** (`evaluation/muril_validator.py`, `edit_audit.py`) — a text-only edit audit (locality, preservation, polarity, health / elder / respect, grammar agreement,
  rule-gate re-check) with a repair step that reverts unsafe edits; centred MuRIL as a drift diagnostic. Detects **97.8%** of corrupted edits vs **36.5%** for the old raw cosine
  while passing **86.9%** of the v5 gold edits (all 29 rejections are grammar errors in the gold labels) ([docs/VALIDATOR.md](docs/VALIDATOR.md)).
* **Validated LLM judge** (`evaluation/judge.py`) — versioned rubric, two model families (Claude Haiku 5.5, Gemini 2.5 Flash; neither is the editor), cached, 99.8% corruption detection
  with both judges, κ = 0.69 between them, calibrated against a human rater (κ = 0.75 for the "either judge accepts" rule); optional run-time second opinion `VALIDATOR_JUDGE=1` (fails open).
* **Multi-story ablation with confidence intervals** over 7 stories (`evaluation/ablation.py`, `aggregate_ablation.py`; [docs/RESULTS.md](docs/RESULTS.md)).
* **Human calibration set** (`evaluation/calibration.py create|analyze`, `calibration_small.py` for a one-rater 50-item set with an HTML form): blinded items validate the automatic metrics once, so humans never have to
  rate the system again (one rater done, the author: 94% of real edits kept, `docs/EVALUATION.md` §4a; a 125-item multi-rater set is ready).
* **Slang Director** (`agents/slang_director.py`, [docs/DIRECTOR.md](docs/DIRECTOR.md)) — a scene-level LLM agent (default on; `SLANG_DIRECTOR=0` restores the line-by-line gates) that decides where slang fits, in which mood and how
  strongly, reading the cast, relationships and emotions of the whole scene. On our 7-story ablation it is level with the hand-written gates on net useful edits; the validator stays as the independent safety check.
* Other: scene segmentation (`eval_segmenter`), voice input (`eval_voice_input`), story fidelity (`eval_fidelity`), TTS naturalness, blinded human study (`evaluation/human_eval.py`).

## Layout

| Path | What |
|---|---|
| `preprocessing/` | transliteration, Scene Segmenter |
| `agents/` | Agent 1–4, Persona Agent, Fidelity Critic, edit gate, GGUF editor, intent classifier |
| `gates/` | the three eligibility gates |
| `evaluation/` | validator, edit audit, ablation, human-eval tooling, metrics |
| `audio/` | ASR, voices (Edge / Studio / Maya), prosody, mixing, ambience |
| `orchestrator/` | LangGraph pipeline |
| `ui/` | Gradio demo |
| `tools/` | GGUF export, validator calibration, Maya batch export / import |
| `deploy/` | Modal serverless-GPU service for Maya |
| `data_prep/`, `kaggle_v*/`, `notebooks/` | training data builders and the Kaggle training notebooks (editor v1–v6) |
| `data/` | demo stories, slang banks, training data, validator calibration |

## Known limitations (honest)

* The slang editor is conservative: on the demo stories about 20–25% of lines are edited (8 of 33, 13 of 66); elder, health, formal and factual lines are deliberately left alone.
* Slang *appropriateness* beyond the rules (is it natural for this character?) is not fully automatable; the validator removes clearly unsafe edits, a human
  review (`review_script.py`) is the route to a polished script.
* The Pass-1 dialogue writer is an LLM; the Fidelity Critic reduces, not eliminates, invented events.
* Maya voices need a GPU (pre-render or serverless); real human voice input has only been tested on a few recordings; local (offline) ASR is weaker than cloud.
* Only the slang editor was trained; everything else uses pretrained models, rules and calibrated thresholds.

## Team & contributing

Group project (Atul, Dia, Anika, Anish). Work on a branch and open a pull request. **Never commit** `.env`, model weights, or virtual environments
(already in `.gitignore`).
