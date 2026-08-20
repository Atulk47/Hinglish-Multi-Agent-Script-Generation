# Hinglish Screenplay Generator — Project Status & Work Plan

_Last updated: 19 Aug 2026_

This document explains **what the project is**, **everything that has been built so far** (and why),
**what is still left to do**, and **how the remaining work is split between the four of us
(Atul, Dia, Anika, Anish)**. It ends with how to **set up and run** the project and how we
**collaborate on GitHub**.

---

## 1. What this project does (in plain words)

You give it a short **story written in Hindi (Devanagari script)**. It gives back a
**screenplay written in "Hinglish"** — the casual Hindi-English mix, in Roman letters, that
young Indians actually speak — with **natural slang** sprinkled in ("yaar", "scene hai",
"bindaas", etc.).

It does this in stages, each handled by a small specialised piece we call an **agent**. Think of
it as an assembly line: each station does one job and passes its result to the next.

---

## 2. The pipeline (the assembly line)

```
Hindi story
   │
   ▼
[1] Preprocess     → convert Devanagari to Roman letters, split into scenes
   │
   ▼
[2] Agent 1        → read each scene, list what happens (the "event chain")
   │
   ▼
[3] Agent 2        → figure out the characters and their mood/stress
   │
   ▼
[4] Agent 3        → write plain (neutral) Hinglish dialogue for each scene   ← "Pass 1"
   │
   ▼
[5] Agent 4        → add slang to that dialogue                                ← "Pass 2"
   │
   ▼
[6] Validation     → check the slang version still means the same thing
   │
   ▼
Final screenplay
```

Everything is glued together by an **orchestrator** that runs the stations in order and shares
data between them through a common **scratchpad** (shared memory).

---

## 3. Every component — what it is, why it exists, and its status

### 3.1 Preprocessing (`preprocessing/`)
- **Transliterator** (`transliterate.py`, class `HindiTransliterator`): converts Devanagari
  Hindi (e.g. "पुस्तकालय") into Roman letters ("pustakalay"). We need this because the rest of
  the system works in Roman-script Hinglish. Uses the Indic NLP library.
  **Status: DONE and working.**
- **Chunker** (`chunker.py`, class `ProseChunker`): splits the romanized story into **scenes**
  by looking for natural breaks (blank lines, scene-change cues). Each scene is handled
  separately downstream.
  **Status: DONE and working.**

### 3.2 Shared memory (`memory/scratchpad.py`)
- **Scratchpad**: a simple shared notebook every agent can read from and write to (event chain,
  character profiles, the neutral script, the slang script, the final script). It saves these to
  disk per session so we can inspect what each stage produced.
- **PipelineState**: the "form" that travels through the pipeline holding all the in-progress
  data (raw text → romanized → scenes → dialogue → slang → final).
  **Status: DONE and working.**

### 3.3 Agent 1 — Narrative Analyst (`agents/agent1_narrative.py`)
- **What it does**: reads each scene and produces an **event chain** — a short structured list of
  what happens: which characters appear, the goal of the scene, the key events, and how the
  scene links to the previous one. Runs on a hosted LLM (Groq).
- **Why**: later stages need to know *what the scene is about* so the dialogue stays faithful to
  the story and characters are handled correctly.
  **Status: DONE and working.**

### 3.4 Agent 2 — Character Analyst (`agents/agent2_character.py`)
- **What it does**: identifies the **characters** (using a name-detection / NER model) and builds
  a small **profile** for each (their role, and a running **mood and stress level** that updates
  scene by scene). Uses two language models: an NER model for names and **MuRIL** (an
  Indian-languages model) for understanding text.
- **Why**: slang choice should depend on *who* is speaking and *how they feel* — an angry
  character and a happy one shouldn't get the same slang.
  **Status: DONE and working.** (Mood/stress currently stays fairly neutral — tuning is a future
  improvement, noted in Pending.)

### 3.5 Agent 3 — Dialogue Synthesizer (`agents/agent3_synthesizer.py`)
- **What it does**: writes **plain, neutral Hinglish dialogue** for each scene ("Pass 1") based
  on the events and character profiles. No slang yet — just clean, correct dialogue that covers
  the story. Runs on Groq.
- **Why**: separating "write correct dialogue" from "add slang" makes both jobs easier and lets
  us check that slang didn't change the meaning (compare Pass 1 vs Pass 2).
  **Status: DONE and working.**

### 3.6 The Three-Gate Controller (`gates/three_gate_controller.py`)
- **What it does**: before adding slang to a line, three quick checks decide whether that line
  *should* get slang:
  1. **Speaker gate** — is this speaker someone who'd use slang?
  2. **Pragmatic gate** — does the line's intent/tone suit slang?
  3. **Density gate** — have we already added enough slang in this scene? (so we don't overdo it)
- **Why**: real dialogue doesn't put slang in every single line; the gates keep it natural.
  **Status: DONE and working.** (Note: the gates sometimes skip too many lines — see Pending,
  "coverage".)

### 3.7 Intent Classifier (`agents/intent_classifier.py`)
- **What it does**: labels each line with an **intent/emotion** (e.g. asserting, stressed, happy).
  Primary version uses Groq; there's also a fallback design.
- **Why**: the slang picker uses the intent to choose slang that fits the line's feeling.
  **Status: DONE and working (Groq version).**

### 3.8 Agent 4 — Slang Rewriter (`agents/agent4_slang_rewriter.py`)
- **What it does**: the "Pass 2" stage. For each eligible line it **adds a fitting slang word**.
  It has two ways to do this:
  - **Trained editor (mT5)** — our own fine-tuned model that inserts slang (see 3.11).
  - **Groq fallback** — picks a slang term from our list (rarity-weighted so it doesn't repeat
    "yaar" every time) and asks Groq to place it naturally.
  It **automatically uses the Groq fallback** if the trained editor can't run, and it avoids
  repeating the same slang back-to-back.
- **Why**: this is the heart of the project — turning plain Hinglish into slangy, natural Hinglish.
  **Status: DONE and working via the Groq fallback.** (The trained editor exists but has an
  integration issue — see Pending.)

### 3.9 Validation (`evaluation/muril_validator.py`)
- **What it does**: after slang is added, checks the slang version against the plain version to
  make sure the **meaning is preserved** (using MuRIL similarity) and the **key events are still
  mentioned** (event coverage). Produces a pass / "redo" decision.
- **Why**: adding slang must not accidentally change or drop parts of the story.
  **Status: WORKING but basic** — it currently rarely rejects anything, so it needs to be made
  stricter/smarter (see Pending — this is assigned work).

### 3.10 Orchestrator (`orchestrator/pipeline.py`)
- **What it does**: wires all the stations above into one flow using a graph, runs them in order,
  passes data along, and assembles the final screenplay. It also contains a **redo loop**:
  scenes that fail validation can be sent back to have their slang re-done (up to a few attempts).
- **Why**: one place that runs the whole thing end-to-end.
  **Status: DONE and working end-to-end.** The redo loop is wired but **not yet doing useful
  work** — making it actually improve failed scenes is assigned work (see Pending).
- **Runner**: `run_pipeline.py` runs the whole pipeline on a story file and saves the outputs.

### 3.11 The trained slang editor (`notebooks/train_slang_editor_kaggle_v3.py`, `models/`)
- **What it is**: our own model (a fine-tuned **mT5-small**) trained to insert slang into a line.
  It's trained on Kaggle (free GPU) and downloaded into `models/slang-editor-full`.
- **The journey (useful for the paper)**: earlier attempts learned bad shortcuts (just copying
  the input, or inserting a meaningless placeholder). We diagnosed each, and the final **full
  fine-tune** genuinely learned to insert real slang. So it works — but its output is a bit
  repetitive, and there's a **version mismatch** that stops it running smoothly in our local
  setup right now (so the pipeline falls back to Groq). Fixing/retraining it is assigned work.
- **Why our own model**: to have a component we trained ourselves (stronger for the report than
  only calling a hosted API), and to reduce reliance on paid API calls.
  **Status: IN PROGRESS (not yet used by the pipeline).** A first version has been trained, but
  its slang is still repetitive and it does not yet run cleanly in our local setup, so the
  pipeline currently uses the Groq path instead. Getting it trained well and usable is
  assigned work (see Pending #2).

### 3.12 Data preparation (`data_prep/`)
- **`clean_felix_dataset.py`** — cleans the raw slang dataset. **DONE.**
- **`build_editor_dataset.py`** — turns the slang data into training examples for the editor
  (plain line → slang line). **DONE.**
- **`augment_slang_dataset.py`** — generates **more, more-diverse** training examples using Groq,
  with quality checks (a second model judges each example). This is how we grow the dataset so the
  editor learns richer slang. **Built and tested, but the full large run is still PENDING** — it
  needs more Groq capacity (assigned work).

### 3.13 Evaluation harness (`evaluation/metrics.py`, `evaluation/evaluate.py`)
- **What it does**: measures the quality of a generated screenplay **with numbers**, so we can
  report results and compare versions. It reads a saved run and reports:
  - **slang coverage** — how many lines got slang
  - **slang variety** — how varied the slang is (not repeating the same word)
  - **slang validity** — how much of the added slang is real/known slang
  - **meaning preserved** — similarity between plain and slang versions
  - **events kept** — key events still present
  - **code-mixing index (CMI)** — how genuinely mixed Hindi+English the output is
  - **(optional) a quality judge** — asks Groq to rate how well the slang fits each scene
- **Why**: a research project needs measurable results, not just "looks good".
  **Status: DONE and working** (produces per-scene and overall tables; can evaluate several
  stories at once).

### 3.14 Tests (`tests/`)
- Test scripts for the earlier weekly milestones (transliteration, gates, weekly flows).
  **Status: present.**

---

## 4. What is DONE (summary)

- ✅ Convert Hindi → Hinglish (Roman letters) and split into scenes.
- ✅ Understand the story (event chain) — Agent 1.
- ✅ Understand characters and mood — Agent 2.
- ✅ Write neutral Hinglish dialogue — Agent 3.
- ✅ Add slang, with gates + intent + variety control — Agent 4 (Groq path).
- ✅ Validate meaning + events — basic version.
- ✅ Full end-to-end pipeline that runs on a story and produces a screenplay.
- ✅ Data cleaning + training-example building + an augmentation script.
- ✅ Evaluation harness with clear numeric metrics.

**The system runs today, end to end, on all three test stories.**

---

## 5. What is PENDING (the work left)

1. **Data augmentation — the big run.** Generate a large, diverse batch of slang training
   examples. The script is ready; it needs enough API capacity to run at scale. This is the key
   to making the editor's slang less repetitive.
2. **Retrain and integrate the slang editor.** Using the augmented data, retrain the editor and
   fix the version mismatch so it runs cleanly in our setup (instead of falling back to Groq).
3. **Make validation stricter/smarter.** Right now the meaning-check almost always passes, so it
   doesn't really catch weak slang. It needs to actually distinguish good from bad, and the tone
   check (does the slang match the scene's emotion?) should feed the decision.
4. **Make the redo loop actually improve scenes.** When a scene fails validation, the loop should
   *fix the specific problem* (too little slang → add more; changed the meaning → use a lighter
   touch) instead of just trying again blindly. It should also catch scenes that got **no** slang.
5. **Improve coverage consistency.** On some stories many lines get no slang at all — the gates
   are too strict in places. Needs tuning so slang is added reliably across scenes.
6. **Tune character mood/stress.** Agent 2's mood currently stays mostly neutral; make it move
   with the story so slang can react to it.
7. **Full evaluation + result tables for the paper.** Run the finished system across all stories,
   including the quality-judge, and produce the final comparison tables.
8. **Write the research paper.** Method, the editor's training journey (a genuinely interesting
   result), the metrics, and the findings.

---

## 6. Who does what — work split (no overlap)

Four owners. Each **owns** their files so we don't step on each other. Dependencies are noted so
we can sequence the work.

### 👤 Atul — Slang Editor (training) & Evaluation
- **Owns:** `notebooks/train_slang_editor_kaggle_v3.py`, the `models/` editor,
  `evaluation/metrics.py`, `evaluation/evaluate.py`.
- **Tasks:**
  - **Train / retrain the slang editor** (Pending #2) on Dia's augmented data (Kaggle GPU),
    aiming for less repetitive slang, and hand the finished model to Anish to wire in.
  - Sort out the **version mismatch** so the trained model can actually run in our setup.
  - Run the **full evaluation** across all stories including the quality-judge, and produce the
    **result tables** for the paper (Pending #7).
- **Depends on:** Dia (augmented data) for the retrain.

### 👤 Dia — Data Augmentation & Dataset  ⟵ **(owns the augmentation task)**
- **Owns:** `data_prep/augment_slang_dataset.py`, `data_prep/build_editor_dataset.py`, the slang
  dataset itself.
- **Tasks:**
  - Run the **large data augmentation** (Pending #1) to produce a big, diverse batch of slang
    examples with the built-in quality checks. This needs enough API capacity — set that up first.
  - Rebuild the editor training set from the augmented data and hand it to Anish.
  - Keep the slang list / lexicon clean and growing.
- **Depends on:** nothing to start (script is ready); unblocks Anish.

### 👤 Anika — Validation & Quality Gate + Redo Loop
- **Owns:** `evaluation/muril_validator.py`, `gates/three_gate_controller.py`, the redo-loop
  logic in `orchestrator/pipeline.py` (in coordination with Anish).
- **Tasks:**
  - Make the **meaning-check actually discriminate** good vs bad slang (Pending #3), and bring the
    **tone/emotion check** into the decision.
  - Make the **redo loop fix the real problem** (Pending #4): too little slang → add more; meaning
    changed → lighter touch; no slang → force some.
  - Tune the gate thresholds with Anish so coverage is reliable.
- **Depends on:** coordinates with Anish on the shared pipeline file.

### 👤 Anish — Orchestration, Integration & Agent Tuning
- **Owns:** `orchestrator/pipeline.py`, `run_pipeline.py`, editor-loading in
  `agents/agent4_slang_rewriter.py`, keeping the whole thing running end-to-end.
- **Tasks:**
  - Keep the end-to-end pipeline healthy as others change their parts.
  - **Integrate the retrained editor** (from Atul) once it's ready — wire it into Agent 4 and the
    pipeline.
  - Improve **coverage consistency** (Pending #5) by tuning the gates, in coordination with Anika.
  - Tune Agent 2's **mood/stress** (Pending #6) so it varies with the story.
- **Depends on:** Atul (the trained editor model).

### 👥 Everyone — the paper (Pending #8)
Each person writes the section for the part they own; Atul stitches it together.

**Suggested order:** Dia starts augmentation immediately → Anika improves validation + loop in
parallel → Atul retrains the editor once Dia's data is ready and runs evaluation → Anish integrates
it and keeps everything running. Paper written throughout.

---

## 7. How to set up and run (from scratch)

Follow these steps in order the first time. It takes ~15–20 minutes, mostly waiting for
downloads. You need **Python 3.10**, **git**, and an **internet connection**.

### Step 1 — Get the code
```bash
git clone https://github.com/YOUR-USERNAME/<repo_name>
cd <folder_name>
```

### Step 2 — Create a Python environment and install the packages
Using conda (recommended):
```bash
conda create -n hinglish python=3.10 -y
conda activate hinglish
pip install -r requirements.txt
```
Or using plain venv:
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate       (Mac/Linux:  source .venv/bin/activate)
pip install -r requirements.txt
```

### Step 3 — Get the transliteration data (needed for Hindi → Roman)
This folder is large so it is **not** in our repo — download it once into the project folder:
```bash
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```
(You should now have an `indic_nlp_resources/` folder inside `hinglish-mas/`.)

### Step 4 — Add the API key
Create a file named `.env` in the project root with this one line (ask the team for the key):
```
GROQ_API_KEY=your_key_here
```
This file is private and is never committed to GitHub.

### Step 5 — Run the pipeline
```bash
python run_pipeline.py --story data/raw/test_story_1.txt --session my_run
```
- **The first run also downloads two models automatically** (the name-detector and MuRIL,
  ~1 GB total) from the internet and caches them — so the first run is slower. Later runs are fast.
- You **don't need the trained slang editor** to run: if it isn't present, the pipeline
  automatically uses the online (Groq) slang path instead. (The trained editor is large and is
  shared separately, not through GitHub.)
- When it finishes, look in the `outputs/` folder:
  - `my_run_script.txt` — the final Hinglish screenplay
  - `my_run_trace.json` — a full record of what every stage produced

### Step 6 — (Optional) Score a run with the evaluation metrics
```bash
python -m evaluation.evaluate outputs/my_run_trace.json
# add --judge to also get the quality-judge scores (this one uses the API):
python -m evaluation.evaluate outputs/my_run_trace.json --judge
```

### Quick troubleshooting
- **"GROQ_API_KEY" error** → you skipped Step 4, or the `.env` file is in the wrong folder.
- **Transliteration / indic error** → you skipped Step 3, or the folder name is wrong.
- **First run is slow / seems stuck** → it's downloading the ~1 GB of models; let it finish once.
- There are three test stories to try: `data/raw/test_story_1.txt`, `_2.txt`, `_3.txt`.

---

## 8. Working together on GitHub

- Each person works on their **own branch**, not directly on `main`:
  ```bash
  git checkout -b your-name/what-you-are-doing
  # ...make changes...
  git add -A && git commit -m "clear message"
  git push -u origin your-name/what-you-are-doing
  ```
  Then open a **Pull Request** so others can review before it merges into `main`.
- **Do not commit** the `.env` file (the API key), the big model files, or the virtual
  environment — these are already excluded in `.gitignore`.
- Because owners are split by file, merge conflicts should be rare; the one shared file
  (`orchestrator/pipeline.py`) is mainly Anish's, so coordinate there.

---

_Questions or anything unclear about a component? Ask the owner listed above, or check that
component's file — each has a short comment at the top explaining what it does._
