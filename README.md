# Hinglish Screenplay Generator

Turn a short **Hindi story (Devanagari)** into a **Hinglish screenplay** — the casual
Hindi-English mix, in Roman letters, that young Indians actually speak — with natural **slang**
added ("yaar", "scene hai", "bindaas", …).

It works as a small **multi-agent pipeline**: each stage does one job and passes its result on.

```
Hindi story
   → Preprocess   (Devanagari → Roman letters, split into scenes)
   → Agent 1      (read the story: characters, goals, key events)
   → Agent 2      (build character profiles + mood/stress)
   → Agent 3      (write plain "neutral" Hinglish dialogue — Pass 1)
   → Agent 4      (add slang to the dialogue — Pass 2)
   → Validation   (check the slang version keeps the same meaning)
   → Final screenplay
```

> For a full explanation of every component, current status, pending work, and how the team
> splits the work, see **[PROJECT_STATUS.md](PROJECT_STATUS.md)**.

---

## Quick start

You need **Python 3.10**, **git**, and an **internet connection**.

**1. Clone the repo**
```bash
git clone https://github.com/YOUR-USERNAME/hinglish-mas.git
cd hinglish-mas
```

**2. Create an environment and install packages**
```bash
conda create -n hinglish python=3.10 -y
conda activate hinglish
pip install -r requirements.txt
```
(Or with venv: `python -m venv .venv` → activate it → `pip install -r requirements.txt`.)

**3. Get the transliteration data** (large, so it's not in the repo — download once into the project)
```bash
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```

**4. Add your API key** — create a file called `.env` in the project root (ask the team for the key):
```
GROQ_API_KEY=your_key_here
```

**5. Run it**
```bash
python run_pipeline.py --story data/raw/test_story_1.txt --session my_run
```
- The **first run downloads ~1 GB of models** (a name-detector + MuRIL) automatically and caches
  them, so it's slower once. Later runs are fast.
- You **don't need the trained slang editor** to run — if it's absent, the pipeline uses the
  online (Groq) slang path automatically.
- Results appear in `outputs/`:
  - `my_run_script.txt` — the final Hinglish screenplay
  - `my_run_trace.json` — a full record of every stage

**6. (Optional) Score a run**
```bash
python -m evaluation.evaluate outputs/my_run_trace.json          # metrics only
python -m evaluation.evaluate outputs/my_run_trace.json --judge  # + quality-judge (uses API)
```

---

## Project layout

| Folder / file | What's in it |
|---|---|
| `preprocessing/` | Devanagari → Roman transliteration, scene splitting |
| `agents/` | Agent 1 (narrative), Agent 2 (characters), Agent 3 (dialogue), Agent 4 (slang), intent classifier |
| `gates/` | The three checks that decide whether a line gets slang |
| `memory/` | Shared scratchpad + the data that flows through the pipeline |
| `orchestrator/` | Wires all the stages together and runs them end-to-end |
| `evaluation/` | Meaning/event validation + the numeric evaluation metrics |
| `data_prep/` | Cleaning + building + augmenting the slang training data |
| `notebooks/` | Training scripts for the slang editor (run on Kaggle) |
| `data/` | Test stories and the slang datasets |
| `run_pipeline.py` | Runs the whole pipeline on a story |
| `config.py` | Model names, paths, and thresholds |

---

## Common issues
- **`GROQ_API_KEY` error** → the `.env` file is missing or in the wrong folder (see step 4).
- **Transliteration / indic error** → you skipped step 3 (the `indic_nlp_resources` folder).
- **First run seems stuck** → it's downloading the ~1 GB of models; let it finish once.

There are three test stories: `data/raw/test_story_1.txt`, `_2.txt`, `_3.txt`.

---

## Team & contributing
This is a group project (Atul, Dia, Anika, Anish). Work areas and ownership are in
**[PROJECT_STATUS.md](PROJECT_STATUS.md)**. Work on your own branch and open a Pull Request:
```bash
git checkout -b your-name/what-you-are-doing
# ...changes...
git add -A && git commit -m "clear message"
git push -u origin your-name/what-you-are-doing
```
**Never commit** the `.env` file, the large model files, or your virtual environment —
these are already excluded in `.gitignore`.
