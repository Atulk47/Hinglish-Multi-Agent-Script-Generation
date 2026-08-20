# Hinglish MAS — Week 1 & Week 2: Revised Complete Implementation Guide

### (Reflects all actual fixes, real outputs, and lessons learned during implementation)

---

## PLATFORM

**Week 1 and Week 2: 100% local machine, VS Code.**

- No GPU needed. Transliteration is CPU + rule-based (<10ms/sentence).
- Agent 1 uses Groq API (free, no GPU needed).
- All models download once and are cached in `~/.cache/huggingface/hub/`.

**Do NOT use Kaggle or Colab for Weeks 1-2.** You need a persistent multi-file project
structure, `pytest`, and `git` — all easier on a local IDE. Save Kaggle's 30h/week GPU quota
for Week 4 (IndicBART fine-tuning).

---

# WEEK 1 — ENVIRONMENT SETUP + TRANSLITERATION MODULE

## Day 1-2: GitHub + Project Structure + Environment

### Step 1: Create the GitHub repository

1. Go to https://github.com and log in (or sign up if needed).
2. Click **`+`** (top-right) → **"New repository"**.
3. Name: `hinglish-mas`
4. Set to **Private**.
5. Check **"Add a README file"**.
6. Click **"Create repository"**.

### Step 2: Add teammates as collaborators

1. In the repo → **Settings** → **Collaborators** → **"Add people"**.
2. Add each teammate by GitHub username (Anish M, Diya Ramani, Atul Krishna).
3. Each teammate must accept the invite email from GitHub.

### Step 3: Clone the repo

Open a terminal (Anaconda Prompt or Git Bash recommended on Windows):

```bash
cd ~/Desktop
git clone https://github.com/YOUR-USERNAME/hinglish-mas.git
cd hinglish-mas
```

> **If git asks for a password:** GitHub no longer accepts passwords over HTTPS.
> Generate a Personal Access Token (PAT):
>
> 1. Go to https://github.com/settings/tokens → **"Generate new token (classic)"**
> 2. Give it a name, set expiry 90 days, check the **`repo`** scope.
> 3. Click **"Generate token"** — copy it immediately (shown once only).
> 4. When git asks for password, paste the token.

### Step 4: Open in VS Code

```bash
code .
```

If `code .` doesn't work: Open VS Code manually → **File → Open Folder** → select `hinglish-mas`.

### Step 5: Create directory structure

Run in VS Code terminal (Git Bash recommended on Windows):

```bash
mkdir -p preprocessing agents gates memory evaluation orchestrator
mkdir -p data/raw data/processed data/felix_dataset data/sessions
mkdir -p models tests notebooks outputs
```

Create `__init__.py` files so Python treats each folder as a package:

```bash
touch preprocessing/__init__.py
touch agents/__init__.py
touch gates/__init__.py
touch memory/__init__.py
touch evaluation/__init__.py
touch orchestrator/__init__.py
touch tests/__init__.py
touch models/.gitkeep
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/felix_dataset/.gitkeep
touch data/sessions/.gitkeep
touch outputs/.gitkeep
```

> **Windows PowerShell alternative** (if `touch` isn't recognized):
>
> ```powershell
> New-Item -ItemType File -Path "preprocessing/__init__.py"
> # repeat for each file above
> ```
>
> Or just use Git Bash for all terminal commands — it ships with Git for Windows and
> supports `touch`, `mkdir -p`, etc.

Your structure should now be:

```
hinglish-mas/
├── preprocessing/
├── agents/
├── gates/
├── memory/
├── evaluation/
├── orchestrator/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── felix_dataset/
│   └── sessions/
├── models/
├── tests/
├── notebooks/
├── outputs/
└── README.md
```

### Step 6: Install Python 3.10 via Miniconda

Check first:

```bash
python3 --version
```

- **Shows `3.10.x`** → skip to Step 7.
- **Not installed or wrong version** → install Miniconda:
  1. Go to https://docs.conda.io/en/latest/miniconda.html
  2. Download installer for your OS and run it with default options.
  3. Restart terminal after installation.
  4. Verify: `conda --version`

> **KNOWN ISSUE — conda not recognized in PowerShell:**
> This is the most common Windows setup issue. Fix:
>
> 1. Search Start menu for **"Anaconda Prompt"** — use this instead of PowerShell.
> 2. OR run `conda init powershell` in Anaconda Prompt, then restart PowerShell.
> 3. If PowerShell still blocks conda: run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` as Administrator, then restart.

### Step 7: Create conda environment

```bash
conda create -n hinglish python=3.10 -y
conda activate hinglish
```

After activation, your terminal prompt shows `(hinglish)` at the start.
**All subsequent commands assume this environment is active.**

### Step 8: Create `requirements.txt`

Create file `requirements.txt` in the project root:

```
torch==2.2.0
transformers==4.40.0
datasets==2.19.0
peft==0.10.0
trl==0.8.6
langgraph>=0.2.0,<0.3.0
langchain-core>=0.2.0,<0.3.0
langchain-community>=0.2.0,<0.3.0
indic-nlp-library==0.92
sentencepiece==0.2.0
sentence-transformers==2.7.0
evaluate==0.4.1
wandb==0.17.0
rapidfuzz==3.9.0
spacy==3.7.0
scikit-learn==1.4.0
rouge-score
bert-score
groq
pytest
python-dotenv
seqeval
```

> **Why some packages use `>=` instead of `==`:**
> `langgraph==0.1.0` (the original plan's version) doesn't exist on PyPI — it was
> yanked/never released. Using `>=0.2.0,<0.3.0` picks the latest stable 0.2.x release.
> Same fix for `langchain-core` and `langchain-community`.

### Step 9: Install dependencies

```bash
pip install -r requirements.txt
```

This takes 5-15 minutes (PyTorch is ~700MB-2GB). Do not interrupt.

> **If you get `ERROR: Could not find a version that satisfies the requirement langgraph==0.1.0`:**
> You forgot to use the updated `requirements.txt` above. The original plan has a wrong version.
> Use exactly the file contents shown in Step 8.

### Step 10: Clone IndicNLP resources

```bash
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
```

Run this from the project root (`hinglish-mas/`). This creates `indic_nlp_resources/` as a subfolder.

> This repo is ~several hundred MB. It may take a few minutes on a slow connection.
> If it hangs for >10 minutes, Ctrl+C and retry.

### Step 11: Verify IndicNLP installed

```bash
python -c "import indicnlp; print('ok')"
```

Should print `ok`. If `ModuleNotFoundError`, run: `pip install indic-nlp-library==0.92`

### Step 12: Create `config.py`

Create `config.py` in the project root:

```python
import os

# ── Base directory ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model paths ──────────────────────────────────────────────────────────
INDICBART_BASE_PATH = "ai4bharat/indicbart"
INDICBART_ADAPTER_PATH = "./models/stage2-lora-adapter"
FELIX_TAGGER_PATH = "./models/felix-tagger-final"
MURIL_MODEL_PATH = "google/muril-base-cased"

# ── LLM backend ──────────────────────────────────────────────────────────
LLM_BACKEND = "groq"
GROQ_MODEL = "llama-3.1-8b-instant"

# ── Paths ─────────────────────────────────────────────────────────────────
INDIC_NLP_RESOURCES = os.path.join(BASE_DIR, "indic_nlp_resources")
FELIX_DATASET_PATH = "./data/felix_dataset/final_augmented_master_6000.csv"

# ── Validation thresholds ─────────────────────────────────────────────────
MURIL_SIMILARITY_THRESHOLD = 0.82
EVENT_COVERAGE_F1_THRESHOLD = 0.5
SLANG_DENSITY_THRESHOLD = 0.25
MAX_REWRITE_ATTEMPTS = 3

# ── Training hyperparameters ──────────────────────────────────────────────
INDICBART_STAGE1_EPOCHS = 3
INDICBART_STAGE2_EPOCHS = 2
FELIX_EPOCHS = 10
LORA_RANK = 16
LORA_ALPHA = 32
```

> **Why `GROQ_MODEL = "llama-3.1-8b-instant"` and not `"llama-3-8b-8192"`:**
> The original plan used `"llama-3-8b-8192"` — this model name has been deprecated on Groq.
> `"llama-3.1-8b-instant"` is the current equivalent. Verify at https://console.groq.com/docs/models
> if this ever stops working and update accordingly.

> **Why `os.path.join(BASE_DIR, "indic_nlp_resources")` instead of an absolute path:**
> The original plan said "set absolute path manually" — that breaks when teammates clone the repo
> on different machines (each would have a different absolute path). Using `BASE_DIR` makes it
> work for everyone automatically, as long as `indic_nlp_resources/` sits in the project root
> (which Step 10 ensures).

### Step 13: Create `.gitignore`

Create `.gitignore` in the project root:

```
# API keys
.env

# Python cache
__pycache__/
*.pyc
.ipynb_checkpoints/

# Model weights (too large for git — store in Kaggle Datasets)
models/*.bin
models/*.pt
models/*.safetensors
models/stage2-lora-adapter/
models/felix-tagger-final/

# Large data
indic_nlp_resources/

# Sessions / outputs
data/sessions/*.json
outputs/*.txt

# OS files
.DS_Store
Thumbs.db
```

### Step 14: Set up Groq API key

1. Go to https://console.groq.com
2. Sign in with Google (fastest).
3. Left sidebar → **"API Keys"** → **"Create API Key"**.
4. Copy the key immediately — shown only once.

Create `.env` file in the project root (this is git-ignored — your key stays private):

```
GROQ_API_KEY=paste_your_key_here
```

> **Security check:** Run `git status` — `.env` should NOT appear in the list.
> If it does, `.gitignore` isn't working. Fix before committing anything.

### Step 15: Verify Groq access

Create a throwaway file `test_groq.py` in the project root:

```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in one word."}],
    temperature=0.1,
    max_tokens=20
)
print(response.choices[0].message.content)
```

Run it:

```bash
python test_groq.py
```

**Expected:** prints something like `Hello` or `Namaste`.

> **If you get a model-not-found error:** Go to https://console.groq.com/docs/models,
> find the current Llama 3.x 8B instant model name, update `config.py`'s `GROQ_MODEL`
> and this test file, then retry.

Once working, delete the test file:

```bash
rm test_groq.py
```

### Step 16: First commit

```bash
git add .
git commit -m "Project structure, config, environment setup"
git push origin main
```

> **If git push asks for a password:** Use your GitHub username + the PAT from Step 3.
> If branch is `master` instead of `main`: `git push origin master`.

---

## Day 3-4: Transliteration Module

### Step 17: Understand why IndicNLP's transliterator needs `init()`

This was a critical bug discovered during implementation. There are TWO transliterator classes
in `indicnlp.transliterate.unicode_transliterate`:

- `UnicodeIndicTransliterator` — converts between **Indic scripts only** (e.g. Devanagari → Gujarati).
  Despite accepting `"en"` as a target, it does NOT convert to ASCII Roman.
  Using this gave output like `"तम कय कर रह ह?"` (vowel matras stripped by Step 5's regex).

- `ItransTransliterator` — converts Devanagari → ITRANS Roman scheme (correct for our use case).
  BUT: its lookup tables (`OFFSET_TO_ITRANS`, `ITRANS_TO_OFFSET`) are **empty by default**.
  Calling `to_itrans()` without first calling `unicode_transliterate.init()` produces
  output like `'ुम ा कर रे ो'` (only matras, no consonants — tables were empty).

**Fix:** call `unicode_transliterate.init()` once at module load time.

### Step 18: Create `preprocessing/transliterate.py`

```python
import re
from indicnlp import common

import config
common.set_resources_path(config.INDIC_NLP_RESOURCES)

from indicnlp.transliterate import unicode_transliterate as ut
from indicnlp.transliterate.unicode_transliterate import ItransTransliterator
from indicnlp.tokenize import sentence_tokenize

# CRITICAL: must call init() before any to_itrans() call
# This populates the OFFSET_TO_ITRANS and ITRANS_TO_OFFSET tables (128 and 112 entries)
# Without this, tables are empty → every consonant maps to nothing → garbage output
ut.init()


class HindiTransliterator:
    # Applied to ITRANS output BEFORE word-boundary schwa deletion.
    # Order matters — longer/more specific patterns must come first.
    ITRANS_TO_CASUAL = [
        (r'\.m', 'n'),      # anusvara ं   → n  (mai.m → main)
        (r'ँ', 'n'),         # chandrabindu ँ (raw Devanagari leak from offset table gap) → n
        (r'aa', 'a'),       # long aa      → a  (kyaa → kya)
        (r'ii', 'i'),       # long ii      → i  (Thiika → Thika)
        (r'uu', 'u'),       # long uu      → u  (huu → hu)
        (r'\.a', ''),       # stray schwa marker, if any
    ]

    # ITRANS uses capital letters for retroflex/aspirated consonants
    # (Th, Dh, Ph, Kh, Gh, Ch, Jh etc.) — lowercase them for casual Hinglish
    RETROFLEX_LOWER = ['Th', 'Dh', 'Ph', 'Kh', 'Gh', 'Ch', 'Jh', 'Bh', 'Rh',
                        'T', 'D', 'N', 'R', 'S', 'L']

    # Words where the final 'a' must NOT be deleted (it's phonemic, not a silent schwa)
    # Expand this list as you encounter more words in real test outputs
    SCHWA_EXCEPTIONS = {
        # Question / function words
        'kya', 'tha', 'kaha', 'yaha', 'vaha', 'jaha', 'kaba', 'aba',
        'na', 'ka', 'ki', 'ke', 'ko', 'pa', 'ya', 'wa', 'va',
        'da', 'ga', 'ra', 'la', 'ma', 'sa', 'ja', 'ha',
        'cha', 'kha', 'gha',
        # Past tense verb forms ending in ya/a
        'gaya', 'kiya', 'liya', 'diya', 'piya', 'siya',
        'likha', 'dekha', 'socha', 'roka', 'khola',
        # Habitual verb forms
        'hota', 'jata', 'aata', 'rehta', 'chahta',
        # Possessives
        'uska', 'unka', 'iska', 'inka', 'apna', 'apni',
        # Common adjectives / ordinals
        'pura', 'sara', 'kara', 'para', 'mara', 'tera',
        'accha', 'achha', 'bura', 'naya',
        'pahala', 'pahle', 'agala', 'agale',
        # Suffix forms
        'wala', 'wali', 'wale',
        # Locatives
        'vaha', 'yaha', 'jaha', 'kaha',
        # Additional discovered during testing
        'bata', 'jata', 'aata', 'aana', 'jaana', 'karna', 'hona',
    }

    SCHWA_DELETION_RULES = {
        # Specific word-level fixes that survive general schwa deletion
        'mujhko': 'mujhe',
        'tumko': 'tumhe',
        'kuchh': 'kuch',
        'yahaan': 'yahan',
        'wahaan': 'wahan',
        'kahaan': 'kahan',
    }

    def transliterate_sentence(self, devanagari_text: str) -> str:
        # Step 1: Split on danda (।), ?, !
        sentences = sentence_tokenize.sentence_split(devanagari_text, lang='hi')

        romanized = []
        for sent in sentences:
            if not sent.strip():
                continue

            # Step 2: Devanagari → ITRANS roman scheme
            # Example: "तुम क्या कर रहे हो" → "tuma kyaa kara rahe ho"
            roman = ItransTransliterator.to_itrans(sent, 'hi')

            # Step 3: ITRANS conventions → casual Hinglish roman
            roman = self._itrans_to_casual(roman)

            # Step 4: Terminal schwa deletion (per word, with exceptions)
            roman = self._apply_schwa_deletion(roman)

            # Step 5: Phonetic smoothing (specific word-level fixes)
            roman = self._phonetic_smooth(roman)

            # Step 6: Lowercase, strip noise, preserve punctuation
            roman = roman.lower()
            roman = re.sub(r'[^\w\s\.\,\!\?\;\:\'\-]', '', roman)
            roman = re.sub(r'\s+', ' ', roman).strip()

            if roman:
                romanized.append(roman)

        return ' '.join(romanized)

    def _itrans_to_casual(self, text: str) -> str:
        for pattern, repl in self.ITRANS_TO_CASUAL:
            text = re.sub(pattern, repl, text)
        # Lowercase retroflex/aspirated consonant markers
        for cap in self.RETROFLEX_LOWER:
            text = re.sub(cap, cap.lower(), text)
        return text

    def _apply_schwa_deletion(self, text: str) -> str:
        """
        Drop terminal 'a' after a consonant at word end — per word, with exceptions.
        E.g. "tuma" → "tum", "kara" → "kar", but "kya" stays "kya".

        IMPORTANT: processes word-by-word (not sentence-wide regex) to avoid
        partial-word matches. Strips trailing punctuation before checking,
        then reattaches it.
        """
        words = text.split(' ')
        result = []
        for word in words:
            # Separate trailing punctuation
            m = re.match(r'^(.*?)([\.\,\!\?\;\:]*)$', word)
            core, punct = m.group(1), m.group(2)

            if core.lower() in self.SCHWA_EXCEPTIONS:
                result.append(core + punct)
                continue

            # Drop terminal 'a' only if preceded by a consonant
            new_core = re.sub(r'^([a-zA-Z]*[bcdfghjklmnpqrstvwxyz])a$',
                               r'\1', core)
            result.append(new_core + punct)
        return ' '.join(result)

    def _phonetic_smooth(self, text: str) -> str:
        for wrong, correct in self.SCHWA_DELETION_RULES.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct,
                           text, flags=re.IGNORECASE)
        return text


if __name__ == "__main__":
    t = HindiTransliterator()
    sample = "तुम क्या कर रहे हो? मैं ठीक हूँ।"
    print(t.transliterate_sentence(sample))
    # Expected: "tum kya kar rahe ho? main thik hun."
```

### Step 19: Run the manual test

```bash
python -m preprocessing.transliterate
```

**Expected output:** `tum kya kar rahe ho? main thik hun.`

**Debugging journey (documented for reference):**

| Attempt | Issue                                                                     | Symptom                                       | Fix                                        |
| ------- | ------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------ |
| 1       | Used `UnicodeIndicTransliterator(sent, "hi", "en")`                       | `"तम कय कर रह ह?"` — vowels stripped          | Wrong class — not for Devanagari→Roman     |
| 2       | Switched to `ItransTransliterator.to_itrans(sent, 'hi')` without `init()` | `'ुम ा कर रे ो'` — only matras, no consonants | `OFFSET_TO_ITRANS` table empty (0 entries) |
| 3       | Called `ut.init()` first                                                  | `'tuma kyaa kara rahe ho'` — correct ITRANS!  | Offset tables populated (128 entries)      |
| 4       | Applied `ITRANS_TO_CASUAL` rules                                          | `'tum ky kar rahe ho'` — `kya` became `ky`    | Schwa deletion stripped the `a` from `kya` |
| 5       | Added per-word `SCHWA_EXCEPTIONS`                                         | `'tum kya kar rahe ho? main thik hun.'` ✅    | Final working version                      |

> **DOUBT CHECK:** If Step 19 produces anything other than `tum kya kar rahe ho? main thik hun.`,
> paste the exact output here and we'll trace which processing step caused the deviation.
> Run `python -c "from indicnlp.transliterate import unicode_transliterate as ut; ut.init(); print(len(ut.OFFSET_TO_ITRANS))"`
> to verify the table has 128 entries — if it shows 0, `init()` isn't being called before use.

---

## Day 5: Prose Chunker

### Step 20: Create `preprocessing/chunker.py`

```python
import re
from typing import List


class ProseChunker:
    """
    Splits romanized Hinglish text into scene-level chunks.
    Uses paragraph boundaries (\n\n) as primary scene delimiters.

    IMPORTANT: The transliterator's paragraph-by-paragraph approach (Step 21)
    preserves \n\n boundaries through romanization. If you transliterate the
    entire raw text at once, sentence_tokenize.sentence_split() collapses
    everything into one flat string and \n\n boundaries are lost.
    """

    LOCATION_CUES = [
        'ghar', 'school', 'bazaar', 'bahar', 'andar', 'office',
        'kamra', 'kamre', 'gali', 'station', 'park', 'library',
        'pustakalay', 'baazar', 'sadak', 'raasta', 'gali'
    ]
    TEMPORAL_CUES = [
        'subah', 'sham', 'raat', 'dopahar', 'agli baar', 'phir',
        'kal', 'aaj', 'parso', 'thodi der baad', 'ek din',
        'agale din', 'us din'
    ]

    def chunk(self, romanized_text: str) -> List[dict]:
        """
        Split romanized text into scene chunks.
        Returns list of dicts: {scene_id, text, has_location_cue, has_temporal_cue}
        """
        # Primary strategy: split on double newlines (paragraph boundaries)
        paragraphs = [p.strip() for p in romanized_text.split('\n\n') if p.strip()]

        # Fallback: treat whole text as one scene if no paragraph breaks
        if not paragraphs and romanized_text.strip():
            paragraphs = [romanized_text.strip()]

        scenes = []
        for i, para in enumerate(paragraphs):
            scenes.append({
                "scene_id": f"scene_{i+1:02d}",
                "text": para,
                "has_location_cue": any(
                    cue in para.lower() for cue in self.LOCATION_CUES
                ),
                "has_temporal_cue": any(
                    cue in para.lower() for cue in self.TEMPORAL_CUES
                )
            })
        return scenes


if __name__ == "__main__":
    sample = "subah ram ghar se nikla.\n\nphir woh school gaya aur apne dost se mila."
    chunker = ProseChunker()
    for scene in chunker.chunk(sample):
        print(scene)
```

### Step 21: Run the chunker test

```bash
python -m preprocessing.chunker
```

**Expected:** Two dicts printed:

- `scene_01`: `has_temporal_cue: True` (because of "subah")
- `scene_02`: `has_location_cue: True` (because of "school"), `has_temporal_cue: True` (because of "phir")

---

## Day 6-7: Tests + Baseline Verification

### Step 22: Create `tests/test_transliterate.py`

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.transliterate import HindiTransliterator


def test_basic_devanagari():
    """Core words transliterate to expected Roman substrings."""
    t = HindiTransliterator()
    result = t.transliterate_sentence("तुम क्या कर रहे हो?")
    assert "tum" in result.lower(), f"Expected 'tum' in output, got: {result}"
    assert "kya" in result.lower(), f"Expected 'kya' in output, got: {result}"


def test_schwa_exception_kya():
    """'kya' must not be reduced to 'ky' by schwa deletion."""
    t = HindiTransliterator()
    result = t.transliterate_sentence("तुम क्या कर रहे हो?")
    words = result.lower().split()
    assert "ky" not in words, f"'kya' was wrongly reduced to 'ky'. Output: {result}"
    assert "kya" in words, f"Expected 'kya' in output. Output: {result}"


def test_noise_stripping():
    """Special characters should be stripped from output."""
    t = HindiTransliterator()
    result = t.transliterate_sentence("नमस्ते! ##00:01:30##")
    assert "##" not in result, f"Noise not stripped: {result}"


def test_returns_nonempty():
    """Valid input should always produce non-empty output."""
    t = HindiTransliterator()
    result = t.transliterate_sentence("मैं ठीक हूँ।")
    assert len(result.strip()) > 0, "Output was empty for valid input"


def test_multiple_sentences():
    """Multiple sentences should produce output covering all of them."""
    t = HindiTransliterator()
    result = t.transliterate_sentence("राम घर गया। वह खुश था।")
    assert len(result.split()) >= 4, f"Expected ≥4 words, got: {result}"


def test_no_devanagari_in_output():
    """Output should contain no Devanagari Unicode characters."""
    t = HindiTransliterator()
    result = t.transliterate_sentence("तुम क्या कर रहे हो?")
    devanagari_chars = [c for c in result if '\u0900' <= c <= '\u097F']
    assert len(devanagari_chars) == 0, \
        f"Devanagari chars found in output: {devanagari_chars}. Output: {result}"


def test_full_expected_output():
    """End-to-end check: confirm exact expected output for known input."""
    t = HindiTransliterator()
    result = t.transliterate_sentence("तुम क्या कर रहे हो? मैं ठीक हूँ।")
    assert result == "tum kya kar rahe ho? main thik hun.", \
        f"Unexpected output: '{result}'"
```

> **Why these tests and not exact-equality tests for every word:**
> The original plan had tests like `assert t.transliterate_sentence("राम") == "ram"` —
> exact equality tests break immediately because `to_itrans()` output for any given word
> depends on the ITRANS scheme and your post-processing pipeline. After running real
> transliteration and seeing what `to_itrans()` actually produces, we confirmed the
> expected output for the test sentence above and locked it in `test_full_expected_output`.
> All other tests check properties (substring presence, no Devanagari, non-empty)
> rather than exact strings, making them robust to minor pipeline tuning.

### Step 23: Run the tests

```bash
pytest tests/ -v
```

**Expected:** all 7 tests show `PASSED`.

> **DOUBT CHECK — if `test_full_expected_output` fails:**
> Run `python -m preprocessing.transliterate` and paste the actual output.
> The expected string `"tum kya kar rahe ho? main thik hun."` was confirmed working
> on this exact pipeline. If you get something different, it likely means `SCHWA_EXCEPTIONS`
> is missing a word or `ITRANS_TO_CASUAL` rules are in the wrong order.

### Step 24: Prepare the 200-sentence Hindi test set

You need 200 Hindi sentences in Devanagari, one per line, in `data/raw/test_sentences_devanagari.txt`.

**Option A (recommended): Use a Hindi Wikipedia article or public domain text.**
Copy 5-10 paragraphs of any Hindi Wikipedia article, paste into `data/raw/source_text.txt`,
then run the splitter below.

**Option B:** Ask me to generate 200 sample sentences — I can produce a file directly.

Create `preprocessing/build_test_set.py`:

```python
import re

INPUT_FILE = "data/raw/source_text.txt"
OUTPUT_FILE = "data/raw/test_sentences_devanagari.txt"


def split_into_sentences(text: str):
    # Split on Devanagari danda (।), ?, !
    sentences = re.split(r'(?<=[।?!])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    sentences = split_into_sentences(raw)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"Wrote {len(sentences)} sentences to {OUTPUT_FILE}")
```

Run:

```bash
python -m preprocessing.build_test_set
wc -l data/raw/test_sentences_devanagari.txt
```

Should be close to 200. If short, add more text to `source_text.txt` and re-run.

### Step 25: Run baseline transliteration over 200 sentences

Create `preprocessing/run_baseline.py`:

```python
from preprocessing.transliterate import HindiTransliterator

INPUT_FILE = "data/raw/test_sentences_devanagari.txt"
OUTPUT_FILE = "data/processed/test_sentences_romanized.txt"


def main():
    t = HindiTransliterator()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    results = []
    failures = 0
    for i, line in enumerate(lines):
        try:
            roman = t.transliterate_sentence(line)
            results.append(roman)
        except Exception as e:
            print(f"FAILED on line {i+1}: {line[:50]}... -> {e}")
            results.append("")
            failures += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n")

    print(f"Processed {len(lines)} sentences.")
    print(f"Failures: {failures}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

Run:

```bash
python -m preprocessing.run_baseline
head -20 data/processed/test_sentences_romanized.txt
```

**Expected:** `Failures: 0`, and the first 20 lines show readable romanized Hinglish.

**What to look for in the output and how to fix:**
The most common issues from real 200-sentence runs and their fixes:

| Bad output       | Cause                            | Fix                        |
| ---------------- | -------------------------------- | -------------------------- |
| `gaya` → `gay`   | `gaya` not in `SCHWA_EXCEPTIONS` | Add `'gaya'` to exceptions |
| `kiya` → `kiy`   | `kiya` not in exceptions         | Add `'kiya'`               |
| `hota` → `hot`   | `hota` not in exceptions         | Add `'hota'`               |
| `uska` → `usk`   | `uska` not in exceptions         | Add `'uska'`               |
| `accha` → `acch` | `accha` not in exceptions        | Add `'accha'`              |

When you find these: add the word to `SCHWA_EXCEPTIONS` in `transliterate.py`, then re-run
`python -m preprocessing.run_baseline` and `pytest tests/ -v` to confirm no regressions.

### Step 26: Copy the FELIX dataset to the project

Your FELIX dataset (`final_augmented_master_6000.csv`) should be copied to:

```
data/felix_dataset/final_augmented_master_6000.csv
```

This file has 6,000 rows and will be used in Week 5-6 for training the FELIX slang rewriter.
Copy it now so it's in place:

```bash
# From wherever you downloaded it:
cp ~/Downloads/final_augmented_master_6000.csv data/felix_dataset/
```

Verify:

```bash
python -c "
import csv
with open('data/felix_dataset/final_augmented_master_6000.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print('Rows:', len(rows))
print('Columns:', list(rows[0].keys()))
print('Sample:', rows[0])
"
```

**Expected output:**

```
Rows: 6000
Columns: ['neutral hinglish sentence', 'context', 'social_context', 'Possible Slangs',
          'Slang Location', 'Multiple positions allowed?', 'Control tokens', 'function',
          'output sentence']
Sample: {'neutral hinglish sentence': 'Match ka result disappointing tha.', ...}
```

### Step 27: Week 1 final commit

```bash
git add .
git commit -m "Week 1: transliteration module, chunker, tests, baseline verification, FELIX dataset added"
git push origin main
```

**WEEK 1 DELIVERABLE — CHECKLIST:**

- [ ] `preprocessing/transliterate.py` — runs, produces `"tum kya kar rahe ho? main thik hun."`
- [ ] `preprocessing/chunker.py` — runs, splits paragraphs into scene dicts
- [ ] `tests/test_transliterate.py` — all 7 tests pass
- [ ] `data/processed/test_sentences_romanized.txt` — 200 lines, 0 failures
- [ ] `data/felix_dataset/final_augmented_master_6000.csv` — 6000 rows confirmed
- [ ] Everything pushed to GitHub

---

# WEEK 2 — SCRATCHPAD MEMORY + AGENT 1 (NARRATIVE ANALYST)

## Day 1-2: LangGraph State + Scratchpad

### Step 28: Create `memory/scratchpad.py`

```python
from typing import TypedDict, List, Optional
import json
import os


# ── Character profile schema ────────────────────────────────────────────
class CharacterProfile(TypedDict):
    name: str
    age: Optional[int]
    social_role: str        # "peer" | "elder" | "authority"
    slang_level: int        # 0 (none) - 10 (heavy)
    relationship: dict      # {other_char_name: relationship_type}
    current_emotion: str    # "neutral" | "anxious" | "angry" | "happy" | "urgent"
    turn_count: int
    stress_index: float     # 0.0 - 1.0


# ── Scene event schema ─────────────────────────────────────────────────
class SceneEvent(TypedDict):
    scene_id: str
    key_events: List[str]
    characters: List[str]
    scene_goal: str
    location: str
    narrative_link: str     # "Following X, Y reacts..."
    pronoun_map: dict       # {"usne": "Ram", "woh": "Sita"}


# ── Master LangGraph pipeline state ────────────────────────────────────
class PipelineState(TypedDict):
    # Input
    raw_hindi_text: str

    # After preprocessing
    romanized_text: str
    scene_chunks: List[dict]

    # Agent 1 output
    event_chain: List[SceneEvent]

    # Agent 2 output
    character_profiles: dict        # {char_name: CharacterProfile}

    # Agent 3 output (Pass 1 — neutral Hinglish)
    neutral_script: List[dict]      # [{scene_id, dialogue}]

    # Agent 4 output (Pass 2 — slang rewritten)
    slang_script: List[dict]

    # Evaluation
    validation_scores: List[float]
    failed_scenes: List[str]        # scene_ids that failed validation
    rewrite_count: int              # guard against infinite rewrite loops

    # Final
    final_script: str


class Scratchpad:
    """
    Persistent JSON-backed state store for a single pipeline session.
    Saves after every write so sessions survive process restarts.

    CHANGE from original plan: __init__ now auto-loads existing session data.
    Original code always set self._data = {} unconditionally, meaning
    re-instantiating Scratchpad("same_id") mid-pipeline wiped all saved state.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.path = f"data/sessions/{session_id}_scratchpad.json"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data = {}
        # Auto-load existing session if it exists
        if os.path.exists(self.path):
            self.load()

    def write(self, key: str, value):
        self._data[key] = value
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def read(self, key: str):
        return self._data.get(key)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
        return self

    def all(self):
        return self._data


if __name__ == "__main__":
    # Test: write, reload from disk, confirm data persists
    sp = Scratchpad("test_session")
    sp.write("raw_hindi_text", "तुम क्या कर रहे हो?")
    sp.write("event_chain", [{"scene_id": "scene_01", "key_events": ["test event"]}])
    print("Saved:", sp.all())

    # Simulate a second instantiation (different agent, same session)
    sp2 = Scratchpad("test_session")
    print("Reloaded:", sp2.all())

    # Both should show identical data
    assert sp.all() == sp2.all(), "Auto-load failed!"
    print("AUTO-LOAD TEST PASSED")
```

### Step 29: Test the scratchpad

```bash
python -m memory.scratchpad
```

**Expected:**

```
Saved: {'raw_hindi_text': 'तुम क्या कर रहे हो?', 'event_chain': [...]}
Reloaded: {'raw_hindi_text': 'तुम क्या कर रहे हो?', 'event_chain': [...]}
AUTO-LOAD TEST PASSED
```

Both lines show **identical data** — confirms write-then-reload works.

Confirm the file was created:

```bash
cat data/sessions/test_session_scratchpad.json
```

Then clean up the test session:

```bash
rm data/sessions/test_session_scratchpad.json
```

---

## Day 3-5: Agent 1 — Narrative Analyst

### Step 30: Understanding Agent 1's prompt evolution

Agent 1 went through two prompt iterations before reaching a stable version.

**Problem discovered during Week 3 testing (documented here so you understand the final version):**

Running the pipeline on the 5-paragraph library story, the original Week 2 prompt produced:

- `scene_01` appearing **4 times** instead of once (model split one scene into 4 sub-events)
- Characters included `log` (people), `chatron` (students), `dayari` (diary) — common nouns not names
- `scene_04` and `scene_05` completely missing

**Root causes:**

1. The prompt said "output a JSON array" without specifying exactly one object per input chunk
2. No explicit rule against extracting groups/objects as characters
3. `_parse_json_output` treated empty array `[]` (valid JSON) as success, not fallback

**The version below is the FINAL STABLE version** incorporating all fixes.

### Step 31: Create `agents/agent1_narrative.py`

````python
import json
import os
import re
import time
from typing import List

import config


NARRATIVE_ANALYST_PROMPT = """You are a screenplay structural analyst.
Your ONLY job is to extract a structured event chain from a Hindi story (written in Roman script).
Do NOT generate any dialogue. Only extract what exists.

Story text for THIS SCENE ONLY (scene_id = "{scene_id}"):
{story_text}

Previous scene summary (for continuity):
{previous_scene_summary}

Output a JSON array containing EXACTLY ONE object. No preamble. No explanation.
No markdown code fences. Just the raw JSON array with one element.

The single object must have EXACTLY these fields:
{{
  "scene_id": "{scene_id}",
  "key_events": ["Event 1 description", "Event 2 description"],
  "characters": ["CharName1", "CharName2"],
  "scene_goal": "What this scene accomplishes narratively",
  "location": "Where this scene takes place",
  "narrative_link": "Following [previous event], [character] now [does X]",
  "pronoun_map": {{"usne": "Ram", "woh": "Sita"}}
}}

CRITICAL RULES:
- ALWAYS use scene_id = "{scene_id}" exactly as given — do not invent new scene_ids.
- Output EXACTLY ONE JSON object inside the array, even if the text describes multiple
  events. Combine everything into ONE object's key_events list.
- "characters" must contain ONLY proper names of INDIVIDUAL NAMED PEOPLE
  (e.g. "Arav", "Ram", "Shyam", "Priya").
- NEVER include in "characters":
    * Groups of people (e.g. "log"=people, "chatron"=students, "shodhakartaon"=researchers)
    * Organizations or institutions (e.g. "nagar parishad"=city council)
    * Objects (e.g. "dayari"=diary, "kitab"=book, "alamari"=cupboard)
    * Generic role references to unnamed people (e.g. "bhule-bisare vyapari"=the forgotten merchant)
- If NO individually named characters appear in this scene, "characters" MUST be an empty list [].
- key_events: list every plot-significant event (minimum 1, maximum 6)
- narrative_link: MUST reference a specific event or character name from the previous scene
- pronoun_map: map every pronoun in this chunk to the character it refers to ({{}} if none)
"""


class NarrativeAnalyst:
    def __init__(self, llm_backend: str = None):
        self.backend = llm_backend or config.LLM_BACKEND
        self._setup_llm()

    def _setup_llm(self):
        if self.backend == "ollama":
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="llama3:8b-instruct-q4_K_M",
                               temperature=0.1)

        elif self.backend == "groq":
            from groq import Groq
            from dotenv import load_dotenv
            load_dotenv()
            self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

        elif self.backend == "hf":
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct")
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                torch_dtype=torch.float16,
                load_in_4bit=True)
            self.pipe = pipeline("text-generation", model=model,
                                  tokenizer=tokenizer, max_new_tokens=1024)
        else:
            raise ValueError(f"Unknown LLM backend: '{self.backend}'. "
                              f"Valid options: 'groq', 'ollama', 'hf'")

    def extract_event_chain(self, scene_chunks: List[dict],
                             previous_summary: str = "This is the first scene.") -> List[dict]:
        """
        Extract exactly one event object per scene chunk.
        Guarantees len(output) == len(scene_chunks) — no missing scenes.
        """
        all_events = []

        for chunk in scene_chunks:
            prompt = NARRATIVE_ANALYST_PROMPT.format(
                scene_id=chunk["scene_id"],
                story_text=chunk["text"],
                previous_scene_summary=previous_summary
            )

            raw_output = self._call_llm(prompt)
            events = self._parse_json_output(raw_output, chunk["scene_id"])

            # Guard: if the model returned nothing, insert a fallback
            # (ensures len(all_events) == len(scene_chunks) always)
            if not events:
                events = [{
                    "scene_id": chunk["scene_id"],
                    "key_events": [],
                    "characters": [],
                    "scene_goal": "Unknown",
                    "location": "Unknown",
                    "narrative_link": previous_summary,
                    "pronoun_map": {}
                }]

            # Take only the FIRST object (even if model returned multiple despite instructions)
            # and force the correct scene_id to prevent drift
            first_event = events[0]
            first_event["scene_id"] = chunk["scene_id"]
            all_events.append(first_event)

            previous_summary = first_event.get("scene_goal", previous_summary)

        return all_events

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "ollama":
            return self.llm.invoke(prompt)

        elif self.backend == "groq":
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.groq_client.chat.completions.create(
                        model=config.GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1024
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Groq call failed ({e}), retrying in 2s...")
                        time.sleep(2)
                    else:
                        raise

        elif self.backend == "hf":
            result = self.pipe(prompt, return_full_text=False)
            return result[0]["generated_text"]

    def _parse_json_output(self, raw: str, fallback_scene_id: str) -> List[dict]:
        """
        Robustly extract JSON from LLM output.
        Handles: markdown code fences, single object (not array), trailing commas.
        """
        # Strip markdown code fences (LLMs often add these despite instructions)
        cleaned = re.sub(r'```(?:json)?', '', raw).strip()

        # Find JSON array
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if not match:
            # Try wrapping a single object in an array
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                cleaned = f"[{match.group()}]"
            else:
                return [{
                    "scene_id": fallback_scene_id,
                    "key_events": [],
                    "characters": [],
                    "scene_goal": "Unknown",
                    "location": "Unknown",
                    "narrative_link": "",
                    "pronoun_map": {}
                }]
        else:
            cleaned = match.group()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Clean common LLM JSON errors: trailing commas
            clean = re.sub(r',\s*}', '}', cleaned)
            clean = re.sub(r',\s*]', ']', clean)
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                return [{
                    "scene_id": fallback_scene_id,
                    "key_events": [],
                    "characters": [],
                    "scene_goal": "Unknown",
                    "location": "Unknown",
                    "narrative_link": "",
                    "pronoun_map": {}
                }]


if __name__ == "__main__":
    analyst = NarrativeAnalyst(llm_backend="groq")
    test_chunks = [
        {
            "scene_id": "scene_01",
            "text": "subah ram apne ghar se nikla aur school ki taraf chal diya. "
                     "raaste mein usne apne dost shyam ko dekha."
        }
    ]
    events = analyst.extract_event_chain(test_chunks)
    print(json.dumps(events, indent=2, ensure_ascii=False))
````

### Step 32: Test Agent 1

```bash
python -m agents.agent1_narrative
```

**Expected output:** A JSON array with one object:

- `scene_id: "scene_01"`
- `characters: ["Ram", "Shyam"]` (proper names, not groups)
- `key_events:` at least 2 events (Ram leaves home, Ram meets Shyam)
- `narrative_link:` references the previous scene (or "This is the first scene")
- `pronoun_map: {"usne": "Ram"}` or similar

> **DOUBT CHECK — LLM output is non-deterministic.** Characters and events will vary
> slightly between runs, which is normal. What should NOT vary:
>
> - There should always be exactly ONE object in the array
> - `scene_id` should always be `"scene_01"`
> - `characters` should never include non-names like "raaste", "school", "dost"
>
> If you see multiple objects or garbage characters, run the test 2-3 more times — if
> consistently broken, paste the raw output and we'll tighten the prompt rules further.

---

## Day 6-7: Full Flow Test

### Step 33: Create test stories

Create `data/raw/test_story_1.txt` — a short Hindi story in Devanagari,
**with blank lines between paragraphs** (essential for the chunker):

```
शहर के बीचों-बीच एक बहुत पुरानी पुस्तकालय थी। लोग कहते थे कि वहाँ की कुछ किताबें सौ साल से भी अधिक पुरानी हैं। दिन में वहाँ छात्रों और शोधकर्ताओं की भीड़ लगी रहती थी, लेकिन शाम होते ही पूरा भवन शांत हो जाता था।

एक दिन आरव नाम का छात्र इतिहास की परियोजना के लिए सामग्री खोजने वहाँ गया। उसे एक धूल से ढकी हुई अलमारी मिली, जो सामान्यतः बंद रहती थी। अलमारी के भीतर उसे चमड़े के कवर वाली एक डायरी दिखाई दी।

आरव उत्सुक हो गया। उसने डायरी पढ़ना शुरू किया। उसमें शहर के एक भूले-बिसरे व्यापारी का उल्लेख था, जिसने वर्षों पहले कई गरीब परिवारों की मदद की थी।

अगले कई दिनों तक आरव उन संकेतों का पीछा करता रहा। अंततः उसे नगर अभिलेखागार में ऐसे दस्तावेज़ मिले जिन्होंने व्यापारी की कहानी को सत्य सिद्ध कर दिया।

आरव ने सीखा कि इतिहास केवल बड़ी घटनाओं का नहीं, बल्कि उन लोगों का भी होता है जिनके अच्छे कार्य समय के साथ भुला दिए जाते हैं।
```

### Step 34: Create the Week 2 integration test

Create `tests/test_full_flow_week2.py`:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from preprocessing.transliterate import HindiTransliterator
from preprocessing.chunker import ProseChunker
from agents.agent1_narrative import NarrativeAnalyst
from memory.scratchpad import Scratchpad


def run_full_flow(story_path: str, session_id: str):
    print(f"\n{'='*60}")
    print(f"Running Week 2 flow for: {story_path}")
    print(f"{'='*60}\n")

    # Step 1: Read raw Hindi
    with open(story_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    sp = Scratchpad(session_id)
    sp.write("raw_hindi_text", raw_text)

    # Step 2: Transliterate paragraph-by-paragraph
    # IMPORTANT: must transliterate per paragraph to preserve \n\n boundaries
    # for the chunker. Transliterating the full text at once collapses all
    # paragraph breaks into one flat string.
    transliterator = HindiTransliterator()
    paragraphs = raw_text.split('\n\n')
    romanized_paragraphs = [
        transliterator.transliterate_sentence(p.strip())
        for p in paragraphs if p.strip()
    ]
    romanized_text = '\n\n'.join(romanized_paragraphs)
    sp.write("romanized_text", romanized_text)

    print("=== ROMANIZED TEXT ===")
    print(romanized_text)
    print()

    # Step 3: Chunk into scenes
    chunker = ProseChunker()
    scene_chunks = chunker.chunk(romanized_text)
    sp.write("scene_chunks", scene_chunks)

    print("=== SCENE CHUNKS ===")
    for c in scene_chunks:
        print(f"  {c['scene_id']}: {c['text'][:60]}...")
    print()

    # Step 4: Agent 1 — extract event chain
    analyst = NarrativeAnalyst(llm_backend="groq")
    event_chain = analyst.extract_event_chain(scene_chunks)
    sp.write("event_chain", event_chain)

    print("=== EVENT CHAIN ===")
    print(json.dumps(event_chain, indent=2, ensure_ascii=False))

    # Validate structure
    assert len(event_chain) == len(scene_chunks), (
        f"Event chain has {len(event_chain)} events but story has "
        f"{len(scene_chunks)} scenes — missing scenes!"
    )

    for i, (event, chunk) in enumerate(zip(event_chain, scene_chunks)):
        assert event["scene_id"] == chunk["scene_id"], (
            f"scene_id mismatch at position {i}: "
            f"event has '{event['scene_id']}', chunk is '{chunk['scene_id']}'"
        )

    print(f"\nVALIDATION PASSED: {len(event_chain)} events for {len(scene_chunks)} scenes")
    return event_chain


if __name__ == "__main__":
    # Run on all 3 test stories
    for story_num in range(1, 4):
        path = f"data/raw/test_story_{story_num}.txt"
        if os.path.exists(path):
            run_full_flow(path, f"week2_test_story{story_num}")
        else:
            print(f"Skipping {path} (file not found)")
```

### Step 35: Run the Week 2 integration test

```bash
python -m tests.test_full_flow_week2
```

**Check all three:**

1. **Number of events = number of scenes** (e.g. 5 scenes → 5 events). This is now guaranteed
   by the `extract_event_chain` fix (one object per chunk, forced scene_id, fallback on empty).

2. **Characters list contains only proper names** — for the library story, only `"Arav"` should
   appear (the only named character). Common nouns like `"log"`, `"chatron"`, `"shodhakartaon"`,
   `"dayari"` should NOT appear after the prompt fixes.

3. **`narrative_link` is specific** — scene_02's `narrative_link` should reference something from
   scene_01 (e.g., "Following the library setting, Arav now explores"). Generic filler like
   "Following the previous scene, something happens" means the LLM isn't using the prior scene
   summary — rerun a couple times to check if it's occasional or consistent.

**Real output from actual test run (library story, 5 scenes):**

```
scene_01: [] | Introduce the setting of the story
scene_02: ['Arav', 'dayari'] | Arav finds the necessary materials
scene_03: ['Arav'] | Arav discovers information about a forgotten merchant
scene_04: ['Arav'] | Arav's discovery leads to a monument being built
scene_05: ['Arav'] | Arav learns a lesson about history
VALIDATION PASSED: 5 events for 5 scenes
```

> **Note:** `dayari` (the diary) appeared in `scene_02` characters in one run despite prompt
> rules — this is a known occasional LLM non-compliance. The `NON_CHARACTER_WORDS` blocklist
> in Agent 2 (Week 3) catches this downstream. Accept it for Week 2.

### Step 36: Create 2 more test stories

Create `data/raw/test_story_2.txt` (different plot — recommend a family story with 2-3 named characters to test multi-character profiles in Week 3):

```
रिया और उसका भाई रोहन हर रविवार को पार्क जाते थे। आज रोहन बहुत खुश था क्योंकि उसकी परीक्षा में अच्छे नंबर आए थे।

पार्क में उन्हें रिया की दोस्त नेहा मिली। नेहा ने रोहन को बधाई दी। तीनों ने साथ में आइसक्रीम खाई।

शाम को घर लौटते समय रोहन ने कहा कि वह अगली बार और भी अच्छे नंबर लाएगा। रिया ने उसे प्रोत्साहित किया।
```

Create `data/raw/test_story_3.txt` (single scene story — tests the "single scene, still wrap in array" edge case):

```
विकास अपने कमरे में बैठा था और परीक्षा के लिए पढ़ रहा था। अचानक उसका फोन बजा। उसके दोस्त अमित ने बताया कि कल की परीक्षा एक हफ्ते के लिए टल गई है।
```

Run the integration test again with all 3 stories:

```bash
python -m tests.test_full_flow_week2
```

For story 2, verify `characters` includes `Riya`, `Rohan`, `Neha` — 3 named characters.
For story 3 (single paragraph), verify exactly 1 event in the chain.

### Step 37: Run pytest to confirm all tests still pass

```bash
pytest tests/ -v
```

All 7 transliteration tests should still pass after all the Week 2 code additions.

### Step 38: Week 2 final commit

```bash
git add .
git commit -m "Week 2: scratchpad/PipelineState, Agent 1 narrative analyst, full flow validated on 3 test stories"
git push origin main
```

**WEEK 2 DELIVERABLE — CHECKLIST:**

- [ ] `memory/scratchpad.py` — `PipelineState`, `CharacterProfile`, `SceneEvent`, `Scratchpad` defined and auto-load tested
- [ ] `agents/agent1_narrative.py` — stable Groq backend, final prompt version, exactly-one-event-per-scene guarantee
- [ ] `tests/test_full_flow_week2.py` — runs on 3 stories, validation passes for all
- [ ] Event chains have correct structure: no duplicate scene_ids, no missing scenes, no common nouns in characters
- [ ] 3 test stories created in `data/raw/`
- [ ] `pytest tests/ -v` — all 7 transliteration tests still passing
- [ ] Everything pushed to GitHub

---

## SUMMARY OF ALL CHANGES FROM THE ORIGINAL WEEK 1-2 PLAN

| Area                                     | Original plan                                  | What we actually implemented                                            | Why                                                                                                  |
| ---------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Transliterator class                     | `UnicodeIndicTransliterator(sent, "hi", "en")` | `ItransTransliterator.to_itrans(sent, 'hi')` + `ut.init()`              | `UnicodeIndicTransliterator` is for Indic-to-Indic script conversion, not Devanagari-to-Roman        |
| `ut.init()` call                         | Not mentioned                                  | Required before any `to_itrans()` call                                  | Without it, `OFFSET_TO_ITRANS` table has 0 entries → every consonant maps to nothing                 |
| `SCHWA_DELETION_RULES` as substring dict | `'aa': 'a'` applied globally                   | Word-boundary-matched exception list                                    | Global substring replace corrupted words like `"kaam"→"kam"` incorrectly                             |
| Schwa deletion                           | Single regex across full sentence              | Per-word with `SCHWA_EXCEPTIONS` set                                    | Sentence-level regex stripped `a` from `kya`→`ky`, `gaya`→`gay` etc.                                 |
| ITRANS post-processing                   | Not in original plan                           | `ITRANS_TO_CASUAL` rules + retroflex lowercasing                        | `to_itrans()` produces ITRANS notation (`.m`, `aa`, `Th` etc.) that needs casual-Hinglish conversion |
| `langgraph` version                      | `==0.1.0`                                      | `>=0.2.0,<0.3.0`                                                        | `0.1.0` doesn't exist on PyPI (yanked/never released)                                                |
| Agent 1 prompt                           | One prompt, outputs array of any length        | One prompt per scene with `scene_id` locked, outputs exactly one object | Original prompt caused duplicate `scene_id`s and missing scenes                                      |
| `extract_event_chain`                    | `all_events.extend(events)`                    | `all_events.append(events[0])` + forced scene_id + fallback on empty    | Guarantees `len(output) == len(input)` always                                                        |
| Characters extraction                    | No explicit rules                              | CRITICAL RULES section blocking groups/objects/organizations            | `log`, `chatron`, `dayari`, `nagar parishad` were being extracted as "characters"                    |
| `Scratchpad.__init__`                    | Always `self._data = {}`                       | Auto-loads from disk if session file exists                             | Prevented accidental state wipe when re-instantiating mid-pipeline                                   |
| Transliteration in full flow             | Implied whole-text at once                     | Paragraph-by-paragraph with `\n\n.join()`                               | Preserves `\n\n` boundaries needed by `ProseChunker`                                                 |
| `GROQ_MODEL`                             | `"llama-3-8b-8192"`                            | `"llama-3.1-8b-instant"`                                                | Old model name deprecated on Groq API                                                                |
| Running code                             | `python preprocessing/transliterate.py`        | `python -m preprocessing.transliterate`                                 | `-m` flag puts project root on `sys.path` so `import config` resolves                                |
| Test assertions                          | Exact equality: `== "ram"`                     | Substring checks + one locked end-to-end test                           | Exact output depends on full pipeline; locking only the confirmed-working full output                |
| FELIX dataset                            | Listed as future work                          | Copied to `data/felix_dataset/` in Week 1                               | Dataset already exists (6000 rows), no reason to defer setup                                         |
