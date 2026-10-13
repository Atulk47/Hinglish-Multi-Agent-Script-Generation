# Hinglish Multi-Agent Script Generation — Complete 2-Month Implementation Plan

**Project ID:** PW26_VRB_07 | **Team:** Anika P, Anish M, Diya Ramani, Atul Krishna  
**Timeframe:** 8 Weeks | **Constraint:** Maximum Cost-Efficiency (Free-tier or near-zero cost)

---

## MASTER TOOL REGISTRY — Every Single Tool You Will Use

Before touching a single line of code, install and account for every tool listed below. This is your canonical reference.

### Language & Environment

| Tool             | Version | Why                                              | Cost |
| ---------------- | ------- | ------------------------------------------------ | ---- |
| Python           | 3.10.x  | All models, NLP libs support it cleanly          | Free |
| conda / venv     | Latest  | Isolated environments per module                 | Free |
| Jupyter Notebook | Latest  | Iterative dev, fast feedback on data exploration | Free |
| VS Code          | Latest  | IDE with Python extension                        | Free |

### Compute (Critical — Free GPU Access)

| Platform              | How to use                                 | GPU                | Free Quota      |
| --------------------- | ------------------------------------------ | ------------------ | --------------- |
| **Kaggle Notebooks**  | Primary GPU platform for fine-tuning       | 2× T4 (16 GB each) | 30 hrs/week     |
| **Google Colab Free** | Secondary, quick experiments               | T4                 | ~12 hrs/session |
| **Google Colab Pro**  | Only if you pool ₹500 (~$6) as a team      | A100               | Optional        |
| **Local CPU**         | Transliteration, Gate logic, preprocessing | CPU only           | Free, always    |

**Strategy:** Split heavy training jobs across Kaggle (primary) and Colab (secondary). Never pay for cloud VMs.

### Multi-Agent Orchestration Framework

**Choice: LangGraph (NOT LangChain standalone, NOT raw Python loops)**

Why LangGraph over alternatives:

- **vs. LangChain alone:** LangChain gives you chains; LangGraph gives you _stateful graphs_ with cycles — exactly what your feedback loop (Eval → Style Rewriter → Synthesizer) needs.
- **vs. AutoGen:** AutoGen is for conversational multi-agent chat (LLM-to-LLM). Yours is a sequential pipeline with one controlled feedback edge. AutoGen is overkill and harder to debug.
- **vs. CrewAI:** CrewAI is opinionated about roles and works best for research agents. Your pipeline has precise, custom logic gates — LangGraph gives you full control.
- **vs. raw Python:** You need state persistence across agents (the scratchpad). LangGraph's `StateGraph` with `TypedDict` state is exactly this pattern, built-in.

```
pip install langgraph langchain-core
```

LangGraph cost: **Free** (open source). No API calls made by the framework itself.

### LLM Access (Agent 1 — Narrative Analyst)

**Choice: Ollama running Llama-3-8B-Instruct locally**

- Ollama runs LLMs locally. Zero API cost. Zero rate limits.
- On Kaggle T4, you can load Llama-3-8B-Instruct in 4-bit quantization (fits in ~6GB VRAM).
- Alternatives considered and rejected:
  - OpenAI GPT-3.5: Costs money per call. Multiplied by all stories = not free.
  - Groq API free tier: 14,400 tokens/minute free. Viable BACKUP if Ollama fails. Use `groq` Python client.
  - HuggingFace Inference API free tier: Slow, rate-limited. Avoid for real runs.

```
# On Kaggle / Colab:
!curl -fsSL https://ollama.com/install.sh | sh
!ollama pull llama3:8b-instruct-q4_K_M   # 4-bit quant, ~4.7GB
```

Or via HuggingFace `transformers` directly (more control):

```
pip install transformers accelerate bitsandbytes
# Load with load_in_4bit=True
```

**Groq backup (truly free):**

```
pip install groq
# Free tier: llama-3-8b-8192 model, 14400 tokens/min
```

### Encoder Model (Agent 2 — Character Analyst + Validation Layer)

**Choice: MuRIL via HuggingFace `transformers` — run locally, no API**

```
pip install transformers torch sentencepiece
# Model: google/muril-base-cased (357MB) — fits on CPU easily
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model = AutoModel.from_pretrained("google/muril-base-cased")
```

Cost: **Free** (download once, run locally forever). The model is 357MB. Cache it in Kaggle's dataset storage so you don't re-download every session.

### Generation Model (Agent 3 — Dialogue Synthesizer)

**Choice: ai4bharat/IndicBART via HuggingFace `transformers`**

```
pip install transformers sentencepiece datasets
# Model: ai4bharat/indicbart (~600MB)
from transformers import MBartForConditionalGeneration, MBartTokenizer
```

Fine-tuning strategy: Use HuggingFace `Trainer` API with `peft` (LoRA) to reduce memory.

```
pip install peft trl datasets evaluate
```

Why LoRA for fine-tuning:

- Full fine-tune of IndicBART on T4 = out of memory.
- LoRA fine-tune: trains only ~1% of parameters. Fits in T4's 16GB.
- TRL's `SFTTrainer` makes this 20 lines of code.

### Transliteration

**Choice: IndicNLP Suite (`indic-nlp-library`) + custom regex**

```
pip install indic-nlp-library
# Also need: git clone https://github.com/anoopkunchukuttan/indic_nlp_resources
```

Set environment variable:

```python
from indicnlp import common
common.set_resources_path('/path/to/indic_nlp_resources')
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
```

This is a **deterministic, rule-based system**. Runs on CPU in <10ms/sentence. No GPU needed.

### Slang Rewriter (Pass 2 — FELIX-style)

**Choice: HuggingFace `bert-base-multilingual-cased` fine-tuned as a token-tagger**

FELIX is a text-editing paradigm. You implement it as:

1. A BERT-based **tagging model** that predicts KEEP/INSERT at each token.
2. An **insertion model** (smaller BERT or lookup) that selects which slang phrase to insert.

For Phase 1 (what you build now): Rule-based FELIX approximation using the dataset you already built (6000 pairs). The model learns from the `(neutral_sentence, slang_phrase, position, output)` schema.

```
pip install transformers torch seqeval
# Base: bert-base-multilingual-cased (fine-tune as sequence labeler)
```

Training this on Kaggle T4 with your 6000-sample dataset: ~30 minutes.

### Intent-Emotion Classifier (Gate 2 — HingBERT)

**Choice: `l3cube-pune/hindi-bert-sentiment-analysis` or `l3cube-pune/hing-bert`**

```
pip install transformers
# Model: l3cube-pune/hing-bert — trained on Hinglish, fits in 512MB
# Zero-shot classify or fine-tune on your intent labels
```

For zero-shot (no training data needed):

```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
# Labels: ["REQUEST_URGENCY","ASSERT_FRUSTRATION","ASSERT_NEUTRAL","ASSERT_POSITIVE"]
```

Cost: **Free**. All local inference.

### NLP Utilities (Character Analyst — NER + Coreference)

**Choice: `spacy` + `https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl`**

```
pip install spacy
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
pip install transformers  # for multilingual NER
```

For Hindi/Hinglish NER (since spaCy has no Hindi model):

```python
from transformers import pipeline
ner = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl",
               aggregation_strategy="simple")
```

Coreference resolution for Hinglish is hard — no off-the-shelf model exists. **Pragmatic solution:** Use the LLaMA-3 Narrative Analyst prompt to also extract the character-pronoun mapping. Ask it to return a `pronoun_map` field in its JSON output. This avoids needing a separate coref model entirely.

### Semantic Similarity (Validation Layer)

**Choice: `sentence-transformers` with MuRIL embeddings (custom wrapper)**

```
pip install sentence-transformers
from sentence_transformers import SentenceTransformer
# Use MuRIL via a SentenceTransformer wrapper:
model = SentenceTransformer("google/muril-base-cased")
```

Or manually: extract CLS token from MuRIL, compute cosine similarity with `sklearn`:

```python
from sklearn.metrics.pairwise import cosine_similarity
```

### Dataset Handling

**Choice: HuggingFace `datasets` library**

```
pip install datasets
# Load IndicDialogue:
from datasets import load_dataset
ds = load_dataset("Arnobio/IndicDialogue", "hi")  # Hindi split

# Load Hinglish-Everyday-Conversations-1M:
ds2 = load_dataset("findnitai/english-to-hinglish")
```

Both datasets are **free** on HuggingFace Hub. Download once, cache locally.

### State Management (Shared Scratchpad)

**Choice: Plain JSON files + Python `dataclasses` / `TypedDict`**

No database. No Redis. No vector store. Just:

```python
import json
from typing import TypedDict, List

# character.json structure
class CharacterProfile(TypedDict):
    name: str
    age: int
    social_role: str  # "peer" | "elder" | "authority"
    slang_level: int  # 0-10
    current_emotion: str
    turn_count: int
    stress_index: float
```

Read/write with `json.load()` / `json.dump()`. This is the scratchpad. LangGraph's `StateGraph` state dict wraps this.

### Evaluation Metrics

```
pip install evaluate rouge-score bert-score rapidfuzz
import evaluate
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
# BERTScore for semantic similarity
bertscore = evaluate.load("bertscore")
```

RapidFuzz for your existing dataset validation:

```python
from rapidfuzz import fuzz
score = fuzz.token_sort_ratio(neutral, slang_stripped_output)
```

### Experiment Tracking

**Choice: Weights & Biases (wandb) — free tier**

```
pip install wandb
wandb login  # free account at wandb.ai
```

Free tier: unlimited runs, 100GB storage. Use this for tracking all your fine-tuning loss curves, BLEU scores, validation thresholds. Critical for your final report.

### Version Control & Collaboration

**Choice: GitHub (private repo)**

```
git init
git remote add origin https://github.com/your-team/hinglish-mas
```

Create a `.gitignore` for model weights (too large). Store model weights in Kaggle Datasets or HuggingFace Hub (both free).

### Dependency Management

```
# requirements.txt — pin every version
torch==2.2.0
transformers==4.40.0
datasets==2.19.0
peft==0.10.0
trl==0.8.6
langgraph==0.1.0
langchain-core==0.2.0
indic-nlp-library==0.92
sentencepiece==0.2.0
sentence-transformers==2.7.0
evaluate==0.4.1
wandb==0.17.0
rapidfuzz==3.9.0
spacy==3.7.0
scikit-learn==1.4.0
```

---

## WEEK-BY-WEEK IMPLEMENTATION PLAN

---

### WEEK 1: Environment Setup + Transliteration Module

**Goal:** Everything installed, transliteration pipeline working end-to-end, tested on 200 sentences.

#### Day 1-2: Setup

1. Create a GitHub private repo. Add all 4 team members.
2. Create directory structure:

```
hinglish-mas/
├── preprocessing/
│   ├── transliterate.py
│   └── chunker.py
├── agents/
│   ├── agent1_narrative.py
│   ├── agent2_character.py
│   ├── agent3_synthesizer.py
│   └── agent4_slang_rewriter.py
├── gates/
│   └── three_gate_controller.py
├── memory/
│   └── scratchpad.py
├── evaluation/
│   ├── muril_validator.py
│   └── metrics.py
├── orchestrator/
│   └── pipeline.py          ← LangGraph StateGraph lives here
├── data/
│   ├── raw/
│   ├── processed/
│   └── felix_dataset/       ← your 6000-pair dataset goes here
├── models/
│   └── .gitkeep             ← weights stored in Kaggle, not git
├── tests/
│   └── test_transliterate.py
├── notebooks/
│   └── 01_data_exploration.ipynb
├── requirements.txt
├── config.py                ← all thresholds, model paths, hyperparams
└── README.md
```

3. Set up conda environment:

```bash
conda create -n hinglish python=3.10
conda activate hinglish
pip install -r requirements.txt
```

4. Download `indic_nlp_resources`:

```bash
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
# Set path in config.py:
INDIC_NLP_RESOURCES = "/absolute/path/to/indic_nlp_resources"
```

#### Day 3-4: Transliteration Module

File: `preprocessing/transliterate.py`

```python
import re
from indicnlp import common
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from indicnlp.tokenize import sentence_tokenize

common.set_resources_path(INDIC_NLP_RESOURCES)

class HindiTransliterator:
    SCHWA_DELETION_RULES = {
        'aa': 'a', 'ii': 'i', 'uu': 'u',
        'kyomki': 'kyunki', 'phir': 'phir',
        # Add 20-30 most frequent corrections here
    }

    def transliterate_sentence(self, devanagari_text: str) -> str:
        # Step 1: Split on danda, ?, !
        sentences = sentence_tokenize.sentence_split(devanagari_text, lang='hi')

        romanized = []
        for sent in sentences:
            # Step 2: UnicodeIndicTransliterator hi → en (ITRANS scheme)
            roman = UnicodeIndicTransliterator.transliterate(sent, "hi", "en")

            # Step 3: Schwa deletion post-processing
            roman = self._apply_schwa_deletion(roman)

            # Step 4: Phonetic smoothing
            roman = self._phonetic_smooth(roman)

            # Step 5: Strip noise, preserve punctuation
            roman = re.sub(r'[^\w\s\.\,\!\?\;\:\'\-]', '', roman)
            roman = re.sub(r'\s+', ' ', roman).strip()
            romanized.append(roman)

        return ' '.join(romanized)

    def _apply_schwa_deletion(self, text: str) -> str:
        # Remove terminal 'a' sounds that don't exist in spoken Hindi
        # e.g., "rama" → "ram"
        text = re.sub(r'([bcdfghjklmnpqrstvwxyz])a\b', r'\1', text)
        return text

    def _phonetic_smooth(self, text: str) -> str:
        for wrong, correct in self.SCHWA_DELETION_RULES.items():
            text = text.replace(wrong, correct)
        return text
```

#### Day 5: Prose Chunker

File: `preprocessing/chunker.py`

```python
import re
from typing import List

class ProseChunker:
    # Boundary signals in transliterated Hinglish
    LOCATION_CUES = ['ghar', 'school', 'bazaar', 'bahar', 'andar', 'office']
    TEMPORAL_CUES = ['subah', 'sham', 'raat', 'dopahar', 'agli baar', 'phir']

    def chunk(self, romanized_text: str) -> List[dict]:
        """Split into scene-level chunks. Returns list of {scene_id, text}"""
        # Strategy: Split on double newlines first (paragraph boundaries)
        paragraphs = [p.strip() for p in romanized_text.split('\n\n') if p.strip()]

        scenes = []
        for i, para in enumerate(paragraphs):
            scenes.append({
                "scene_id": f"scene_{i+1:02d}",
                "text": para,
                "has_location_cue": any(cue in para.lower() for cue in self.LOCATION_CUES),
                "has_temporal_cue": any(cue in para.lower() for cue in self.TEMPORAL_CUES)
            })
        return scenes
```

#### Day 6-7: Test + Baseline Verification

Write `tests/test_transliterate.py`:

```python
import pytest
from preprocessing.transliterate import HindiTransliterator

def test_basic_devanagari():
    t = HindiTransliterator()
    result = t.transliterate_sentence("तुम क्या कर रहे हो?")
    assert "tum" in result.lower()
    assert "kya" in result.lower()

def test_schwa_deletion():
    t = HindiTransliterator()
    result = t.transliterate_sentence("राम")
    assert result.strip().lower() == "ram"

def test_noise_stripping():
    t = HindiTransliterator()
    result = t.transliterate_sentence("नमस्ते! ##00:01:30##")
    assert "##" not in result
```

Run: `pytest tests/ -v`

**Week 1 Deliverable:** `transliterate.py` converts Devanagari → Romanized deterministically, passes all tests, processes 200-sentence held-out set. Push to GitHub.

---

### WEEK 2: Scratchpad Memory + Agent 1 (Narrative Analyst)

**Goal:** LangGraph state defined. Agent 1 produces valid JSON event chains from any input chunk.

#### Day 1-2: LangGraph State + Scratchpad

File: `memory/scratchpad.py`

```python
from typing import TypedDict, List, Optional, Annotated
import operator, json, os

# ── Character profile ──────────────────────────────────────────────────
class CharacterProfile(TypedDict):
    name: str
    age: Optional[int]
    social_role: str        # "peer" | "elder" | "authority"
    slang_level: int        # 0 (none) – 10 (heavy)
    relationship: dict      # {other_char_name: relationship_type}
    current_emotion: str    # "neutral" | "anxious" | "angry" | "happy" | ...
    turn_count: int
    stress_index: float     # 0.0 – 1.0

# ── Scene event ────────────────────────────────────────────────────────
class SceneEvent(TypedDict):
    scene_id: str
    key_events: List[str]
    characters: List[str]
    scene_goal: str
    location: str
    narrative_link: str     # "Following X, Y reacts..."

# ── Master LangGraph state ─────────────────────────────────────────────
class PipelineState(TypedDict):
    # Input
    raw_hindi_text: str

    # After preprocessing
    romanized_text: str
    scene_chunks: List[dict]

    # Agent 1 output
    event_chain: List[SceneEvent]

    # Agent 2 output
    character_profiles: dict   # {char_name: CharacterProfile}

    # Agent 3 output (Pass 1)
    neutral_script: List[dict]  # [{scene_id, speaker, dialogue}]

    # Agent 4 output (Pass 2)
    slang_script: List[dict]

    # Evaluation
    validation_scores: List[float]
    failed_scenes: List[str]    # scene_ids that need rewriting
    rewrite_count: int          # guard against infinite loops

    # Final
    final_script: str

class Scratchpad:
    def __init__(self, session_id: str):
        self.path = f"data/sessions/{session_id}_scratchpad.json"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._data = {}

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
```

#### Day 3-5: Agent 1 — Narrative Analyst

File: `agents/agent1_narrative.py`

The prompt is the most critical engineering task for Agent 1. Bad prompt = broken event chain = broken pipeline.

```python
import json, re
from typing import List

# ── Prompt template ────────────────────────────────────────────────────
NARRATIVE_ANALYST_PROMPT = """You are a screenplay structural analyst.
Your ONLY job is to extract a structured event chain from a Hindi story (written in Roman script).
Do NOT generate any dialogue. Only extract what exists.

Story text:
{story_text}

Previous scene summary (for continuity):
{previous_scene_summary}

Output a JSON array ONLY. No preamble. No explanation. Just the JSON.
Each element must have EXACTLY these fields:
{{
  "scene_id": "scene_01",
  "key_events": ["Event 1 description", "Event 2 description"],
  "characters": ["CharName1", "CharName2"],
  "scene_goal": "What this scene accomplishes narratively",
  "location": "Where this scene takes place",
  "narrative_link": "Following [previous event], [character] now [does X]",
  "pronoun_map": {{"usne": "Ram", "woh": "Sita"}}
}}

Rules:
- scene_id must be "scene_01", "scene_02" etc in order
- key_events: list every plot-significant event (minimum 2, maximum 6)
- narrative_link: MUST reference what happened in the previous scene
- pronoun_map: map every pronoun in this chunk to the character it refers to
- If the story has only one scene, still wrap it in an array: [{{...}}]
"""

class NarrativeAnalyst:
    def __init__(self, llm_backend: str = "ollama"):
        # llm_backend: "ollama" | "groq" | "hf"
        self.backend = llm_backend
        self._setup_llm()

    def _setup_llm(self):
        if self.backend == "ollama":
            # Ollama runs locally
            import subprocess
            # Start ollama server if not running
            # subprocess.Popen(["ollama", "serve"])
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="llama3:8b-instruct-q4_K_M",
                              temperature=0.1)  # low temp = deterministic extraction

        elif self.backend == "groq":
            # Free Groq API backup
            import os
            from groq import Groq
            self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

        elif self.backend == "hf":
            # Run Llama-3-8B-Instruct locally via transformers
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

    def extract_event_chain(self, scene_chunks: List[dict],
                             previous_summary: str = "This is the first scene.") -> List[dict]:
        all_events = []

        for chunk in scene_chunks:
            prompt = NARRATIVE_ANALYST_PROMPT.format(
                story_text=chunk["text"],
                previous_scene_summary=previous_summary
            )

            raw_output = self._call_llm(prompt)
            events = self._parse_json_output(raw_output, chunk["scene_id"])

            all_events.extend(events)
            # Update previous_summary for next scene
            if events:
                previous_summary = events[-1].get("scene_goal", previous_summary)

        return all_events

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "ollama":
            return self.llm.invoke(prompt)
        elif self.backend == "groq":
            response = self.groq_client.chat.completions.create(
                model="llama-3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content
        elif self.backend == "hf":
            result = self.pipe(prompt, return_full_text=False)
            return result[0]["generated_text"]

    def _parse_json_output(self, raw: str, fallback_scene_id: str) -> List[dict]:
        """Extract JSON from LLM output robustly."""
        # Find JSON array in output
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            # Try wrapping in array
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                raw = f"[{match.group()}]"
            else:
                # Return fallback structure
                return [{"scene_id": fallback_scene_id, "key_events": [],
                         "characters": [], "scene_goal": "Unknown",
                         "location": "Unknown", "narrative_link": "",
                         "pronoun_map": {}}]

        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            # Try to clean common LLM JSON errors
            clean = re.sub(r',\s*}', '}', raw)
            clean = re.sub(r',\s*]', ']', clean)
            try:
                return json.loads(clean)
            except:
                return []  # Escalate to rewrite
```

#### Day 6-7: Test Agent 1

Create a test story (write a 3-paragraph Hindi short story in Devanagari). Run the full flow:
`raw Hindi → transliterate → chunk → Agent1.extract_event_chain()` and inspect the JSON output.

Fix prompt iteratively until event chains are complete and the `narrative_link` field correctly references the prior scene.

**Week 2 Deliverable:** Full flow from raw Hindi text to a structured event chain JSON. Validated on 3 different test stories. Push to GitHub.

---

### WEEK 3: Agent 2 — Character Analyst + State Updater

**Goal:** `character.json` populated correctly for any story. State updater working. Gate 1 logic implemented.

#### Day 1-3: Character Analyst

File: `agents/agent2_character.py`

```python
from transformers import pipeline, AutoTokenizer, AutoModel
import torch, json
from memory.scratchpad import CharacterProfile, Scratchpad

# ── NER pipeline (multilingual) ───────────────────────────────────────
NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"

SOCIAL_ROLE_KEYWORDS = {
    "elder": ["pitaji", "mataji", "dada", "nana", "bade", "uncle", "aunty",
              "dadaji", "naani", "chacha", "maama"],
    "authority": ["sir", "madam", "doctor", "inspector", "principal",
                   "sahib", "ji"],
    "peer": []  # default
}

class CharacterAnalyst:
    def __init__(self, scratchpad: Scratchpad):
        self.scratchpad = scratchpad
        self.ner = pipeline("ner", model=NER_MODEL,
                            aggregation_strategy="simple")

        # MuRIL for embeddings
        self.muril_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
        self.muril_model = AutoModel.from_pretrained("google/muril-base-cased")
        self.muril_model.eval()

    def build_profiles(self, scene_chunks: List[dict],
                        event_chain: List[dict]) -> dict:
        profiles = {}

        for chunk, events in zip(scene_chunks, event_chain):
            # Extract character names from event chain (more reliable than raw NER)
            char_names = events.get("characters", [])

            # Supplement with NER on raw text
            ner_results = self.ner(chunk["text"])
            ner_names = [e["word"] for e in ner_results if e["entity_group"] == "PER"]

            all_names = list(set(char_names + ner_names))

            for name in all_names:
                if name not in profiles:
                    profiles[name] = self._build_initial_profile(
                        name, chunk["text"], events)

        # Apply pronoun map from event chain
        for events in event_chain:
            pronoun_map = events.get("pronoun_map", {})
            self.scratchpad.write("pronoun_map", pronoun_map)

        self.scratchpad.write("character_profiles", profiles)
        return profiles

    def _build_initial_profile(self, name: str, context_text: str,
                                events: dict) -> CharacterProfile:
        # Infer social role from context words near the name
        text_lower = context_text.lower()
        role = "peer"  # default
        for role_type, keywords in SOCIAL_ROLE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                role = role_type
                break

        # Slang level heuristic: elders get 0, peers get 5 (default)
        slang_level = 0 if role in ["elder", "authority"] else 5

        return CharacterProfile(
            name=name,
            age=None,
            social_role=role,
            slang_level=slang_level,
            relationship={},
            current_emotion="neutral",
            turn_count=0,
            stress_index=0.0
        )

    def update_state(self, char_name: str, scene_events: List[str],
                      scene_goal: str):
        """Rule-based state updater: update emotion and stress after each scene."""
        profiles = self.scratchpad.read("character_profiles") or {}
        if char_name not in profiles:
            return

        profile = profiles[char_name]

        # Simple keyword-based emotion update
        events_text = " ".join(scene_events).lower()

        EMOTION_TRIGGERS = {
            "angry": ["gussa", "naraaz", "chillaya", "maara", "ladai"],
            "anxious": ["dar", "ghabra", "pareshan", "chinta", "tension"],
            "happy": ["khush", "muskuraya", "hansa", "mila", "success"],
            "sad": ["roya", "udaas", "dukh", "aansu", "chala gaya"],
            "urgent": ["jaldi", "abhi", "turant", "emergency", "rukna"]
        }

        new_emotion = profile["current_emotion"]  # persist unless triggered
        for emotion, triggers in EMOTION_TRIGGERS.items():
            if any(t in events_text for t in triggers):
                new_emotion = emotion
                break

        profile["current_emotion"] = new_emotion
        profile["turn_count"] = profile.get("turn_count", 0) + 1

        # Stress index: accumulates with anxious/urgent, decays with happy
        stress = profile.get("stress_index", 0.0)
        if new_emotion in ["anxious", "angry", "urgent"]:
            stress = min(1.0, stress + 0.2)
        elif new_emotion == "happy":
            stress = max(0.0, stress - 0.3)
        profile["stress_index"] = stress

        profiles[char_name] = profile
        self.scratchpad.write("character_profiles", profiles)
```

#### Day 4-5: Gate 1 Implementation

File: `gates/three_gate_controller.py`

```python
import json
from memory.scratchpad import Scratchpad

class ThreeGateController:
    def __init__(self, scratchpad: Scratchpad,
                 slang_density_threshold: float = 0.25):
        self.scratchpad = scratchpad
        self.density_threshold = slang_density_threshold
        self._scene_slang_counts = {}   # {scene_id: {total: int, slang: int}}

    # ── GATE 1: Speaker Eligibility ───────────────────────────────────
    def gate1_speaker_eligible(self, speaker_name: str) -> bool:
        """Returns False if speaker should NEVER receive slang."""
        profiles = self.scratchpad.read("character_profiles") or {}
        profile = profiles.get(speaker_name, {})
        role = profile.get("social_role", "peer")
        slang_level = profile.get("slang_level", 5)

        # Hard block for elders and authority figures
        if role in ["elder", "authority"]:
            return False

        # Soft block if slang_level is explicitly set to 0
        if slang_level == 0:
            return False

        return True

    # ── GATE 2: Pragmatic Eligibility ─────────────────────────────────
    def gate2_pragmatic_eligible(self, utterance: str,
                                   intent_classifier) -> tuple[bool, str]:
        """Returns (eligible, intent_label)"""
        SLANG_PERMITTED_INTENTS = [
            "REQUEST_URGENCY", "ASSERT_FRUSTRATION",
            "ASSERT_POSITIVE", "EXPRESS_STRESS", "CASUAL_ASSERTION"
        ]
        SLANG_BLOCKED_INTENTS = [
            "INFORM_NEUTRAL", "ASSERT_NEUTRAL", "QUESTION_FORMAL"
        ]

        intent = intent_classifier.classify(utterance)
        eligible = intent in SLANG_PERMITTED_INTENTS
        return eligible, intent

    # ── GATE 3: Density Control ────────────────────────────────────────
    def gate3_density_ok(self, scene_id: str, utterance_token_count: int) -> bool:
        """Returns False if scene has reached slang density budget."""
        if scene_id not in self._scene_slang_counts:
            self._scene_slang_counts[scene_id] = {"total": 0, "slang": 0}

        counts = self._scene_slang_counts[scene_id]
        counts["total"] += utterance_token_count

        if counts["total"] == 0:
            return True

        current_density = counts["slang"] / counts["total"]
        return current_density < self.density_threshold

    def record_slang_insertion(self, scene_id: str, slang_token_count: int):
        if scene_id in self._scene_slang_counts:
            self._scene_slang_counts[scene_id]["slang"] += slang_token_count

    def run_all_gates(self, speaker: str, utterance: str, scene_id: str,
                       intent_classifier) -> dict:
        """Run gates sequentially. Returns gate results + go/no-go decision."""
        g1 = self.gate1_speaker_eligible(speaker)
        if not g1:
            return {"go": False, "gate_stopped": 1, "intent": None}

        g2, intent = self.gate2_pragmatic_eligible(utterance, intent_classifier)
        if not g2:
            return {"go": False, "gate_stopped": 2, "intent": intent}

        token_count = len(utterance.split())
        g3 = self.gate3_density_ok(scene_id, token_count)
        if not g3:
            return {"go": False, "gate_stopped": 3, "intent": intent}

        return {"go": True, "gate_stopped": None, "intent": intent}
```

#### Day 6-7: Intent Classifier (Gate 2 Core)

File: `agents/intent_classifier.py`

```python
from transformers import pipeline

class IntentClassifier:
    """Zero-shot intent classifier using BART-MNLI."""

    CANDIDATE_LABELS = [
        "REQUEST_URGENCY",    # "jaldi karo", "abhi batao"
        "ASSERT_FRUSTRATION", # "pagal hai kya", "bore ho gaya"
        "ASSERT_POSITIVE",    # "bahut acha hai", "maza aa gaya"
        "EXPRESS_STRESS",     # "tension mat lo", "pareshan hun"
        "CASUAL_ASSERTION",   # general casual statements
        "INFORM_NEUTRAL",     # neutral information delivery
        "QUESTION_FORMAL",    # formal questions
    ]

    def __init__(self):
        # facebook/bart-large-mnli is free, no auth required
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU is fine for inference
        )

    def classify(self, text: str) -> str:
        result = self.classifier(text, self.CANDIDATE_LABELS,
                                  multi_label=False)
        return result["labels"][0]
```

**Week 3 Deliverable:** Character profiles written to scratchpad correctly. State updater runs after each scene. Gate 1 blocks elders. Gate 2 classifies intents correctly. Gate 3 tracks density. All 3 gates integrated in `ThreeGateController.run_all_gates()`.

---

### WEEK 4: Agent 3 — Dialogue Synthesizer Fine-tuning (Pass 1)

**Goal:** IndicBART fine-tuned in two stages. Pass 1 produces neutral, semantically faithful Hinglish dialogue.

**This entire week runs on Kaggle T4 GPU notebooks.**

#### Day 1-2: Dataset Preparation

Notebook: `notebooks/02_indicbart_finetune_stage1.ipynb`

**Stage 1 Dataset: IndicDialogue (Hindi subtitles)**

```python
from datasets import load_dataset
import pandas as pd

# Load Hindi subset (~6.85M dialogue lines)
ds = load_dataset("Arnobio/IndicDialogue", "hi", split="train")

# Format for seq2seq: context → next_utterance
# We treat 3-turn context as input, next utterance as target
def format_for_seq2seq(examples):
    inputs = []
    targets = []

    for i in range(2, len(examples["text"])):
        context = f"SPEAKER_A: {examples['text'][i-2]}\nSPEAKER_B: {examples['text'][i-1]}"
        target = examples["text"][i]

        # Add screenplay formatting prefix
        if i % 2 == 0:
            target = f"SPEAKER_A: {target}"
        else:
            target = f"SPEAKER_B: {target}"

        inputs.append(context)
        targets.append(target)

    return {"input": inputs, "target": targets}

# Take 100K samples (enough, faster training)
small_ds = ds.select(range(100000))
```

**Stage 2 Dataset: Hinglish-Everyday-Conversations-1M**

```python
ds2 = load_dataset("findnitai/english-to-hinglish", split="train")
# This has "en" (English) and "hi_ng" (Hinglish) columns
# Format: English context → Hinglish response

def format_hinglish(examples):
    return {
        "input": examples["en"],
        "target": examples["hi_ng"]
    }
```

#### Day 3-5: IndicBART Fine-tuning with LoRA

```python
from transformers import (MBartForConditionalGeneration, MBartTokenizer,
                           DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                           Seq2SeqTrainer)
from peft import LoraConfig, get_peft_model, TaskType
import torch

MODEL_NAME = "ai4bharat/indicbart"

tokenizer = MBartTokenizer.from_pretrained(MODEL_NAME)
# Set source and target language tokens
tokenizer.src_lang = "hi_IN"
tokenizer.tgt_lang = "hi_IN"  # Hinglish is treated as Hindi for tokenization

model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

# ── LoRA config ─────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,              # rank — higher = more capacity but more VRAM
    lora_alpha=32,     # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers to adapt
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params ~1.7M / total ~611M = ~0.28%

# ── Tokenize ──────────────────────────────────────────────────────────
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 128

def tokenize(batch):
    model_inputs = tokenizer(
        batch["input"], max_length=MAX_INPUT_LEN,
        truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"], max_length=MAX_TARGET_LEN,
            truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = formatted_ds.map(tokenize, batched=True,
                                  remove_columns=["input","target"])

# ── Training args ─────────────────────────────────────────────────────
args = Seq2SeqTrainingArguments(
    output_dir="./indicbart-stage1",
    num_train_epochs=3,
    per_device_train_batch_size=8,   # fits T4 with LoRA
    gradient_accumulation_steps=4,   # effective batch = 32
    learning_rate=5e-4,
    warmup_steps=500,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    predict_with_generate=True,
    fp16=True,                        # half precision = 2× speed on T4
    report_to="wandb",
    logging_steps=100,
    load_best_model_at_end=True
)

# ── Stage 1 Trainer ───────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

trainer.train()
# Save LoRA adapter only (not full model — ~8MB vs 600MB)
model.save_pretrained("./stage1-lora-adapter")
tokenizer.save_pretrained("./stage1-lora-adapter")
```

**Stage 2 Fine-tuning:** Same code, but:

- Start from Stage 1 checkpoint
- Use Hinglish-Everyday-Conversations-1M dataset
- Lower learning rate: `1e-4`
- Fewer epochs: 2

#### Day 6: Agent 3 Inference Wrapper

File: `agents/agent3_synthesizer.py`

```python
from transformers import MBartForConditionalGeneration, MBartTokenizer
from peft import PeftModel
import torch

PASS1_PROMPT_TEMPLATE = """Generate neutral Hinglish screenplay dialogue. No slang.
Scene context: {scene_goal}
Location: {location}
Previous event: {narrative_link}
Speaker: {speaker_name} (role: {speaker_role}, emotion: {emotion})
Listener: {listener_name}
Intent: {utterance_intent}
Events to cover: {key_events}

Output ONLY the dialogue line(s) in format: SPEAKER_NAME: dialogue
No scene directions. No explanation."""

class DialogueSynthesizer:
    def __init__(self, model_path: str, adapter_path: str):
        self.tokenizer = MBartTokenizer.from_pretrained(adapter_path)
        base_model = MBartForConditionalGeneration.from_pretrained(model_path)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def generate_pass1(self, scene: dict, character_profiles: dict,
                        event_chain_entry: dict) -> str:
        speaker = event_chain_entry["characters"][0] if event_chain_entry["characters"] else "SPEAKER"
        listener = event_chain_entry["characters"][1] if len(event_chain_entry["characters"]) > 1 else "OTHER"

        profile = character_profiles.get(speaker, {})

        prompt = PASS1_PROMPT_TEMPLATE.format(
            scene_goal=event_chain_entry["scene_goal"],
            location=event_chain_entry["location"],
            narrative_link=event_chain_entry["narrative_link"],
            speaker_name=speaker,
            speaker_role=profile.get("social_role", "peer"),
            emotion=profile.get("current_emotion", "neutral"),
            listener_name=listener,
            utterance_intent="dialogue",
            key_events="; ".join(event_chain_entry["key_events"])
        )

        inputs = self.tokenizer(prompt, return_tensors="pt",
                                 max_length=512, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Day 7: Integration Test

Wire together: `transliterate → chunk → agent1 → agent2 → agent3` on a test story. Verify the Pass 1 output is dialogue-formatted and covers the key events.

**Week 4 Deliverable:** IndicBART fine-tuned (Stage 1 + Stage 2). Agent 3 inference wrapper working. Pass 1 produces neutral Hinglish dialogue for a test story.

---

### WEEK 5: FELIX Slang Rewriter (Pass 2) + Dataset Finalization

**Goal:** FELIX tagger trained on your 6000-pair dataset. Pass 2 inserts slang correctly.

#### Day 1-2: Load and Validate Your 6000-Pair Dataset

```python
import pandas as pd, json
from rapidfuzz import fuzz

# Load your existing dataset (CSV or JSONL format)
df = pd.read_csv("data/felix_dataset/slang_pairs.csv")
# Expected columns: neutral, slang_phrase, position, intent, function_type, output

# Validation checks
def validate_row(row):
    errors = []
    # Check 1: slang phrase present in output
    if row["slang_phrase"] not in row["output"]:
        errors.append("slang_missing")

    # Check 2: semantic preservation (RapidFuzz)
    # Strip slang phrase from output before comparing
    stripped = row["output"].replace(row["slang_phrase"], "").strip()
    similarity = fuzz.token_sort_ratio(row["neutral"], stripped)
    if similarity < 70:
        errors.append(f"semantic_drift_{similarity:.0f}")

    # Check 3: position verification
    output_tokens = row["output"].split()
    slang_tokens = row["slang_phrase"].split()
    # START: slang appears in first 3 tokens
    if row["position"] == "START":
        if not any(t in output_tokens[:3] for t in slang_tokens):
            errors.append("position_mismatch")

    return errors

# Run validation
df["errors"] = df.apply(validate_row, axis=1)
bad_rows = df[df["errors"].apply(len) > 0]
print(f"Bad rows: {len(bad_rows)}/{len(df)}")
# Remove bad rows
df_clean = df[df["errors"].apply(len) == 0]
print(f"Clean dataset: {len(df_clean)} rows")
```

#### Day 3-5: FELIX Token Tagger Training

The FELIX paradigm for your task:

- Input: tokenized `neutral_sentence`
- Label per token: `KEEP` (1) or `INSERT_BEFORE` (2) or `INSERT_AFTER` (3)
- Separately: given the insertion position, predict which slang phrase

```python
from transformers import (BertTokenizerFast, BertForTokenClassification,
                           TrainingArguments, Trainer, DataCollatorForTokenClassification)
from datasets import Dataset
import torch

# ── Model: multilingual BERT as token tagger ─────────────────────────
MODEL = "bert-base-multilingual-cased"
tokenizer = BertTokenizerFast.from_pretrained(MODEL)

LABEL2ID = {"KEEP": 0, "INSERT_BEFORE": 1, "INSERT_AFTER": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def create_token_labels(neutral: str, output: str,
                          slang_phrase: str, position: str) -> dict:
    """Create KEEP/INSERT labels for each token in neutral_sentence."""
    neutral_tokens = neutral.split()
    labels = [LABEL2ID["KEEP"]] * len(neutral_tokens)

    if position == "START":
        labels[0] = LABEL2ID["INSERT_BEFORE"]
    elif position == "END":
        labels[-1] = LABEL2ID["INSERT_AFTER"]
    elif position == "MID":
        mid = len(neutral_tokens) // 2
        labels[mid] = LABEL2ID["INSERT_BEFORE"]

    return {"tokens": neutral_tokens, "labels": labels}

def tokenize_and_align(examples):
    tokenized = tokenizer(examples["neutral"], truncation=True,
                           max_length=128, is_split_into_words=True)

    label_ids = []
    for i, labels in enumerate(examples["word_labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)  # special tokens
            elif word_id != prev_word_id:
                aligned.append(labels[word_id])
            else:
                aligned.append(-100)  # continuation tokens
            prev_word_id = word_id
        label_ids.append(aligned)

    tokenized["labels"] = label_ids
    return tokenized

# Prepare dataset
label_data = df_clean.apply(
    lambda r: create_token_labels(r["neutral"], r["output"],
                                    r["slang_phrase"], r["position"]), axis=1)

hf_ds = Dataset.from_dict({
    "neutral": df_clean["neutral"].tolist(),
    "word_labels": [d["labels"] for d in label_data],
    "slang_phrase": df_clean["slang_phrase"].tolist(),
    "intent": df_clean["intent"].tolist()
})

tokenized_ds = hf_ds.map(tokenize_and_align, batched=True)
split = tokenized_ds.train_test_split(test_size=0.1)

# ── Train FELIX tagger ────────────────────────────────────────────────
model = BertForTokenClassification.from_pretrained(
    MODEL, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)

args = TrainingArguments(
    output_dir="./felix-tagger",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    learning_rate=3e-5,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

trainer.train()
model.save_pretrained("./felix-tagger-final")
```

Training time: ~15 min on T4 for 6000 samples, 10 epochs.

#### Day 6-7: Full Slang Rewriter

File: `agents/agent4_slang_rewriter.py`

```python
from transformers import BertTokenizerFast, BertForTokenClassification
import torch, random
from gates.three_gate_controller import ThreeGateController

class SLangRewriter:
    def __init__(self, tagger_path: str, scratchpad, intent_classifier):
        self.tokenizer = BertTokenizerFast.from_pretrained(tagger_path)
        self.tagger = BertForTokenClassification.from_pretrained(tagger_path)
        self.tagger.eval()
        self.gate_controller = ThreeGateController(scratchpad)
        self.intent_classifier = intent_classifier

        # Slang lexicon: {function_type: [slang_phrases]}
        self.slang_lexicon = self._load_slang_lexicon()
        self._usage_tracker = {}  # for weighted selection

    def _load_slang_lexicon(self) -> dict:
        """Load from your dataset's unique slang markers."""
        import pandas as pd
        df = pd.read_csv("data/felix_dataset/slang_pairs.csv")
        lexicon = {}
        for _, row in df.iterrows():
            fn = row["function_type"]
            if fn not in lexicon:
                lexicon[fn] = []
            if row["slang_phrase"] not in lexicon[fn]:
                lexicon[fn].append(row["slang_phrase"])
        return lexicon

    def rewrite(self, neutral_utterance: str, speaker: str,
                 scene_id: str) -> str:
        # Run all gates
        gate_result = self.gate_controller.run_all_gates(
            speaker, neutral_utterance, scene_id, self.intent_classifier)

        if not gate_result["go"]:
            return neutral_utterance  # Return Pass 1 as-is

        intent = gate_result["intent"]

        # Predict insertion positions using FELIX tagger
        insertion_positions = self._predict_insertions(neutral_utterance)

        if not insertion_positions:
            return neutral_utterance

        # Select slang phrase (weighted by usage frequency)
        slang_phrase = self._select_slang(intent, scene_id)

        if not slang_phrase:
            return neutral_utterance

        # Apply insertion
        result = self._apply_insertion(neutral_utterance,
                                        insertion_positions[0], slang_phrase)

        # Record for Gate 3
        self.gate_controller.record_slang_insertion(
            scene_id, len(slang_phrase.split()))

        return result

    def _predict_insertions(self, text: str) -> list:
        inputs = self.tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.tagger(**inputs).logits

        predictions = torch.argmax(logits, dim=-1)[0].tolist()
        word_ids = inputs.word_ids()

        insertion_positions = []
        for i, (pred, word_id) in enumerate(zip(predictions, word_ids)):
            if word_id is not None and pred in [1, 2]:  # INSERT_BEFORE or INSERT_AFTER
                insertion_positions.append((word_id, pred))

        return insertion_positions

    def _select_slang(self, intent: str, scene_id: str) -> str:
        """Weighted selection — down-weight already-used markers."""
        # Map intent to slang function type
        INTENT_TO_FUNCTION = {
            "REQUEST_URGENCY": "urgency_marker",
            "ASSERT_FRUSTRATION": "frustration_marker",
            "ASSERT_POSITIVE": "positive_evaluative",
            "EXPRESS_STRESS": "stress_marker",
            "CASUAL_ASSERTION": "filler"
        }
        fn_type = INTENT_TO_FUNCTION.get(intent, "filler")
        candidates = self.slang_lexicon.get(fn_type,
                       self.slang_lexicon.get("filler", ["yaar"]))

        # Down-weight frequently used phrases in this scene
        tracker_key = f"{scene_id}_{fn_type}"
        usage = self._usage_tracker.get(tracker_key, {})
        weights = [1.0 / (1 + usage.get(c, 0)) for c in candidates]

        # Weighted random choice
        selected = random.choices(candidates, weights=weights, k=1)[0]

        # Update tracker
        if tracker_key not in self._usage_tracker:
            self._usage_tracker[tracker_key] = {}
        self._usage_tracker[tracker_key][selected] = \
            self._usage_tracker[tracker_key].get(selected, 0) + 1

        return selected

    def _apply_insertion(self, text: str, word_position: tuple,
                          slang: str) -> str:
        tokens = text.split()
        word_idx, insert_type = word_position
        word_idx = min(word_idx, len(tokens) - 1)

        if insert_type == 1:  # INSERT_BEFORE
            tokens.insert(word_idx, slang)
        else:  # INSERT_AFTER
            tokens.insert(word_idx + 1, slang)

        return " ".join(tokens)
```

**Week 5 Deliverable:** FELIX tagger trained. Full Pass 2 rewrites Pass 1 output with contextually appropriate slang, gated by all 3 gates.

---

### WEEK 6: MuRIL Validation Layer + LangGraph Orchestration

**Goal:** Full pipeline connected via LangGraph. Validation feedback loop working.

#### Day 1-2: MuRIL Validation Layer

File: `evaluation/muril_validator.py`

```python
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch, numpy as np

class MuRILValidator:
    SIMILARITY_THRESHOLD = 0.82  # Empirically determined — tune in Week 7

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
        self.model = AutoModel.from_pretrained("google/muril-base-cased")
        self.model.eval()

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=512,
                                 padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return cls_embedding

    def compute_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return float(cosine_similarity(emb1, emb2)[0][0])

    def validate_scene(self, pass1_dialogue: str, pass2_dialogue: str,
                        scene_events: list) -> dict:
        """Returns validation result dict."""
        # Check 1: Semantic similarity between Pass 1 and Pass 2
        p1_p2_similarity = self.compute_similarity(pass1_dialogue, pass2_dialogue)

        # Check 2: Event coverage (Precision/Recall/F1)
        event_coverage = self._compute_event_coverage(
            pass2_dialogue, scene_events)

        passed = (p1_p2_similarity >= self.SIMILARITY_THRESHOLD and
                   event_coverage["f1"] >= 0.5)

        return {
            "passed": passed,
            "p1_p2_similarity": p1_p2_similarity,
            "event_coverage": event_coverage,
            "action": "accept" if passed else "rewrite"
        }

    def _compute_event_coverage(self, dialogue: str, events: list) -> dict:
        """Simple keyword overlap for event mention checking."""
        dialogue_lower = dialogue.lower()
        covered = 0
        for event in events:
            # Check if key words from event appear in dialogue
            event_words = set(event.lower().split()) - {"the","a","an","is","was","in","of","and"}
            if event_words and any(w in dialogue_lower for w in event_words):
                covered += 1

        precision = covered / max(len(events), 1)
        recall = covered / max(len(events), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {"precision": precision, "recall": recall, "f1": f1,
                "covered": covered, "total": len(events)}
```

#### Day 3-5: LangGraph Pipeline Orchestration

File: `orchestrator/pipeline.py`

This is the central piece — the LangGraph `StateGraph` that connects everything.

```python
from langgraph.graph import StateGraph, END
from typing import Literal
import json

from memory.scratchpad import PipelineState, Scratchpad
from preprocessing.transliterate import HindiTransliterator
from preprocessing.chunker import ProseChunker
from agents.agent1_narrative import NarrativeAnalyst
from agents.agent2_character import CharacterAnalyst
from agents.agent3_synthesizer import DialogueSynthesizer
from agents.agent4_slang_rewriter import SLangRewriter
from agents.intent_classifier import IntentClassifier
from evaluation.muril_validator import MuRILValidator
import config

MAX_REWRITE_ATTEMPTS = 3

class HinglishPipeline:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.scratchpad = Scratchpad(session_id)

        # Initialize all components
        self.transliterator = HindiTransliterator()
        self.chunker = ProseChunker()
        self.agent1 = NarrativeAnalyst(llm_backend=config.LLM_BACKEND)
        self.agent2 = CharacterAnalyst(self.scratchpad)
        self.agent3 = DialogueSynthesizer(config.INDICBART_BASE_PATH,
                                           config.INDICBART_ADAPTER_PATH)
        self.intent_clf = IntentClassifier()
        self.agent4 = SLangRewriter(config.FELIX_TAGGER_PATH,
                                     self.scratchpad, self.intent_clf)
        self.validator = MuRILValidator()

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(PipelineState)

        # Add nodes (one per pipeline stage)
        workflow.add_node("preprocess", self._node_preprocess)
        workflow.add_node("plan_narrative", self._node_plan_narrative)
        workflow.add_node("build_characters", self._node_build_characters)
        workflow.add_node("generate_pass1", self._node_generate_pass1)
        workflow.add_node("inject_slang", self._node_inject_slang)
        workflow.add_node("validate", self._node_validate)
        workflow.add_node("assemble_output", self._node_assemble_output)

        # Define edges (flow)
        workflow.set_entry_point("preprocess")
        workflow.add_edge("preprocess", "plan_narrative")
        workflow.add_edge("plan_narrative", "build_characters")
        workflow.add_edge("build_characters", "generate_pass1")
        workflow.add_edge("generate_pass1", "inject_slang")
        workflow.add_edge("inject_slang", "validate")

        # Conditional edge: validate → rewrite (if failed) or assemble (if passed)
        workflow.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {
                "rewrite": "inject_slang",     # Cycle back for failed scenes
                "assemble": "assemble_output", # Proceed to final output
                "force_assemble": "assemble_output"  # Max retries exceeded
            }
        )
        workflow.add_edge("assemble_output", END)

        return workflow.compile()

    # ── Node functions ─────────────────────────────────────────────────
    def _node_preprocess(self, state: PipelineState) -> PipelineState:
        romanized = self.transliterator.transliterate_sentence(
            state["raw_hindi_text"])
        chunks = self.chunker.chunk(romanized)
        state["romanized_text"] = romanized
        state["scene_chunks"] = chunks
        return state

    def _node_plan_narrative(self, state: PipelineState) -> PipelineState:
        event_chain = self.agent1.extract_event_chain(state["scene_chunks"])
        state["event_chain"] = event_chain
        self.scratchpad.write("event_chain", event_chain)
        return state

    def _node_build_characters(self, state: PipelineState) -> PipelineState:
        profiles = self.agent2.build_profiles(
            state["scene_chunks"], state["event_chain"])
        state["character_profiles"] = profiles
        return state

    def _node_generate_pass1(self, state: PipelineState) -> PipelineState:
        neutral_script = []
        for chunk, events in zip(state["scene_chunks"], state["event_chain"]):
            dialogue = self.agent3.generate_pass1(
                chunk, state["character_profiles"], events)
            neutral_script.append({
                "scene_id": events["scene_id"],
                "dialogue": dialogue,
                "events": events
            })
        state["neutral_script"] = neutral_script
        return state

    def _node_inject_slang(self, state: PipelineState) -> PipelineState:
        slang_script = []
        failed_scenes = set(state.get("failed_scenes", []))

        for scene in state["neutral_script"]:
            scene_id = scene["scene_id"]
            # Skip scenes that already passed validation
            if (state.get("slang_script") and
                scene_id not in failed_scenes):
                # Carry over already-validated scene
                existing = next((s for s in state["slang_script"]
                                  if s["scene_id"] == scene_id), None)
                if existing:
                    slang_script.append(existing)
                    continue

            # Parse dialogue lines from Pass 1
            lines = scene["dialogue"].split('\n')
            rewritten_lines = []
            for line in lines:
                if ':' in line:
                    speaker, utterance = line.split(':', 1)
                    speaker = speaker.strip()
                    utterance = utterance.strip()
                    rewritten = self.agent4.rewrite(utterance, speaker, scene_id)
                    rewritten_lines.append(f"{speaker}: {rewritten}")
                else:
                    rewritten_lines.append(line)

            slang_script.append({
                "scene_id": scene_id,
                "dialogue": '\n'.join(rewritten_lines),
                "events": scene["events"]
            })

        state["slang_script"] = slang_script
        return state

    def _node_validate(self, state: PipelineState) -> PipelineState:
        failed_scenes = []
        scores = []

        for neutral_scene, slang_scene in zip(
                state["neutral_script"], state["slang_script"]):

            result = self.validator.validate_scene(
                neutral_scene["dialogue"],
                slang_scene["dialogue"],
                neutral_scene["events"].get("key_events", [])
            )

            scores.append(result["p1_p2_similarity"])

            if not result["passed"]:
                failed_scenes.append(slang_scene["scene_id"])

        state["validation_scores"] = scores
        state["failed_scenes"] = failed_scenes
        state["rewrite_count"] = state.get("rewrite_count", 0) + 1
        return state

    def _route_after_validation(self, state: PipelineState) -> Literal["rewrite","assemble","force_assemble"]:
        if not state["failed_scenes"]:
            return "assemble"
        if state["rewrite_count"] >= MAX_REWRITE_ATTEMPTS:
            return "force_assemble"  # Accept best effort
        return "rewrite"

    def _node_assemble_output(self, state: PipelineState) -> PipelineState:
        script_lines = ["# HINGLISH SCREENPLAY\n"]

        for scene in state["slang_script"]:
            scene_id = scene["scene_id"]
            events = scene.get("events", {})

            script_lines.append(f"\n## {scene_id.upper()}")
            script_lines.append(f"*{events.get('location', 'LOCATION')}*\n")
            script_lines.append(scene["dialogue"])
            script_lines.append("\n---")

        state["final_script"] = '\n'.join(script_lines)
        return state

    def run(self, hindi_story: str) -> str:
        initial_state = PipelineState(
            raw_hindi_text=hindi_story,
            romanized_text="",
            scene_chunks=[],
            event_chain=[],
            character_profiles={},
            neutral_script=[],
            slang_script=[],
            validation_scores=[],
            failed_scenes=[],
            rewrite_count=0,
            final_script=""
        )

        final_state = self.graph.invoke(initial_state)
        return final_state["final_script"]
```

#### Day 6-7: End-to-End Integration Test

```python
# run_pipeline.py
from orchestrator.pipeline import HinglishPipeline

story = """
राम और श्याम बचपन के दोस्त थे। एक दिन राम को पता चला कि उसका नया काम
मिल गया है। वो बहुत खुश था और श्याम को बताने के लिए भागा।
श्याम घर पर था। राम ने दरवाजा खटखटाया।
"""

pipeline = HinglishPipeline(session_id="test_001")
script = pipeline.run(story)
print(script)
```

Verify: Is the output in screenplay format? Does it cover the events? Does it have slang in peer characters' dialogue but not in elders'?

**Week 6 Deliverable:** Full end-to-end pipeline running via LangGraph. Feedback loop cycling on failed scenes. Final script assembled.

---

### WEEK 7: Evaluation Suite + Threshold Tuning + Bug Fixes

**Goal:** Quantitative evaluation working. Thresholds empirically tuned. All bugs from integration fixed.

#### Day 1-2: Build Evaluation Suite

File: `evaluation/metrics.py`

```python
import evaluate
from rapidfuzz import fuzz
import numpy as np

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def compute_bleu(predictions: list, references: list) -> dict:
    return bleu.compute(predictions=predictions,
                         references=[[r] for r in references])

def compute_rouge(predictions: list, references: list) -> dict:
    return rouge.compute(predictions=predictions, references=references)

def compute_slang_density(dialogue: str, slang_lexicon: list) -> float:
    tokens = dialogue.split()
    slang_count = sum(1 for t in tokens if t in slang_lexicon)
    return slang_count / max(len(tokens), 1)

def compute_event_coverage_f1(generated_dialogue: str,
                                expected_events: list) -> float:
    """What % of expected events are mentioned in the dialogue?"""
    dialogue_lower = generated_dialogue.lower()
    covered = sum(1 for event in expected_events
                   if any(w in dialogue_lower for w in event.lower().split()))
    precision = covered / max(len(expected_events), 1)
    return precision  # simplified for now

def run_full_evaluation(neutral_script: list, slang_script: list,
                          event_chain: list, slang_lexicon: list) -> dict:
    results = {
        "per_scene": [],
        "aggregate": {}
    }

    for neutral, slang, events in zip(neutral_script, slang_script, event_chain):
        scene_result = {
            "scene_id": slang["scene_id"],
            "slang_density": compute_slang_density(
                slang["dialogue"], slang_lexicon),
            "event_coverage": compute_event_coverage_f1(
                slang["dialogue"], events.get("key_events", [])),
            "semantic_similarity": fuzz.token_sort_ratio(
                neutral["dialogue"], slang["dialogue"]) / 100
        }
        results["per_scene"].append(scene_result)

    results["aggregate"] = {
        "mean_slang_density": np.mean([s["slang_density"] for s in results["per_scene"]]),
        "mean_event_coverage": np.mean([s["event_coverage"] for s in results["per_scene"]]),
        "mean_semantic_similarity": np.mean([s["semantic_similarity"] for s in results["per_scene"]])
    }

    return results
```

#### Day 3-4: Threshold Tuning Experiments

Run the pipeline on 10 different Hindi stories (write or source from Hindi Wikipedia / Premchand short stories — all copyright-free). Record:

- MuRIL similarity distribution (Pass 1 vs Pass 2)
- Event coverage F1 scores
- Slang density per scene

Tune `SIMILARITY_THRESHOLD` in `muril_validator.py` and `density_threshold` in `three_gate_controller.py` to find the sweet spot where:

- Enough slang is injected to sound authentic
- Not so much that semantic drift occurs

Use W&B to log all experiment results. Compare runs visually.

#### Day 5-7: Bug Fixing Checklist

Work through every known failure mode:

1. **JSON parse failure in Agent 1:** Add retry logic (call LLM again if JSON fails). Max 3 retries.
2. **IndicBART hallucination:** If Pass 1 output doesn't contain any character name from the event chain, regenerate with higher `num_beams`.
3. **Gate 3 blocking too aggressively:** If >60% of scenes have 0 slang, lower `density_threshold`.
4. **Character profile missing for a speaker:** Add a default profile for unknown speakers.
5. **LangGraph cycle detection:** Verify `rewrite_count` correctly prevents infinite loops.
6. **Scratchpad corruption:** Add `try/except` around all JSON reads.

**Week 7 Deliverable:** Pipeline running on 10 test stories. Thresholds tuned. W&B dashboard showing evaluation metrics. All known bugs fixed.

---

### WEEK 8: Human Evaluation + Report + Demo

**Goal:** Human evaluation protocol executed. Final dissertation-ready results. Demo script working.

#### Day 1-2: Human Evaluation Protocol

Design a Google Form (free) with these criteria for each generated scene:

**Evaluators:** 5 native Hindi speakers (classmates, family members — no payment needed)

**Criteria (5-point Likert scale):**

1. **Plot Fidelity:** Does the dialogue faithfully reflect all events from the original story?
2. **Character Consistency:** Does each character speak in a way consistent with their role/personality?
3. **Linguistic Authenticity:** Does the Hinglish sound natural (not robotic or over-slanged)?
4. **Slang Appropriateness:** Is the slang contextually appropriate (not random)?
5. **Overall Quality:** Would this dialogue work in a real Bollywood web series?

**Test set:** Take your 10 test stories, show evaluators:

- Original Hindi story (translated to English for their reference)
- Your system's generated Hinglish script
- A baseline: raw IndicBART output WITHOUT agents (single-model baseline)

Ask them to rate both on the 5 criteria. This gives you the comparison your dissertation needs.

#### Day 3-4: Results Compilation

Create `evaluation/human_eval_results.py` to aggregate form responses and compute:

- Mean scores per criterion
- Inter-annotator agreement (Fleiss' Kappa or Krippendorff's Alpha)
- Your system vs. baseline head-to-head win rates

```python
from scipy import stats
import pandas as pd

def compute_krippendorff_alpha(ratings_matrix):
    """ratings_matrix: shape (n_annotators, n_items)"""
    # Use krippendorff library
    import krippendorff
    return krippendorff.alpha(reliability_data=ratings_matrix)
```

#### Day 5-6: Report Updates + Code Cleanup

1. Update Chapter 6 (Results) with all quantitative metrics and human evaluation scores.
2. Write up the W&B training curves for the IndicBART fine-tuning.
3. Document the Three-Gate system test results (table is already in the report).
4. Expand Chapter 7 (Future Work) with concrete next steps based on what you learned.

Clean up code:

```bash
# Remove debug prints
grep -r "print(" agents/ | grep -v test | grep -v logger
# Add docstrings to all public methods
# Write README with setup instructions
```

#### Day 7: Demo + Final Push

Create `demo.py`:

```python
"""
Quick demo script for capstone presentation.
Usage: python demo.py --story "path/to/story.txt"
"""
import argparse
from orchestrator.pipeline import HinglishPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--story", required=True)
    parser.add_argument("--session", default="demo_001")
    args = parser.parse_args()

    with open(args.story, 'r', encoding='utf-8') as f:
        story = f.read()

    print("🎬 Hinglish Script Generator")
    print("=" * 50)
    print("Processing story...")

    pipeline = HinglishPipeline(args.session)
    script = pipeline.run(story)

    print("\n📜 GENERATED SCRIPT:")
    print("=" * 50)
    print(script)

    # Save output
    with open(f"outputs/{args.session}_script.txt", 'w', encoding='utf-8') as f:
        f.write(script)
    print(f"\n✅ Script saved to outputs/{args.session}_script.txt")

if __name__ == "__main__":
    main()
```

Push everything to GitHub. Tag the release: `git tag v1.0-capstone`.

---

## COST BREAKDOWN — ZERO BUDGET

| Component                             | Tool                            | Monthly Cost |
| ------------------------------------- | ------------------------------- | ------------ |
| GPU compute (fine-tuning)             | Kaggle 30 hr/week + Colab free  | **₹0**       |
| Narrative Analyst LLM                 | Ollama (local) + Groq free tier | **₹0**       |
| MuRIL (Character Analyst + Validator) | Local download from HuggingFace | **₹0**       |
| IndicBART (Dialogue Synthesizer)      | Local fine-tune on Kaggle       | **₹0**       |
| FELIX BERT tagger                     | Local fine-tune on Kaggle       | **₹0**       |
| Intent Classifier                     | BART-MNLI zero-shot, local      | **₹0**       |
| Experiment tracking                   | W&B free tier                   | **₹0**       |
| Dataset storage                       | HuggingFace Hub private (free)  | **₹0**       |
| Code repository                       | GitHub private repo (free)      | **₹0**       |
| **TOTAL**                             |                                 | **₹0**       |

---

## TEAM RESPONSIBILITY MATRIX

| Module                                      | Owner        | Reviewer  |
| ------------------------------------------- | ------------ | --------- |
| Preprocessing (transliterate + chunker)     | Atul         | Anish     |
| Agent 1 (Narrative Analyst) + LangGraph     | Anish        | Anika     |
| Agent 2 (Character Analyst) + State Updater | Diya         | Atul      |
| Three-Gate Controller + Intent Classifier   | Anika        | Diya      |
| IndicBART fine-tuning (Agent 3)             | Anish + Diya | All       |
| FELIX dataset + Tagger training (Agent 4)   | Anika + Atul | All       |
| MuRIL Validation Layer                      | Diya         | Anish     |
| LangGraph Orchestration                     | Anish        | Anika     |
| Human Evaluation + Metrics                  | All          | Prof. VRB |
| Report + Demo                               | All          | Prof. VRB |

---

## RISK REGISTER & CONTINGENCY PLANS

| Risk                                   | Likelihood | Mitigation                                       |
| -------------------------------------- | ---------- | ------------------------------------------------ |
| Ollama too slow on Kaggle CPU          | Medium     | Switch to Groq free API (14K tokens/min)         |
| IndicBART fine-tune OOM on T4          | Medium     | Reduce batch size to 4, gradient accumulate more |
| Agent 1 JSON output malformed          | High       | Retry loop + regex extraction fallback           |
| Groq rate limit hit during testing     | Low        | Cache all LLM outputs to disk                    |
| HuggingFace download fails on Kaggle   | Low        | Download once, save to Kaggle Dataset, re-use    |
| MuRIL similarity threshold too strict  | Medium     | Tune down from 0.82 → 0.75 if >30% scenes fail   |
| 6000-sample Felix dataset insufficient | Low        | Augment to 10K using same Qwen pipeline          |

---

## CONFIG FILE

File: `config.py`

```python
# ── Model paths ─────────────────────────────────────────────────────
INDICBART_BASE_PATH = "ai4bharat/indicbart"
INDICBART_ADAPTER_PATH = "./models/stage2-lora-adapter"
FELIX_TAGGER_PATH = "./models/felix-tagger-final"
MURIL_MODEL_PATH = "google/muril-base-cased"

# ── LLM backend ──────────────────────────────────────────────────────
LLM_BACKEND = "groq"  # "ollama" | "groq" | "hf"
GROQ_MODEL = "llama-3-8b-8192"

# ── Paths ─────────────────────────────────────────────────────────────
INDIC_NLP_RESOURCES = "./indic_nlp_resources"
FELIX_DATASET_PATH = "./data/felix_dataset/slang_pairs_clean.csv"

# ── Validation thresholds ─────────────────────────────────────────────
MURIL_SIMILARITY_THRESHOLD = 0.82
EVENT_COVERAGE_F1_THRESHOLD = 0.5
SLANG_DENSITY_THRESHOLD = 0.25
MAX_REWRITE_ATTEMPTS = 3

# ── Training hyperparameters ──────────────────────────────────────────
INDICBART_STAGE1_EPOCHS = 3
INDICBART_STAGE2_EPOCHS = 2
FELIX_EPOCHS = 10
LORA_RANK = 16
LORA_ALPHA = 32
```

---

## WEEK-BY-WEEK MILESTONE CHECKLIST

```
Week 1 ✓ Transliteration pipeline working, all tests pass
       ✓ Prose chunker produces scene-level segments
       ✓ GitHub repo set up, directory structure created

Week 2 ✓ LangGraph PipelineState defined
       ✓ Scratchpad JSON read/write working
       ✓ Agent 1 produces valid event chain JSON
       ✓ Narrative continuity links across scenes

Week 3 ✓ Character profiles in scratchpad for all speakers
       ✓ State updater tracking emotions across scenes
       ✓ Gate 1: elders blocked correctly
       ✓ Gate 2: intent classifier running (zero-shot)
       ✓ Gate 3: density counter working

Week 4 ✓ IndicBART Stage 1 fine-tune complete (Kaggle)
       ✓ IndicBART Stage 2 fine-tune complete (Kaggle)
       ✓ Agent 3 inference wrapper generating Pass 1 dialogue
       ✓ Pass 1 output covers key events from event chain

Week 5 ✓ 6000-pair dataset validated and cleaned
       ✓ FELIX tagger trained on Kaggle T4
       ✓ Agent 4 (Slang Rewriter) producing Pass 2 output
       ✓ All 3 gates integrated in run_all_gates()

Week 6 ✓ MuRIL validation computing similarity scores
       ✓ LangGraph pipeline connecting all nodes
       ✓ Feedback loop cycling on failed scenes
       ✓ End-to-end pipeline produces final script

Week 7 ✓ 10 test stories processed end-to-end
       ✓ Thresholds tuned with W&B experiments
       ✓ All major bugs fixed
       ✓ Evaluation metrics computed and logged

Week 8 ✓ Human evaluation form created and filled (5 evaluators)
       ✓ Results compiled and analyzed
       ✓ Report Chapter 6 updated with real numbers
       ✓ Demo script working
       ✓ GitHub repo cleaned and tagged v1.0
```

---

## FIRST COMMANDS TO RUN RIGHT NOW

```bash
# 1. Clone your repo (after creating it on GitHub)
git clone https://github.com/YOUR-ORG/hinglish-mas.git
cd hinglish-mas

# 2. Create environment
conda create -n hinglish python=3.10 -y
conda activate hinglish

# 3. Install all dependencies
pip install torch==2.2.0 transformers==4.40.0 datasets==2.19.0 \
    peft==0.10.0 trl==0.8.6 langgraph==0.1.0 langchain-core==0.2.0 \
    sentencepiece==0.2.0 sentence-transformers==2.7.0 evaluate==0.4.1 \
    wandb==0.17.0 rapidfuzz==3.9.0 spacy==3.7.0 scikit-learn==1.4.0 \
    rouge-score bert-score groq

# 4. Clone IndicNLP resources
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git

# 5. Install indic-nlp-library
pip install indic-nlp-library

# 6. Set up W&B
wandb login

# 7. Create directory structure
mkdir -p preprocessing agents gates memory evaluation orchestrator \
         data/{raw,processed,felix_dataset,sessions} models tests notebooks outputs

# 8. Start building transliterate.py — you're off!
```

---

_Generated for PW26_VRB_07 — Anika, Anish, Diya, Atul — PES University Capstone Phase 2_
_All tools selected for zero-cost operation. Every component locally runnable or free-tier API._
