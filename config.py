import os

# ── Base directory ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model paths ─────────────────────────────────────────────────────
INDICBART_BASE_PATH = "ai4bharat/indicbart"
INDICBART_ADAPTER_PATH = "./models/stage2-lora-adapter"
FELIX_TAGGER_PATH = "./models/felix-tagger-final"
MURIL_MODEL_PATH = "google/muril-base-cased"

# ── LLM backend ──────────────────────────────────────────────────────
LLM_BACKEND = "groq"  # "ollama" | "groq" | "hf"
# NOTE: Groq retired llama-3.1-8b-instant. Current text models on this key:
#   openai/gpt-oss-20b (fast, used here), openai/gpt-oss-120b, qwen/qwen3.6-27b.
# gpt-oss/qwen are reasoning models -> pass reasoning_effort="low" (see GROQ_REASONING_EFFORT)
# so the visible content isn't starved by reasoning tokens.
# Default 20b (fast/cheap). If it hits its daily token cap, override without editing
# code:  GROQ_MODEL=openai/gpt-oss-120b python run_pipeline.py ...  (120b = separate budget)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_REASONING_EFFORT = "low"

# ── Paths ─────────────────────────────────────────────────────────────
INDIC_NLP_RESOURCES = os.path.join(BASE_DIR, "indic_nlp_resources")
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


# ── FELIX / Agent 4 dataset ───────────────────────────────────────────
FELIX_DATASET_RAW   = "data/felix_dataset/final_augmented_master_6000.csv"
FELIX_DATASET_CLEAN = "data/felix_dataset/slang_pairs_clean.csv"
SLANG_LEXICON_PATH  = "data/felix_dataset/slang_lexicon.json"

# ── Agent 4 editor ────────────────────────────────────────────────────
# Qwen LoRA editor (preferred): a small instruction LLM LoRA-fine-tuned to PLACE a
# given slang naturally. Far better generation than mt5-small. See notebooks/
# train_slang_editor_lora_qwen.py.
QWEN_BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"       # must match what was trained
# the downloaded LoRA adapter; override per-run to A/B two adapters without editing code:
#   QWEN_LORA_PATH=./models/slang-editor-lora-v1 python run_pipeline.py ...
QWEN_LORA_PATH      = os.getenv("QWEN_LORA_PATH", "./models/slang-editor-lora-v4")
# Precision the Qwen base is loaded in on CPU: "auto" (default), "float32" or "bfloat16".
#   float32  ~6 GB, fast.   bfloat16 ~3 GB but ~25 s/line on a CPU without native bf16.
# "auto" = float32 if >=7.5 GB RAM is free when the editor loads, else bfloat16 (so it fits
# instead of failing with Windows "paging file too small (os error 1455)").
# Force one per-run, e.g. PowerShell:  $env:QWEN_DTYPE="bfloat16"
QWEN_DTYPE          = os.getenv("QWEN_DTYPE", "auto")
# Runtime slang lexicon = the VETTED dataset (346 slangs, every one has a >=4/5 natural example),
# NOT the old slang_pairs_clean.csv (464 slangs incl. junk like "kaam ka", "peti-pack").
SLANG_RUNTIME_SRC   = os.getenv("SLANG_RUNTIME_SRC", "data/felix_dataset/slang_bank_v3.csv")
# Per-line naturalness judge (Groq): every edited clause is rated 1-5 and kept only if
# >= QWEN_JUDGE_MIN and the meaning is preserved. Fails OPEN (guards still apply) if Groq
# is unreachable. Turn off with  $env:QWEN_JUDGE="0"
QWEN_JUDGE          = os.getenv("QWEN_JUDGE", "1") not in ("0", "false", "False", "")
QWEN_JUDGE_MIN      = int(os.getenv("QWEN_JUDGE_MIN", "4"))
# Local-editor attempts per line before giving up on it. Each attempt costs ~30 s on this CPU
# (bf16) and retrying the same model rarely fixes a paraphrase, so the default is 1.
QWEN_MAX_TRIES      = int(os.getenv("QWEN_MAX_TRIES", "1"))
# When the guards reject the local editor's edit, let Groq place a slang for that line
# instead (fast; its edit goes through the same guards; each line is tagged with its
# engine). Set to 0 for a PURE local-editor run, e.g. to measure the editor on its own:
#   PowerShell:  $env:QWEN_GROQ_FALLBACK="0"
QWEN_GROQ_FALLBACK  = os.getenv("QWEN_GROQ_FALLBACK", "0") not in ("0", "false", "False", "")   # OFF: the trained editor must work on its own
# Shortlist size shown to the editor, per adapter prompt-version. v1 was trained on lists
# of exactly 5, so it stays at 5 (untested otherwise); v2 is trained on 5-8, so it gets 8
# (a bigger list = more chance that at least one candidate genuinely fits the line).
QWEN_SHORTLIST_K    = {1: 5, 2: 8}
# Legacy seq2seq editor (mt5) — kept as a fallback path:
EDITOR_BASE_MODEL   = "google/mt5-small"
EDITOR_FULL_PATH    = "./models/slang-editor-full"      # v3 FULL fine-tuned model
EDITOR_ADAPTER_PATH = "./models/slang-editor-lora-mt5"  # old mt5 LoRA adapter (renamed to avoid clash)
EDITOR_MAX_INPUT    = 96
EDITOR_MAX_TARGET   = 96
EDITOR_NUM_BEAMS    = 4

# ── Gate 2 control-token classifier ───────────────────────────────────
CONTROL_CLF_PATH    = "./models/control-clf"            # fine-tuned mBERT classifier
CONTROL_CLF_BASE    = "bert-base-multilingual-cased"
USE_TRAINED_INTENT_CLF = True   # False = fall back to zero-shot BART-MNLI

# ── Validation thresholds (Week 6) ────────────────────────────────────
MURIL_SIMILARITY_THRESHOLD   = 0.82   # tune in Week 7
EVENT_COVERAGE_RECALL_THRESHOLD = 0.5
MAX_REWRITE_ATTEMPTS = 3

# ── Anika: stricter validation + smart redo-loop thresholds ───────────
# Per-line MuRIL similarity floor. Whole-scene similarity is coarse (a few
# corrupted lines get averaged out), so we also flag any single line whose
# neutral-vs-slang similarity drops below this.
MURIL_LINE_SIM_THRESHOLD = 0.72

# Below this whole-scene similarity we classify the scene as MEANING_DRIFT
# (a distinct, more urgent failure than a borderline pass).
MURIL_MEANING_DRIFT_THRESHOLD = 0.78

# Minimum fraction of dialogue lines that should carry slang in a scene.
# Below this we classify the scene as LOW_COVERAGE (or NO_SLANG if it is 0).
MIN_SLANG_COVERAGE_PER_SCENE = 0.15

# A scene is MONOTONOUS if more than this fraction of its applied slang
# words are the same token (e.g. every line got "yaar").
MAX_SLANG_REPETITION_RATIO = 0.60

# Multipliers used by the redo loop to temporarily adjust the gate-3 density
# budget for a failing scene on retry.
DENSITY_MULTIPLIER_LOW_COVERAGE = 2.0    # loosen a bit
DENSITY_MULTIPLIER_MEANING_DRIFT = 0.5   # tighten (fewer insertions)
# Groq-fallback attempts per line (each: fresh shortlist -> guards -> judge). Cheap (~2 s each).
QWEN_FALLBACK_TRIES = int(os.getenv("QWEN_FALLBACK_TRIES", "3"))


# ---- slang editor: generalisation knobs (apply to ANY narrative) ----
SLANG_SEED          = int(os.getenv("SLANG_SEED", "7"))          # reproducible shortlists
# Retrieve-then-rerank: pull a wide tone-matched pool, then an LLM keeps only the slangs that
# fit THIS clause. Empty result = nothing fits = leave the line neutral (content-aware abstain,
# and it skips a ~30 s local generation). Off: $env:QWEN_RERANK="0"
QWEN_RERANK         = os.getenv("QWEN_RERANK", "1") not in ("0", "false", "False", "")
QWEN_RERANK_POOL    = int(os.getenv("QWEN_RERANK_POOL", "30"))
QWEN_RERANK_ROUNDS  = int(os.getenv("QWEN_RERANK_ROUNDS", "2"))   # fresh pool if nothing fit
# When the word-list tone detector is unsure, ask the LLM (generalises beyond our word lists)
QWEN_LLM_TONE       = os.getenv("QWEN_LLM_TONE", "1") not in ("0", "false", "False", "")

# Unattributed dialogue tags ("SPEAKER:", "UNKNOWN:") have an unknown register, so by default the
# editor leaves them alone. Set to 1 to let them be edited like peers (NARRATOR is never edited).
# Only worth it for stories where Agents 1/3 fail to name any speakers.   $env:SLANG_ALLOW_UNATTRIBUTED="1"
SLANG_ALLOW_UNATTRIBUTED = os.getenv("SLANG_ALLOW_UNATTRIBUTED", "0") in ("1", "true", "True")

# Editing unit for the Qwen editor: "sentence" (v3, whole sentences of 4-24 words) or "clause"
# (older comma-split fragments). The v3 adapter is trained on sentences.   $env:QWEN_UNIT="clause"
QWEN_UNIT = os.getenv("QWEN_UNIT", "sentence")

# Strict tone handling in the editor: mood-loaded slang (fire, jhakaas, panga ...) is only offered/accepted
# when the clause's tone is KNOWN and matches; unknown/neutral tone -> mood-neutral fillers only.
SLANG_STRICT_TONE = os.getenv("SLANG_STRICT_TONE", "1") in ("1", "true", "True")

# Edit acceptance gate: "rules" = deterministic local gate (agents/edit_gate.py, default);
# anything else = the older Groq LLM naturalness judge (QWEN_JUDGE).   $env:QWEN_GATE="llm"
QWEN_GATE = os.getenv("QWEN_GATE", "rules")

# CPU speed (all measured by evaluation/bench_speed.py; every option leaves the output TEXT identical):
QWEN_MERGE = os.getenv("QWEN_MERGE", "1") not in ("0", "false", "False")      # merge LoRA into the weights
QWEN_LOOKUP = int(os.getenv("QWEN_LOOKUP", "10"))   # prompt-lookup decoding draft length (0 = off)
QWEN_SLOT_PRECHECK = os.getenv("QWEN_SLOT_PRECHECK", "1") not in ("0", "false", "False")   # skip generations the gate would reject anyway
