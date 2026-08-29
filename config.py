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

# ── Agent 4 editor (seq2seq) ──────────────────────────────────────────
EDITOR_BASE_MODEL   = "google/mt5-small"
EDITOR_FULL_PATH    = "./models/slang-editor-full"      # v3 FULL fine-tuned model (preferred)
EDITOR_ADAPTER_PATH = "./models/slang-editor-lora"      # v1/v2 LoRA adapter (fallback)
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