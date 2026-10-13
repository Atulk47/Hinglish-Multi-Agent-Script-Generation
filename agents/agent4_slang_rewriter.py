"""
agents/agent4_slang_rewriter.py
Agent 4 — Pass 2 slang injection.

Two selectable backends behind ONE interface (`rewrite_line`, `rewrite_script`):

  backend="mt5"  : a trained mT5 "slang editor" (LoRA adapter at config.EDITOR_ADAPTER_PATH)
                   that learned neutral -> slang insertion from the slang dataset.
                   This is the research target (uses the dataset in the pipeline).
  backend="groq" : a Groq stopgap used until the editor is trained. It SELECTS a
                   specific, non-repeating slang in code (weighted by rarity + the
                   speaker's emotion + the utterance intent, drawn from the full
                   576-phrase lexicon) and asks Groq only to PLACE it naturally.
                   Selecting-in-code (not letting the LLM choose) is what breaks the
                   "everything becomes bhai/yaar" monotony.

Either way, the Three-Gate controller decides WHETHER a line may receive slang.
`backend="auto"` uses mt5 if the trained adapter exists, else groq.
"""
from __future__ import annotations
import csv
import json
import os
import random
import re
import time
from collections import defaultdict
from typing import List, Tuple

from groq import Groq
from dotenv import load_dotenv

import config
from gates.three_gate_controller import ThreeGateController

load_dotenv()

# Gate intent (small set) -> a single dataset-style control token for the mT5 editor
# (the editor was trained conditioned on the dataset's canonical control tokens).
INTENT_TO_EDITOR_CONTROL = {
    "ASSERT_POSITIVE":    "ASSERT_POSITIVE",
    "ASSERT_FRUSTRATION": "ASSERT_FRUSTRATION",
    "EXPRESS_STRESS":     "ASSERT_STRESS",
    "REQUEST_URGENCY":    "REQUEST_URGENCY",
    "CASUAL_ASSERTION":   "ASSERT_NEUTRAL",
}

# Gate/intent-classifier labels (small set) -> dataset `control` tokens
GATE_INTENT_TO_CONTROLS = {
    "ASSERT_POSITIVE":    {"ASSERT_POSITIVE", "ASSERT_HAPPY", "ASSERT_EXCITED"},
    "ASSERT_FRUSTRATION": {"ASSERT_FRUSTRATION", "ASSERT_ANGER", "ASSERT_NEGATIVE"},
    "EXPRESS_STRESS":     {"ASSERT_STRESS", "ASSERT_ANXIETY", "ASSERT_SADNESS"},
    "REQUEST_URGENCY":    {"REQUEST_URGENCY", "ASSERT_STRESS"},
    "CASUAL_ASSERTION":   {"CASUAL_ASSERTION", "ASSERT_POSITIVE"},
}

# The speaker's tracked emotion enriches a generic intent so the rich,
# emotion-specific slang actually gets used.
EMOTION_TO_CONTROLS = {
    "happy":   {"ASSERT_HAPPY", "ASSERT_POSITIVE", "ASSERT_EXCITED"},
    "angry":   {"ASSERT_FRUSTRATION", "ASSERT_ANGER", "ASSERT_NEGATIVE"},
    "anxious": {"ASSERT_ANXIETY", "ASSERT_STRESS"},
    "urgent":  {"REQUEST_URGENCY", "ASSERT_STRESS"},
    "sad":     {"ASSERT_SADNESS"},
    "neutral": set(),
}

# When the pool is still thin, pull colourful (non-filler) functions before
# falling back to plain fillers, so we don't default to bhai/yaar.
COLOURFUL_FUNCTIONS = ["idiom", "intensifier", "slang_eval", "positive_eval",
                       "negative_eval", "descriptor", "modern_slang", "metaphor"]

# ── Sentiment + register guards ────────────────────────────────────────
# Control tokens whose slang carries a negative/harsh connotation, vs positive.
NEGATIVE_CONTROLS = {"ASSERT_NEGATIVE", "ASSERT_ANGER", "ASSERT_SADNESS",
                     "ASSERT_FEAR", "ASSERT_WARNING", "ASSERT_ANXIETY",
                     "ASSERT_STRESS", "ASSERT_FRUSTRATION"}
POSITIVE_CONTROLS = {"ASSERT_POSITIVE", "ASSERT_HAPPY", "ASSERT_EXCITED"}
# Only in these contexts is negative/harsh slang appropriate.
NEGATIVE_INTENTS = {"ASSERT_FRUSTRATION", "EXPRESS_STRESS"}
NEGATIVE_EMOTIONS = {"angry", "anxious", "sad"}
# Clearly insulting/teasing terms that should never land on a positive line,
# regardless of how the lexicon labels them.
NEGATIVE_SLANG_HARD = {
    "muft-khor", "muftkhor", "muft khor", "kaand", "kaand ho gaya", "kamina",
    "kamine", "bakwaas", "bakwas", "ghatiya", "faltu", "chindi", "scam",
    "nikamma", "chirkut", "dhakkan", "bewakoof", "bevkoof", "gadha",
}
# Gratitude / respect / farewell markers — lines carrying these get NO slang.
RESPECT_MARKERS = [
    "dhanyavaad", "dhanyavad", "dhanyawad", "dhanyawaad", "shukriya", "shukria",
    "aabhar", "abhaar", "pranam", "pranaam", "namaste", "namaskar", "kshama",
    "kshma", "maaf", "maafi", "maphi", "alvida", "aadar", "sammaan", "aashirwad",
    "kripya", "krupya",
]

PLACE_PROMPT = """You are a Hinglish dialogue editor for a casual web-series.
Rewrite the neutral line to sound casual by inserting ONE slang expression.

Choose the ONE expression from this shortlist that fits this line most naturally:
{candidates}

Hard rules:
- You MUST use exactly one expression from the shortlist above (you may position/inflect it naturally).
- Do NOT change the meaning or any fact. Do NOT add new information.
- Keep it in romanized Hinglish (Roman script). Return exactly ONE line.
- Output ONLY the rewritten line: no speaker name, no quotes, no explanation.

Neutral line: {utterance}
Rewritten line:"""

# ── Qwen LoRA editor ───────────────────────────────────────────────────
# The prompt MUST match the format the adapter was trained on. Two versions exist:
#   v1: shortlist-pick (first 1.5B adapter) - always picks something.
#   v2: shortlist-pick + ABSTAIN ("none fits -> return the line unchanged"), trained on
#       shortlists of 5-8 candidates incl. examples where no candidate fits.
# The notebook writes `prompt_version.txt` into the adapter folder; if it is missing the
# adapter is treated as v1. (Keep these strings identical to data_prep/build_lora_dataset.py
# and notebooks/train_slang_editor_lora_qwen.py - verify with data_prep/check_prompt_sync.py.)
QWEN_SYSTEM_V1 = ("You are a Hinglish (romanized Hindi-English) dialogue editor. Given a sentence and "
                  "a few candidate slang expressions, pick the ONE that fits best and insert it "
                  "naturally, keeping the meaning EXACTLY the same. Reply with ONLY the rewritten sentence.")
QWEN_USER_V1 = ("Pick the slang from this list that best fits the line, and insert it naturally "
                "without changing the meaning:\nSlangs: {slangs}\nLine: {line}")

QWEN_SYSTEM_V2 = ("You are a Hinglish (romanized Hindi-English) dialogue editor. Given a sentence and "
                  "a few candidate slang expressions, pick the ONE that fits best and insert it "
                  "naturally, keeping the meaning EXACTLY the same. If none of the candidates fits "
                  "naturally, reply with the sentence UNCHANGED. Reply with ONLY the sentence.")
QWEN_USER_V2 = ("Pick the slang from this list that best fits the line, and insert it naturally "
                "without changing the meaning. If none of them fits naturally, return the line "
                "UNCHANGED:\nSlangs: {slangs}\nLine: {line}")

# Lexical line-polarity estimate (used by the polarity-flip guard). Deliberately small and
# conservative: it only returns +1/-1 when the evidence is clear, else 0 (= no opinion).
_POL_NEGATIONS = {"nahi", "nahin", "nhi", "mat"}
_POL_NEG_WORDS = {"bekaar", "bura", "buri", "kharab", "bakwaas", "ghatiya", "boring", "bore",
                  "pareshan", "dukhi", "gussa", "thak", "thaka", "fail", "problem", "dikkat",
                  "mushkil", "galat", "worst", "sad", "tension", "stress", "ghabrahat",
                  "darr", "dar", "nakam", "nakaam", "ghabra", "ghabraya", "udaas", "dukh", "afsos", "niraash",
                  "niraasha", "nirasha", "chinta", "haar", "haara", "musibat", "bekar", "pareshani", "jalna",
                  "jal", "bigad", "kami", "kamzor", "dukhi", "rona", "royi"}
_POL_POS_WORDS = {"accha", "acchi", "achha", "achhi", "acha", "achi", "badiya", "badhiya", "mast",
                  "sahi", "shaandaar", "zabardast", "kamaal", "behtareen", "khush", "maza",
                  "mazedaar", "pasand", "best", "great", "superb", "sundar", "khushi"}

INTENT_TO_SENTIMENT = {
    "ASSERT_POSITIVE": "positive", "ASSERT_FRUSTRATION": "frustrated",
    "EXPRESS_STRESS": "stressed", "REQUEST_URGENCY": "urgent",
    "CASUAL_ASSERTION": "casual", "INFORM_NEUTRAL": "casual", "QUESTION_FORMAL": "casual",
}


JUDGE_LINE_PROMPT = """You are a strict native speaker of casual Indian Hinglish judging an auto-edited line.
Original: {neutral}
Edited:   {edited}
Slang used: {slang}
Rate 1-5 how NATURAL the edited line sounds: grammatical, the slang is used correctly and fits the meaning and the tone, nothing forced or tacked on.
5 = a real person would say exactly this; 4 = natural; 3 = a bit off; 2 = awkward/forced; 1 = wrong or nonsense. Also say whether the meaning is unchanged.
Score 2 or lower if the slang's MOOD does not match the sentence's mood (e.g. a celebratory word like "fire" or "jhakaas" on a fearful, sad or worried sentence, or a complaint word on a happy one), or if a filler / exclamation is tacked onto a purely factual or technical explanation.
Return ONLY JSON: {{"score": <1-5>, "meaning_preserved": <true/false>}}"""


RERANK_PROMPT = """You are helping add ONE casual slang to a line in a Hinglish screenplay.
Line: {clause}
Tone of the line: {tone}

Candidate slangs:
{options}

Choose up to 6 candidates that a real young Indian would NATURALLY say as part of THIS line: they must fit what the line is about, its tone, and sound right when added at the start or end of it. Reject any that belong to some other situation, or would sound forced, rude or odd here. If the line is a factual, technical or procedural explanation (how something works, what was built or done), or its mood is unclear, return an empty list. If none fit, return an empty list.
Return ONLY JSON: {{"fit": ["slang one", "slang two"]}}  (best first, exact text from the list)"""

TONE_PROMPT = """Classify the emotional tone of this Hinglish line. Answer with exactly one word: positive, negative or neutral.
negative = fear, worry, doubt about failing, sadness, disappointment, frustration, a problem or failure (e.g. "mujhe darr hai ki main nakam ho jaunga", "sab bekaar ho gaya").
positive = joy, pride, praise, excitement, success, a good idea (e.g. "hamari jeet hui", "yeh bahut accha idea hai").
neutral = plain facts, instructions, explanations, questions with no feeling.
Line: {core}
Tone:"""

# Slang that fits almost any sentence of the right mood (judge acceptance 60-85% in the v3 study). Used only when the
# LLM reranker is unavailable, so a failed API call can never let context-specific slang through.
SAFE_GENERIC_SLANG = {"zabardast", "kamaal", "badiya", "mast", "lajawaab", "kadak", "jhakaas", "gazab", "solid",
                      "ghatiya", "bekaar", "bakwaas", "panga", "locha", "lafda", "full tension",
                      "dil baith raha hai", "fatti padi hai", "waat lag gayi"}

UNATTRIBUTED_SPEAKERS = {"SPEAKER", "NARRATOR", "UNKNOWN", "VOICE", "NARRATION"}
# Group / authority speaker tags are formal by nature (Gate 1 only knows individual honorifics).
FORMAL_SPEAKER_WORDS = {"COMMITTEE", "PANEL", "JURY", "JUDGE", "JUDGES", "BOARD", "ANNOUNCER",
                        "HOST", "EMCEE", "ANCHOR", "REPORTER", "OFFICIAL", "OFFICER", "MINISTER",
                        "PRINCIPAL", "TEACHER", "PROFESSOR", "DOCTOR", "INSPECTOR", "CHAIRMAN",
                        "MAYOR", "MAGISTRATE", "COMMISSIONER", "JUDGEMENT"}
# Elder / kinship tags. Gate 1 only knows a few spellings ("dadi" yes, "DAADI" no), so a story whose
# grandmother is tagged DAADI would get slang. Matching ignores doubled letters, so
# DAADI / Dadi / NAANI / Nani / Dadaji / AMMA all hit.
ELDER_TAG_WORDS = {"DADI", "DADA", "DADAJI", "DADIMA", "NANI", "NANA", "NANAJI", "NANIMA",
                   "GRANDMOTHER", "GRANDFATHER", "GRANDMA", "GRANDPA", "GRANNY", "MAA", "MA",
                   "MUMMY", "MOM", "MOTHER", "PAPA", "PITAJI", "PITA", "FATHER", "DAD", "DADDY",
                   "BABA", "BABUJI", "BAUJI", "AMMA", "AMMAJI", "MATAJI", "UNCLE", "AUNTY",
                   "AUNT", "CHACHA", "CHACHI", "CHACHAJI", "MAMA", "MAMI", "BUA", "BUAJI",
                   "MASI", "MAUSI", "TAYA", "TAYI", "KAKA", "KAKI", "ELDER", "ELDERS", "BUZURG",
                   "BUJURG", "BUZURGON", "SASUR", "SAAS"}


def _squash(w: str) -> str:
    """lowercase and collapse repeated letters: DAADI -> dadi, Naani -> nani"""
    return re.sub(r"(.)\1+", r"\1", w.lower())


_FORMAL_OR_ELDER_TAGS = {_squash(w) for w in (FORMAL_SPEAKER_WORDS | ELDER_TAG_WORDS)}
PLAIN_FILLER_WORDS = {"yaar", "bhai", "bro", "boss", "abe", "abey", "arre", "re", "bhaiya", "yr"}


PLACE_PROMPT_STRICT = """You are a Hinglish dialogue editor. Use ONE slang from this shortlist so the line sounds like a real young person talking.
Shortlist:
{candidates}

Rules (strict):
- Keep the meaning, facts, names and every other word EXACTLY as written. Add no new information or feeling.
- Preferred: SUBSTITUTE the slang for an evaluative word or intensifier in the line ("bahut accha" / "uttam" / "kharab" / "sundar" / "bahut" ...), e.g. "Yeh bahut accha idea hai" -> "Yeh ekdum badiya idea hai".
- Or add a short filler (yaar, bhai, arre, chill ...) right after a name or at the very start, e.g. "Neha, humein jaldi nikalna chahiye" -> "Neha yaar, humein jaldi nikalna chahiye".
- Do NOT tack an idiom or exclamation onto the end of the sentence, and do not repeat the same praise twice ("accha ... badiya").
- Use exactly one expression from the shortlist, spelled exactly as written there.
- If none of them fits naturally, return the line unchanged.
- Output ONLY the resulting line: no quotes, no speaker name, no explanation.

Line: {utterance}
Result:"""


class SlangRewriter:
    def __init__(self, scratchpad, intent_classifier,
                 backend: str = "auto",
                 slang_density_threshold: float = None,
                 dataset_path: str = None):
        self.scratchpad = scratchpad
        self.intent_clf = intent_classifier
        self._rng = random.Random(getattr(config, "SLANG_SEED", 7))   # reproducible shortlists
        self.stats = defaultdict(int)       # e.g. groq_errors: >0 means judge/rerank/tone degraded
        self._tone_cache = {}
        threshold = (slang_density_threshold if slang_density_threshold is not None
                     else config.SLANG_DENSITY_THRESHOLD)
        self.gate = ThreeGateController(scratchpad, slang_density_threshold=threshold)
        self.by_control, self.by_function, self.filler_pool = self._load_lexicon(dataset_path)
        self._neg_slang = self._compute_negative_slang()   # harsh/teasing slang to keep off positive lines
        self._pos_slang = self._compute_positive_slang()   # positive-only slang (polarity-flip guard)
        self._used = defaultdict(int)   # slang -> times used in this story
        self._last = None               # last slang used (avoid back-to-back repeats)

        self.backend = self._resolve_backend(backend)
        if self.backend == "qwen":
            try:
                self._init_qwen()         # Qwen base + LoRA adapter
                self._smoke_test_qwen()   # verify it generates a real insertion
            except Exception as e:
                print(f"SlangRewriter: Qwen LoRA editor unusable ({e}); falling back to Groq.")
                self.backend = "groq"
        if self.backend == "qwen" and (getattr(config, "QWEN_GROQ_FALLBACK", True)
                                       or getattr(config, "QWEN_JUDGE", True)):
            try:
                self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
                _j = ("ON (min %s/5)" % config.QWEN_JUDGE_MIN) if config.QWEN_JUDGE else "off"
                print("SlangRewriter: Groq fallback %s, per-line naturalness judge %s."
                      % ("ON" if config.QWEN_GROQ_FALLBACK else "off", _j))
            except Exception as e:
                self.groq_client = None
                print(f"SlangRewriter: Groq fallback unavailable ({e}).")
        if self.backend == "mt5":
            try:
                self._init_mt5()          # prints which editor (full vs LoRA) it loaded
                self._smoke_test_mt5()    # verify it actually generates (raises if not)
            except Exception as e:
                # e.g. a model saved by a newer transformers than this env pins
                # (see EDITOR_FULL_PATH); don't silently emit empty slang.
                print(f"SlangRewriter: mT5 editor loaded but unusable ({e}); "
                      f"falling back to Groq.")
                self.backend = "groq"
        if self.backend == "groq":        # not `else`: also the fallback path
            self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
            print(f"SlangRewriter: Groq stopgap (select-in-code). "
                  f"Lexicon: {sum(len(v) for v in self.by_control.values())} entries, "
                  f"{len(self.filler_pool)} fillers.")

    def _smoke_test_mt5(self):
        """A freshly loaded mT5 editor must actually produce text. A model trained
        on a newer transformers than this env can load cleanly yet decode to '' —
        catch that here so __init__ can fall back to Groq."""
        out = self._mt5_edit("Aaj mera din accha tha.", "ASSERT_NEUTRAL")
        if not out or not out.strip():
            raise RuntimeError("editor returned empty output on smoke test")

    def _resolve_backend(self, backend):
        if backend in ("qwen", "mt5", "groq"):
            return backend
        # auto: prefer the Qwen LoRA editor, else the mt5 model, else groq
        if os.path.isdir(getattr(config, "QWEN_LORA_PATH", "")):
            return "qwen"
        if os.path.isdir(getattr(config, "EDITOR_FULL_PATH", "")) or \
           os.path.isdir(getattr(config, "EDITOR_ADAPTER_PATH", "")):
            return "mt5"
        return "groq"

    # ── Qwen LoRA editor ───────────────────────────────────────────────
    def _init_qwen(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        self._torch = torch
        base = config.QWEN_BASE_MODEL
        self.qwen_tok = AutoTokenizer.from_pretrained(base)
        # Precision trade-off on CPU (measured on this machine, 12 cores, no native bf16):
        #   float32  : ~6 GB for the model, FAST (normal CPU matmuls)
        #   bfloat16 : ~3 GB, but ~25 s per line (bf16 is emulated here)
        # "auto" picks float32 only if enough physical RAM is free right now (Agent 2's NER +
        # MuRIL are already loaded at this point), else bfloat16 so we never hit Windows'
        # "paging file too small (os error 1455)" and silently fall back to Groq.
        want = getattr(config, "QWEN_DTYPE", "auto")
        if want == "auto":
            try:
                import psutil
                free_gb = psutil.virtual_memory().available / 1e9
            except Exception:
                free_gb = 0.0
            dtype = torch.float32 if free_gb >= 7.5 else torch.bfloat16
            print(f"SlangRewriter: {free_gb:.1f} GB RAM free -> loading Qwen in "
                  f"{str(dtype).replace('torch.', '')}"
                  + ("" if dtype == torch.float32 else
                     " (works but slow: ~25-30 s/line measured; with >=7.5 GB free it uses "
                     "float32 instead - run `python -m evaluation.bench_editor` to compare)"))
        else:
            dtype = getattr(torch, want)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=dtype, low_cpu_mem_usage=True)
        except (OSError, MemoryError, RuntimeError) as e:
            if dtype == torch.bfloat16:
                raise
            import gc; gc.collect()
            print(f"SlangRewriter: {str(dtype).replace('torch.', '')} load failed ({e}); "
                  f"retrying in bfloat16.")
            model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.qwen = PeftModel.from_pretrained(model, config.QWEN_LORA_PATH)
        if getattr(config, "QWEN_MERGE", True):
            try:                                   # fold the adapter into the weights: no per-layer LoRA overhead
                self.qwen = self.qwen.merge_and_unload()
            except Exception as e:
                print(f"SlangRewriter: LoRA merge skipped ({str(e)[:80]}); using the unmerged adapter.")
        self.qwen.eval()
        # which prompt format was this adapter trained on? (marker written by the notebook)
        try:
            with open(os.path.join(config.QWEN_LORA_PATH, "prompt_version.txt")) as f:
                self._qwen_ver = int(f.read().strip())
        except Exception:
            self._qwen_ver = 1
        ks = getattr(config, "QWEN_SHORTLIST_K", {1: 5, 2: 8})
        self._qwen_k = ks.get(self._qwen_ver, 5)
        print(f"SlangRewriter: Qwen LoRA editor ({base} + {config.QWEN_LORA_PATH}) "
              f"| prompt v{self._qwen_ver}, shortlist k={self._qwen_k}.")

    def _smoke_test_qwen(self):
        # verify the model loads and generates; 2 tokens is enough (a full sentence cost ~25 s here)
        user = QWEN_USER_V2.replace("{slangs}", "mast, solid").replace("{line}", "Aaj ka match accha tha.")
        enc = self.qwen_tok.apply_chat_template(
            [{"role": "system", "content": QWEN_SYSTEM_V2}, {"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with self._torch.no_grad():
            out = self.qwen.generate(**enc, max_new_tokens=2, do_sample=False,
                                     pad_token_id=self.qwen_tok.eos_token_id)
        if out.shape[1] <= enc["input_ids"].shape[1]:
            raise RuntimeError("Qwen editor returned no tokens on smoke test")

    def _qwen_edit(self, utterance: str, shortlist) -> str:
        # shortlist format: the model picks the best-fitting slang from the candidates
        # (code supplies colourful options) and inserts it naturally. On a v2 adapter it
        # may also return the line unchanged to signal "none of these fit".
        sys_p, user_t = ((QWEN_SYSTEM_V2, QWEN_USER_V2) if self._qwen_ver >= 2
                         else (QWEN_SYSTEM_V1, QWEN_USER_V1))
        user = user_t.replace("{slangs}", ", ".join(shortlist)).replace("{line}", utterance)
        enc = self.qwen_tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with self._torch.no_grad():
            # cap generation by line length: a good edit is the line + a short slang, so a
            # model that starts paraphrasing/rambling must not run to the full limit
            n_line = len(self.qwen_tok(utterance, add_special_tokens=False)["input_ids"])
            max_new = max(24, min(110, int(n_line * 1.4) + 12))
            kw = {}
            if int(getattr(config, "QWEN_LOOKUP", 10)) > 0:
                kw["prompt_lookup_num_tokens"] = int(config.QWEN_LOOKUP)
            out = self.qwen.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=self.qwen_tok.eos_token_id, **kw)
        text = self.qwen_tok.decode(out[0][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
        return text.split("\n")[0].strip().strip('"')

    @staticmethod
    def _inserted_phrase(neutral: str, output: str) -> str:
        """Best-effort: the words in output that weren't in the neutral line = the slang."""
        nset = set(re.findall(r"[a-z]+", neutral.lower()))
        new = [w for w in re.findall(r"[a-z]+", output.lower()) if w not in nset]
        return " ".join(new[:3]) if new else None

    _SEG_RE = re.compile(r"[^.!?;]+[.!?;]*\s*|[.!?;]+\s*")

    _SENT_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)\s*")

    def _pick_sentence(self, utt: str):
        """v3: edit ONE complete sentence (4-28 words; the v3 adapter is trained on 4-24). The v3 adapter is trained on whole
        Agent-3-style sentences, and a complete sentence keeps its own context; comma fragments
        ("kyunki ...", "aur ...") start mid-thought and read badly with slang tacked on.
        Prefers a sentence with a clear sentiment word, then '!' over '.', and avoids questions."""
        best = None
        units = []
        for m in self._SENT_RE.finditer(utt):
            a, b = m.start(), m.end()
            if len(utt[a:b].split()) > 32:
                # over-long sentence: use its ';'-separated parts (each a complete thought)
                units += [(a + q.start(), a + q.end()) for q in re.finditer(r"[^;]+;?", utt[a:b])]
            else:
                units.append((a, b))
        for a, b in units:
            core = utt[a:b].strip()
            n = len(core.split())
            if n < 4 or n > 32:
                continue
            score = ((2.0 if self._line_polarity(core) != 0 else 0.0)
                     + (1.0 if core.endswith("!") else 0.0)
                     - (1.5 if core.endswith("?") else 0.0) - n / 60)
            if best is None or score > best[0]:
                best = (score, a, b)
        return (best[1], best[2]) if best else None

    def _pick_segment(self, utt: str):
        """Choose ONE short clause of the line to edit -> (start, end) or None.
        The editor was trained on short single sentences (5-16 words); on a whole long Agent-3
        line it paraphrases and invents clauses. Editing one clause puts it back in-distribution
        (and is faster: fewer tokens). Sentences >14 words are split at the comma nearest the
        middle. Prefers clauses that carry a clear sentiment word."""
        if getattr(config, "QWEN_UNIT", "sentence") == "sentence":
            # v3: a complete sentence or nothing - comma fragments ("jaise ki ...") start mid-thought
            return self._pick_sentence(utt)
        spans = [[m.start(), m.end()] for m in self._SEG_RE.finditer(utt)]
        if not spans or spans[0][0] != 0 or spans[-1][1] != len(utt):
            spans = [[0, len(utt)]]
        parts = []
        for a, b in spans:
            t = utt[a:b]
            if len(t.split()) > 14 and "," in t:
                best = None
                for i, ch in enumerate(t):
                    if ch == ",":
                        l, r = len(t[:i + 1].split()), len(t[i + 1:].split())
                        if l >= 4 and r >= 4 and (best is None or abs(l - r) < best[0]):
                            best = (abs(l - r), i)
                if best:
                    parts += [[a, a + best[1] + 1], [a + best[1] + 1, b]]
                    continue
            parts.append([a, b])
        best = None
        for a, b in parts:
            core = utt[a:b].strip()
            n = len(core.split())
            if n < 4 or n > 16:
                continue
            score = ((2.0 if self._line_polarity(core) != 0 else 0.0)
                     + (1.0 if core.endswith("!") else 0.0)
                     - (1.0 if core.endswith("?") else 0.0) - n / 40)
            # Prefer a clause that ENDS a sentence: slang appended there sounds natural.
            # A fragment ending in a comma/semicolon is mid-sentence, so slang tacked on
            # after it ("...add karenge, fit hai, taki robot...") reads as broken.
            if core.endswith((".", "!", "?")):
                score += 1.5
            elif core.endswith((",", ";")):
                score -= 1.5
            if best is None or score > best[0]:
                best = (score, a, b)
        if best:
            return best[1], best[2]
        return (0, len(utt)) if len(utt.split()) <= 22 else None

    def _judge_ok(self, neutral: str, edited: str, slang):
        """LLM naturalness gate for one edited clause -> (ok, reason_if_rejected).
        Fails OPEN (accepts) if the judge is disabled/unreachable; the lexical guards
        have already run by this point."""
        # DEFAULT: deterministic local rule gate (agents/edit_gate.py) - no network, no LLM judge.
        # (The Groq judge rejected clearly good edits such as "Yeh kamaal idea hai, Arav." = 2/5.)
        if getattr(config, "QWEN_GATE", "rules") == "rules":
            from agents.edit_gate import rule_gate
            ok, why = rule_gate(neutral, edited, slang or "")
            return ok, ("" if ok else "gate:" + why)
        if not getattr(config, "QWEN_JUDGE", True) or getattr(self, "groq_client", None) is None:
            return True, ""
        try:
            r = self.groq_client.chat.completions.create(
                model=config.GROQ_MODEL, temperature=0.0, max_tokens=400,
                reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
                messages=[{"role": "user", "content": JUDGE_LINE_PROMPT.format(
                    neutral=neutral, edited=edited, slang=slang or "")}])
            m = re.search(r"\{.*\}", r.choices[0].message.content or "", re.DOTALL)
            d = json.loads(m.group())
            score = float(d.get("score", 0))
            keeps = bool(d.get("meaning_preserved", True))
        except Exception:
            self.stats["groq_errors"] += 1
            return True, ""
        ok = score >= getattr(config, "QWEN_JUDGE_MIN", 4) and keeps
        return ok, ("" if ok else "judge:%g%s" % (score, "" if keeps else ",meaning"))

    @staticmethod
    def _has_filler(core: str) -> bool:
        """Clause already contains a plain casual filler (yaar/bhai/bro...): don't stack more."""
        return any(w in PLAIN_FILLER_WORDS for w in re.findall(r"[a-z]+", core.lower()))

    def _groq_text(self, prompt: str, max_tokens: int = 400) -> str:
        r = self.groq_client.chat.completions.create(
            model=config.GROQ_MODEL, temperature=0.0, max_tokens=max_tokens,
            reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""

    def _clause_polarity(self, core: str) -> int:
        """+1 / -1 / 0. Word-list first (free); if it can't tell, ask the LLM (cached) so
        tone detection is not limited to the words we happened to list."""
        # A reassuring line ("Chinta mat karo, ... galti hai ...") mentions a problem but the SPEAKER is calm
        # and supportive: its tone is neutral (-> a filler like "yaar"), not negative (-> "bakwaas", "panga").
        if re.match(r"^\s*(chinta mat|tension mat|ghabrao mat|dhairya rakho|fikar mat|koi baat nahi)", core.lower()):
            return 0
        p = self._line_polarity(core)
        if p != 0:
            return p
        if getattr(self, "groq_client", None) is None or not getattr(config, "QWEN_LLM_TONE", True):
            return 0
        key = core.lower()
        if key in self._tone_cache:
            return self._tone_cache[key]
        pol = 0
        try:
            w = self._groq_text(TONE_PROMPT.format(core=core), 300).strip().lower()
            pol = 1 if w.startswith("pos") else (-1 if w.startswith("neg") else 0)
        except Exception:
            self.stats["groq_errors"] += 1
        self._tone_cache[key] = pol
        return pol

    def _rerank(self, core: str, pool, tone: str):
        """LLM keeps only the pool slangs that fit THIS clause. Returns a list (maybe empty) or
        None if the reranker failed (caller then falls back to the plain random shortlist)."""
        try:
            txt = self._groq_text(RERANK_PROMPT.format(
                clause=core, tone=tone, options="\n".join("- " + p for p in pool)), 500)
            fit = json.loads(re.search(r"\{.*\}", txt, re.DOTALL).group()).get("fit", [])
        except Exception:
            self.stats["groq_errors"] += 1
            return None
        by_lower = {p.lower(): p for p in pool}
        out = []
        for x in fit if isinstance(fit, list) else []:
            p = by_lower.get(str(x).strip().lower())
            if p and p not in out:
                out.append(p)
        return out[:6]

    def _fit_shortlist(self, core: str, intent: str, emotion: str, pol: int):
        """Retrieve-then-rerank. [] = nothing fits this clause (leave it neutral);
        None = reranker unavailable/off (use the plain random shortlist)."""
        if not getattr(config, "QWEN_RERANK", True) or getattr(self, "groq_client", None) is None:
            return None
        tone = {1: "positive", -1: "negative", 0: "neutral"}[pol]
        for _ in range(max(1, int(getattr(config, "QWEN_RERANK_ROUNDS", 2)))):
            pool = self._select_candidates(intent, emotion,
                                           k=int(getattr(config, "QWEN_RERANK_POOL", 30)), polarity=pol)
            if not pool:
                return []
            fit = self._rerank(core, pool, tone)
            if fit is None:
                # The reranker FAILED (Groq error / rate limit). It is the check that vetoes context-specific
                # slang ("paisa vasool") and technical lines, so fail CLOSED instead of using a random shortlist:
                #   neutral tone -> no edit; known tone -> only generic slang that fits almost any sentence.
                if pol == 0:
                    return []
                safe = [x for x in pool if x.lower() in SAFE_GENERIC_SLANG]
                return safe[:6] or []
            if fit:
                return fit
        return []

    def _vet_edit(self, utterance: str, cand: str, candidates, pol_hint: int = 0):
        """Single place that decides whether an editor output is acceptable.
        Returns (ok, used_slang, reason_if_rejected). Used for BOTH the local Qwen edit and
        the Groq fallback, so neither can slip a bad line past the guards."""
        if not cand or cand.strip().lower() == utterance.strip().lower():
            return False, None, "abstained"
        # the slang must be NEWLY inserted: one already present in the original clause (e.g.
        # "saaf", "sahi") would otherwise let a no-op or a mere repetition pass as an edit
        low_orig = utterance.lower()
        used = next((c for c in candidates
                     if c.lower() in cand.lower() and c.lower() not in low_orig), None)
        if used is None:
            return False, None, "off-list"
        if not self._keeps_meaning(utterance, cand):
            return False, used, "drift"
        # the word-list polarity is only a partial view ("darr hai ... nakam ho jaunga" has no listed
        # word); fall back to the clause tone the editor already worked out (LLM tone) so a positive
        # slang can never land on a fearful / sad clause, and vice versa
        pol, u = (self._line_polarity(utterance) or pol_hint), used.lower()
        if (pol < 0 and u in self._pos_slang) or (pol > 0 and u in self._neg_slang):
            return False, used, "polarity-flip"
        if (pol == 0 and getattr(config, "SLANG_STRICT_TONE", True)
                and (u in self._pos_slang or u in self._neg_slang)):
            return False, used, "tone-unknown"
        return True, used, ""

    @staticmethod
    def _keeps_meaning(neutral: str, output: str) -> bool:
        """Lexical drift guard, calibrated on the gold pairs.

        Good edits often SWAP a short evaluative phrase for slang ("bahut accha" ->
        "ekdum jhakaas"), so a flat "keep >=X% of words" rule is wrong in both directions:
        the old 0.6 floor let through a line that lost "shayad wiring mein" (83% kept) and
        flipped its meaning. What separates bad from good is line LENGTH:
          * short lines (<10 content words) may lose up to half their words (phrase swap)
          * long lines may lose at most 2 content words
          * and an edit may not add more than 7 new words.
        Known limit: this is lexical only - it cannot see a meaning flip that keeps the
        words. A semantic check (centred MuRIL) is the proper long-term fix."""
        nlist = re.findall(r"[a-z]+", neutral.lower())
        nwords = {w for w in nlist if len(w) >= 3}              # content words
        if not nwords:
            return True
        owords = set(re.findall(r"[a-z]+", output.lower()))
        dropped = len(nwords - owords)
        added = len(owords - set(nlist))
        allowed_drop = 2 if len(nwords) >= 10 else (len(nwords) + 1) // 2
        return dropped <= allowed_drop and added <= 7

    # ── Trained mT5 editor ─────────────────────────────────────────────
    def _init_mt5(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._torch = torch
        full = getattr(config, "EDITOR_FULL_PATH", "")
        if full and os.path.isdir(full):
            # v3: a full fine-tuned model (its own generation_config suppresses sentinels).
            # .float() upcasts a fp16-saved model to fp32 so CPU inference is safe.
            self.mt5_tok = AutoTokenizer.from_pretrained(full)
            self.mt5 = AutoModelForSeq2SeqLM.from_pretrained(full).float()
            print(f"SlangRewriter: full mT5 editor ({full}).")
        else:
            # v1/v2: base model + LoRA adapter
            from peft import PeftModel
            self.mt5_tok = AutoTokenizer.from_pretrained(config.EDITOR_BASE_MODEL)
            base = AutoModelForSeq2SeqLM.from_pretrained(config.EDITOR_BASE_MODEL)
            self.mt5 = PeftModel.from_pretrained(base, config.EDITOR_ADAPTER_PATH)
            print(f"SlangRewriter: LoRA mT5 editor ({config.EDITOR_ADAPTER_PATH}).")
        self.mt5.eval()

    def _mt5_edit(self, utterance: str, control: str) -> str:
        # Same input format the training script uses (see data_prep/build_editor_dataset.py)
        src = f"insert slang | control: {control} | {utterance}"
        inputs = self.mt5_tok(src, return_tensors="pt", truncation=True,
                              max_length=config.EDITOR_MAX_INPUT)
        with self._torch.no_grad():
            out = self.mt5.generate(**inputs, max_new_tokens=config.EDITOR_MAX_TARGET,
                                    num_beams=config.EDITOR_NUM_BEAMS, early_stopping=True)
        text = self.mt5_tok.decode(out[0], skip_special_tokens=True).strip()
        return self._dedupe_wrap(text)

    @staticmethod
    def _dedupe_wrap(text: str) -> str:
        """The full mT5 editor tends to wrap a filler at BOTH ends
        ("Yaar, <line> yaar"). If a leading "Word," is repeated as the final
        word, drop the trailing copy. Only triggers on that exact pattern."""
        toks = text.split()
        if len(toks) >= 3 and toks[0].endswith(","):
            first = toks[0].rstrip(",").lower()
            last = toks[-1].rstrip(".,!?").lower()
            if first == last:
                return " ".join(toks[:-1])
        return text

    # ── Lexicon ────────────────────────────────────────────────────────
    def _load_lexicon(self, dataset_path):
        path = dataset_path or getattr(config, "SLANG_RUNTIME_SRC", None) or config.FELIX_DATASET_CLEAN
        if not os.path.exists(path):
            path = config.FELIX_DATASET_CLEAN
        if not os.path.exists(path):
            path = config.FELIX_DATASET_RAW
        by_control, by_function, filler_pool = defaultdict(set), defaultdict(set), set()
        try:
            with open(path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                is_clean = "control" in cols
                for row in reader:
                    if is_clean:
                        control = (row.get("control") or "").strip().upper()
                        func = (row.get("function") or "").strip().lower()
                        slangs = [s.strip() for s in (row.get("slangs") or "").split("|") if s.strip()]
                    else:
                        control = (row.get("Control tokens") or "").strip().strip("[]").upper()
                        func = (row.get("function") or "").strip().lower()
                        slangs = [s.strip() for s in re.split(r"[,/|]", row.get("Possible Slangs") or "") if s.strip()]
                    for s in slangs:
                        s = s.strip()
                        if not s:
                            continue
                        if control:
                            by_control[control].add(s)
                        if func:
                            by_function[func].add(s)
                        if func == "filler":
                            filler_pool.add(s)
        except Exception as e:
            print(f"  Lexicon load warning: {e} — minimal fallback pool.")
        if not filler_pool:
            filler_pool = {"yaar", "bhai", "bro", "boss"}
        return ({k: sorted(v) for k, v in by_control.items()},
                {k: sorted(v) for k, v in by_function.items()},
                sorted(filler_pool))

    def _compute_negative_slang(self) -> set:
        """Slang that reads as negative/harsh: it appears under a negative control
        and NOT under any positive control, plus a hard list of insulting terms.
        These are excluded when a line's tone is positive/neutral."""
        neg, pos = set(), set()
        for ctrl, items in self.by_control.items():
            bucket = neg if ctrl in NEGATIVE_CONTROLS else (pos if ctrl in POSITIVE_CONTROLS else None)
            if bucket is not None:
                bucket.update(s.lower() for s in items)
        risky = {s for s in neg if s not in pos}          # negative-only slang
        risky |= {s.lower() for s in NEGATIVE_SLANG_HARD}  # always-insulting terms
        return risky

    def _compute_positive_slang(self) -> set:
        """Slang that appears under a positive control and NOT under any negative one."""
        neg, pos = set(), set()
        for ctrl, items in self.by_control.items():
            bucket = neg if ctrl in NEGATIVE_CONTROLS else (pos if ctrl in POSITIVE_CONTROLS else None)
            if bucket is not None:
                bucket.update(s.lower() for s in items)
        return {s for s in pos if s not in neg}

    @staticmethod
    def _line_polarity(line: str) -> int:
        """+1 positive, -1 negative, 0 unclear. A positive word next to a negation counts as
        NEGATIVE ("bilkul acchi nahi thi" = not good), which is the case a plain word-list
        gets backwards."""
        toks = re.findall(r"[a-z]+", line.lower())
        neg = sum(t in _POL_NEG_WORDS for t in toks)
        pos = 0
        for i, t in enumerate(toks):
            if t in _POL_POS_WORDS:
                if any(w in _POL_NEGATIONS for w in toks[max(0, i - 1):i + 3]):
                    neg += 1
                else:
                    pos += 1
        return -1 if neg > pos else (1 if pos > neg else 0)

    def _is_respectful(self, utterance: str) -> bool:
        """True if the line is grateful/respectful/farewell — skip slang entirely."""
        low = utterance.lower()
        return any(re.search(rf"\b{re.escape(m)}\b", low) for m in RESPECT_MARKERS)

    def _pool_for(self, intent: str, emotion: str, polarity: int = 0) -> List[str]:
        # The clause's OWN polarity beats the (often wrong) intent label: a negative clause
        # ("circuit baar baar burn out ho raha hai") gets negative-tone slang, a positive one
        # positive-tone slang. Without this, lines the classifier called "casual" were offered
        # positive slang, which the polarity guard then (rightly) rejected -> left neutral.
        if polarity < 0:
            controls = {"ASSERT_FRUSTRATION", "ASSERT_ANGER", "ASSERT_NEGATIVE",
                        "ASSERT_STRESS", "ASSERT_ANXIETY", "ASSERT_SADNESS"}
        elif polarity > 0:
            controls = {"ASSERT_POSITIVE", "ASSERT_HAPPY", "ASSERT_EXCITED"}
        elif getattr(config, "SLANG_STRICT_TONE", True):
            # Tone unknown / neutral -> only mood-neutral fillers (yaar, arre, chill ...). Offering
            # mood-loaded slang here is how "fire" ended up on a fearful line. The Agent-2 emotion
            # label is deliberately NOT used (it is "happy" for everyone - known no-op).
            return sorted(self.filler_pool)
        else:
            controls = set(GATE_INTENT_TO_CONTROLS.get(intent, set()))
            controls |= EMOTION_TO_CONTROLS.get((emotion or "neutral").lower(), set())
        pool = set()
        for c in controls:
            pool.update(self.by_control.get(c, []))
        if len(pool) < 6:  # enrich generic/casual lines with colourful functions
            for fn in COLOURFUL_FUNCTIONS:
                pool.update(self.by_function.get(fn, []))
        if len(pool) < 4:
            pool.update(self.filler_pool)
        return sorted(pool)

    def _select_candidates(self, intent: str, emotion: str, k: int = 5,
                           polarity: int = 0) -> List[str]:
        """Rarity-weighted shortlist of DISTINCT slang; bhai/yaar down-weighted,
        immediate repeats avoided. Groq then picks the best-fitting one."""
        pool = self._pool_for(intent, emotion, polarity) or list(self.filler_pool)
        # Sentiment guard: on a positive/neutral line, drop harsh/teasing slang
        # (e.g. "muft-khor", "kaand") so it can't be inserted where it clashes.
        neg_ok = (intent in NEGATIVE_INTENTS or polarity < 0
                  or (emotion or "neutral").lower() in NEGATIVE_EMOTIONS)
        if not neg_ok:
            filtered = [s for s in pool if s.lower() not in self._neg_slang]
            pool = filtered or pool   # never empty the pool entirely
        weights = []
        for s in pool:
            w = 1.0 / (1 + self._used[s])
            if s == self._last:
                w *= 0.05
            if s.lower() in {"bhai", "yaar", "ekdum"}:
                w *= 0.35
            weights.append(w)
        chosen, remaining, rw = [], list(pool), list(weights)
        for _ in range(min(k, len(remaining))):
            pick = self._rng.choices(remaining, weights=rw, k=1)[0]
            i = remaining.index(pick)
            remaining.pop(i); rw.pop(i)
            chosen.append(pick)
        return chosen

    def _note_used(self, output: str, candidates: List[str]):
        low = output.lower()
        used = next((c for c in candidates if c.lower() in low), None)
        if used:
            self._used[used] += 1
            self._last = used
        return used

    # ── Per-line rewrite ───────────────────────────────────────────────
    def reset(self):
        self.gate._scene_slang_counts = {}
        self._used = defaultdict(int)
        self._last = None

    def _emotion_of(self, speaker: str) -> str:
        from agents.name_utils import match_profile
        profiles = self.scratchpad.read("character_profiles") or {}
        return (match_profile(profiles, speaker) or {}).get("current_emotion", "neutral")

    def rewrite_line(self, speaker: str, utterance: str, scene_id: str) -> Tuple[str, dict]:
        # Upstream failure marker (Agent 3 writes "[generation failed after N attempts]" when
        # Groq is rate-limited). It is not dialogue - never "add slang" to it (a real run once
        # produced "Generation failed after 3 attempts, ekdum solid!").
        if "[generation failed" in utterance.lower():
            return utterance, {"applied": False, "go": False, "gate_stopped": None,
                               "intent": None,
                               "reason": "skipped: upstream Pass-1 generation failed"}
        _tag = speaker.strip().upper()
        _allow = (getattr(config, "SLANG_ALLOW_UNATTRIBUTED", False)
                  and _tag in {"SPEAKER", "UNKNOWN", "VOICE"})
        if (_tag in UNATTRIBUTED_SPEAKERS and not _allow) or ({_squash(t) for t in re.findall(r"[A-Za-z]+", _tag)}
                                                              & _FORMAL_OR_ELDER_TAGS):
            # unknown or formal speaker = register unknown/formal; the safe default is to leave
            # the line alone
            return utterance, {"applied": False, "go": False, "gate_stopped": None, "intent": None,
                               "reason": "skipped: unattributed, formal or elder speaker"}
        decision = self.gate.run_all_gates(speaker, utterance, scene_id, self.intent_clf)
        meta = {"applied": False, **decision}
        if not decision["go"]:
            return utterance, meta

        # Register guard: a grateful / respectful / farewell line gets NO slang,
        # even from a peer (avoids e.g. "dhanyavaad ... muft-khor").
        if self._is_respectful(utterance):
            meta["reason"] = "respectful/grateful line — slang skipped"
            return utterance, meta

        intent = decision["intent"]
        if self.backend == "qwen":
            new_line, slang, notes, rejects, engine = None, None, [], [], None
            span = self._pick_segment(utterance)
            if span is None:
                notes.append("no-editable-clause")
            else:
                a, b = span
                seg = utterance[a:b]
                lead = seg[:len(seg) - len(seg.lstrip())]
                trail = seg[len(seg.rstrip()):]
                core = seg.strip()
                emotion = self._emotion_of(speaker)
                if self._has_filler(core):
                    notes.append("already-casual")
                else:
                    pol = self._clause_polarity(core)
                    if (pol != 0 and getattr(config, "QWEN_GATE", "rules") == "rules"
                            and getattr(config, "QWEN_SLOT_PRECHECK", True)):
                        from agents.edit_gate import has_slot
                        if not has_slot(core):
                            # a mood-slang edit needs an evaluative / emotion word to replace; without one the
                            # rule gate would reject every output, so skip the shortlist call + ~20 s generation
                            shortlist = []
                            notes.append("no-slot")
                        else:
                            shortlist = self._fit_shortlist(core, intent, emotion, pol)
                    else:
                        shortlist = self._fit_shortlist(core, intent, emotion, pol)
                    if shortlist is not None and not shortlist:
                        notes.append("no-fit")      # content-aware abstain: nothing suits this clause
                    else:
                        if shortlist is None:       # reranker off/unavailable -> plain random shortlist
                            shortlist = self._select_candidates(intent, emotion, k=self._qwen_k, polarity=pol)
                        edited = None
                        for attempt in range(max(1, int(getattr(config, "QWEN_MAX_TRIES", 1)))):
                            if not shortlist:
                                break
                            cand = self._qwen_edit(core, shortlist)
                            ok, used, why = self._vet_edit(core, cand, shortlist, pol)
                            if ok:
                                ok, why = self._judge_ok(core, cand, used)
                            if ok:
                                edited, slang, engine = cand, self._note_used(cand, shortlist), "qwen-lora"
                                break
                            notes.append(why)
                            rejects.append((why, cand or ""))
                        # local edit rejected -> Groq places one of the SAME fit-checked slangs
                        if (edited is None and shortlist and getattr(self, "groq_client", None) is not None
                                and getattr(config, "QWEN_GROQ_FALLBACK", True)):
                            for _ in range(max(1, int(getattr(config, "QWEN_FALLBACK_TRIES", 3)))):
                                cand = self._place_with_groq(core, shortlist, strict=True)
                                ok, used, why = self._vet_edit(core, cand, shortlist, pol)
                                if ok:
                                    ok, why = self._judge_ok(core, cand, used)
                                if ok:
                                    edited, slang, engine = cand, self._note_used(cand, shortlist), "groq-fallback"
                                    break
                                notes.append("groq:" + why)
                                rejects.append(("groq:" + why, cand or ""))
                        if edited:
                            edited = edited.strip()
                            tail = core[-1:] if core[-1:] in ",;.!?" else ""
                            if tail in (",", ";"):
                                # the clause was a FRAGMENT of a longer sentence: it must stay one
                                # (a model-added "." or "!" would split the sentence mid-thought)
                                edited = edited.rstrip(".!?;,") + tail
                            elif tail and edited[-1:] not in ",;.!?":
                                edited += tail              # keep the clause's closing punctuation
                            edited = re.sub(r"([!?])\s*,\s*", r"\1 ", edited)       # "jeet!, jhakaas!" -> "jeet! jhakaas!"
                            new_line = utterance[:a] + lead + edited + trail + utterance[b:]
                            meta["edited_clause"] = core
            if engine:
                meta["engine"] = engine
            if new_line is None and notes:
                meta["editor_note"] = "kept neutral (" + ", ".join(notes) + ")"
                meta["editor_rejects"] = [{"why": w, "text": t} for w, t in rejects]
                # live visibility: show what the model wrote and which guard stopped it
                for w, t in rejects[:2]:
                    if w != "abstained":
                        print(f"      ~ {speaker}: rejected [{w}] -> {t[:110]}")
        elif self.backend == "mt5":
            control = INTENT_TO_EDITOR_CONTROL.get(intent, "ASSERT_NEUTRAL")
            new_line = self._mt5_edit(utterance, control)
            slang = None
        else:
            candidates = self._select_candidates(intent, self._emotion_of(speaker))
            new_line = self._place_with_groq(utterance, candidates)
            slang = self._note_used(new_line, candidates) if new_line else None

        if not new_line or new_line.strip().lower() == utterance.strip().lower():
            return utterance, meta

        added = max(1, len(new_line.split()) - len(utterance.split()))
        self.gate.record_slang_insertion(scene_id, added)
        meta.update({"applied": True, "slang": slang, "slang_tokens_added": added})
        return new_line, meta

    def _place_with_groq(self, utterance: str, candidates: List[str], max_retries: int = 3,
                         strict: bool = False) -> str:
        shortlist = "\n".join(f"- {c}" for c in candidates)
        # strict = insert-only (copy the line exactly, add one slang at an edge): used for the
        # fallback, because free rewriting was the main cause of meaning-changing edits
        prompt = (PLACE_PROMPT_STRICT if strict else PLACE_PROMPT).format(
            candidates=shortlist, utterance=utterance)
        for attempt in range(max_retries):
            try:
                resp = self.groq_client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=250,
                    reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
                )
                out = (resp.choices[0].message.content or "").strip()
                out = out.split("\n")[0].strip().strip('"').strip()
                if ":" in out and len(out.split(":", 1)[0].split()) <= 2:
                    head, rest = out.split(":", 1)
                    if head.isupper() or head.istitle():
                        out = rest.strip()
                return out
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"    Groq place failed ({e})")
        return ""

    # ── Whole-script pass ──────────────────────────────────────────────
    def rewrite_script(self, neutral_script: List[dict]) -> List[dict]:
        slang_script = []
        for scene in neutral_script:
            sid = scene["scene_id"]
            out_lines, decisions = [], []
            for line in scene["dialogue"].split("\n"):
                if ":" in line:
                    speaker, utt = line.split(":", 1)
                    speaker, utt = speaker.strip(), utt.strip()
                    if not utt:
                        continue
                    new_utt, meta = self.rewrite_line(speaker, utt, sid)
                    out_lines.append(f"{speaker}: {new_utt}")
                    decisions.append({"speaker": speaker, **meta})
                elif line.strip():
                    out_lines.append(line)
            slang_script.append({"scene_id": sid, "dialogue": "\n".join(out_lines),
                                 "decisions": decisions})
            used = [d.get("slang") for d in decisions if d.get("applied")]
            eng = {}
            for d in decisions:
                if d.get("applied") and d.get("engine"):
                    eng[d["engine"]] = eng.get(d["engine"], 0) + 1
            extra = ("  [" + ", ".join(f"{k}: {v}" for k, v in eng.items()) + "]") if eng else ""
            print(f"  {sid}: {len(used)}/{len(decisions)} lines slanged -> {used}{extra}")
        return slang_script


if __name__ == "__main__":
    from memory.scratchpad import Scratchpad
    from agents.intent_classifier import GroqIntentClassifier
    sp = Scratchpad("agent4_smoke")
    sp.write("character_profiles", {"Arav": {"name": "Arav", "social_role": "peer",
             "slang_level": 5, "current_emotion": "happy", "stress_index": 0.0}})
    rw = SlangRewriter(sp, GroqIntentClassifier(), backend="groq")
    demo = [{"scene_id": "scene_01",
             "dialogue": "ARAV: Yeh purani diary bahut interesting hai.\n"
                         "ARAV: Mujhe iske baare mein aur jaanna hai.\n"
                         "ARAV: Chalo ise abhi padhte hain."}]
    for s in rw.rewrite_script(demo):
        print(f"\n[{s['scene_id']}]\n{s['dialogue']}")
