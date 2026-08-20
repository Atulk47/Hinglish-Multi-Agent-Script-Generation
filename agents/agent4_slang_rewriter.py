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
    "CASUAL_ASSERTION":   set(),
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


class SlangRewriter:
    def __init__(self, scratchpad, intent_classifier,
                 backend: str = "auto",
                 slang_density_threshold: float = None,
                 dataset_path: str = None):
        self.scratchpad = scratchpad
        self.intent_clf = intent_classifier
        threshold = (slang_density_threshold if slang_density_threshold is not None
                     else config.SLANG_DENSITY_THRESHOLD)
        self.gate = ThreeGateController(scratchpad, slang_density_threshold=threshold)
        self.by_control, self.by_function, self.filler_pool = self._load_lexicon(dataset_path)
        self._used = defaultdict(int)   # slang -> times used in this story
        self._last = None               # last slang used (avoid back-to-back repeats)

        self.backend = self._resolve_backend(backend)
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
        if backend == "mt5":
            return "mt5"
        if backend == "groq":
            return "groq"
        # auto: prefer the v3 full model, else the v1/v2 LoRA adapter, else groq
        if os.path.isdir(getattr(config, "EDITOR_FULL_PATH", "")) or \
           os.path.isdir(config.EDITOR_ADAPTER_PATH):
            return "mt5"
        return "groq"

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
        path = dataset_path or config.FELIX_DATASET_CLEAN
        if not os.path.exists(path):
            path = config.FELIX_DATASET_RAW
        by_control, by_function, filler_pool = defaultdict(set), defaultdict(set), set()
        try:
            with open(path, encoding="utf-8") as f:
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

    def _pool_for(self, intent: str, emotion: str) -> List[str]:
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

    def _select_candidates(self, intent: str, emotion: str, k: int = 5) -> List[str]:
        """Rarity-weighted shortlist of DISTINCT slang; bhai/yaar down-weighted,
        immediate repeats avoided. Groq then picks the best-fitting one."""
        pool = self._pool_for(intent, emotion) or list(self.filler_pool)
        weights = []
        for s in pool:
            w = 1.0 / (1 + self._used[s])
            if s == self._last:
                w *= 0.05
            if s.lower() in {"bhai", "yaar"}:
                w *= 0.35
            weights.append(w)
        chosen, remaining, rw = [], list(pool), list(weights)
        for _ in range(min(k, len(remaining))):
            pick = random.choices(remaining, weights=rw, k=1)[0]
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
        profiles = self.scratchpad.read("character_profiles") or {}
        for k, v in profiles.items():
            if k.lower() == speaker.lower():
                return v.get("current_emotion", "neutral")
        return "neutral"

    def rewrite_line(self, speaker: str, utterance: str, scene_id: str) -> Tuple[str, dict]:
        decision = self.gate.run_all_gates(speaker, utterance, scene_id, self.intent_clf)
        meta = {"applied": False, **decision}
        if not decision["go"]:
            return utterance, meta

        intent = decision["intent"]
        if self.backend == "mt5":
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

    def _place_with_groq(self, utterance: str, candidates: List[str], max_retries: int = 3) -> str:
        shortlist = "\n".join(f"- {c}" for c in candidates)
        prompt = PLACE_PROMPT.format(candidates=shortlist, utterance=utterance)
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
            print(f"  {sid}: {len(used)}/{len(decisions)} lines slanged -> {used}")
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
