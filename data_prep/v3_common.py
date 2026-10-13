"""
data_prep/v3_common.py
Shared helpers for the v3 "on-distribution" slang-editor dataset (offline teacher distillation).

The teacher (Claude) is used ONLY to write training data. At runtime the small Qwen LoRA works alone.
"""
from __future__ import annotations
import csv, json, os, re, time, threading
from collections import defaultdict
from dotenv import load_dotenv
import anthropic

load_dotenv()

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LEX_SRC = "data/felix_dataset/final_dataset.csv"

# ($/1M input, $/1M output)
PRICING = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0),
           "claude-opus-5-5": (4.0, 20.0)}

POS_CTRL = {"ASSERT_POSITIVE", "ASSERT_HAPPY", "ASSERT_EXCITED", "POSITIVE_EVAL"}
NEG_CTRL = {"ASSERT_FRUSTRATION", "ASSERT_ANGER", "ASSERT_NEGATIVE", "ASSERT_SADNESS", "ASSERT_STRESS",
            "ASSERT_ANXIETY", "ASSERT_WARNING", "ASSERT_FEAR", "EXPRESS_SADNESS", "EXPRESS_STRESS",
            "REQUEST_FRUSTRATION"}
PLAIN_FILLERS = {"yaar", "bhai", "bro", "boss", "abe", "abey", "re", "yr", "bhaiya", "arre"}


def load_lexicon():
    """-> {slang: tone} where tone in positive/negative/neutral (majority control of the vetted rows)."""
    votes = defaultdict(lambda: defaultdict(int))
    with open(LEX_SRC, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s = (r.get("slangs") or "").strip()
            c = (r.get("control") or "").strip().upper()
            if not s:
                continue
            t = "positive" if c in POS_CTRL else ("negative" if c in NEG_CTRL else "neutral")
            votes[s][t] += 1
    lex = {}
    for s, v in votes.items():
        lex[s] = max(v, key=v.get)
    return lex


class Teacher:
    """Thin Anthropic wrapper with a running cost estimate and a hard spend cap."""

    def __init__(self, model: str, max_usd: float):
        self.model, self.max_usd = model, max_usd
        self.client = anthropic.Anthropic()
        self.pin, self.pout = PRICING.get(model, (2.0, 10.0))
        self.in_tok = self.out_tok = self.calls = 0
        self._lock = threading.Lock()

    @property
    def cost(self) -> float:
        return self.in_tok / 1e6 * self.pin + self.out_tok / 1e6 * self.pout

    def ask(self, system: str, user: str, max_tokens: int = 4000) -> str:
        if self.cost >= self.max_usd:
            raise RuntimeError(f"spend cap ${self.max_usd} reached (~${self.cost:.2f})")
        think = ({"type": "between_tools"} if "sonnet" in self.model
                 else {"type": "disabled"} if "haiku" in self.model else None)
        kw = dict(model=self.model, max_tokens=max_tokens, output_config={"effort": "low"},
                  system=system, messages=[{"role": "user", "content": user}])
        if think:
            kw["thinking"] = think
        last = None
        for attempt in range(4):
            try:
                r = self.client.messages.create(**kw)
                with self._lock:
                    self.calls += 1
                    self.in_tok += r.usage.input_tokens
                    self.out_tok += r.usage.output_tokens
                return "".join(b.text for b in r.content if b.type == "text")
            except anthropic.APIStatusError as e:
                last = e
                if getattr(e, "status_code", 0) in (400, 401, 402, 403):
                    raise
                time.sleep(2 + 3 * attempt)
            except Exception as e:
                last = e
                time.sleep(2 + 3 * attempt)
        raise RuntimeError(f"teacher failed: {str(last)[:150]}")


def parse_json(txt: str):
    """Extract the first JSON array/object from model text, tolerating trailing commas."""
    for pat in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pat, txt or "", re.DOTALL)
        if not m:
            continue
        for cand in (m.group(), re.sub(r",\s*([\]}])", r"\1", m.group())):
            try:
                return json.loads(cand)
            except Exception:
                pass
    return None


# ── the runtime clause segmentation, copied from agents/agent4_slang_rewriter.py so that the
#    training clauses are cut EXACTLY the way the pipeline cuts them ─────────────────────────────
_SEG_RE = re.compile(r"[^.!?;]+[.!?;]*\s*|[.!?;]+\s*")


def candidate_clauses(utt: str):
    """All clauses (4-16 words) the runtime picker could choose from one line."""
    spans = [[m.start(), m.end()] for m in _SEG_RE.finditer(utt)]
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
    out = []
    for a, b in parts:
        core = utt[a:b].strip()
        if 4 <= len(core.split()) <= 16:
            out.append(core)
    return out


def candidate_units(utt: str, lo: int = 4, hi: int = 24):
    """Complete sentences (lo..hi words). The v3 editor works on whole sentences, not comma
    fragments: fragments start mid-sentence ("aur...", "kyunki...") and read awkwardly with slang."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", (utt or "").strip()) if x.strip()]
    return [x for x in sents if lo <= len(x.split()) <= hi]


# ── the runtime drift guard (identical logic to SlangRewriter._keeps_meaning) ───────────────────
def keeps_meaning(neutral: str, output: str) -> bool:
    nlist = re.findall(r"[a-z]+", neutral.lower())
    nwords = {w for w in nlist if len(w) >= 3}
    if not nwords:
        return True
    owords = set(re.findall(r"[a-z]+", output.lower()))
    dropped = len(nwords - owords)
    added = len(owords - set(nlist))
    allowed_drop = 2 if len(nwords) >= 10 else (len(nwords) + 1) // 2
    return dropped <= allowed_drop and added <= 7


def newly_inserted(neutral: str, output: str, slang: str) -> bool:
    return slang.lower() in output.lower() and slang.lower() not in neutral.lower()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    out.append(json.loads(ln))
    return out
