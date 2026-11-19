import json
import os
import re
import time
from typing import List

import config


# ── Prompt template ────────────────────────────────────────────────────
NARRATIVE_ANALYST_PROMPT = """You are a screenplay structural analyst.
Your ONLY job is to extract a structured event chain from ONE scene (scene_id = "{scene_id}") of a Hindi story. Do NOT generate any dialogue. Only extract what exists.

{story_block}

Previous scene summary (for continuity):
{previous_scene_summary}

Output a JSON array containing EXACTLY ONE object. No preamble. No explanation. No markdown code fences. Just the raw JSON array with one element.

The single object must have EXACTLY these fields:
{{
  "scene_id": "{scene_id}",
  "key_events": ["Event 1 description", "Event 2 description"],
  "event_evidence": ["a short QUOTATION (at most 15 words) copied exactly from the scene text that shows event 1", "quotation for event 2"],
  "characters": ["CharName1", "CharName2"],
  "scene_goal": "What this scene accomplishes narratively",
  "location": "The place, or Unknown",
  "location_evidence": "a quotation copied from the scene text that names or clearly shows that place, or an empty string",
  "narrative_link": "Following [previous event], [character] now [does X]",
  "pronoun_map": {{"usne": "Ram", "woh": "Sita"}}
}}

CRITICAL RULES:
- ALWAYS use scene_id = "{scene_id}" exactly - do not create additional scene_ids, do not split this text into multiple scenes.
- Output EXACTLY ONE JSON object inside the array, even if the scene text describes multiple events or covers a time span. Combine everything into ONE object's key_events list.
- LANGUAGE OF YOUR ANSWER: write every field in English, except that character names must be written in Roman letters (Hori, Dhaniya, Bhola) and
  quotations are copied exactly as they appear in the text. Never answer in Devanagari.
- GROUNDING: every event must be something the scene text actually says or shows, in the meaning the words really have. Read unusual, literary or
  regional words in the ORIGINAL text and take their true meaning (do not guess from the romanised spelling). If you cannot quote the text for
  an event, do not list it. "event_evidence" must have exactly one quotation per event, in the same order, copied character for character.
- location: ONLY a place that the scene text names or clearly shows (a house, a field, a road, a market). If the text does not say, write
  "Unknown" and leave location_evidence empty. NEVER guess, NEVER describe uncertainty ("likely", "probably", "where X is"), NEVER write a sentence.
- "characters" must contain ONLY proper names of individual named people (e.g. "Arav", "Ram", "Shyam").
- NEVER include: groups of people (e.g. "log"/people, "chatron"/students, "shodhakartaon"/researchers), organizations or institutions (e.g. "nagar parishad"/city council), objects (e.g. "dayari"/diary), or generic role references to unnamed people (e.g. "bhule-bisare vyapari"/the forgotten merchant) UNLESS that is literally the character's only name given in the story.
- If NO named individual characters appear in this scene, "characters" must be an empty list [].
- key_events: list every plot-significant event (minimum 1, maximum 6)
- narrative_link: MUST reference what happened in the previous scene
- pronoun_map: map every pronoun in this chunk to the character it refers to (use {{}} if none)
"""


def story_block(roman: str, deva: str = "") -> str:
    """The scene text the analyst reads. The Devanagari original (when the story came in Devanagari) is the authority; the romanisation is only an aid."""
    if deva and deva.strip():
        return ("SCENE TEXT - ORIGINAL (Devanagari, exactly as the author wrote it; this is the authority):\n" + deva.strip()[:2600] +
                "\n\nSCENE TEXT - ROMANISED (automatic transliteration, spellings can be wrong; use it only to read names):\n" + roman.strip()[:2600])
    return "SCENE TEXT:\n" + roman.strip()[:3200]


class NarrativeAnalyst:
    def __init__(self, llm_backend: str = None):
        # llm_backend: "ollama" | "groq" | "hf"
        self.backend = llm_backend or config.LLM_BACKEND
        self._setup_llm()

    def _setup_llm(self):
        if self.backend == "ollama":
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="llama3:8b-instruct-q4_K_M",
                               temperature=0.1)

        elif self.backend == "groq":
            from dotenv import load_dotenv
            load_dotenv()
            from llm_client import make_client          # OpenRouter first, Groq fallback
            self.groq_client = make_client("extract")

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
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    # Kinship words / titles that are used AS A NAME ("Dadi", "Coach"). The LLM sometimes drops them as "generic roles"
    # (scene 3 of the science-fair story lost Dadi), which then makes Agent 3 invent a "Narrator" speaker.
    KINSHIP_NAMES = {"dadi": "Dadi", "dadaji": "Dadaji", "dada": "Dada", "dadiji": "Dadiji", "nani": "Nani",
                     "naniji": "Naniji", "nana": "Nana", "nanaji": "Nanaji", "mummy": "Mummy", "papa": "Papa",
                     "pitaji": "Pitaji", "maa": "Maa", "amma": "Amma", "babuji": "Babuji", "chacha": "Chacha",
                     "chachi": "Chachi", "mausi": "Mausi", "bua": "Bua", "coach": "Coach", "koch": "Coach", "principal": "Principal"}

    @classmethod
    def _add_kinship(cls, characters: list, text: str) -> list:
        """Add kinship/title names that occur in the scene text but are missing from the LLM's character list."""
        chars = list(characters or [])
        have = " ".join(c.lower() for c in chars)
        for term, canon in cls.KINSHIP_NAMES.items():
            if re.search(r"\b" + term + r"\b", text.lower()) and term not in have and canon.lower() not in have:
                chars.append(canon)
        return chars

    @staticmethod
    def _canonicalize(events: List[dict]) -> None:
        """Unify name variants across scenes ("Riy"/"Riya", "Sharm"/"Sharma"): a name that is a prefix of a longer
        name (<=2 extra letters, shorter >=3 letters) is replaced by the longer one everywhere (characters + pronoun_map)."""
        from collections import Counter
        # pass 0: transliteration artefacts and non-people. "Koch Varma" -> "Coach Varma"; "Tim"/"Kap" (team, cup) are objects
        NON_PEOPLE = {"tim", "team", "kap", "cup", "skul", "school", "maich", "match", "stadium", "maidan", "robot"}
        for e in events:
            fixed = []
            for n in (e.get("characters") or []):
                if not isinstance(n, str) or n.lower() in NON_PEOPLE:
                    continue
                fixed.append(re.sub(r"^Koch\b", "Coach", n))
            e["characters"] = fixed
            pm = e.get("pronoun_map")
            if isinstance(pm, dict):
                e["pronoun_map"] = {k: (re.sub(r"^Koch\b", "Coach", v) if isinstance(v, str) else v) for k, v in pm.items()}
        # pass 0b: a bare word that is part of exactly one multi-word name ("Varma", "Coach" in "Coach Varma") is that person
        multi = {n for e in events for n in e["characters"] if " " in n}
        alias = {}
        for e in events:
            for n in e["characters"]:
                if " " not in n:
                    hits = [m for m in multi if n.lower() in m.lower().split()]
                    if len(hits) == 1:
                        alias[n] = hits[0]
        for e in events:
            e["characters"] = [alias.get(n, n) for n in e["characters"]]
            pm = e.get("pronoun_map")
            if isinstance(pm, dict):
                e["pronoun_map"] = {k: alias.get(v, v) if isinstance(v, str) else v for k, v in pm.items()}
        # pass 0c: "Grandma" listed next to "Dadiji" (the story's own word): the same person, the Hindi word the story uses wins
        KIN_EN = {"grandma": {"dadi", "dadiji", "nani", "naniji"}, "grandmother": {"dadi", "dadiji", "nani", "naniji"},
                  "grandpa": {"dada", "dadaji", "nana", "nanaji"}, "grandfather": {"dada", "dadaji", "nana", "nanaji"},
                  "mother": {"maa", "amma", "mummy", "mataji"}, "mom": {"maa", "amma", "mummy", "mataji"},
                  "father": {"papa", "pitaji", "babuji"}, "dad": {"papa", "pitaji", "babuji"}}
        def merge_kin():
            every = {n for e in events for n in e["characters"]}
            kin = {}
            for en, hi in KIN_EN.items():
                have_en = [n for n in every if n.lower() == en]
                have_hi = [n for n in every if n.lower() in hi]
                if have_en and len(have_hi) == 1:
                    kin[have_en[0]] = have_hi[0]
            if kin:
                for e in events:
                    e["characters"] = list(dict.fromkeys(kin.get(n, n) for n in e["characters"]))
                    pm = e.get("pronoun_map")
                    if isinstance(pm, dict):
                        e["pronoun_map"] = {k: kin.get(v, v) if isinstance(v, str) else v for k, v in pm.items()}
        merge_kin()
        count = Counter(n for e in events for n in (e.get("characters") or []) if isinstance(n, str))
        reps, canon = [], {}
        # the MORE FREQUENT spelling wins (Rohan x9 beats a stray Rohana); ties go to the longer one (Riya beats Riy)
        for n in sorted(count, key=lambda x: (-count[x], -len(x), x)):
            low = n.lower()

            def _variant(r, low=low):
                a, b = sorted((low.replace(" ", ""), r.lower().replace(" ", "")), key=len)       # "DamadSahab" / "Damad Sahab" differ only in spacing
                if a == b or (len(a) >= 3 and b.startswith(a) and len(b) - len(a) <= 2):
                    return True
                import difflib
                return len(a) >= 6 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.9         # Dadiiji / Dadiji, Pradeepa / Pradeep
            hit = next((r for r in reps if _variant(r)), None)
            if hit is None:
                reps.append(n)
                canon[n] = n
            else:
                canon[n] = hit
        for e in events:
            seen, out = set(), []
            for n in (e.get("characters") or []):
                c = canon.get(n, n)
                if c.lower() not in seen:
                    seen.add(c.lower())
                    out.append(c)
            e["characters"] = out
            pm = e.get("pronoun_map")
            if isinstance(pm, dict):
                e["pronoun_map"] = {k: canon.get(v, v) if isinstance(v, str) else v for k, v in pm.items()}
        merge_kin()                                                  # a spelling merge may just have produced the Hindi kin name ("Dadiiji" -> "Dadiji")

    _DEVA = re.compile(r"[ऀ-ॿ]")

    @staticmethod
    def _romanise_fields(ev: dict) -> dict:
        """The analyst reads Devanagari and sometimes answers in it. Names, events, goal and location must be Roman-script text for every later agent
        (speaker labels are built from them); a leftover Devanagari word is transliterated with the project's own rule-based transliterator.
        Quotations (event_evidence, location_evidence) are left untouched: they must match the source."""
        if not any(NarrativeAnalyst._DEVA.search(str(v)) for k, v in ev.items() if k not in ("event_evidence", "location_evidence")):
            return ev
        from preprocessing.transliterate import HindiTransliterator
        tr = HindiTransliterator()
        fix = lambda t: tr.transliterate_sentence(t).strip() if NarrativeAnalyst._DEVA.search(t) else t
        for k in ("scene_goal", "location", "narrative_link"):
            if isinstance(ev.get(k), str):
                ev[k] = fix(ev[k])
        if isinstance(ev.get("key_events"), list):
            ev["key_events"] = [fix(x) if isinstance(x, str) else x for x in ev["key_events"]]
        if isinstance(ev.get("characters"), list):
            ev["characters"] = [(fix(x).title() if NarrativeAnalyst._DEVA.search(x) else x) if isinstance(x, str) else x for x in ev["characters"]]
        if isinstance(ev.get("pronoun_map"), dict):
            ev["pronoun_map"] = {k: (fix(v) if isinstance(v, str) else v) for k, v in ev["pronoun_map"].items()}
        return ev

    @staticmethod
    def _ground(ev: dict, roman: str, deva: str = "") -> dict:
        """Provenance check on one scene's extraction: an event survives only with a quotation that is really in the scene text, and a location only if
        it is unhedged and quoted (agents/grounding.py). What is removed is recorded in ev['grounding'] so a run can be audited."""
        from agents import grounding as g
        sources = [x for x in (deva, roman) if x and x.strip()]
        ev = NarrativeAnalyst._romanise_fields(ev)
        kept, dropped = g.ground_events(ev.get("key_events"), ev.get("event_evidence"), *sources)
        loc_before = ev.get("location")
        ev["key_events"] = kept
        ev["location"] = g.clean_location(loc_before, ev.get("location_evidence", ""), *sources)
        ev["grounding"] = {"events_dropped": dropped, "location_before": loc_before if ev["location"] != loc_before else None,
                           "evidence_given": bool(ev.get("event_evidence"))}
        # a character must be IN the story: a model that has to name people in a story that names nobody invents "Sara" (and the whole script then speaks as her)
        chars = [n for n in (ev.get("characters") or []) if isinstance(n, str)]
        real = [n for n in chars if g.name_grounded(n, roman)]
        if len(real) != len(chars):
            gone = [n for n in chars if n not in real]
            ev["characters"] = real
            ev["grounding"]["characters_dropped"] = gone
            if isinstance(ev.get("pronoun_map"), dict):
                ev["pronoun_map"] = {k: v for k, v in ev["pronoun_map"].items() if not (isinstance(v, str) and v in gone)}
        return ev

    @staticmethod
    def _coerce(ev: dict) -> dict:
        """Models sometimes return the right keys with the wrong types ("characters": "Ram", "key_events": 5, "location": 7): make them what the pipeline expects."""
        as_list = lambda v: [str(x) for x in v if x is not None] if isinstance(v, list) else ([str(v)] if isinstance(v, (str, int, float)) and str(v).strip() else [])
        ev["key_events"] = as_list(ev.get("key_events"))
        ev["event_evidence"] = as_list(ev.get("event_evidence"))
        ev["characters"] = as_list(ev.get("characters"))
        for k in ("scene_goal", "location", "location_evidence", "narrative_link"):
            v = ev.get(k)
            ev[k] = v if isinstance(v, str) else ("" if v is None else str(v))
        if not isinstance(ev.get("pronoun_map"), dict):
            ev["pronoun_map"] = {}
        return ev

    def _extract_one(self, chunk: dict, deva: str, previous_summary: str) -> dict:
        prompt = NARRATIVE_ANALYST_PROMPT.format(scene_id=chunk["scene_id"], story_block=story_block(chunk["text"], deva),
                                                 previous_scene_summary=previous_summary)
        # One model call can come back empty or cut off (a reasoning model spends its token budget thinking, a provider hiccups). That used to drop the scene's
        # events AND its cast for good - a missing character then lost his lines for the whole story. So: retry with a bigger budget, then once with a
        # second (non-reasoning) model from the same gateway, before giving up.
        ev = None
        size = len(chunk.get("text", "")) + len(deva or "")
        first = int(min(7000, 2500 + 1.6 * size))                    # reasoning tokens + the JSON with quoted Devanagari evidence grow with the scene; 1500 truncated the JSON
        for k, (budget, alt) in enumerate(((first, False), (int(min(12000, first * 1.6)), True), (int(min(12000, first * 1.6)), False))):        # primary, then the OTHER model (an intermittent failure of one model is not cured by asking it again), then primary with room to spare
            client = None
            if alt:
                try:
                    from llm_client import make_client
                    client = getattr(self, "_alt_client", None) or make_client("dialogue")
                    self._alt_client = client
                except Exception:
                    client = None
            try:
                raw = self._call_llm(prompt, budget) if k == 0 else self._call_llm(prompt, budget, client)
            except Exception as exc:                                                                    # a provider error is a failed attempt, not a crash
                print(f"  Agent 1: {chunk['scene_id']} model call raised {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                raw = ""
            events = self._parse_json_output(raw, chunk["scene_id"])
            cand = (events or [{}])[0]
            if cand.get("key_events") or str(cand.get("scene_goal", "Unknown")).strip().lower() not in ("unknown", ""):
                ev = cand
                break
            seen = (raw or "").strip().replace("\n", " ")[:70]
            print(f"  Agent 1: {chunk['scene_id']} came back empty / unparsable (attempt {k + 1}/3, {'second model' if alt else 'primary model'}; "
                  f"answer: {seen!r})" + ("; retrying" if k < 2 else "; giving up"), flush=True)
        if ev is None:
            ev = {"scene_id": chunk["scene_id"], "key_events": [], "characters": [], "scene_goal": "Unknown",
                  "location": "Unknown", "narrative_link": previous_summary, "pronoun_map": {}, "analysis_failed": True}
        ev["scene_id"] = chunk["scene_id"]
        ev = self._coerce(ev)
        ev["characters"] = self._add_kinship(ev.get("characters") or [], chunk["text"])
        return self._ground(ev, chunk["text"], deva)

    def _extract_parallel(self, scene_chunks: List[dict], workers: int, deva_scenes: List[str] = None) -> List[dict]:
        """Scenes are independent except for the 'previous scene' hint, so they are extracted concurrently and the narrative links are
        filled in afterwards from the previous scene's goal (5-10x less waiting than one call after another)."""
        from concurrent.futures import ThreadPoolExecutor
        deva = deva_scenes if deva_scenes and len(deva_scenes) == len(scene_chunks) else [""] * len(scene_chunks)

        def one(args):
            chunk, d = args
            try:
                return self._extract_one(chunk, d, "(see the story text; the previous scene is not needed here)")
            except Exception as exc:                                                                    # one scene's failure must not lose the others
                print(f"  Agent 1: {chunk['scene_id']} failed unexpectedly ({type(exc).__name__}: {str(exc)[:80]})", flush=True)
                return {"scene_id": chunk["scene_id"], "key_events": [], "characters": [], "scene_goal": "Unknown", "location": "Unknown",
                        "narrative_link": "", "pronoun_map": {}, "analysis_failed": True}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            all_events = list(ex.map(one, zip(scene_chunks, deva)))
        prev = "This is the first scene."
        for ev in all_events:
            ev["narrative_link"] = (f"Following the previous scene ({prev}), " + str(ev.get("narrative_link") or "").split(",", 1)[-1].strip()
                                    if prev != "This is the first scene." else str(ev.get("narrative_link") or "This is the first scene."))
            prev = ev.get("scene_goal") or prev
        self._canonicalize(all_events)
        return all_events

    def extract_event_chain(self, scene_chunks: List[dict], previous_summary: str = "This is the first scene.",
                            deva_scenes: List[str] = None) -> List[dict]:
        """deva_scenes: the Devanagari text of each scene (same order as scene_chunks) when the story came in Devanagari. The analyst then reads the
        ORIGINAL words (the romanisation of literary Hindi is easy to misread) and every claim must carry a verifiable quotation."""
        workers = int(getattr(config, "AGENT1_PARALLEL", 4))
        if workers > 1 and len(scene_chunks) > 2:
            return self._extract_parallel(scene_chunks, workers, deva_scenes)
        deva = deva_scenes if deva_scenes and len(deva_scenes) == len(scene_chunks) else [""] * len(scene_chunks)
        all_events = []
        for chunk, d in zip(scene_chunks, deva):
            first_event = self._extract_one(chunk, d, previous_summary)
            all_events.append(first_event)
            previous_summary = first_event.get("scene_goal", previous_summary)
        self._canonicalize(all_events)
        return all_events

    def _call_llm(self, prompt: str, max_tokens: int = 1500, client=None) -> str:
        if self.backend == "ollama":
            return self.llm.invoke(prompt)

        elif self.backend == "groq":
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = (client or self.groq_client).chat.completions.create(
                        model=config.GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=max_tokens,
                        reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"),
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    msg = str(e).lower()
                    # A daily/per-minute token cap won't clear on a 2s retry, so
                    # fail soft: return "" and let the caller insert a fallback
                    # empty event for this scene instead of crashing the whole run.
                    if "429" in msg or "rate limit" in msg or "quota" in msg:
                        print(f"  Agent 1: rate limit reached — skipping this scene "
                              f"(empty event). Detail: {str(e)[:100]}")
                        return ""
                    if attempt < max_retries - 1:
                        print(f"  Groq call failed ({e}), retrying in 2s...")
                        time.sleep(2)
                    else:
                        print(f"  Agent 1: giving up after {max_retries} attempts "
                              f"— skipping this scene (empty event).")
                        return ""

        elif self.backend == "hf":
            result = self.pipe(prompt, return_full_text=False)
            return result[0]["generated_text"]

    def _parse_json_output(self, raw: str, fallback_scene_id: str) -> List[dict]:
        """Extract JSON from LLM output robustly. Never raises: None / text / wrong shapes all end in a list of dicts (possibly the empty-scene fallback)."""
        raw = "" if raw is None else str(raw)
        out = self._parse_json_output_raw(raw, fallback_scene_id)
        out = [x for x in (out if isinstance(out, list) else [out]) if isinstance(x, dict)]
        failed = (not out) or (not out[0].get("key_events") and str(out[0].get("scene_goal", "Unknown")).strip().lower() in ("unknown", ""))
        if failed:
            rescued = self._salvage_truncated_json(raw)                      # an answer cut off at the token budget still holds the first fields
            if rescued:
                out = [rescued]
        return out or [{"scene_id": fallback_scene_id, "key_events": [], "characters": [], "scene_goal": "Unknown", "location": "Unknown",
                        "narrative_link": "", "pronoun_map": {}}]

    @staticmethod
    def _salvage_truncated_json(raw: str):
        """A JSON answer cut off mid-way ('...", "event_evidence": ["घर पहुँ') is closed at its last complete element: key_events and the fields before the cut survive."""
        i = raw.find("[")
        j = raw.find("{")
        start = min(x for x in (i, j) if x >= 0) if (i >= 0 or j >= 0) else -1
        if start < 0:
            return None
        txt = raw[start:]
        cuts = sorted({m.end() for m in re.finditer(r'"\s*,|"\s*\]|\}\s*,|\]\s*,|"\s*\}', txt)}, reverse=True)[:60]
        for cut in cuts:
            cand = txt[:cut].rstrip().rstrip(",")
            stack, in_str, esc = [], False, False
            for ch in cand:
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch in "[{":
                    stack.append("]" if ch == "[" else "}")
                elif ch in "]}" and stack:
                    stack.pop()
            if in_str:
                continue
            try:
                val = json.loads(cand + "".join(reversed(stack)))
            except Exception:
                continue
            val = val[0] if isinstance(val, list) and val else val
            if isinstance(val, dict) and val.get("key_events"):
                return val
        return None

    def _parse_json_output_raw(self, raw: str, fallback_scene_id: str):
        """(the original parser)"""
        # Strip markdown code fences if present
        cleaned = re.sub(r'```(?:json)?', '', raw).strip()

        # Find JSON array in output
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if not match:
            # Try wrapping a single object in an array
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                cleaned = f"[{match.group()}]"
            else:
                # Return fallback structure
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
            # Try to clean common LLM JSON errors (trailing commas)
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
        {"scene_id": "scene_01", "text": "subah ram apne ghar se nikla aur school ki taraf chal diya. raaste mein usne apne dost shyam ko dekha."}
    ]
    events = analyst.extract_event_chain(test_chunks)
    print(json.dumps(events, indent=2, ensure_ascii=False))