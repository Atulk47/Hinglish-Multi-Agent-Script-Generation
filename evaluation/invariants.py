"""
evaluation/invariants.py  -  properties that must hold for ANY story, checked automatically (no model, no human reading).

Why this exists: every new story used to reveal a new bug, because fixes were verified only on the story that exposed them. These checks encode what a correct run
looks like independent of the story, so a corpus of very different stories (evaluation/stress_run.py) finds failures systematically.

Each check returns violations  {"code", "level" ("error" = a real defect, "warn" = suspicious / metric), "scene", "detail"}.

    check_structure(original_text, deva_scenes, ...)        scenes: count, coverage of the text, alignment
    check_pass1(deva_scenes, roman_scenes, result, ...)    the neutral script: speakers, language, grounding, speech coverage, leaks
    check_slang(neutral_script, slang_script, profiles, setting)   the edited script: audit, register, gender, frames, fillers
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional

from agents import grounding as g
from agents.name_utils import same_person

DEVA = re.compile(r"[ऀ-ॿ]")
ALLOWED_NON_ASCII = set("—–…‘’“” ")
COMMON_VOCATIVES = {"haan", "han", "nahi", "nahin", "accha", "achha", "acha", "theek", "thik", "arre", "are", "aree", "dekho", "suno", "chalo", "aao", "jao", "beta", "beti",
                    "bhaiya", "bhai", "didi", "dada", "dadi", "dadaji", "nana", "nani", "mummy", "papa", "maa", "amma", "ammi", "abbu", "sir", "madam", "maharaj", "wah",
                    "waah", "waaw", "haha", "dhanyavaad", "shukriya", "please", "sorry", "dear", "click", "clear", "bilkul", "bas", "oho", "hey", "hello", "namaste",
                    "ji", "kya", "kyun", "oh", "yes", "no", "okay", "ok", "hmm", "uff", "shabash", "congratulations", "kal", "aaj", "abhi", "phir", "toh", "to", "sach",
                    "dost", "doston", "bachcho", "bacchon", "log", "sab", "sabhi", "agar", "lekin", "par", "magar", "waise", "ruko", "ruk", "ghar", "chalo"}
META_SPEAKERS = {"narrator", "narration", "voice", "author", "speaker", "note", "wait", "correct", "output", "scene", "location", "assistant"}


def _in_source(label: str, source: str) -> bool:
    """The same test the sanitiser and Agent 1 apply (spelling-tolerant), so checker and code cannot disagree."""
    return g.name_grounded(label, source, min_tokens=1)


def _v(code, level, scene, detail):
    return {"code": code, "level": level, "scene": scene, "detail": str(detail)[:200]}


def _lines(dialogue: str):
    out = []
    for ln in (dialogue or "").split("\n"):
        if ln.strip():
            m = re.match(r"^\s*([^:\n]{1,40}):\s*(.*)$", ln)
            out.append((m.group(1).strip(), m.group(2).strip(), ln) if m else (None, ln.strip(), ln))
    return out


# --------------------------------------------------------------------------------------------------------------------------- structure
def check_structure(original_text: str, deva_scenes: List[str], event_chain: Optional[list] = None, neutral: Optional[list] = None, max_scenes: int = 15):
    V = []
    n = len(deva_scenes)
    if n < 1:
        V.append(_v("no-scenes", "error", None, "the segmenter returned no scene"))
    if n > max_scenes:
        V.append(_v("too-many-scenes", "error", None, f"{n} scenes (cap {max_scenes})"))
    norm = lambda t: re.sub(r"\s+", "", t or "")
    if norm(" ".join(deva_scenes)) != norm(original_text):
        V.append(_v("text-not-preserved", "error", None, "the scenes do not add up to the original text (text lost or duplicated)"))
    if any(not (s or "").strip() for s in deva_scenes):
        V.append(_v("empty-scene-source", "error", None, "a scene has no source text"))
    if event_chain is not None and len(event_chain) != n:
        V.append(_v("event-chain-misaligned", "error", None, f"{len(event_chain)} event entries for {n} scenes"))
    if neutral is not None and len(neutral) != n:
        V.append(_v("script-misaligned", "error", None, f"{len(neutral)} scripted scenes for {n} scenes"))
    return V


# ---------------------------------------------------------------------------------------------------------------------------- pass 1
def check_pass1(deva_scenes: List[str], roman_scenes: List[str], result: dict, max_lines: int = 14):
    """result = {"event_chain": [...], "persona": {...}, "neutral": [{scene_id, dialogue, fidelity}]}"""
    from agents.agent3_synthesizer import _is_english_meta, _leaks, _LEAK_POS, _LEAK_NEG, _LEAK_TROUBLE
    V: List[dict] = []
    persona = result.get("persona") or {}
    known = set(persona.get("characters") or {})
    extras = {x for sc in (persona.get("scene_cast") or {}).values() for x in (sc.get("extra_speakers") or [])}
    all_speakers: Dict[str, int] = {}
    ev_by = {e["scene_id"]: e for e in result.get("event_chain", [])}
    sure_leaks = _LEAK_POS | _LEAK_NEG | _LEAK_TROUBLE | {"yaar", "bro", "arre", "abe", "abey"}
    for i, sc in enumerate(result.get("neutral", [])):
        sid = sc.get("scene_id")
        dlg = sc.get("dialogue") or ""
        src_d = deva_scenes[i] if i < len(deva_scenes) else ""
        src_r = roman_scenes[i] if i < len(roman_scenes) else ""
        lines = _lines(dlg)
        spoken_src = g.spoken_lines(src_d)
        fid = sc.get("fidelity") or {}
        e = ev_by.get(sid, {})

        # language / format artefacts
        if DEVA.search(dlg):
            V.append(_v("devanagari-in-output", "error", sid, DEVA.search(dlg).group()))
        import unicodedata
        odd = {c for c in dlg if ord(c) > 127 and c not in ALLOWED_NON_ASCII and unicodedata.category(c)[0] in ("L", "M")}      # stray LETTERS ("sajī"); symbols such as the rupee sign are fine
        if odd:
            V.append(_v("non-ascii-artefact", "error", sid, "".join(sorted(odd))))
        if re.search(r"\*\s*\*|\(no dialogue\)|\[generation failed", dlg, re.I):
            V.append(_v("artefact-in-dialogue", "error", sid, re.search(r"\*\s*\*|\(no dialogue\)|\[generation failed", dlg, re.I).group()))
        if len(lines) > max_lines:
            V.append(_v("too-many-lines", "warn", sid, f"{len(lines)} lines"))
        seen = set()
        for spk, body, raw in lines:
            if spk is None:
                V.append(_v("unlabelled-line", "error", sid, raw))
                continue
            all_speakers[spk] = all_speakers.get(spk, 0) + 1
            first = (re.findall(r"[a-z]+", spk.lower()) or [""])[0]
            if spk.lower() in META_SPEAKERS or first in {"wait", "note", "correct", "actually", "hmm"}:
                V.append(_v("meta-speaker", "error", sid, spk))
            if not body:
                V.append(_v("empty-line", "error", sid, raw))
            if _is_english_meta(body):
                V.append(_v("english-prose-line", "error", sid, body))
            if g.narrates_own_gesture(body):
                V.append(_v("gesture-as-speech", "error", sid, body))
            if (spk, body) in seen:
                V.append(_v("duplicate-line", "error", sid, body))
            seen.add((spk, body))
            if len(body) > 500:
                V.append(_v("overlong-line", "warn", sid, len(body)))
            leaked = sorted((set(re.findall(r"[a-z]+", body.lower())) & sure_leaks) - set(re.findall(r"[a-z]+", (src_r or "").lower())))        # the author's own words are not leaks
            if leaked:
                V.append(_v("slang-leaked-into-neutral", "error", sid, leaked))
            # a person named in a vocative must be in the scene's own text (or the cast)
            m = re.match(r"^([A-Z][a-z]{2,})\b[,!]", body)
            if m and m.group(1).lower() not in COMMON_VOCATIVES and not (_in_source(m.group(1), src_r) or m.group(1).lower() in {k.lower().split()[0] for k in known | extras}):
                if not any(same_person(m.group(1), k) for k in known | extras):
                    V.append(_v("name-not-in-source", "warn", sid, m.group(1)))
            if spk not in known and spk not in extras and not any(same_person(spk, k) for k in known | extras) and not _in_source(spk, src_r):
                V.append(_v("speaker-not-grounded", "error", sid, spk))

        # speech the author wrote must survive
        if spoken_src and not lines and not fid.get("narration_only"):
            V.append(_v("spoken-words-lost", "error", sid, f"{len(spoken_src)} spoken clause(s) in the source, no dialogue"))
        elif spoken_src and len(lines) < max(1, len(spoken_src) // 2):
            V.append(_v("few-spoken-words-kept", "warn", sid, f"{len(spoken_src)} clauses, {len(lines)} lines"))
        if not lines and not fid.get("narration_only") and (e.get("characters") or spoken_src):
            V.append(_v("empty-scene-with-people", "warn", sid, "no dialogue although people or speech are in the source"))

        # provenance
        loc = e.get("location")
        if loc and not g.is_unknown_location(loc) and "location_evidence" in e:          # (runs older than the grounding work carry no evidence to check)
            if not g.span_supported(e.get("location_evidence", ""), src_d, src_r):
                V.append(_v("location-not-grounded", "error", sid, loc))
        if e.get("analysis_failed"):
            V.append(_v("analysis-failed", "warn", sid, "Agent 1 could not analyse this scene after all retries"))
        if any(DEVA.search(str(x)) for x in (e.get("characters") or []) + [loc or ""]):
            V.append(_v("devanagari-in-names", "error", sid, e.get("characters")))
        for issue in fid.get("final_issues", []):
            V.append(_v("critic:" + issue["type"], "warn", sid, issue.get("detail", "")))

    # one name per person across the script
    names = list(all_speakers)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if names[a].lower() != names[b].lower() and same_person(names[a], names[b]):
                V.append(_v("alias-duplicate", "warn", None, f"{names[a]} / {names[b]}"))
    for s in persona.get("secrets", []):
        if not g.shows_concealment(s.get("evidence", s.get("what", ""))) and "evidence" in s:
            V.append(_v("secret-without-concealment", "error", None, s.get("what")))
    return V


# -------------------------------------------------------------------------------------------------------------------------- slang stage
URBAN_FILLERS = {"bro", "boss", "chill", "abey", "abe"}


def check_slang(neutral_script: List[dict], slang_script: List[dict], profiles: Optional[dict] = None, setting: Optional[dict] = None):
    from agents import edit_gate as eg
    from evaluation import edit_audit
    from evaluation.compare import _parse_turns
    from agents.name_utils import match_profile
    V: List[dict] = []
    profiles = profiles or {}
    trad = bool((setting or {}).get("traditional"))
    fem = set(eg.FEMALE_NAMES) | {t for n, p in profiles.items() if p.get("gender") == "female" for t in re.findall(r"[a-z]{3,}", n.lower())}
    nmap = {s["scene_id"]: s for s in neutral_script}
    for sc in slang_script:
        sid = sc["scene_id"]
        n_turns = _parse_turns((nmap.get(sid) or {}).get("dialogue", ""))
        s_turns = _parse_turns(sc.get("dialogue", ""))
        decs = sc.get("decisions") or []
        for i, st in enumerate(s_turns):
            nt = n_turns[i]["text"] if i < len(n_turns) else ""
            if st["text"].strip() == nt.strip():
                continue
            slang = (decs[i].get("slang") if i < len(decs) else None) or edit_audit._infer_slang(edit_audit.diff_edit(nt, st["text"])["added"])
            ed = st["text"]
            a = edit_audit.audit_edit(nt, ed, slang)
            if not a["ok"]:
                V.append(_v("edit-fails-audit", "error", sid, f"{[f['code'] for f in a['flags'] if f['blocking']]} | {ed}"))
            low = ed.lower()
            if trad:
                bad = [w for w, r in eg.REGISTER.items() if r == "urban_youth" and re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low) and w not in nt.lower()]
                bad += [w for w in URBAN_FILLERS if re.search(r"\b" + w + r"\b", low) and w not in nt.lower()]
                if bad:
                    V.append(_v("urban-slang-in-traditional-story", "error", sid, f"{bad} | {ed}"))
            names_in = {w.lower() for w in re.findall(r"[A-Za-z]+", ed)}
            if names_in & fem and re.search(r"\b(bro|bhai)\b", low) and not re.search(r"\b(bro|bhai)\b", nt.lower()):
                V.append(_v("male-address-for-a-girl", "error", sid, ed))
            prof = match_profile(profiles, st["speaker"]) if profiles else {}
            gender = {"male": "m", "female": "f"}.get(prof.get("gender"))
            if slang and not eg.person_agreement_ok(ed, slang, gender):
                V.append(_v("gender-disagreement", "error", sid, ed))
            if slang and not eg.experiencer_frame_ok(ed, slang):
                V.append(_v("experiencer-frame", "error", sid, ed))
            if slang and not eg.copula_ok(ed, slang):
                V.append(_v("predicate-idiom-without-copula", "error", sid, ed))
            if len(re.findall(r"\b(yaar|bro|bhai|arre|boss)\b", low)) > 1 + len(re.findall(r"\b(yaar|bro|bhai|arre|boss)\b", nt.lower())):
                V.append(_v("stacked-fillers", "warn", sid, ed))
            if re.search(r",\s*,|\s,", ed):
                V.append(_v("punctuation-glitch", "error", sid, ed))
    return V


def summarize(violations: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in violations:
        k = f"{v['level']}:{v['code']}"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
