"""v7 data pipeline: the situation rule, the prompt, the safety filter and the resumable teacher loop (with a stub model, no spend)."""
import json

import pytest

from data_prep import v7_filter, v7_format as F, v7_teacher as T


def ctx(line="Yeh idea bahut accha hai.", sp_role="peer", li_role="peer", register="casual", traditional=False, emotion="happy", intensity=2, max_edits=2, **kw):
    c = {"id": "c1", "story": "s", "scene": "scene_01", "setting": {"era": "contemporary", "milieu": "campus", "formality": "casual", "traditional": traditional},
         "speaker": {"name": "Karan", "role": sp_role, "gender": "male", "emotion": emotion, "intensity": intensity},
         "listener": {"name": "Neha", "role": li_role, "gender": "female", "register": register}, "gist": "friends plan a film", "previous": "Neha: Pehle kahani sunao.",
         "line": line, "hints": ["solid", "kadak"], "max_edits": max_edits}
    c.update(kw)
    c["allows_edit"] = F.context_allows_edit(c)
    return c


@pytest.mark.parametrize("kw,allowed", [({}, True), ({"sp_role": "elder"}, False), ({"li_role": "authority"}, False), ({"register": "respectful"}, False),
                                        ({"traditional": True}, False), ({"emotion": "sad", "intensity": 3}, False), ({"emotion": "sad", "intensity": 1}, True),
                                        ({"line": "Sorry yaar, meri galti thi."}, False), ({"line": "Aap bahut accha kaam karte hain."}, False)])
def test_the_situation_rule(kw, allowed):
    assert ctx(**kw)["allows_edit"] is allowed


def test_prompt_and_chat_carry_the_whole_situation_and_the_hints():
    u = F.render_user(ctx())
    for part in ("SETTING: modern campus, casual", "SPEAKER: Karan (peer, male, feels happy 2)", "LISTENER: Neha (peer, female, register casual)", "SCENE: friends plan a film",
                 "PREVIOUS LINE:", "MAX EDITS: 2", "HINTS (optional): solid, kadak", "LINE: Yeh idea bahut accha hai."):
        assert part in u
    assert "traditional" in F.render_user(ctx(traditional=True)) and "HINTS" not in F.render_user(ctx(hints=[]))
    chat = F.make_chat(ctx(), "Yeh idea ekdum solid hai.")["messages"]
    assert [m["role"] for m in chat] == ["system", "user", "assistant"] and chat[2]["content"] == "Yeh idea ekdum solid hai."
    assert F.parse_output('"Yeh idea ekdum solid hai."\nextra', "x y z") == "Yeh idea ekdum solid hai." and F.parse_output("", "Yeh hai.") == "Yeh hai."


@pytest.mark.parametrize("edited,ok", [
    ("Yeh idea bahut accha hai.", True),                                  # unchanged: a valid keep-as-is
    ("Yeh idea ekdum solid hai.", True),
    ("Yeh idea bahut kadak hai, yaar.", True),
    ("Yeh idea bahut accha hai, yaar bro arre.", False),                  # stacked fillers
    ("Yeh idea bahut accha hai, yaar.", False),                           # a single misplaced filler: the runtime gate still rejects it
    ("Yeh idea ekdum solid hai!!! Sach mein kamaal hai, ekdum next level wala plan hai bhai.", False),   # length / too many edits
    ("Yeh idea nahi solid hai.", False),                                  # negation added
    ("यह idea solid hai.", False),                                          # Devanagari
    ("Yeh idea bahut chutiya hai.", False),                               # profanity
])
def test_filter_on_a_plain_line(edited, ok):
    assert v7_filter.check(ctx(), edited)["ok"] is ok


def test_filter_never_accepts_an_edit_where_none_is_allowed_but_accepts_keeping_the_line():
    c = ctx(sp_role="elder")
    assert v7_filter.check(c, "Yeh idea ekdum solid hai.")["reasons"][0] == "edit-where-none-allowed"
    assert v7_filter.check(c, c["line"]) == {"ok": True, "changed": False, "reasons": [], "flags": []}


def test_filter_keeps_gender_forms_numbers_names_and_question_form():
    c = ctx(line="Karan, main 5 baje aa rahi hoon, tum kahan ho?")
    assert not v7_filter.check(c, "Karan, main 5 baje aa raha hoon, tum kahan ho?")["ok"]                 # rahi -> raha
    assert not v7_filter.check(c, "Karan, main 6 baje aa rahi hoon, tum kahan ho?")["ok"]                 # number changed
    assert not v7_filter.check(c, "Karan, main 5 baje aa rahi hoon, tum kahan ho.")["ok"]                 # question became a statement
    assert v7_filter.is_filler_only(ctx(), "Yeh idea bahut accha hai yaar.") and not v7_filter.is_filler_only(ctx(), "Yeh idea ekdum solid hai.")


def test_teacher_loop_is_resumable_budgeted_and_only_sends_lines_that_allow_slang(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "D", str(tmp_path))
    rows = [ctx(line=f"Yeh plan bahut accha hai number {i}.", id=f"c{i}") for i in range(5)] + [dict(ctx(sp_role="elder"), id="cx")]
    (tmp_path / "contexts.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    seen = []

    def stub(msgs):
        ids = [l.split("id: ")[1] for l in msgs[1]["content"].split("\n") if l.startswith("### id: ")]
        seen.append(ids)
        return json.dumps([{"id": i, "edited": "Yeh plan ekdum solid hai.", "edits": ["bahut accha -> ekdum solid"]} for i in ids])

    T.run(10, 5.0, "m", batch_size=2, caller=stub)
    assert "cx" not in sum(seen, []) and len(sum(seen, [])) == 5
    raw = [json.loads(x) for x in (tmp_path / "teacher_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(raw) == 5 and all(1 <= r["max_edits"] <= 3 for r in raw)
    T.run(10, 5.0, "m", batch_size=2, caller=stub)                                  # nothing new to do
    assert len((tmp_path / "teacher_raw.jsonl").read_text(encoding="utf-8").splitlines()) == 5


def test_judge_step_scores_only_filter_survivors_and_accepts_by_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "D", str(tmp_path))
    rows = [ctx(id=f"c{i}") for i in range(3)]
    (tmp_path / "contexts.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    raw = [{"id": "c0", "max_edits": 2, "edited": "Yeh idea ekdum solid hai."}, {"id": "c1", "max_edits": 2, "edited": "Yeh idea nahi solid hai."},       # c1 fails the filter
           {"id": "c2", "max_edits": 2, "edited": "Yeh idea bahut kadak hai."}]
    (tmp_path / "teacher_raw.jsonl").write_text("\n".join(json.dumps(r) for r in raw), encoding="utf-8")
    asked = []

    def stub(msgs):
        ids = [l.split("id: ")[1] for l in msgs[1]["content"].split("\n") if l.startswith("### id: ")]
        asked.extend(ids)
        return json.dumps([{"id": i, "natural": 5 if i == "c0" else 3, "meaning_kept": True, "register_ok": True} for i in ids])

    T.judge(5.0, "m", caller=stub)
    assert asked == ["c0", "c2"] and T.accepted() == {"c0"}                       # c2 is only 3/5: odd but understandable is not good enough
