"""Pass 1 must not contain the model's English self-notes (seen once in the 7-story set: fin_t1 scene_02)."""
import glob
import json
import os

from agents.agent3_synthesizer import strip_meta

LEAK = ("Arav: Main itihaas ki pariyojana ke liye kuch saamagri dhoondhne aaya hoon. Yahan bahut accha laga.\n"
        "Narrator's listener is not a character in the story, so the scene's sole speaker is Arav, who addresses himself as the listener.\n"
        "Arav: Yeh alamari toh dhool se bhari hai.")


def test_unprefixed_english_note_is_dropped():
    out = strip_meta(LEAK)
    assert "listener" not in out and "sole speaker" not in out
    assert out.count("Arav:") == 2 and "Yahan bahut accha laga." in out


def test_note_inside_a_spoken_line_is_cut_but_the_dialogue_stays():
    out = strip_meta("Arav: Yahan bahut accha laga. Narrator's listener is not a character in the story, so the sole speaker is Arav.\nRiya: Chalo.")
    assert out == "Arav: Yahan bahut accha laga.\nRiya: Chalo."


def test_note_prefix_forms():
    assert strip_meta("Arav: Chalo.\n(Note: I kept the dialogue short.)\nRiya: Haan.") == "Arav: Chalo.\nRiya: Haan."


def test_hinglish_with_english_loanwords_is_untouched():
    t = ("Riya: Science fair ka announcement hua hai, hum is baar robot banayenge.\n"
         "Aryan: Mujhe lagta hai ki the main problem battery hai, lekin team ready hai.\n"
         "Kabir: Yeh project deadline se pehle submit karna hai.")
    assert strip_meta(t) == t


def test_a_scene_is_never_emptied():
    only_note = "The sole speaker is the listener of the story, which is not a character."
    assert strip_meta(only_note) == only_note


def test_no_false_positive_on_any_saved_neutral_script():
    """Every neutral line of every saved trace is kept; the only change allowed is the one known leak."""
    changed = []
    for f in glob.glob(os.path.join("outputs", "*trace*.json")):
        try:
            t = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for sc in (t.get("neutral_script") or []):
            d = sc.get("dialogue", "")
            if isinstance(d, str) and strip_meta(d) != d:
                changed.append((os.path.basename(f), sc.get("scene_id")))
    assert all(f.startswith("fin_t1_trace") for f, _ in changed) or not changed, changed
