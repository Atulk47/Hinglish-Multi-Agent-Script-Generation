"""
data_prep/slang_bank_v3.py
The CURATED slang bank used for BOTH the v3 training data and (via slang_bank_v3.csv) the runtime
shortlist. The old runtime lexicon (final_dataset.csv, 427 raw strings) contained fragments and
odd entries ("gajab scene hai", "sir pe", "scam", "be", "Oye", case duplicates) that produced
nonsense edits however good the model was. Only expressions that insert naturally into a
sentence are kept here.

tone: P positive, N negative, F neutral filler.
controls: dataset control tokens the runtime pool uses (see GATE_INTENT_TO_CONTROLS).

  python -m data_prep.slang_bank_v3      # (re)writes data/felix_dataset/slang_bank_v3.csv
"""
import csv, os

P_EXC = ("ASSERT_POSITIVE", "ASSERT_EXCITED")
P_HAP = ("ASSERT_POSITIVE", "ASSERT_HAPPY")
P_GEN = ("ASSERT_POSITIVE",)
N_FRU = ("ASSERT_FRUSTRATION", "ASSERT_NEGATIVE")
N_STR = ("ASSERT_STRESS", "ASSERT_ANXIETY")
N_SAD = ("ASSERT_SADNESS", "ASSERT_NEGATIVE")
N_GEN = ("ASSERT_NEGATIVE",)

# slang -> (tone, controls, function)
BANK = {
    # ── positive ─────────────────────────────────────────────
    "mast": ("P", P_GEN, "slang_eval"), "kadak": ("P", P_GEN, "slang_eval"),
    "solid": ("P", P_GEN, "slang_eval"), "lajawaab": ("P", P_GEN, "slang_eval"),
    "jhakaas": ("P", P_EXC, "slang_eval"), "badiya": ("P", P_GEN, "slang_eval"),
    "zabardast": ("P", P_GEN, "slang_eval"), "gazab": ("P", P_EXC, "slang_eval"),
    "faadu": ("P", P_EXC, "slang_eval"), 
    "dhamaal": ("P", P_EXC, "slang_eval"), 
    "top class": ("P", P_GEN, "slang_eval"), "fire": ("P", P_EXC, "slang_eval"),
    "on point": ("P", P_GEN, "slang_eval"), "paisa vasool": ("P", P_HAP, "idiom"),
    "kya baat": ("P", P_EXC, "idiom"),
    "sahi hai": ("P", P_GEN, "idiom"), 
    "dhasu": ("P", P_EXC, "slang_eval"), 
    "ek number": ("P", P_GEN, "slang_eval"), 
    
    "jalwa": ("P", P_EXC, "idiom"), 
    "dil jeet liya": ("P", P_HAP, "idiom"), 
    "full masti": ("P", P_HAP, "idiom"), "chaka-chak": ("P", P_GEN, "slang_eval"),
    "phaad diya": ("P", P_EXC, "idiom"), 
    
    "banger": ("P", P_EXC, "slang_eval"), 
    "lit": ("P", P_EXC, "slang_eval"), 
    "makkhan": ("P", P_GEN, "slang_eval"), 
    
    "kamaal": ("P", P_GEN, "slang_eval"), "sorted": ("P", P_GEN, "slang_eval"),
    
    "chutki mein": ("P", P_GEN, "idiom"), "number one": ("P", P_GEN, "slang_eval"),
    
    "sab set hai": ("P", P_GEN, "idiom"),
    
    
    # ── negative ─────────────────────────────────────────────
    "bakwaas": ("N", N_FRU, "slang_eval"), "bekaar": ("N", N_FRU, "slang_eval"),
    "ghatiya": ("N", N_FRU, "slang_eval"),
    "flop": ("N", N_FRU, "slang_eval"), 
    "jhand": ("N", N_FRU, "slang_eval"),
    "kya bakwaas hai": ("N", N_FRU, "idiom"),
    "kachra": ("N", N_FRU, "slang_eval"),
    
    
    "panga": ("N", N_FRU, "slang_eval"),
    "dimaag fry": ("N", N_STR, "idiom"), 
    
    "full tension": ("N", N_STR, "intensifier"), 
    "jaan nikal gayi": ("N", N_STR, "idiom"), "dil baith raha hai": ("N", N_STR, "idiom"),
    "fatti padi hai": ("N", N_STR, "idiom"),
    
    
    
    
    "waat lag gayi": ("N", N_GEN, "idiom"), 
    
    "lafda": ("N", N_FRU, "slang_eval"), "gadbad": ("N", N_GEN, "slang_eval"),
    "ghapla": ("N", N_GEN, "slang_eval"),
    "ulta pad gaya": ("N", N_GEN, "idiom"), 
    "locha": ("N", N_GEN, "slang_eval"), "jhol": ("N", N_GEN, "slang_eval"),
    # ── neutral fillers ──────────────────────────────────────
    "yaar": ("F", ("CASUAL_ASSERTION",), "filler"), "bhai": ("F", ("CASUAL_ASSERTION",), "filler"),
    "arre": ("F", ("CASUAL_ASSERTION",), "filler"), "abey": ("F", ("CASUAL_ASSERTION",), "filler"),
    "boss": ("F", ("CASUAL_ASSERTION",), "filler"), "chill": ("F", ("CASUAL_ASSERTION",), "filler"),
    "kasam se": ("F", ("CASUAL_ASSERTION",), "filler"), "bro": ("F", ("CASUAL_ASSERTION",), "filler"),
}

TONE_NAME = {"P": "positive", "N": "negative", "F": "neutral"}


def write_csv(path="data/felix_dataset/slang_bank_v3.csv"):
    """Runtime lexicon in the schema Agent 4 already reads (control, function, slangs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for s, (tone, controls, fn) in BANK.items():
        for c in controls:
            rows.append({"control": c, "function": fn, "slangs": s})
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["control", "function", "slangs"])
        w.writeheader(); w.writerows(rows)
    return path, len(BANK), len(rows)


if __name__ == "__main__":
    p, n, r = write_csv()
    from collections import Counter
    print(f"wrote {p}: {n} slangs, {r} rows;", dict(Counter(v[0] for v in BANK.values())))
