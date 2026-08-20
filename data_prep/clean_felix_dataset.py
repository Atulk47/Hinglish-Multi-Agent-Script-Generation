# data_prep/clean_felix_dataset.py
"""
Clean and normalize final_augmented_master_6000.csv into:
  - slang_pairs_clean.csv   (training-ready)
  - slang_lexicon.json      (control_token -> [slang phrases], for fallback + density)
Run from repo root:  python -m data_prep.clean_felix_dataset
"""
import re
import json
import pandas as pd
from collections import defaultdict, Counter
from rapidfuzz import fuzz
import config

RAW   = config.FELIX_DATASET_RAW
CLEAN = config.FELIX_DATASET_CLEAN
LEX   = config.SLANG_LEXICON_PATH

RAW_COLS = {
    "neutral hinglish sentence": "neutral",
    "context": "context",
    "social_context": "social_context",
    "Possible Slangs": "slangs_raw",
    "Slang Location": "location_raw",
    "Multiple positions allowed?": "multi_allowed",
    "Control tokens": "control_raw",
    "function": "function_raw",
    "output sentence": "output",
}

VALID_POS = {"START", "MIDDLE", "END"}


def split_slangs(s: str) -> list:
    """Split on , and / ; drop parenthetical glosses; trim; dedupe order-preserving."""
    parts = re.split(r"[,/]", str(s))
    out, seen = [], set()
    for p in parts:
        p = re.sub(r"\(.*?\)", "", p).strip()      # drop "(pro)" style glosses
        p = re.sub(r"\s+", " ", p)
        if p and p.lower() not in seen:
            out.append(p)
            seen.add(p.lower())
    return out


def norm_positions(s: str) -> list:
    toks = re.split(r"[,/]", str(s))
    out = []
    for t in toks:
        t = t.strip().upper()
        if t in VALID_POS and t not in out:
            out.append(t)
    return out or ["END"]   # default safety


def norm_control(s: str) -> str:
    return str(s).strip().strip("[]").upper().replace(" ", "_")


def norm_function(s: str) -> str:
    base = str(s).split("+")[0].strip().lower()
    base = re.sub(r"\s+", "_", base)
    return base or "filler"


def slang_present(slangs: list, output: str) -> bool:
    """True if at least one slang phrase (or its stem) is in output.

    Hindi idioms are stored in citation form (`Ghasna`, `Patti padhana`) but
    appear inflected in the output (`ghas raha hai`, `patti padha raha hai`).
    A whole-phrase exact/fuzzy match misses these, so we also match on word
    STEMS (first 4 chars). This recovers ~13 idiom rows that a strict check
    wrongly quarantines, while still rejecting genuine no-slang rows.
    """
    out = str(output).lower()
    out_words = re.sub(r"[^\w\s]", " ", out).split()
    for sl in slangs:
        s = sl.lower()
        if s in out:                                   # exact phrase
            return True
        if fuzz.partial_ratio(s, out) >= 80:           # minor spelling drift
            return True
        for w in re.sub(r"[^\w\s]", " ", s).split():   # per-word stem match
            stem = w[:4] if len(w) >= 5 else w
            if stem and any(ow.startswith(stem) for ow in out_words):
                return True
    return False


def main():
    df = pd.read_csv(RAW)
    df = df.rename(columns=RAW_COLS)

    df["slangs"]    = df["slangs_raw"].apply(split_slangs)
    df["positions"] = df["location_raw"].apply(norm_positions)
    df["control"]   = df["control_raw"].apply(norm_control)
    df["function"]  = df["function_raw"].apply(norm_function)
    df["multi_allowed"] = df["multi_allowed"].astype(str).str.strip().str.lower().eq("yes")
    df["neutral"]   = df["neutral"].astype(str).str.strip()
    df["output"]    = df["output"].astype(str).str.strip()

    # --- quality filters -------------------------------------------------
    before = len(df)
    df["slang_ok"] = df.apply(lambda r: slang_present(r["slangs"], r["output"]), axis=1)
    df["changed"]  = df["neutral"].str.lower() != df["output"].str.lower()
    df["nonempty"] = (df["neutral"].str.len() > 0) & (df["output"].str.len() > 0)

    quarantine = df[~(df["slang_ok"] & df["changed"] & df["nonempty"])].copy()
    df_clean   = df[df["slang_ok"] & df["changed"] & df["nonempty"]].copy()

    # --- dedup: KEEP label variety -------------------------------------
    # The dataset has only ~1082 unique neutral sentences, each augmented many
    # times (mean 4.1 distinct outputs each). Deduping on (neutral, output)
    # alone collapses ~1539 rows AND arbitrarily keeps whichever control token
    # came first, throwing away legitimate label variety. Audit shows the SAME
    # edit (e.g. insert "chal") is validly labeled ASSERT_SADNESS / NEGATIVE /
    # FRUSTRATION — never crossing into positive. Keeping that variety makes the
    # editor robust to a noisy Gate-2 prediction, so we dedup on
    # (neutral, output, control) and only drop truly redundant rows.
    df_clean = df_clean.drop_duplicates(
        subset=["neutral", "output", "control"]).reset_index(drop=True)

    print(f"Raw rows:        {before}")
    print(f"Quarantined:     {len(quarantine)}  (saved to quarantine.csv for manual review)")
    print(f"Clean rows:      {len(df_clean)}")

    # --- save clean csv --------------------------------------------------
    out_cols = ["neutral", "output", "control", "function",
                "positions", "multi_allowed", "social_context", "context"]
    save = df_clean[out_cols].copy()
    save["slangs"]    = df_clean["slangs"].apply(lambda x: "|".join(x))
    save["positions"] = save["positions"].apply(lambda x: "|".join(x))
    save.to_csv(CLEAN, index=False)
    quarantine.to_csv("data/felix_dataset/quarantine.csv", index=False)

    # --- build lexicon: control_token -> {slang: count} ------------------
    lex = defaultdict(Counter)
    for _, r in df_clean.iterrows():
        for sl in r["slangs"]:
            lex[r["control"]][sl.lower()] += 1
    # also a function-keyed lexicon as a secondary fallback
    func_lex = defaultdict(Counter)
    for _, r in df_clean.iterrows():
        for sl in r["slangs"]:
            func_lex[r["function"]][sl.lower()] += 1

    lexicon = {
        "by_control":  {k: dict(v) for k, v in lex.items()},
        "by_function": {k: dict(v) for k, v in func_lex.items()},
    }
    with open(LEX, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)

    print(f"\nLexicon written to {LEX}")
    print("Control tokens covered:", len(lexicon["by_control"]))
    for c in sorted(lexicon["by_control"], key=lambda k: -sum(lexicon['by_control'][k].values()))[:5]:
        top = sorted(lexicon["by_control"][c].items(), key=lambda x: -x[1])[:6]
        print(f"  {c:20s} -> {[w for w, _ in top]}")


if __name__ == "__main__":
    main()