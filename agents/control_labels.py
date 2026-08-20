# agents/control_labels.py
# Canonical control-token label space for Gate 2 + editor conditioning.
# Rare raw tokens are merged into a frequent sibling to keep training stable.

RAW_TO_CANON = {
    "ASSERT_POSITIVE": "ASSERT_POSITIVE",
    "POSITIVE_EVAL":   "ASSERT_POSITIVE",
    "ASSERT_HAPPY":    "ASSERT_HAPPY",
    "ASSERT_EXCITED":  "ASSERT_EXCITED",
    "ASSERT_SURPRISE": "ASSERT_EXCITED",
    "ASSERT_NEGATIVE": "ASSERT_NEGATIVE",
    "ASSERT_FRUSTRATION": "ASSERT_FRUSTRATION",
    "ASSERT_ANGER":    "ASSERT_FRUSTRATION",
    "REQUEST_FRUSTRATION": "ASSERT_FRUSTRATION",
    "ASSERT_STRESS":   "ASSERT_STRESS",
    "EXPRESS_STRESS":  "ASSERT_STRESS",
    "ASSERT_ANXIETY":  "ASSERT_ANXIETY",
    "ASSERT_FEAR":     "ASSERT_ANXIETY",
    "ASSERT_SADNESS":  "ASSERT_SADNESS",
    "EXPRESS_SADNESS": "ASSERT_SADNESS",
    "ASSERT_NEUTRAL":  "ASSERT_NEUTRAL",
    "INFO_STATEMENT":  "ASSERT_NEUTRAL",
    "ASSERT_CONFUSION":"ASSERT_CONFUSION",
    "ASSERT_WARNING":  "ASSERT_WARNING",
    "REQUEST_NEUTRAL": "REQUEST_NEUTRAL",
    "REQUEST_URGENCY": "REQUEST_URGENCY",
}

CANON_LABELS = sorted(set(RAW_TO_CANON.values()))   # ~13 stable classes

# Which control tokens permit slang injection at all (Gate 2 eligibility).
# Neutral/info/formal-request intents are blocked so slang stays pragmatically motivated.
SLANG_PERMITTED = {
    "ASSERT_POSITIVE", "ASSERT_HAPPY", "ASSERT_EXCITED",
    "ASSERT_NEGATIVE", "ASSERT_FRUSTRATION", "ASSERT_STRESS",
    "ASSERT_ANXIETY", "ASSERT_SADNESS", "ASSERT_CONFUSION",
    "ASSERT_WARNING", "REQUEST_URGENCY",
}
# Blocked (kept neutral): ASSERT_NEUTRAL, REQUEST_NEUTRAL

def canon(token: str) -> str:
    return RAW_TO_CANON.get(str(token).strip().strip("[]").upper().replace(" ", "_"),
                            "ASSERT_NEUTRAL")