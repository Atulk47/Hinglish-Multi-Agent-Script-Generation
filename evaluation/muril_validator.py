"""
evaluation/muril_validator.py
Layer 5 — MuRIL Validation agent.

Checks that Pass-2 (slang) dialogue stays semantically faithful to Pass-1
(neutral) dialogue, and that the scene's key events remain covered. Emits an
accept/rewrite decision that the orchestrator's feedback loop consumes.

Only needs google/muril-base-cased (already cached locally).
"""
from __future__ import annotations
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

import config

_STOPWORDS = {
    "the", "a", "an", "is", "was", "in", "of", "and", "to", "ka", "ki", "ke",
    "ko", "se", "par", "hai", "tha", "thi", "ne", "aur", "woh", "yeh", "ek",
    "mein", "hai.", "kar", "raha", "rahi", "rahe",
}


class MuRILValidator:
    def __init__(self, similarity_threshold: float = None,
                 event_recall_threshold: float = None, shared_muril=None):
        self.sim_threshold = (similarity_threshold if similarity_threshold is not None
                              else config.MURIL_SIMILARITY_THRESHOLD)
        self.event_threshold = (event_recall_threshold if event_recall_threshold is not None
                                else getattr(config, "EVENT_COVERAGE_RECALL_THRESHOLD",
                                             getattr(config, "EVENT_COVERAGE_F1_THRESHOLD", 0.5)))
        if shared_muril is not None:
            self.tokenizer, self.model = shared_muril
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.MURIL_MODEL_PATH)
            self.model = AutoModel.from_pretrained(config.MURIL_MODEL_PATH)
        self.model.eval()
        print("MuRILValidator initialized (google/muril-base-cased).")

    def _embed(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            text = "."
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=256, padding=True)
        with torch.no_grad():
            out = self.model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return (summed / counts)[0].numpy()

    def compute_similarity(self, text1: str, text2: str) -> float:
        a, b = self._embed(text1), self._embed(text2)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _event_coverage(self, dialogue: str, events: list) -> dict:
        dl = dialogue.lower()
        covered = 0
        for ev in events:
            words = {w for w in re.findall(r"[a-z]+", ev.lower()) if w not in _STOPWORDS}
            if words and any(w in dl for w in words):
                covered += 1
        total = max(len(events), 1)
        return {"recall": covered / total, "covered": covered, "total": len(events)}

    def validate_scene(self, pass1_dialogue: str, pass2_dialogue: str,
                       key_events: list) -> dict:
        sim = self.compute_similarity(pass1_dialogue, pass2_dialogue)
        cov = self._event_coverage(pass2_dialogue, key_events or [])
        passed = (sim >= self.sim_threshold) and (cov["recall"] >= self.event_threshold)
        return {
            "passed": passed,
            "similarity": round(sim, 4),
            "event_recall": round(cov["recall"], 4),
            "event_coverage": cov,
            "action": "accept" if passed else "rewrite",
        }


if __name__ == "__main__":
    v = MuRILValidator()
    p1 = "ARAV: Mujhe ek purani diary mili hai library mein."
    p2 = "ARAV: Yaar mujhe ek purani diary mili hai library mein!"
    print(v.validate_scene(p1, p2, ["Arav finds a diary", "in the library"]))
