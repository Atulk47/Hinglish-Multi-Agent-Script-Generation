import os
import re
import time


class GroqIntentClassifier:
    """
    Groq-backed drop-in replacement for IntentClassifier (same .classify() API).

    Used for the "Groq-first" pipeline run: no 1.6GB BART-MNLI download, no slow
    CPU zero-shot inference, and it keeps the exact label space the Three-Gate
    controller expects. Results are cached per-utterance to avoid repeat calls.
    """

    CANDIDATE_LABELS = [
        "REQUEST_URGENCY",
        "ASSERT_FRUSTRATION",
        "ASSERT_POSITIVE",
        "EXPRESS_STRESS",
        "CASUAL_ASSERTION",
        "INFORM_NEUTRAL",
        "QUESTION_FORMAL",
    ]

    _PROMPT = """Classify the pragmatic intent of this romanized Hindi/Hinglish line.
Answer with EXACTLY ONE label from this list, nothing else:
REQUEST_URGENCY, ASSERT_FRUSTRATION, ASSERT_POSITIVE, EXPRESS_STRESS, CASUAL_ASSERTION, INFORM_NEUTRAL, QUESTION_FORMAL

Line: {text}
Label:"""

    def __init__(self):
        from groq import Groq
        from dotenv import load_dotenv
        load_dotenv()
        import config
        self._config = config
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self._cache = {}
        print(f"Intent classifier: Groq ({config.GROQ_MODEL}) backend.")

    def classify(self, text: str) -> str:
        if not text or not text.strip():
            return "INFORM_NEUTRAL"
        key = text.strip().lower()
        if key in self._cache:
            return self._cache[key]

        label = "CASUAL_ASSERTION"
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self._config.GROQ_MODEL,
                    messages=[{"role": "user", "content": self._PROMPT.format(text=text)}],
                    temperature=0.0,
                    max_tokens=80,
                    reasoning_effort=getattr(self._config, "GROQ_REASONING_EFFORT", "low"),
                )
                raw = (resp.choices[0].message.content or "").upper()
                m = re.search(r"[A-Z_]+", raw)
                cand = m.group(0) if m else ""
                label = cand if cand in self.CANDIDATE_LABELS else "CASUAL_ASSERTION"
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    intent classify failed ({e}) -> CASUAL_ASSERTION")
        self._cache[key] = label
        return label


from transformers import pipeline


class IntentClassifier:
    """
    Zero-shot intent classifier using facebook/bart-large-mnli.
    Downloads ~1.6GB on first run. Free, no auth required.
    Runs on CPU — ~1-2 seconds per call, acceptable for our use case.
    """

    CANDIDATE_LABELS = [
        "REQUEST_URGENCY",      # "jaldi karo", "abhi batao"
        "ASSERT_FRUSTRATION",   # "pagal hai kya", "bore ho gaya"
        "ASSERT_POSITIVE",      # "bahut acha hai", "maza aa gaya"
        "EXPRESS_STRESS",       # "tension mat lo", "pareshan hun"
        "CASUAL_ASSERTION",     # general casual statements
        "INFORM_NEUTRAL",       # neutral information delivery
        "QUESTION_FORMAL",      # formal questions
    ]

    def __init__(self):
        print("Loading BART-MNLI intent classifier (~1.6GB on first run)...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1   # CPU
        )
        print("Intent classifier loaded.")

    def classify(self, text: str) -> str:
        """Returns the top intent label for the given text."""
        if not text or not text.strip():
            return "INFORM_NEUTRAL"

        result = self.classifier(
            text,
            self.CANDIDATE_LABELS,
            multi_label=False
        )
        return result["labels"][0]

    def classify_with_scores(self, text: str) -> dict:
        """Returns all labels with their scores — useful for debugging."""
        if not text or not text.strip():
            return {"labels": self.CANDIDATE_LABELS,
                    "scores": [0.0] * len(self.CANDIDATE_LABELS)}
        return self.classifier(text, self.CANDIDATE_LABELS, multi_label=False)


if __name__ == "__main__":
    clf = IntentClassifier()

    test_utterances = [
        "yaar jaldi karo please",
        "main bahut pareshan hun aaj kal",
        "maza aa gaya sach mein",
        "library mein bahut si kitabein hain",
        "aap kya keh rahe hain sir",
    ]

    for utt in test_utterances:
        label = clf.classify(utt)
        print(f"'{utt}' -> {label}")