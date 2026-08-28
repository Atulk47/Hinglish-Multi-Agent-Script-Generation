import re
from typing import List


class ProseChunker:
    """Split a romanized story into scene-level chunks.

    Two strategies, in order of preference:
      1. Blank-line paragraphs — if the author separated scenes with blank lines,
         trust that (one paragraph = one scene).
      2. Sentence grouping — if the text is one continuous block (no blank lines),
         split it into sentences and group them into scenes, starting a NEW scene
         at a narrative boundary (a time/place shift) or once a scene grows past a
         size cap. This is what lets the pipeline handle text that has no '\\n\\n'.
    """

    # In-scene signals (kept for downstream flags)
    LOCATION_CUES = ['ghar', 'school', 'bazaar', 'bahar', 'andar', 'office',
                      'kamra', 'kamre', 'gali', 'station', 'park', 'chhat',
                      'college', 'hall', 'manch', 'daftar', 'dukaan']
    TEMPORAL_CUES = ['subah', 'sham', 'raat', 'dopahar', 'agli baar', 'phir',
                      'kal', 'aaj', 'parso', 'thodi der baad']

    # Phrases that, when they OPEN a sentence, signal the start of a new scene
    # (a jump in time or place). Matched against the sentence's first few words.
    BOUNDARY_CUES = [
        'agle din', 'agle', 'us din', 'ek din', 'kuch din', 'kai din',
        'agli subah', 'agli', 'usi raat', 'us raat', 'uske baad', 'iske baad',
        'baad mein', 'subah', 'sham', 'shaam', 'raat', 'dopahar', 'phir',
        'akhirkar', 'antdt', 'antatah', 'jab', 'us shaam', 'is shaam',
    ]

    MAX_SENTENCES = 4    # cap scene length in the sentence-grouping fallback
    MAX_WORDS = 60

    def chunk(self, romanized_text: str) -> List[dict]:
        paragraphs = [p.strip() for p in romanized_text.split('\n\n') if p.strip()]

        # Fallback: no usable blank-line structure -> group sentences into scenes.
        if len(paragraphs) <= 1:
            block = paragraphs[0] if paragraphs else romanized_text.strip()
            grouped = self._group_sentences(block)
            paragraphs = grouped if grouped else ([block] if block else [])

        scenes = []
        for i, para in enumerate(paragraphs):
            low = para.lower()
            scenes.append({
                "scene_id": f"scene_{i+1:02d}",
                "text": para,
                "has_location_cue": any(cue in low for cue in self.LOCATION_CUES),
                "has_temporal_cue": any(cue in low for cue in self.TEMPORAL_CUES),
            })
        return scenes

    # ── sentence-grouping fallback ────────────────────────────────────────
    def _split_sentences(self, text: str) -> List[str]:
        """Split on sentence-final punctuation (., ?, !, Hindi danda) followed by
        whitespace. Robust enough for romanized Hinglish, where the transliterator
        has already converted the danda to a period in most cases."""
        parts = re.split(r'(?<=[\.\?\!।])\s+', text.strip())
        return [s.strip() for s in parts if s.strip()]

    def _starts_new_scene(self, sentence: str) -> bool:
        first_words = ' '.join(sentence.lower().split()[:3])
        return any(
            first_words == cue or first_words.startswith(cue + ' ')
            for cue in self.BOUNDARY_CUES
        )

    def _group_sentences(self, text: str) -> List[str]:
        sents = self._split_sentences(text)
        if not sents:
            return []
        scenes, cur, cur_words = [], [], 0
        for s in sents:
            boundary = self._starts_new_scene(s)
            over_cap = len(cur) >= self.MAX_SENTENCES or cur_words >= self.MAX_WORDS
            if cur and (boundary or over_cap):
                scenes.append(' '.join(cur))
                cur, cur_words = [], 0
            cur.append(s)
            cur_words += len(s.split())
        if cur:
            scenes.append(' '.join(cur))
        return scenes


if __name__ == "__main__":
    chunker = ProseChunker()
    print("--- blank-line input ---")
    a = "subah ram ghar se nikla.\n\nphir woh school gaya aur apne dost se mila."
    for sc in chunker.chunk(a):
        print(sc["scene_id"], ":", sc["text"])

    print("--- single-block input (no blank lines) ---")
    b = ("ram subah uthta hai aur chai peeta hai. woh apne dost ko phone karta hai. "
         "agle din woh school jaata hai aur exam deta hai. sham ko woh ghar wapas "
         "aata hai aur thak kar so jaata hai.")
    for sc in chunker.chunk(b):
        print(sc["scene_id"], ":", sc["text"])
