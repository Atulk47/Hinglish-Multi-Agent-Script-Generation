import re
from typing import List


class ProseChunker:
    # Boundary signals in transliterated Hinglish
    LOCATION_CUES = ['ghar', 'school', 'bazaar', 'bahar', 'andar', 'office',
                      'kamra', 'kamre', 'gali', 'station', 'park']
    TEMPORAL_CUES = ['subah', 'sham', 'raat', 'dopahar', 'agli baar', 'phir',
                      'kal', 'aaj', 'parso', 'thodi der baad']

    def chunk(self, romanized_text: str) -> List[dict]:
        """Split into scene-level chunks. Returns list of {scene_id, text, has_location_cue, has_temporal_cue}"""
        # Strategy: Split on double newlines first (paragraph boundaries)
        paragraphs = [p.strip() for p in romanized_text.split('\n\n') if p.strip()]

        # Fallback: if no double-newlines exist (single block of text),
        # treat the whole thing as one paragraph
        if not paragraphs and romanized_text.strip():
            paragraphs = [romanized_text.strip()]

        scenes = []
        for i, para in enumerate(paragraphs):
            scenes.append({
                "scene_id": f"scene_{i+1:02d}",
                "text": para,
                "has_location_cue": any(cue in para.lower() for cue in self.LOCATION_CUES),
                "has_temporal_cue": any(cue in para.lower() for cue in self.TEMPORAL_CUES)
            })
        return scenes


if __name__ == "__main__":
    sample = "subah ram ghar se nikla.\n\nphir woh school gaya aur apne dost se mila."
    chunker = ProseChunker()
    for scene in chunker.chunk(sample):
        print(scene)