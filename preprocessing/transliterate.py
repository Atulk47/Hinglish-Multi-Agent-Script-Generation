import re
from indicnlp import common

import config
common.set_resources_path(config.INDIC_NLP_RESOURCES)

from indicnlp.transliterate import unicode_transliterate as ut
from indicnlp.transliterate.unicode_transliterate import ItransTransliterator
from indicnlp.tokenize import sentence_tokenize

ut.init() 

class HindiTransliterator:
    # Applied to ITRANS output BEFORE word-boundary schwa deletion.
    # Order matters: longer/more specific patterns first.
    ITRANS_TO_CASUAL = [
        (r'\.m', 'n'),      # anusvara ं  -> n   (mai.m -> main)
        (r'ँ', 'n'),         # chandrabindu (raw Devanagari leak) -> n
        (r'aa', 'a'),       # long aa -> a       (kyaa -> kya)
        (r'ii', 'i'),       # long ii -> i       (Thiika -> Thika)
        (r'uu', 'u'),       # long uu -> u       (huu -> hu)
        (r'\.a', ''),       # stray schwa marker, if any
    ]

    # Casing fixes: ITRANS uses capitals for retroflex/aspirated consonants
    # (Th, Dh, Ph, Kh, Gh, Ch, Jh, etc.) - lowercase them for casual Hinglish
    RETROFLEX_LOWER = ['Th', 'Dh', 'Ph', 'Kh', 'Gh', 'Ch', 'Jh', 'Bh', 'Rh',
                        'T', 'D', 'N', 'R', 'S', 'L']

    SCHWA_DELETION_RULES = {
        'mujhko': 'mujhe',
        'tumko': 'tumhe',
        'kuchh': 'kuch',
        'yahaan': 'yahan',
        'wahaan': 'wahan',
        'kahaan': 'kahan',
    }

    def transliterate_sentence(self, devanagari_text: str) -> str:
        sentences = sentence_tokenize.sentence_split(devanagari_text, lang='hi')

        romanized = []
        for sent in sentences:
            if not sent.strip():
                continue

            # Step 2: Devanagari -> ITRANS roman scheme
            roman = ItransTransliterator.to_itrans(sent, 'hi')

            # Step 3: ITRANS conventions -> casual romanization
            roman = self._itrans_to_casual(roman)

            # Step 4: terminal schwa deletion (drop trailing 'a' after consonants)
            roman = self._apply_schwa_deletion(roman)

            # Step 5: phonetic smoothing (specific word fixes)
            roman = self._phonetic_smooth(roman)

            # Step 6: lowercase, strip noise, preserve punctuation
            roman = roman.lower()
            roman = re.sub(r'[^\w\s\.\,\!\?\;\:\'\-]', '', roman)
            roman = re.sub(r'\s+', ' ', roman).strip()

            if roman:
                romanized.append(roman)

        return ' '.join(romanized)

    def _itrans_to_casual(self, text: str) -> str:
        for pattern, repl in self.ITRANS_TO_CASUAL:
            text = re.sub(pattern, repl, text)
        # Lowercase remaining retroflex/capital consonant markers
        for cap in self.RETROFLEX_LOWER:
            text = re.sub(cap, cap.lower(), text)
        return text
    
    SCHWA_EXCEPTIONS = {
        # existing ones...
    'kya', 'tha', 'kaha', 'yaha', 'vaha', 'jaha',

    # Add these now based on real output:
    'gaya', 'kiya', 'liya', 'diya', 'piya', 'siya',   # common past-tense verbs ending in ya
    'likha', 'dekha', 'socha', 'roka', 'khola',        # past tense verbs ending in a
    'hota', 'jata', 'aata', 'rehta', 'chahta',         # habitual verb forms
    'uska', 'unka', 'iska', 'inka', 'apna', 'apni',    # possessives
    'pura', 'sara', 'kara', 'para', 'mara', 'tera',    # adjectives/pronouns ending in a
    'accha', 'achha', 'bura', 'naya', 'pura',          # common adjectives
    'wala', 'wali', 'wale',                             # suffix forms
    'pahala', 'pahle', 'agala', 'agale',                # ordinals/sequence words
    'vaha', 'yaha', 'jaha', 'kaha',
    }

    def _apply_schwa_deletion(self, text: str) -> str:
        words = text.split(' ')
        result = []
        for word in words:
            # Separate trailing punctuation so the check/regex works on the word itself
            m = re.match(r'^(.*?)([\.\,\!\?\;\:]*)$', word)
            core, punct = m.group(1), m.group(2)

            if core.lower() in self.SCHWA_EXCEPTIONS:
                result.append(core + punct)
                continue

            new_core = re.sub(r'^([a-zA-Z]*[bcdfghjklmnpqrstvwxyz])a$', r'\1', core)
            result.append(new_core + punct)
        return ' '.join(result)

    def _phonetic_smooth(self, text: str) -> str:
        for wrong, correct in self.SCHWA_DELETION_RULES.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
        return text


if __name__ == "__main__":
    t = HindiTransliterator()
    sample = "तुम क्या कर रहे हो? मैं ठीक हूँ।"
    print(t.transliterate_sentence(sample))