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
        (r'aa(?![A-Za-z.])', '\u00a7'),   # word-FINAL long aa is protected (marker) so schwa deletion cannot eat it:
                                          # riyaa -> riya, sharmaa -> sharma, meeraa -> mira, raadhaa -> radha
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

    # Nukta letters (ज़ फ़ ख़ ग़ क़ ड़ ढ़) are written either as ONE precomposed code point (U+0958-095F) or as base + U+093C. The ITRANS converter
    # drops the first form ("ज़ोया" -> "oya") and adds a stray vowel to the second ("फ़िर" -> "phair", "ज़ोया" -> "jaoya"), so both are rewritten
    # to their usual casual spelling BEFORE conversion (z, f, kh, g, q, d, dh).
    _NUKTA_PRECOMPOSED = {"क़": "क़", "ख़": "ख़", "ग़": "ग़", "ज़": "ज़",
                          "ड़": "ड़", "ढ़": "ढ़", "फ़": "फ़", "य़": "य़"}
    _NUKTA_LATIN = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f", "य": "y"}

    _MATRA_ITRANS = {"ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "े": "e", "ै": "ai", "ो": "o", "ौ": "au"}

    @classmethod
    def _normalize_nukta(cls, text: str) -> str:
        for k, v in cls._NUKTA_PRECOMPOSED.items():
            text = text.replace(k, v)

        def repl(m):
            latin, tail = cls._NUKTA_LATIN[m.group(1)], m.group(2)
            if tail == "्":                 # conjunct (ज़्यादा): the consonant without its inherent vowel
                return latin
            if tail in cls._MATRA_ITRANS:         # the converter loses an ASCII letter that is followed by a vowel sign: write the vowel in Latin too
                return latin + cls._MATRA_ITRANS[tail]
            return latin + "a" + tail             # inherent 'a'
        return re.sub("([कखगजडढफय])़([ा-्ॢॣ]?)", repl, text)

    # English words written in Devanagari whose spelling the generic converter cannot recover (they contain the candra-o vowel, see _normalize_candra). Whole words only.
    _LOANWORDS = {"वॉर्डन": "warden", "वार्डन": "warden", "कॉलेज": "college", "हॉस्टल": "hostel", "डॉक्टर": "doctor", "ऑफिस": "office", "ऑफ़िस": "office",
                  "लैपटॉप": "laptop", "ऑर्डर": "order", "बॉलर": "bowler", "रिकॉर्डिंग": "recording", "रिकॉर्ड": "record", "कॉल": "call", "फ़ॉर्म": "form",
                  "ऑनलाइन": "online", "ऑटो": "auto", "बॉस": "boss", "शॉपिंग": "shopping", "मॉल": "mall", "कॉफ़ी": "coffee", "कॉफी": "coffee",
                  "फ़ुटबॉल": "football", "फुटबॉल": "football", "बास्केटबॉल": "basketball", "वॉलीबॉल": "volleyball", "हॉल": "hall", "स्कॉलरशिप": "scholarship",
                  "कॉमन": "common", "रूम": "room", "कॉम्पिटिशन": "competition", "शॉर्ट": "short", "शॉट": "shot", "ऑडिटोरियम": "auditorium", "ऑडिशन": "audition",
                  "कॉन्फ़िडेंस": "confidence", "कॉन्फिडेंस": "confidence", "ऑफ़र": "offer", "ऑफर": "offer", "स्टॉप": "stop", "बॉक्स": "box", "कॉन्सर्ट": "concert"}

    @classmethod
    def _normalize_candra(cls, text: str) -> str:
        """The candra vowels of English loanwords (ॉ ऑ ॅ ऍ: doctor, college, hostel, office, laptop) were DROPPED by the ITRANS converter ("डॉक्टर" -> "dktara", "कॉलेज" -> "kleja") or
        leaked as raw Devanagari ("ऑफिस" -> "ऑphisa"). Whole known loanwords become their English spelling; any other candra vowel is read as the plain o / e."""
        for w, en in cls._LOANWORDS.items():
            text = re.sub(r"(?<![\u0900-\u097F])" + re.escape(w) + r"(?![\u0900-\u097F])", en, text)
        return text.replace("\u0949", "\u094b").replace("\u0911", "\u0913").replace("\u0945", "\u0947").replace("\u090d", "\u090f")

    def transliterate_sentence(self, devanagari_text: str) -> str:
        devanagari_text = self._normalize_candra(devanagari_text)
        devanagari_text = self._normalize_nukta(devanagari_text)
        devanagari_text = re.sub(r'(?<![\u0900-\u097F])\u0938\u0930(?![\u0900-\u097F])', '\u0938\u0930\u094d', devanagari_text)  # "सर" -> sar
        devanagari_text = re.sub(r'(?<![ऀ-ॿ])मा(?:ँ|ं)(?![ऀ-ॿ])', 'mummy', devanagari_text)  # "माँ" -> mummy (not "man"); also matches KINSHIP_NAMES
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
            roman = roman.replace('\u00a7', 'a')          # restore the protected word-final long aa

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
            m = re.match(r'^(["\'“‘(]*)(.*?)([\.\,\!\?\;\:"\'”’)]*)$', word, re.DOTALL)
            lead, core, punct = m.group(1), m.group(2), m.group(3)   # a leading quote must not hide the word

            if core.lower() in self.SCHWA_EXCEPTIONS:
                result.append(lead + core + punct)
                continue

            new_core = re.sub(r'^([a-zA-Z]*[bcdfghjklmnpqrstvwxyz])a$', r'\1', core)
            result.append(lead + new_core + punct)
        return ' '.join(result)

    def _phonetic_smooth(self, text: str) -> str:
        for wrong, correct in self.SCHWA_DELETION_RULES.items():
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
        return text


if __name__ == "__main__":
    t = HindiTransliterator()
    sample = "तुम क्या कर रहे हो? मैं ठीक हूँ।"
    print(t.transliterate_sentence(sample))