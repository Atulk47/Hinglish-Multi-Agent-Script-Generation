"""The candra-o vowel of English loanwords (doctor, college, hostel, office, laptop, warden) used to be dropped ("dktara", "kleja") or leak as raw Devanagari ("ऑphisa");
found when the demo story's "वॉर्डन शर्मा सर" came out as the first name "Wardhan Sharma"."""
import re

import pytest

from preprocessing.transliterate import HindiTransliterator

T = HindiTransliterator()


@pytest.mark.parametrize("deva,roman", [
    ("वॉर्डन शर्मा सर", "warden sharma sar"), ("वार्डन", "warden"), ("डॉक्टर", "doctor"), ("कॉलेज के हॉस्टल में", "college ke hostel men"),
    ("ऑफिस", "office"), ("ऑफ़िस", "office"), ("लैपटॉप", "laptop"), ("ऑर्डर", "order"), ("बॉलर", "bowler"), ("फ़ुटबॉल", "football"),
])
def test_known_loanwords_get_their_english_spelling(deva, roman):
    assert T.transliterate_sentence(deva) == roman


@pytest.mark.parametrize("deva", ["कॉमेडी", "ऑपरेशन", "स्टॉप", "शॉट", "बॉक्स", "ऑटोग्राफ़", "कॉन्सर्ट", "ऍप", "बॅट"])
def test_any_other_candra_vowel_is_read_as_a_plain_vowel_and_never_leaks_devanagari(deva):
    out = T.transliterate_sentence(deva)
    assert out and not re.search(r"[ऀ-ॿ]", out) and re.fullmatch(r"[a-z .'-]+", out)


def test_a_word_that_merely_contains_a_loanword_is_not_rewritten_and_plain_hindi_is_unchanged():
    assert T.transliterate_sentence("कॉलेजों") != "colleges" and "college" not in T.transliterate_sentence("कॉलेजों").split()
    plain = "रोहन ने आर्यन को बताया। वह कैंटीन में चाय पी रहा था।"
    assert HindiTransliterator._normalize_candra(plain) == plain                     # the new step does not touch text without a candra vowel
