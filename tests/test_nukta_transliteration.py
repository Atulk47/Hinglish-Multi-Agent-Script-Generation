"""Nukta letters (ज़ फ़ ख़ ग़ क़ ड़ ढ़) must romanize the same way whether typed precomposed (U+0958-095F) or as base + U+093C."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from preprocessing.transliterate import HindiTransliterator

T = HindiTransliterator()
_PRE = {"क़": "क़", "ख़": "ख़", "ग़": "ग़", "ज़": "ज़", "ड़": "ड़",
        "ढ़": "ढ़", "फ़": "फ़"}


def _precomposed(w):
    for k, v in _PRE.items():
        w = w.replace(k, v)
    return w


def _rom(w):
    return T.transliterate_sentence("वह " + w + " है।").replace("vaha ", "").replace(" hai.", "")


@pytest.mark.parametrize("word,expected", [("ज़ोया", "zoya"), ("फ़िर", "fir"), ("ज़रूर", "zarur"), ("ज़्यादा", "zyada"), ("ख़ुशी", "khushi"),
                                           ("ग़ुस्सा", "gussa"), ("क़रीब", "qarib"), ("फ़ोन", "fon"), ("ज़िंदगी", "zindagi"), ("बड़ा", "bada"),
                                           ("आख़िर", "akhir"), ("ग़लत", "galat")])
def test_nukta_words_romanize_correctly_in_both_unicode_forms(word, expected):
    assert _rom(word) == expected and _rom(_precomposed(word)) == expected


def test_plain_words_are_unchanged():
    assert _rom("जया") == "jaya" and _rom("दोस्त") == "dost"
