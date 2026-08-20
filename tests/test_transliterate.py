import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.transliterate import HindiTransliterator


def test_basic_devanagari():
    t = HindiTransliterator()
    result = t.transliterate_sentence("तुम क्या कर रहे हो?")
    assert "tum" in result.lower()
    assert "kya" in result.lower()


def test_noise_stripping():
    t = HindiTransliterator()
    result = t.transliterate_sentence("नमस्ते! ##00:01:30##")
    assert "##" not in result


def test_returns_nonempty_for_valid_input():
    t = HindiTransliterator()
    result = t.transliterate_sentence("मैं ठीक हूँ।")
    assert len(result.strip()) > 0


def test_multiple_sentences():
    t = HindiTransliterator()
    result = t.transliterate_sentence("राम घर गया। वह खुश था।")
    # Should produce output covering both sentences, joined with a space
    assert len(result.split()) >= 4