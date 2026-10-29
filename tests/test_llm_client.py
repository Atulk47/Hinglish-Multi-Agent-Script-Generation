"""
tests/test_llm_client.py - failover / budget logic of llm_client.py with fake backends (no network, no cost).
    python -m pytest tests/test_llm_client.py -q
"""
import os

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-test")
os.environ.setdefault("GROQ_API_KEY", "gsk_test")
import config
import llm_client as L


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    L._STATE.update({"or_disabled": False, "or_disabled_why": "", "announced": True, "last_warn": 0.0})
    for k in ("calls", "or_calls", "groq_calls", "fallbacks", "errors", "in", "out"):
        L.STATS[k] = 0
    L.STATS["cost"] = 0.0
    L.STATS["by_model"] = {}
    monkeypatch.setattr(config, "LLM_BACKEND", "auto", raising=False)
    monkeypatch.setattr(config, "LLM_COST_CAP_USD", 0.5, raising=False)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)


MSG = [{"role": "user", "content": "hi"}]


def test_openrouter_first_and_cost_tracked(monkeypatch):
    def ok(task, messages, temperature, max_tokens):
        L._record("openai/gpt-oss-20b", {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001})
        return "OR-answer", "stop"
    monkeypatch.setattr(L, "_openrouter", ok)
    monkeypatch.setattr(L, "_groq", lambda *a, **k: pytest.fail("Groq must not be called when OpenRouter works"))
    r = L.make_client("small").chat.completions.create(model="ignored", messages=MSG, reasoning_effort="low")
    assert r.choices[0].message.content == "OR-answer"
    assert L.STATS["cost"] == pytest.approx(0.001)


def test_falls_back_to_groq_on_rate_limit(monkeypatch):
    monkeypatch.setattr(L, "_openrouter", lambda *a, **k: (_ for _ in ()).throw(L.LLMError("OpenRouter HTTP 429 slow down")))
    monkeypatch.setattr(L, "_groq", lambda *a, **k: ("groq-answer", "stop"))
    r = L.make_client("dialogue").chat.completions.create(messages=MSG)
    assert r.choices[0].message.content == "groq-answer"
    assert L.STATS["fallbacks"] == 1


def test_groq_first_mode(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "groq", raising=False)
    monkeypatch.setattr(L, "_groq", lambda *a, **k: ("g", "stop"))
    monkeypatch.setattr(L, "_openrouter", lambda *a, **k: pytest.fail("OpenRouter must not be called first in groq mode"))
    assert L.make_client("small").chat.completions.create(messages=MSG).choices[0].message.content == "g"


def test_cost_cap_skips_openrouter(monkeypatch):
    L.STATS["cost"] = 0.51
    monkeypatch.setattr(L, "_openrouter", lambda *a, **k: pytest.fail("cap exceeded -> OpenRouter must be skipped"))
    monkeypatch.setattr(L, "_groq", lambda *a, **k: ("g", "stop"))
    assert L.make_client("small").chat.completions.create(messages=MSG).choices[0].message.content == "g"


def test_all_backends_fail_raises_with_status_text(monkeypatch):
    monkeypatch.setattr(L, "_openrouter", lambda *a, **k: (_ for _ in ()).throw(L.LLMError("OpenRouter HTTP 402 credit")))
    monkeypatch.setattr(L, "_groq", lambda *a, **k: (_ for _ in ()).throw(L.LLMError("Groq: 429 rate limit")))
    with pytest.raises(L.LLMError) as e:
        L.make_client("small").chat.completions.create(messages=MSG)
    assert "429" in str(e.value) and "402" in str(e.value)       # callers string-match on 429 / rate limit


def test_model_params():
    assert L._extras("openai/gpt-oss-20b") == {"reasoning": {"effort": "low"}}
    assert L._extras("anthropic/claude-haiku-5.5") == {"reasoning": {"enabled": False}}
    assert L._task_model("dialogue").startswith("anthropic/")
