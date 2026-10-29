"""
llm_client.py - one place for every CLOUD LLM call in the pipeline (Agents 1 and 3, intent classifier, the slang
editor's tone / shortlist calls).

* OpenRouter first (pay-per-token, no tight daily caps), Groq as the automatic fallback (or the other way round with
  LLM_BACKEND=groq). If one backend fails (429 / 5xx / timeout / 402 out of credit / bad request) the call is retried
  briefly and then handed to the other backend, so one rate limit can no longer kill a run.
* Model per TASK (see config.OPENROUTER_MODELS):  extract -> gpt-oss-120b, dialogue -> Claude Haiku 5.5,
  small (intent / tone / shortlist) -> gpt-oss-20b. Reasoning is kept minimal (gpt-oss effort=low, Haiku reasoning off)
  because reasoning tokens are billed as output.
* Exposes the SAME interface the code already used for Groq:  client.chat.completions.create(...) ->
  resp.choices[0].message.content   (the `model=` / `reasoning_effort=` arguments are accepted and ignored),
  so call sites only change how the client is constructed.
* Tracks real cost from OpenRouter's usage report and refuses to spend more than config.LLM_COST_CAP_USD per process
  (then it falls back to Groq).  print(llm_client.cost_summary()) shows the totals.
"""
from __future__ import annotations
import json
import os
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import config

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_LOCK = threading.Lock()
STATS = {"calls": 0, "or_calls": 0, "groq_calls": 0, "fallbacks": 0, "errors": 0, "in": 0, "out": 0,
         "cost": 0.0, "by_model": {}}
_STATE = {"or_disabled": False, "or_disabled_why": "", "announced": False, "last_warn": 0.0, "groq": None}


class LLMError(RuntimeError):
    pass


def _warn(msg: str):
    now = time.time()
    if now - _STATE["last_warn"] > 15:                 # at most one warning per 15 s
        _STATE["last_warn"] = now
        print("  [llm] " + msg, flush=True)


def _task_model(task: str) -> str:
    return getattr(config, "OPENROUTER_MODELS", {}).get(task) or "openai/gpt-oss-20b"


def _extras(model: str) -> dict:
    if model.startswith("openai/gpt-oss"):
        return {"reasoning": {"effort": "low"}}
    if model.startswith("anthropic/"):
        return {"reasoning": {"enabled": False}}
    return {}


def _backend_order() -> list:
    mode = str(getattr(config, "LLM_BACKEND", "auto")).lower()
    has_or = bool(os.environ.get("OPENROUTER_API_KEY")) and not _STATE["or_disabled"]
    cap = float(getattr(config, "LLM_COST_CAP_USD", 0.50))
    if STATS["cost"] >= cap:
        has_or = False
    has_gq = bool(os.environ.get("GROQ_API_KEY"))
    if mode == "groq":
        order = ["groq", "openrouter"]
    else:
        order = ["openrouter", "groq"]
    return [b for b in order if (b == "openrouter" and has_or) or (b == "groq" and has_gq)]


def _record(model: str, usage: dict):
    with _LOCK:
        STATS["calls"] += 1
        STATS["in"] += int(usage.get("prompt_tokens") or 0)
        STATS["out"] += int(usage.get("completion_tokens") or 0)
        c = float(usage.get("cost") or 0.0)
        STATS["cost"] += c
        m = STATS["by_model"].setdefault(model, {"calls": 0, "cost": 0.0})
        m["calls"] += 1
        m["cost"] += c


def _openrouter(task: str, messages, temperature: float, max_tokens: int):
    model = _task_model(task)
    mt = int(max_tokens or 500)
    if model.startswith("openai/gpt-oss"):
        mt = max(mt, 400)                              # reasoning tokens count against max_tokens
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": mt,
            "usage": {"include": True}}
    body.update(_extras(model))
    sort = getattr(config, "OPENROUTER_PROVIDER_SORT", "throughput")
    if sort:                                           # measured: gpt-oss-120b 11.3 s -> 1.2 s per call (Cerebras vs CoreWeave)
        body["provider"] = {"sort": sort}
    last = None
    for attempt in range(3):
        req = urllib.request.Request(
            OPENROUTER_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/Atulk47/Hinglish-Multi-Agent-Script-Generation",
                     "X-Title": "Hinglish MAS"})
        try:
            with urllib.request.urlopen(req, timeout=75) as r:
                out = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            txt = ""
            try:
                txt = e.read().decode("utf-8")[:240]
            except Exception:
                pass
            last = f"OpenRouter HTTP {e.code} {txt}"
            if e.code == 402:
                _STATE["or_disabled"], _STATE["or_disabled_why"] = True, "out of credit / key limit (402)"
                raise LLMError(last)
            if e.code in (429, 500, 502, 503, 504):
                try:
                    wait = min(8.0, float(e.headers.get("Retry-After", "0") or 0))
                except Exception:
                    wait = 0.0
                time.sleep(max(wait, 1.5 * (attempt + 1)))
                continue
            raise LLMError(last)                       # 400 / 401 / 403 / 404: not worth retrying
        except Exception as e:                         # timeout, connection reset ...
            last = f"OpenRouter {type(e).__name__}: {str(e)[:120]}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if "error" in out and "choices" not in out:
            last = "OpenRouter error: " + json.dumps(out["error"])[:240]
            code = (out["error"] or {}).get("code")
            if code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise LLMError(last)
        ch = out["choices"][0]
        content = (ch.get("message") or {}).get("content") or ""
        _record(model, out.get("usage") or {})
        if not content.strip() and ch.get("finish_reason") == "length" and attempt < 2:
            body["max_tokens"] = int(body["max_tokens"] * 2)       # reasoning ate the budget: one bigger retry
            continue
        with _LOCK:
            STATS["or_calls"] += 1
        return content, ch.get("finish_reason") or "stop"
    raise LLMError(last or "OpenRouter failed")


def _groq(task: str, messages, temperature: float, max_tokens: int):
    from groq import Groq
    if _STATE["groq"] is None:
        _STATE["groq"] = Groq(api_key=os.environ["GROQ_API_KEY"])
    last = None
    for attempt in range(2):
        try:
            r = _STATE["groq"].chat.completions.create(
                model=config.GROQ_MODEL, messages=messages, temperature=temperature, max_tokens=max(int(max_tokens or 500), 300),
                reasoning_effort=getattr(config, "GROQ_REASONING_EFFORT", "low"))
            with _LOCK:
                STATS["calls"] += 1
                STATS["groq_calls"] += 1
            return (r.choices[0].message.content or ""), (r.choices[0].finish_reason or "stop")
        except Exception as e:
            last = f"Groq: {str(e)[:200]}"
            if "429" in last or "rate limit" in last.lower():
                break                                   # a Groq cap will not clear in 2 s
            time.sleep(2)
    raise LLMError(last or "Groq failed")


def complete(task: str, messages, temperature: float = 0.0, max_tokens: int = 500):
    """-> (text, finish_reason). Tries the backends in order; raises LLMError (message includes the status text) if all fail."""
    order = _backend_order()
    if not order:
        raise LLMError("no LLM backend available (set OPENROUTER_API_KEY or GROQ_API_KEY)")
    if not _STATE["announced"]:
        _STATE["announced"] = True
        print(f"  [llm] backends in order: {order} | models: {getattr(config, 'OPENROUTER_MODELS', {})}", flush=True)
    errors = []
    for i, b in enumerate(order):
        try:
            res = _openrouter(task, messages, temperature, max_tokens) if b == "openrouter" else _groq(task, messages, temperature, max_tokens)
            if i > 0:
                with _LOCK:
                    STATS["fallbacks"] += 1
            return res
        except LLMError as e:
            with _LOCK:
                STATS["errors"] += 1
            errors.append(str(e))
            _warn(f"{b} failed ({str(e)[:110]}); " + ("trying next backend" if i + 1 < len(order) else "no backend left"))
    raise LLMError(" | ".join(errors))


class ChatClient:
    """Drop-in for the Groq client: client.chat.completions.create(...) -> resp.choices[0].message.content"""

    def __init__(self, task: str):
        self.task = task
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model=None, messages=None, temperature=0.0, max_tokens=500, reasoning_effort=None, **_ignored):
        text, fin = complete(self.task, messages, temperature, max_tokens)
        msg = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=fin)])


def make_client(task: str) -> ChatClient:
    return ChatClient(task)


def cost_summary() -> str:
    s = STATS
    parts = ", ".join(f"{m.split('/')[-1]}: {v['calls']} calls ${v['cost']:.4f}" for m, v in s["by_model"].items())
    return (f"LLM usage: {s['calls']} calls ({s['or_calls']} OpenRouter, {s['groq_calls']} Groq, {s['fallbacks']} fallbacks, "
            f"{s['errors']} backend errors) | {s['in']} in / {s['out']} out tokens | OpenRouter cost ${s['cost']:.4f} "
            f"(cap ${float(getattr(config, 'LLM_COST_CAP_USD', 0.5)):.2f}) | {parts}")
