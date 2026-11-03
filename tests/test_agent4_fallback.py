"""Regression tests for Agent 4 backend selection and graceful fallback.

These tests stub model loading and network clients: they run quickly, do not
need model weights or API keys, and protect the requested-vs-actual backend
contract used by experiment manifests.
"""
from unittest.mock import patch

from agents.agent4_slang_rewriter import SlangRewriter


def _base_patches():
    return [
        patch.object(SlangRewriter, "_load_lexicon", return_value=({}, {}, [])),
        patch.object(SlangRewriter, "_compute_negative_slang", return_value=set()),
        patch.object(SlangRewriter, "_compute_positive_slang", return_value=set()),
    ]


def test_auto_backend_prefers_qwen_then_mt5_then_groq():
    editor = SlangRewriter.__new__(SlangRewriter)
    with patch("agents.agent4_slang_rewriter.config.QWEN_LORA_PATH", "qwen-adapter"), \
         patch("agents.agent4_slang_rewriter.config.EDITOR_FULL_PATH", "mt5-full"), \
         patch("agents.agent4_slang_rewriter.config.EDITOR_ADAPTER_PATH", "mt5-adapter"), \
         patch("agents.agent4_slang_rewriter.os.path.isdir",
               side_effect=lambda path: path in {"qwen-adapter", "mt5-full", "mt5-adapter"}):
        assert editor._resolve_backend("auto") == "qwen"
        with patch("agents.agent4_slang_rewriter.os.path.isdir",
                   side_effect=lambda path: path in {"mt5-full", "mt5-adapter"}):
            assert editor._resolve_backend("auto") == "mt5"
        with patch("agents.agent4_slang_rewriter.os.path.isdir", return_value=False):
            assert editor._resolve_backend("auto") == "groq"


def test_qwen_load_failure_falls_back_to_groq_with_reason():
    client = object()
    patches = _base_patches()
    with patches[0], patches[1], patches[2], \
         patch.object(SlangRewriter, "_resolve_backend", return_value="qwen"), \
         patch.object(SlangRewriter, "_init_qwen", side_effect=FileNotFoundError("adapter weights missing")), \
         patch("llm_client.make_client", return_value=client) as make_client:
        editor = SlangRewriter(object(), object(), backend="qwen")

    assert editor.requested_backend == "qwen"
    assert editor.backend == "groq"
    assert editor.editor_fallback is True
    assert "adapter weights missing" in editor.fallback_reason
    assert editor.groq_client is client
    make_client.assert_called_once_with("small")


def test_qwen_smoke_test_failure_also_falls_back():
    client = object()
    patches = _base_patches()
    with patches[0], patches[1], patches[2], \
         patch.object(SlangRewriter, "_resolve_backend", return_value="qwen"), \
         patch.object(SlangRewriter, "_init_qwen"), \
         patch.object(SlangRewriter, "_smoke_test_qwen", side_effect=RuntimeError("no generated tokens")), \
         patch("llm_client.make_client", return_value=client):
        editor = SlangRewriter(object(), object(), backend="qwen")

    assert editor.backend == "groq"
    assert editor.editor_fallback is True
    assert "no generated tokens" in editor.fallback_reason
    assert editor.groq_client is client


def test_mt5_load_failure_falls_back_and_records_reason():
    client = object()
    patches = _base_patches()
    with patches[0], patches[1], patches[2], \
         patch.object(SlangRewriter, "_resolve_backend", return_value="mt5"), \
         patch.object(SlangRewriter, "_init_mt5", side_effect=OSError("incompatible checkpoint")), \
         patch("llm_client.make_client", return_value=client) as make_client:
        editor = SlangRewriter(object(), object(), backend="mt5")

    assert editor.requested_backend == "mt5"
    assert editor.backend == "groq"
    assert editor.editor_fallback is True
    assert "incompatible checkpoint" in editor.fallback_reason
    assert editor.groq_client is client
    make_client.assert_called_once_with("small")


def test_successful_qwen_does_not_get_marked_as_fallback():
    patches = _base_patches()
    with patches[0], patches[1], patches[2], \
         patch.object(SlangRewriter, "_resolve_backend", return_value="qwen"), \
         patch.object(SlangRewriter, "_init_qwen"), \
         patch.object(SlangRewriter, "_smoke_test_qwen"), \
         patch("agents.agent4_slang_rewriter.config.QWEN_GROQ_FALLBACK", False), \
         patch("agents.agent4_slang_rewriter.config.QWEN_JUDGE", False):
        editor = SlangRewriter(object(), object(), backend="qwen")

    assert editor.requested_backend == "qwen"
    assert editor.backend == "qwen"
    assert editor.editor_fallback is False
    assert editor.fallback_reason is None
