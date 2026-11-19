"""
evaluation/fluency.py  -  FLUENCY CHECK for slang edits: did the edit make the sentence less natural?

Rules and embeddings can verify WHAT changed but not whether the result still reads like a sentence a person would say
("jo ek sahi hai sanket hai", "mujhe thodi full tension ho rahi hai").  A language model can: score both versions of the line by their
average negative log-likelihood under the local GGUF model (llama.cpp, the same file the editor uses, no network, ~0.2 s per line) and compare.

    delta = nll_per_token(edited) - nll_per_token(neutral)       (>0: the edit made the line LESS likely)

The score is a SOFT signal (a rare but fine slang word also raises the NLL): it is used to rank / warn, and to block only when
it is far beyond what good edits show (threshold calibrated on judged edits by evaluation/eval_validator.py).
"""
from __future__ import annotations
import math
import os
from typing import Dict, List, Optional

import numpy as np


class FluencyScorer:
    def __init__(self, gguf_path: Optional[str] = None, n_threads: Optional[int] = None):
        import config
        from llama_cpp import Llama
        path = gguf_path or getattr(config, "QWEN_GGUF_PATH", "./models/gguf/slang-editor-v6-Q5_K_M.gguf")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.llm = Llama(model_path=path, n_ctx=160, n_batch=160, logits_all=True, verbose=False,
                         n_threads=n_threads or max(4, (os.cpu_count() or 8) - 2), use_mmap=True)
        self._cache: Dict[str, float] = {}

    def nll(self, text: str) -> float:
        """Mean negative log-likelihood per token of `text` (nats); lower = more natural."""
        text = (text or "").strip()
        if text in self._cache:
            return self._cache[text]
        toks = self.llm.tokenize(text.encode("utf-8"), add_bos=True)[:150]
        if len(toks) < 3:
            return 0.0
        self.llm.reset()
        self.llm.eval(toks)
        logits = np.asarray(self.llm.scores[:len(toks)], dtype=np.float32)          # [n, vocab]
        m = logits.max(axis=1, keepdims=True)
        logp = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
        v = float(-np.mean([logp[i, toks[i + 1]] for i in range(len(toks) - 1)]))
        self._cache[text] = v
        return v

    def delta(self, neutral: str, edited: str) -> float:
        if (neutral or "").strip() == (edited or "").strip():
            return 0.0
        return self.nll(edited) - self.nll(neutral)
